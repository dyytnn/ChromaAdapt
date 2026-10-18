import gc
import os
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from logging import Logger
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm

from abs_interface import AbsAlgorithmProcessHandler
from data import dataset as dataset_module
from data.transformation import build_transforms
from utils import c_, is_main_process, log_on_main_process, save_on_master, sr_

from ..log import StatAccumulator
from ..registers import (
    domain_fusing_training_segmentation_handler_register,
    training_segmentation_handler_register,
)


@domain_fusing_training_segmentation_handler_register.register("MultiHeadTrainGenerator")
@training_segmentation_handler_register.register("MultiHeadTrainGenerator")
class MultiHeadTrainGenerator(AbsAlgorithmProcessHandler):
    def __init__(self, config: dict):
        super(MultiHeadTrainGenerator, self).__init__()
        self.config = config

    def _pre_process(self, data: dict):
        is_train = data.get("is_train", None)
        
        num_workers = self.config.get("num_workers", 10) if is_train else 1
        
        for mode in ['train', 'val']:
            transformation = build_transforms(self.config.get("transformation")[mode])
            batch_size = self.config.get("batch_size")[mode]
            
            dataset_class = getattr(dataset_module, self.config.get("dataset"))
            
            dataset = dataset_class(image_dir=self.config.get(f"{mode}_data_dir", None), transformation=transformation, equalization_method_config = self.config.get("equalization_method"))
            
            print(f"{c_} [INFO] number of instances in {mode} dataset: {dataset.__len__()} {sr_}")
            
            sampler = RandomSampler(dataset) if os.environ.get("local_rank", -1) == -1 else DistributedSampler(dataset)

            dataloader = DataLoader(dataset, sampler = sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

            data.update(
                {
                    f"{mode}_dataloader": dataloader,
                    f"{mode}_dataset": dataset
                }
            )
    
        return data
    
    def _main_process(self, data: dict):
        train_logger = data.get("train_logger", None)
        val_logger = data.get("val_logger", None)
        grad_logger = data.get("grad_logger", None)
        stat_accumulator_instance = data.get("stat_accumulator_instance", None)
                
        last_model: nn.Module = data.get("last_model", None)
        
        start_epoch: int = data.get("start_epoch", 0)
        best_loss = data.get("best_loss", np.inf)
        
        train_loader = data.get("train_dataloader", None)
        val_loader = data.get("val_dataloader", None)
        optimizer = data.get("optimizer", None)
        scheduler = data.get("scheduler", None)
        fp16_scaler = data.get("fp16_scaler", None)
        criterion = data.get("criterion", None)
        
        history = defaultdict(list)
        
        for epoch in range(start_epoch, self.config.get("epochs")):
            gc.collect()
            print(f'Epoch {epoch}/{self.config.get("epochs")}', end='')
            
            train_loss = self.training_one_epoch(last_model, optimizer, scheduler, fp16_scaler, criterion, train_loader, epoch, train_logger, grad_logger, stat_accumulator_instance)
            
            valid_loss = self.valid_one_epoch(last_model, val_loader, criterion, epoch, val_logger)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(valid_loss)
            
            save_dict = {
                'last_model': last_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'start_epoch': epoch + 1,
                'loss': criterion.state_dict(),
                'best_loss': best_loss,  # Always save current best_loss
            }

            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()

            if valid_loss <= best_loss:
                log_on_main_process(f"{c_}Valid loss Improved ({best_loss} ---> {valid_loss}) {sr_}")
                best_loss = valid_loss
                best_model_wts = deepcopy(last_model.state_dict())
                save_dict["best_loss"] = best_loss  # Update with new best_loss
                if is_main_process():
                    torch.save(best_model_wts, os.path.join(data.get("save_path"), "best_model.pth"))
                    print(f"Best model saved at {os.path.join(data.get('save_path'), 'best_model.pth')}")

            if is_main_process():
                save_on_master(save_dict, os.path.join(data.get("save_path"), "last_save_dict.pth"))
                print(f"Last checkpoint saved at {os.path.join(data.get('save_path'), 'last_save_dict.pth')}")
                       
        data.update(
            {
                "save_dict": save_dict,
                "history": history
            }
        )
        
        if dist.is_initialized():
            dist.destroy_process_group()
        return data
    
    def training_one_epoch(
        self,
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler.LRScheduler, 
        fp16_scaler: torch.amp.GradScaler,
        criterion: nn.CrossEntropyLoss, 
        dataloader: DataLoader,
        epoch: int, 
        train_logger: Logger, 
        grad_logger: Logger, 
        stat_accumulator_instance: StatAccumulator
    ) -> dict:
        model.train()
        device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = criterion.to(device)
        dataset_size = 0
        running_loss = 0.0
        
        header = 'Epoch: [{}/{}]'.format(epoch, self.config.get("epochs"))
        if is_main_process():
            pbar: tqdm[Tuple[int | gc.Any]] = tqdm(enumerate(dataloader), total=len(dataloader), desc=header)
        else:
            pbar = enumerate(dataloader)
        
        for it, batches in pbar:
            images, segmentation_labels = batches
            
            images: Tensor = images.to(device)
            segmentation_labels: Tuple[Tensor] = tuple(t.to(device) for t in segmentation_labels)
            
            batch_size = images.shape[0]

            if it % self.config.get("gradient_accumulation_steps", 1) == 0:
                optimizer.zero_grad()

            autocast_context: torch.autocast | nullcontext[None] = (
                torch.amp.autocast(enabled=True, device_type='cuda') 
                if fp16_scaler else nullcontext()
            )
            
            with autocast_context:
                _, segmentation_pred = model(images)
                seg_loss = criterion(segmentation_pred, segmentation_labels)
                loss = seg_loss['total_loss'] / self.config.get("gradient_accumulation_steps", 1)

            if fp16_scaler:
                fp16_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (it + 1) % self.config.get("gradient_accumulation_steps", 1) == 0 or (it + 1) == len(dataloader):
                if fp16_scaler:
                    fp16_scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get("max_grad_norm", 1.0)) if self.config.get("max_grad_norm") else None

                if fp16_scaler:
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()
                else:
                    optimizer.step()

                if is_main_process():
                    scheduler.step()
                
            torch.cuda.synchronize()

            running_loss += (seg_loss['total_loss'].item() * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                mem_local = torch.cuda.max_memory_allocated(current_device) / 1E9

                if dist.is_initialized():
                    mem_tensor = torch.tensor(mem_local).cuda()
                    dist.all_reduce(mem_tensor, op=dist.ReduceOp.SUM)
                    mem = mem_tensor.item()
                else:
                    mem = mem_local
            else:
                mem = 0
            current_lr = optimizer.param_groups[0]['lr']

            del segmentation_pred, seg_loss, loss
            torch.cuda.empty_cache()

            if is_main_process() and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix( epoch=f'{epoch}',
                                train_loss=f'{epoch_loss:0.4f}',
                                lr=f'{current_lr:0.5f}',
                                gpu_mem=f'{mem:0.2f} GB')

        if dist.is_initialized():
            running_loss_tensor = torch.tensor(running_loss).cuda()
            dataset_size_tensor = torch.tensor(dataset_size).cuda()
            dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(dataset_size_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = running_loss_tensor.item() / dataset_size_tensor.item()

        if grad_logger and is_main_process():
            stat_accumulator_instance.log_epoch_summary(grad_logger, prefix=f"[Epoch {epoch}] ")
        if train_logger:
            rank = dist.get_rank() if dist.is_initialized() else 0
            num_gpus = dist.get_world_size() if dist.is_initialized() else 1
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}] - Train Loss: {epoch_loss:.4f} - lr: {current_lr:.5f} - Total GPU Mem ({num_gpus} GPUs): {mem:.2f} GB")

        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss

    @torch.no_grad()
    def valid_one_epoch(
        self,
        model: nn.Module, 
        dataloader: DataLoader, 
        criterion: nn.CrossEntropyLoss, 
        epoch: int,
        val_logger: Logger
    ):
        model.eval()
        device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset_size = 0
        running_loss = 0.0
        
        if is_main_process():
            pbar = tqdm(enumerate(dataloader), total = len(dataloader), desc = 'Validation')
        else:
            pbar = enumerate(dataloader)
        with torch.no_grad():
            for _, batches in pbar:
                
                images, segmentation_labels = batches
                images = images.to(device)
                segmentation_labels = tuple(t.to(device) for t in segmentation_labels)
            
                batch_size = images.shape[0]    
                
                segmentation_pred = model(images)
                seg_loss = criterion(segmentation_pred, segmentation_labels)
                running_loss += (seg_loss["total_loss"].item() * batch_size)
                dataset_size += batch_size
                
                epoch_loss = running_loss / dataset_size

                if torch.cuda.is_available():
                    current_device = torch.cuda.current_device()
                    mem_local = torch.cuda.memory_allocated(current_device) / 1E9

                    if dist.is_initialized():
                        mem_tensor = torch.tensor(mem_local).cuda()
                        dist.all_reduce(mem_tensor, op=dist.ReduceOp.SUM)
                        mem = mem_tensor.item()
                    else:
                        mem = mem_local
                else:
                    mem = 0
                if is_main_process() and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(epoch=f'{epoch}',
                                    valid_loss=f'{epoch_loss:0.4f}',
                                    gpu_memory=f'{mem:0.2f} GB')

        if dist.is_initialized():
            running_loss_tensor = torch.tensor(running_loss).cuda()
            dataset_size_tensor = torch.tensor(dataset_size).cuda()
            dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(dataset_size_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = running_loss_tensor.item() / dataset_size_tensor.item()

        if val_logger:
            rank = dist.get_rank() if dist.is_initialized() else 0
            num_gpus = dist.get_world_size() if dist.is_initialized() else 1
            val_logger.info(f"[Process: {rank}] [Epoch {epoch}] - Val Loss: {epoch_loss:.4f} - Total GPU Mem ({num_gpus} GPUs): {mem:.2f} GB")
        
        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss
    
    def _post_process(self, data):
        return data
    
    def handle(self, data):
        data = self.process(data)
        return super().handle(data)
