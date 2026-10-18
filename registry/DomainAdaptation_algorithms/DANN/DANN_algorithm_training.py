import gc
import os
from ast import List
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from logging import Logger
from typing import Any, List, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm

from abs_interface import AbsAlgorithmProcessHandler
from data import (
    DANNCombinedDataset,
    build_transforms,
    dann_collate_fn,
    real_dataset_module,
    synthetic_dataset_module,
)

# Import KoLeoLoss for feature spreading
from registry.DINOv3.loss.koleo_loss import KoLeoLoss
from registry.DomainAdaptation_algorithms.DANN.utils.DANN_model_wrapper import (
    DANNModelWrapper,
)

# Import debugging tools
from registry.DomainAdaptation_algorithms.UDA.gradient_debug_tools import (
    GradientClipperWithNaNHandling,
    GradientNaNDetector,
    ModelStabilizer,
    apply_emergency_fixes,
    debug_tensor_health,
)
from registry.log import StatAccumulator
from registry.registers import (
    domain_fusing_training_segmentation_handler_register,
    scheduler_register,
)
from utils import (
    c_,
    is_dist_available_and_initialized,
    is_main_process,
    log_on_main_process,
    save_on_master,
    sr_,
)


def get_unwrapped_model(model):
    """Get the unwrapped model, handling DistributedDataParallel wrapping"""
    if hasattr(model, 'module'):
        return model.module
    return model


@domain_fusing_training_segmentation_handler_register.register("UnsupervisedDomainAdaptationAlgorithm")
class UnsupervisedDomainAdaptationAlgorithm(AbsAlgorithmProcessHandler):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
    def datatransformation(self, key: str = "val"):
        return build_transforms(self.config.get("transformation").get(key))

    @property
    def real_dataset(self):
        dataset_class_name = self.config.get("RealData").get("dataset")
        real_config = self.config.get("RealData").get("config")
        real_config.update({
            "transformation": self.datatransformation("train")
        })
        dataset = getattr(real_dataset_module, dataset_class_name)(**real_config)

        return dataset

    def synthetic_dataset(self, key: str = "val"):
        dataset_class_name = self.config.get("SyntheticData").get("dataset")
        synthetic_config = self.config.get("SyntheticData").get("config")[key]
        synthetic_config.update({
            "transformation": self.datatransformation(key)
        })
        dataset = getattr(synthetic_dataset_module, dataset_class_name)(**synthetic_config)

        return dataset
    
    def get_lambda_scheduler(self, niter_per_ep: int):
        scheduler_params = self.config.get("lambda_scheduler")["params"].copy()
        scheduler_params["niter_per_ep"] = niter_per_ep
        return scheduler_register.get(self.config.get("lambda_scheduler")["name"], **scheduler_params)
    
    def _pre_process(self, data: dict) -> dict:
        is_train = data.get("is_train", None)

        num_workers = self.config.get("num_workers", 10) if is_train else 1

        train_synthetic_ds = self.synthetic_dataset("train")
        val_synthetic_ds = self.synthetic_dataset("val")
        real_ds = self.real_dataset

        # Create combined dataset for balanced training (handles imbalanced dataset sizes)
        train_combined_ds = DANNCombinedDataset(
            source_dataset=train_synthetic_ds,
            target_dataset=real_ds,
            balance_strategy=self.config.get("balance_strategy", "oversample_minority")
        )

        # Combined dataloader with balanced sampling
        train_combined_dataloader = DataLoader(
            train_combined_ds,
            batch_size=self.config.get("batch_size", 8),
            sampler=RandomSampler(train_combined_ds) if not is_dist_available_and_initialized() else
            DistributedSampler(train_combined_ds, shuffle=True),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dann_collate_fn,
            drop_last=True  # Ensure consistent batch sizes
        )

        val_synthetic_dataloader = DataLoader(
            val_synthetic_ds,
            batch_size=self.config.get("batch_size", 8),
            sampler=RandomSampler(val_synthetic_ds) if not is_dist_available_and_initialized() else
            DistributedSampler(val_synthetic_ds),
            num_workers=num_workers,
            pin_memory=True,
        )

        data.update(
            {
                "train_combined_dataloader": train_combined_dataloader,
                "val_synthetic_dataloader": val_synthetic_dataloader,
            }
        )

        return data
    
    def _main_process(self, data):
        train_logger: Logger = data.get("train_logger", None)
        val_logger: Logger = data.get("val_logger", None)
        grad_logger: Logger = data.get("grad_logger", None)
        stat_accumulator_instance: StatAccumulator = data.get("stat_accumulator_instance", None)

        last_model: DANNModelWrapper = data.get("last_model", None)

        start_epoch: int = data.get("start_epoch", 0)
        best_loss: float | Any = data.get("best_loss", np.inf)

        train_combined_dataloader: DataLoader = data.get("train_combined_dataloader", None)
        val_synthetic_dataloader: DataLoader = data.get("val_synthetic_dataloader", None)

        optimizer: torch.optim.Optimizer = data.get("optimizer", None)
        scheduler: list | np.ndarray = data.get("scheduler", None)
        fp16_scaler: torch.amp.GradScaler = data.get("fp16_scaler", None)

        label_loss: Any = data.get("criterion", None)
        domain_classification_loss: nn.BCEWithLogitsLoss = data.get("domain_classification_loss", None)

        # Calculate niter_per_ep from combined dataloader
        num_iters = len(train_combined_dataloader)
        lambda_scheduler = self.get_lambda_scheduler(niter_per_ep=num_iters)

        history = defaultdict(list)

        for epoch in range(start_epoch, self.config.get("epochs")):
            # Set epoch for distributed samplers to ensure proper shuffling
            if is_dist_available_and_initialized():
                train_combined_dataloader.sampler.set_epoch(epoch)
                val_synthetic_dataloader.sampler.set_epoch(epoch)

            gc.collect()
            print(f'Epoch {epoch}/{self.config.get("epochs")}', end='')

            train_loss = self.train_one_epoch(
                model = last_model,
                optimizer = optimizer,
                scheduler = scheduler,
                fp16_scaler = fp16_scaler,
                epoch = epoch,
                combined_loader = train_combined_dataloader,
                label_loss = label_loss,
                domain_classification_loss = domain_classification_loss,
                train_logger = train_logger,
                grad_logger = grad_logger,
                stat_accumulator_instance = stat_accumulator_instance,
                lambda_scheduler = lambda_scheduler
            )
            
            valid_loss = self.valid_one_epoch(
                model = last_model,
                dataloader = val_synthetic_dataloader,
                criterion = label_loss,
                epoch = epoch,
                val_logger = val_logger,
            )
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(valid_loss)
            
            # Get unwrapped model for consistent state dict access
            unwrapped_model = get_unwrapped_model(last_model)

            save_dict = {
                'last_model': unwrapped_model.base_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'start_epoch': epoch + 1,
                'criterion': label_loss.state_dict(),
            }

            save_dict_domain_classifier = {
                "domain_classification_loss": domain_classification_loss.state_dict(),
                "domain_classifier": unwrapped_model.domain_classifier.state_dict(),
            }

            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()

            # Always save current best_loss
            save_dict["best_loss"] = best_loss

            # Synchronize all processes before saving to ensure consistency
            if is_dist_available_and_initialized():
                try:
                    dist.barrier()  # 60 second timeout
                except Exception as e:
                    log_on_main_process(f"Warning: Distributed barrier failed: {e}. Continuing without synchronization.")

            if valid_loss <= best_loss:
                log_on_main_process(f"{c_}Valid loss Improved ({best_loss} ---> {valid_loss}) {sr_}")
                best_loss = valid_loss

                unwrapped_model = get_unwrapped_model(last_model)
                best_model_wts = deepcopy(last_model.state_dict())
                best_domain_classifier_heads = unwrapped_model.domain_classifier.state_dict()

                save_dict["best_loss"] = best_loss  # Update with new best_loss
                if is_main_process():
                    torch.save(best_model_wts, os.path.join(data.get("save_path"), "best_model.pth"))
                    torch.save(best_domain_classifier_heads, os.path.join(data.get("gradient_reversal_save_path"), "best_model.pth"))

                    print(f"Best model saved at {os.path.join(data.get('save_path'), 'best_model.pth')}")
                    print(f"Domain classifier saved at {os.path.join(data.get('gradient_reversal_save_path'), 'best_model.pth')}")

            if is_main_process():
                save_on_master(save_dict, os.path.join(data.get("save_path"), "last_save_dict.pth"))
                save_on_master(save_dict_domain_classifier, os.path.join(data.get("gradient_reversal_save_path"), "last_save_dict.pth"))

                print(f"Last checkpoint saved at {os.path.join(data.get('save_path'), 'last_save_dict.pth')}")
                print(f"Domain classifier checkpoint saved at {os.path.join(data.get('gradient_reversal_save_path'), 'last_save_dict.pth')}")
                       
        data.update(
            {
                "save_dict": save_dict,
                "history": history
            }
        )
        
        if dist.is_initialized():
            dist.destroy_process_group()
        return data
    
    def train_one_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        fp16_scaler: torch.amp.GradScaler,
        epoch: int,
        combined_loader: DataLoader,
        label_loss: Any,
        domain_classification_loss: Any,
        train_logger: Logger,
        grad_logger: Logger,
        stat_accumulator_instance: StatAccumulator,
        lambda_scheduler: Union[List[float], np.ndarray],
    ) -> dict:
        model.train()
        device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize debugging tools
        nan_detector = GradientNaNDetector(model, verbose=False)  # Reduce verbosity
        gradient_clipper = GradientClipperWithNaNHandling(
            max_norm=self.config.get("max_grad_norm", 1.0),
            handle_nan='zero',  # Set NaN gradients to zero instead of crashing
            verbose=False  # Reduce verbosity
        )

        # Initialize KoLeoLoss for source feature spreading
        koleo_loss_fn = KoLeoLoss().to(device) if self.config.get("use_koleo_loss", False) else None
        koleo_loss_weight = self.config.get("koleo_loss_weight", 0.1)

        # Apply model stabilization at the beginning of each epoch
        if epoch == 0:
            print("🔧 Applying model stabilization techniques...")
            ModelStabilizer.add_weight_noise(model, noise_std=1e-6)
            ModelStabilizer.initialize_batch_norm(model)

        dataset_size = 0
        running_loss = 0.0
        nan_count_this_epoch = 0

        # Track component losses separately for detailed logging
        running_seg_loss = 0.0
        running_seg_chromosome_loss = 0.0
        running_seg_overlap_loss = 0.0
        running_source_domain_loss = 0.0
        running_target_domain_loss = 0.0
        running_koleo_loss = 0.0

        header = 'Epoch: [{}/{}]'.format(epoch, self.config.get("epochs"))
        num_iters = len(combined_loader)

        pbar = tqdm(enumerate(combined_loader), total=num_iters, desc=header)

        for it, (source_batches, target_batches) in pbar:
            # Calculate global iteration index across all epochs
            global_it = it + epoch * num_iters

            # Calculate adaptation parameter p following original DANN paper:
            # p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            if lambda_scheduler is not None:
                p = lambda_scheduler[global_it]
            else:
                # Original DANN formula: p = global_iteration / (total_epochs * iterations_per_epoch)
                total_iterations = self.config.get("epochs") * num_iters
                p = float(global_it) / total_iterations

            epsilon = 2. / (1. + np.exp(-10 * p)) - 1
            
            # todo: Training with source data
            source_images, source_labels, source_domain_label = source_batches
            
            source_images: Tensor = source_images.to(device)
            source_labels: Union[Tensor, List[Tensor]] = source_labels.to(device) if isinstance(source_labels, Tensor) else [labels.to(device) for labels in source_labels]
            source_domain_label = source_domain_label.to(device)
            
            target_images, target_domain_label = target_batches
            target_images: Tensor = target_images.to(device)
            target_domain_label: Tensor = target_domain_label.to(device)
                
            batch_size = source_images.shape[0]
            
            if it % self.config.get("gradient_accumulation_steps", 1) == 0:
                optimizer.zero_grad()

            autocast_context: Union[torch.autocast, nullcontext] = (
                torch.amp.autocast(enabled=True, device_type='cuda')
                if fp16_scaler else nullcontext()
            )

            with autocast_context:
                # Get source features for KoLeoLoss (if enabled)
                if koleo_loss_fn is not None:
                    source_segment_output, source_domain_output, source_features = model(
                        x=source_images, epsilon=epsilon, return_features=True
                    )
                else:
                    source_segment_output, source_domain_output = model(x=source_images, epsilon=epsilon)

                source_segment_loss = label_loss(source_segment_output, source_labels)
                source_domain_loss = domain_classification_loss(source_domain_output.flatten(), source_domain_label.float())

                # todo: Training with target data

                _, target_domain_output = model(x = target_images, epsilon = epsilon)
                target_domain_loss = domain_classification_loss(target_domain_output.flatten(), target_domain_label.float())

                # DIAGNOSTIC: Check domain classifier predictions every 100 iterations
                if it % 100 == 0 and is_main_process():
                    with torch.no_grad():
                        # Get domain predictions (apply sigmoid to logits)
                        source_probs = torch.sigmoid(source_domain_output.flatten())
                        target_probs = torch.sigmoid(target_domain_output.flatten())

                        # Expected: source_label=0, target_label=1
                        # So source_probs should be ~0.5 and target_probs should be ~0.5 for confusion

                        print(f"\n[DOMAIN DIAGNOSTIC - Iter {it}, Epoch {epoch}]")
                        print(f"  Domain Loss: Src={source_domain_loss.item():.4f}, Tgt={target_domain_loss.item():.4f}, Total={source_domain_loss.item()+target_domain_loss.item():.4f}")
                        print(f"  Domain Probs: Src={source_probs.mean().item():.4f}±{source_probs.std().item():.4f}, Tgt={target_probs.mean().item():.4f}±{target_probs.std().item():.4f}")
                        print(f"  Expected for confusion: both ≈ 0.5 (but source labeled as 0, target as 1)")

                        # Check for mode collapse
                        if koleo_loss_fn is not None and 'source_features' in locals():
                            feat_norm = source_features.norm(dim=1).mean().item()
                            feat_std = source_features.std().item()

                            # Pairwise distance check
                            sample_size = min(100, source_features.shape[0])
                            sample_feats = source_features[:sample_size]
                            sample_feats_norm = torch.nn.functional.normalize(sample_feats, p=2, dim=1)
                            pw_dist = torch.pdist(sample_feats_norm).mean().item()

                            print(f"  Features: L2={feat_norm:.3f}, std={feat_std:.3f}, pairwise_dist={pw_dist:.3f}")
                            print(f"  Collapse check: pairwise_dist <0.5=COLLAPSE, >0.8=GOOD, ~1.4=RANDOM")

                            if pw_dist < 0.5:
                                print(f"  🚨 MODE COLLAPSE DETECTED! Features clustering together!")
                            elif pw_dist > 0.8:
                                print(f"  ✅ Features well-spread (good domain adaptation)")

                        print(f"  Epsilon (GRL strength): {epsilon:.4f}")

                # Scale domain losses to prevent gradient explosion
                domain_loss_weight = self.config.get("domain_loss_weight", 1.0)
                scaled_domain_loss = (source_domain_loss + target_domain_loss) * domain_loss_weight

                # Apply KoLeoLoss to source features (encourage feature spreading for generalization)
                koleo_loss = torch.tensor(0.0, device=device)
                if koleo_loss_fn is not None:
                    koleo_loss = koleo_loss_fn(source_features) * koleo_loss_weight

                # print(f"Domain losses require_grad: source={source_domain_loss.requires_grad},  target={target_domain_loss.requires_grad}")
                total_loss = (source_segment_loss['total_loss'] + scaled_domain_loss + koleo_loss) / self.config.get("gradient_accumulation_steps", 1)

                # Only debug total loss if it has issues
                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                    debug_tensor_health(total_loss, f"total_loss_iter_{it}_PROBLEMATIC")

            # Check for NaN/Inf before backward pass
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print(f"🚨 NaN/Inf detected in total_loss at iteration {it}! Applying emergency fixes...")
                apply_emergency_fixes(model, optimizer)
                continue

            if fp16_scaler:
                fp16_scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # Check gradients after backward pass
            has_nan, _ = nan_detector.check_gradients(step=global_it)
            if has_nan:
                nan_count_this_epoch += 1
                print(f"⚠️  NaN gradients detected at iteration {it} (count this epoch: {nan_count_this_epoch})")

                # Apply emergency fixes if too many NaN occurrences
                if nan_count_this_epoch >= 3:
                    print("🚑 Too many NaN occurrences, applying emergency fixes...")
                    apply_emergency_fixes(model, optimizer)
                    nan_count_this_epoch = 0  # Reset counter after fixes

            if (it + 1) % self.config.get("gradient_accumulation_steps", 1) == 0 or (it + 1) == num_iters:
                if fp16_scaler:
                    # Unscale gradients before clipping to get correct gradient norms
                    fp16_scaler.unscale_(optimizer)

                # First check for NaN gradients and handle them
                _, had_nan_in_clip = gradient_clipper.handle_nan_gradients(model)
                if had_nan_in_clip:
                    print(f"🔧 NaN gradients handled during clipping")

                # Then apply standard gradient clipping
                if self.config.get("max_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get("max_grad_norm", 1.0))

                if fp16_scaler:
                    fp16_scaler.step(optimizer)  # Apply gradients
                    fp16_scaler.update()  # Update the scaler for next iteration
                else:
                    optimizer.step()  # Apply gradients

                scheduler.step()  # Update learning rate on all processes
                
            torch.cuda.synchronize()
            
            # Store loss values before cleanup
            total_loss_item = total_loss.item()

            running_loss += (total_loss_item * batch_size)
            dataset_size += batch_size

            # Track component losses for detailed logging
            running_seg_loss += (source_segment_loss['total_loss'].item() * batch_size)
            running_seg_chromosome_loss += (source_segment_loss.get('loss_chromosome', torch.tensor(0.0)).item() * batch_size)
            running_seg_overlap_loss += (source_segment_loss.get('loss_overlap', torch.tensor(0.0)).item() * batch_size)
            running_source_domain_loss += (source_domain_loss.item() * batch_size)
            running_target_domain_loss += (target_domain_loss.item() * batch_size)
            running_koleo_loss += (koleo_loss.item() * batch_size)

            epoch_loss = running_loss / dataset_size
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']

            postfix_dict = {
                "epoch": f"{epoch}",
                "train_loss": f"{epoch_loss:0.4f}",
                "model_lr": f"{current_lr:0.5f}",
                "gpu_mem": f"{mem:0.2f} GB",
                "epsilon": f"{epsilon:0.4f}",  # Monitor gradient reversal strength
                "src_dom": f"{source_domain_loss:0.4f}",  # Shortened for space
                "tgt_dom": f"{target_domain_loss:0.4f}"   # Shortened for space
            }
            if koleo_loss_fn is not None:
                postfix_dict["koleo"] = f"{koleo_loss:0.4f}"  # Monitor feature spreading
            pbar.set_postfix(**postfix_dict)

            # Explicit memory cleanup to prevent accumulation
            del source_segment_output, source_domain_output, target_domain_output
            del source_segment_loss, source_domain_loss, target_domain_loss, total_loss
            torch.cuda.empty_cache()

        # Synchronize loss across all processes for consistency
        if dist.is_initialized():
            # Correct distributed loss: sum running_loss and dataset_size across GPUs, then divide
            running_loss_tensor = torch.tensor(running_loss).cuda()
            dataset_size_tensor = torch.tensor(dataset_size).cuda()
            running_seg_loss_tensor = torch.tensor(running_seg_loss).cuda()
            running_seg_chromosome_loss_tensor = torch.tensor(running_seg_chromosome_loss).cuda()
            running_seg_overlap_loss_tensor = torch.tensor(running_seg_overlap_loss).cuda()
            running_source_domain_loss_tensor = torch.tensor(running_source_domain_loss).cuda()
            running_target_domain_loss_tensor = torch.tensor(running_target_domain_loss).cuda()
            running_koleo_loss_tensor = torch.tensor(running_koleo_loss).cuda()

            dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(dataset_size_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(running_seg_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(running_seg_chromosome_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(running_seg_overlap_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(running_source_domain_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(running_target_domain_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(running_koleo_loss_tensor, op=dist.ReduceOp.SUM)

            epoch_loss = running_loss_tensor.item() / dataset_size_tensor.item()
            epoch_seg_loss = running_seg_loss_tensor.item() / dataset_size_tensor.item()
            epoch_seg_chromosome_loss = running_seg_chromosome_loss_tensor.item() / dataset_size_tensor.item()
            epoch_seg_overlap_loss = running_seg_overlap_loss_tensor.item() / dataset_size_tensor.item()
            epoch_source_domain_loss = running_source_domain_loss_tensor.item() / dataset_size_tensor.item()
            epoch_target_domain_loss = running_target_domain_loss_tensor.item() / dataset_size_tensor.item()
            epoch_koleo_loss = running_koleo_loss_tensor.item() / dataset_size_tensor.item()
        else:
            # Ensure consistent loss computation for single GPU mode
            epoch_loss = running_loss / dataset_size
            epoch_seg_loss = running_seg_loss / dataset_size
            epoch_seg_chromosome_loss = running_seg_chromosome_loss / dataset_size
            epoch_seg_overlap_loss = running_seg_overlap_loss / dataset_size
            epoch_source_domain_loss = running_source_domain_loss / dataset_size
            epoch_target_domain_loss = running_target_domain_loss / dataset_size
            epoch_koleo_loss = running_koleo_loss / dataset_size

        # Print debugging summary for this epoch
        debug_summary = nan_detector.get_summary()
        print(f"\n📊 Epoch {epoch} Debugging Summary:")
        print(f"   - NaN events this epoch: {nan_count_this_epoch}")
        print(f"   - Total NaN events: {debug_summary['total_nan_events']}")
        print(f"   - NaN gradients clipped: {gradient_clipper.nan_count}")
        if debug_summary['total_nan_events'] > 0:
            print(f"   - Affected parameters: {len(debug_summary.get('affected_parameters', set()))}")

        if grad_logger and is_main_process():
            stat_accumulator_instance.log_epoch_summary(grad_logger, prefix=f"[Epoch {epoch}] ")

        if train_logger:
            rank = dist.get_rank() if dist.is_initialized() else 0
            # Comprehensive loss logging with all components
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}] ============== EPOCH SUMMARY ==============")
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}] Total Loss: {epoch_loss:.6f} | LR: {current_lr:.5e} | GPU Mem: {mem:.2f} GB | NaN Events: {nan_count_this_epoch}")
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}] --- Segmentation Losses ---")
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}]   Seg Total:       {epoch_seg_loss:.6f}")
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}]   Chromosome:      {epoch_seg_chromosome_loss:.6f}")
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}]   Overlap:         {epoch_seg_overlap_loss:.6f}")
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}] --- Domain Losses (scaled by {self.config.get('domain_loss_weight', 1.0)}) ---")
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}]   Source Domain:   {epoch_source_domain_loss:.6f}")
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}]   Target Domain:   {epoch_target_domain_loss:.6f}")
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}]   Domain Total:    {(epoch_source_domain_loss + epoch_target_domain_loss):.6f}")
            if self.config.get("use_koleo_loss", False):
                train_logger.info(f"[Process: {rank}] [Epoch {epoch}] --- Feature Spreading ---")
                train_logger.info(f"[Process: {rank}] [Epoch {epoch}]   KoLeo Loss:      {epoch_koleo_loss:.6f}")
            train_logger.info(f"[Process: {rank}] [Epoch {epoch}] ==============================================")

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
                
                images, segmentation_labels, _ = batches
                images = images.to(device)
                segmentation_labels: Union[Tensor, List[Tensor]] = segmentation_labels.to(device) if isinstance(segmentation_labels, Tensor) else [labels.to(device) for labels in segmentation_labels]
                
                batch_size = images.shape[0]    
                
                segmentation_pred = model(images, epsilon = None)
                seg_loss = criterion(segmentation_pred, segmentation_labels)
                running_loss += (seg_loss['total_loss'].item() * batch_size)
                dataset_size += batch_size
                
                epoch_loss = running_loss / dataset_size
        
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
                if is_main_process() and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(epoch=f'{epoch}', valid_loss=f'{epoch_loss:0.4f}', gpu_memory=f'{mem:0.2f} GB')

        # Synchronize loss across all processes for consistency
        if dist.is_initialized():
            # Correct distributed loss: sum running_loss and dataset_size across GPUs, then divide
            running_loss_tensor = torch.tensor(running_loss).cuda()
            dataset_size_tensor = torch.tensor(dataset_size).cuda()
            dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(dataset_size_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = running_loss_tensor.item() / dataset_size_tensor.item()
        else:
            # Ensure consistent loss computation for single GPU mode
            epoch_loss = running_loss / dataset_size

        if val_logger:
            rank = dist.get_rank() if dist.is_initialized() else 0
            val_logger.info(f"[Process: {rank}] [Epoch {epoch}] - Val Loss: {epoch_loss:.4f}")
        
        torch.cuda.empty_cache()
        gc.collect()
        return epoch_loss
    
    def _post_process(self, data):
        return data
    
    
    def handle(self, data):
        data = self.process(data)
        return super().handle(data)