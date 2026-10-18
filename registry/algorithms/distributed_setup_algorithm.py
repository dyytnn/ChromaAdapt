import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from abs_interface import AbsAlgorithmProcessHandler
from utils import (
    c_,
    get_params_groups,
    init_distributed_mode,
    init_seeds,
    restart_from_checkpoint,
    sr_,
)
from utils.freeze_utils import freeze_layers, count_trainable_params

from ..registers import (
    domain_fusing_training_segmentation_handler_register,
    scheduler_register,
    segmentation_loss_register,
    training_segmentation_handler_register,
)


@domain_fusing_training_segmentation_handler_register.register("InitTrainingHandler")
@training_segmentation_handler_register.register("InitTrainingHandler")
class InitTrainingHandler(AbsAlgorithmProcessHandler):
    def __init__(self, config: dict, is_init_distributed: bool = True):
        super(InitTrainingHandler, self).__init__()
        self.is_init_distributed = is_init_distributed
        self.config = config
        
        if self.is_init_distributed:
            if dist.is_initialized():
                dist.destroy_process_group()
        
            init_distributed_mode(self.config)
            init_seeds(self.config["seed"])    

    def _pre_process(self, data):
        return data
    
    def get_loss_function(self):
        loss_config = self.config.get("loss", {})
        loss_name = loss_config.get("name", "")
        loss_params = loss_config.get("params")
        
        return (
            getattr(torch.nn, loss_name.replace("pytorch_", ""))(**loss_params) if loss_params else getattr(torch.nn, loss_name.replace("pytorch_", ""))()
        ) if loss_name.startswith("pytorch") else (
            segmentation_loss_register.get(loss_name, **loss_params) if loss_params else segmentation_loss_register.get(loss_name)
        )

    def _main_process(self, data: dict):
        model: nn.Module = data.get("model", None)
        model.to(self.config.get("device"))
        pretrained_path = data.get("pretrained_path", None)
        
        start_epoch = 0
        warmup_epochs = self.config.get("warmup_epochs", 10)
        best_loss = np.inf


        if self.is_init_distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.config.get("gpu")],
                find_unused_parameters=True,
                broadcast_buffers=False
            )

        for key in ["eps", "lr"]:
            if key in self.config.get("optimizer", {}).get("params", {}):
                self.config["optimizer"]["params"][key] = float(self.config["optimizer"]["params"].pop(key))

        custom_param_groups_fn = data.get("custom_param_groups_fn", None)
        base_lr = float(self.config.get("optimizer")["params"]["lr"])

        if custom_param_groups_fn is not None:
            print(f"{c_}[INFO] Using custom parameter groups with separate head LR{sr_}")
            head_lr_multiplier = self.config.get("head_lr_multiplier", 10.0)
            param_groups = custom_param_groups_fn(model, base_lr=base_lr, head_lr_multiplier=head_lr_multiplier)
        else:
            param_groups = get_params_groups(model)

        optimizer = getattr(torch.optim, self.config.get("optimizer")["name"])(
            params=param_groups,
            **self.config.get("optimizer")["params"]
        )

        pretrained_mode = data.get("pretrained_mode", "train")
        scheduler = None

        if pretrained_mode != "resume":
            self.config.get("scheduler")["params"].update({"optimizer": optimizer})
            scheduler = scheduler_register.get(
                self.config.get("scheduler")["name"],
                **self.config.get("scheduler")["params"]
            )

        fp16_scaler = torch.amp.GradScaler() if self.config.get("fp16_scaler") else None

        loss = self.get_loss_function()

        if pretrained_path:
            if pretrained_mode == "finetune":
                checkpoint_path = os.path.join(pretrained_path, "best_model.pth")
                print(f"{c_}[INFO] Finetune mode: Loading model weights from {checkpoint_path}{sr_}")

                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

                print(f"{c_}[DEBUG] Checkpoint has {len(checkpoint)} keys{sr_}")
                print(f"{c_}[DEBUG] First few keys: {list(checkpoint.keys())[:3]}{sr_}")
                print(f"{c_}[DEBUG] Model expects 'module.' prefix: {self.is_init_distributed}{sr_}")

                if self.is_init_distributed:
                    if any(k.startswith('module.') for k in checkpoint.keys()):
                        print(f"{c_}[INFO] Loading with module prefix (checkpoint has it){sr_}")
                        msg = model.load_state_dict(checkpoint, strict=False)
                    else:
                        print(f"{c_}[INFO] Adding module prefix to checkpoint keys{sr_}")
                        checkpoint_with_prefix = {f"module.{k}": v for k, v in checkpoint.items()}
                        msg = model.load_state_dict(checkpoint_with_prefix, strict=False)
                else:
                    if any(k.startswith('module.') for k in checkpoint.keys()):
                        print(f"{c_}[INFO] Removing module prefix from checkpoint keys{sr_}")
                        checkpoint_no_prefix = {k.replace("module.", ""): v for k, v in checkpoint.items()}
                        msg = model.load_state_dict(checkpoint_no_prefix, strict=False)
                    else:
                        print(f"{c_}[INFO] Loading checkpoint directly (no prefix changes){sr_}")
                        msg = model.load_state_dict(checkpoint, strict=False)

                print(f"{c_}[INFO] Loaded model weights: {msg}{sr_}")

                if msg.missing_keys and len(msg.missing_keys) > len(checkpoint) * 0.5:
                    print(f"{c_}[ERROR] More than 50% of keys are missing! Weight loading likely failed.{sr_}")
                    print(f"{c_}[ERROR] Missing keys: {msg.missing_keys[:10]}...{sr_}")
                    print(f"{c_}[ERROR] Check that the model architecture matches the checkpoint.{sr_}")

                print(f"{c_}[INFO] Resetting BatchNorm running statistics for domain adaptation{sr_}")
                bn_count = 0
                for module in model.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                        module.reset_running_stats()
                        module.momentum = 0.1
                        bn_count += 1

                if bn_count > 0:
                    print(f"{c_}[INFO] Reset {bn_count} BatchNorm layers for domain adaptation{sr_}")
                else:
                    print(f"{c_}[WARNING] No BatchNorm layers found - this might be okay{sr_}")

                last_model = model
                start_epoch = 0
                best_loss = np.inf
                warmup_epochs = self.config.get("warmup_epochs", 10)
                print(f"{c_}[INFO] Fine-tuning: Starting from epoch 0 with fresh optimizer{sr_}")

            else:
                checkpoint_path = os.path.join(pretrained_path, "last_save_dict.pth")
                print(f"{c_}[INFO] Resume mode: Loading checkpoint from {checkpoint_path}{sr_}")

                to_restore = {
                    "start_epoch": None,
                    "best_loss": None,
                    "last_model": model,
                    "optimizer": optimizer,
                    "fp16_scaler": fp16_scaler,
                    "loss": loss
                }

                restart_from_checkpoint(
                    checkpoint_path,
                    run_variables=to_restore,
                    last_model=model,
                    optimizer=optimizer,
                    fp16_scaler=fp16_scaler,
                    loss=loss,
                    best_loss=None,
                    start_epoch=None,
                    is_distributed=self.is_init_distributed
                )

                last_model = to_restore["last_model"]
                optimizer = to_restore["optimizer"] if to_restore["optimizer"] is not None else optimizer
                fp16_scaler = to_restore["fp16_scaler"] if to_restore["fp16_scaler"] is not None else fp16_scaler
                loss = to_restore["loss"]
                start_epoch = to_restore.get("start_epoch") or 0
                best_loss = to_restore.get("best_loss") or np.inf
                warmup_epochs = 0
                print(f"{c_}[INFO] Resuming from epoch {start_epoch} with best_loss={best_loss:.4f}{sr_}")

                for param_group in optimizer.param_groups:
                    if 'initial_lr' not in param_group:
                        param_group['initial_lr'] = param_group['lr']
                        print(f"{c_}[DEBUG] Set initial_lr={param_group['lr']:.6f} for param_group{sr_}")

                self.config.get("scheduler")["params"].update({"optimizer": optimizer})
                scheduler = scheduler_register.get(
                    self.config.get("scheduler")["name"],
                    **self.config.get("scheduler")["params"]
                )
                print(f"{c_}[INFO] Created scheduler with last_epoch={self.config.get('scheduler')['params'].get('last_epoch', -1)}{sr_}")
        else:
            last_model = model

        freeze_config = data.get("freeze_layers_config", {})
        if freeze_config.get("enabled", False):
            strategy = freeze_config.get("strategy", "freeze_backbone")

            model_to_freeze = last_model.module if self.is_init_distributed else last_model

            print(f"{c_}[INFO] Applying layer freezing strategy: {strategy}{sr_}")
            freeze_layers(model_to_freeze, strategy=strategy, verbose=True)

            if custom_param_groups_fn is not None:
                head_lr_multiplier = self.config.get("head_lr_multiplier", 10.0)
                param_groups_after_freeze = custom_param_groups_fn(last_model, base_lr=base_lr, head_lr_multiplier=head_lr_multiplier)
            else:
                param_groups_after_freeze = get_params_groups(last_model)

            optimizer = getattr(torch.optim, self.config.get("optimizer")["name"])(
                params=param_groups_after_freeze,
                **self.config.get("optimizer")["params"]
            )

            self.config.get("scheduler")["params"].update({"optimizer": optimizer})
            scheduler = scheduler_register.get(
                self.config.get("scheduler")["name"],
                **self.config.get("scheduler")["params"]
            )

        return last_model, optimizer, scheduler, fp16_scaler, loss, start_epoch, warmup_epochs, best_loss

    def _post_process(self, data):
        return data
    
    def handle(self, data):
        last_model, optimizer, scheduler, fp16_scaler, loss, start_epoch, warmup_epochs, best_loss = self.process(data)
        data.update(
            {
                "last_model": last_model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "fp16_scaler": fp16_scaler,
                "criterion": loss, 
                "start_epoch": start_epoch,
                "best_loss": best_loss,
                "warmup_epochs": warmup_epochs
            }
        )
        return super().handle(data)
