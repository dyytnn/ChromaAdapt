"""
FineTune Model Building - Simplified model loading for fine-tuning.

This class provides a streamlined interface for fine-tuning pretrained models
without needing to re-specify architecture configs.
"""
import os
from datetime import datetime

import torch.distributed as dist
import yaml
from coolname import generate_slug
from omegaconf import OmegaConf

from abs_interface import AbsAlgorithmProcessHandler
from configs.config import configurable
from registry import (
    domain_fusing_training_segmentation_handler_register,
    training_segmentation_handler_register,
)
from registry.architecture import ArchitectureBuilder
from registry.log import StatAccumulator, enable_epoch_logging, setup_logger
from utils import is_main_process
from utils.color import c_, sr_


@training_segmentation_handler_register.register("FineTuneModelBuilding")
@domain_fusing_training_segmentation_handler_register.register("FineTuneModelBuilding")
class FineTuneModelBuilding(AbsAlgorithmProcessHandler):
    """
    Simplified ModelBuilding for fine-tuning that only requires pretrained_path.

    Automatically loads:
    - Architecture from arch.yaml or params.yaml in pretrained_path
    - Weights from best_model.pth
    - Creates finetune subfolder automatically

    Config example:
        FineTuneModelBuilding:
          config:
            pretrained_path: "outputs/training/susy/run_name"
            freeze_layers:  # Optional
              enabled: true
              strategy: "freeze_backbone"
    """

    @configurable
    def __init__(self, *, config: dict):
        super(FineTuneModelBuilding, self).__init__()
        self.config = config

        if not self.config.get("pretrained_path"):
            raise ValueError("FineTuneModelBuilding requires 'pretrained_path' in config")

    @classmethod
    def from_config(cls, config: dict):
        return {"config": config}

    def _pre_process(self, data):
        pretrained_path = self.config.get("pretrained_path")

        conf = OmegaConf.load(os.path.join(pretrained_path, "params.yaml"))
        architecture_config = conf.cfg.pipeline.ModelBuilding.config.architecture_config

        builder = ArchitectureBuilder(architecture_config)

        if self.config.get("is_validate", True):
            builder.validate()

        model = builder.model

        return model, pretrained_path

    def _main_process(self, data):
        """
        Main processing: Create finetune folder and setup logging.
        """
        model, pretrained_path = data
        task = "FineTunedModel"

        save_path = None
        if is_main_process():
            finetune_folder_name = f"[FINETUNE]{datetime.now().strftime('%Y-%m-%d')}"
            base_finetune_path = os.path.join(pretrained_path, finetune_folder_name)

            run_number = 1
            while True:
                save_path = os.path.join(base_finetune_path, f"{task}_{run_number}")
                if not os.path.exists(save_path):
                    break
                run_number += 1

            os.makedirs(save_path, exist_ok=True)

            config_save_path = os.path.join(save_path, "finetune_params.yaml")
            with open(config_save_path, 'w') as f:
                yaml.dump({"finetune_config": self.config, "source_model": pretrained_path}, f)

            print(f"{c_}[INFO] FineTune mode: Loading from {pretrained_path}/best_model.pth{sr_}")
            print(f"{c_}[INFO] FineTune mode: Saving to {save_path}{sr_}")

        if dist.is_available() and dist.is_initialized():
            save_path_list = [save_path]
            dist.broadcast_object_list(save_path_list, src=0)
            save_path = save_path_list[0]
        elif not is_main_process():
            raise RuntimeError("FineTuneModelBuilding running on non-main process without distributed training")

        stat_accumulator_instance = None
        train_logger = None
        val_logger = None
        grad_logger = None

        logging_config = self.config.get("logging", {})
        if logging_config.get("enabled", True) and is_main_process():
            train_logger = setup_logger(log_dir=save_path, logger_name="TrainLogger", log_prefix="train")
            val_logger = setup_logger(log_dir=save_path, logger_name="ValLogger", log_prefix="val")

            if logging_config.get("log_grad", False):
                grad_logger = setup_logger(log_dir=save_path, logger_name="GradLogger", log_prefix="grad")

            stat_accumulator_instance = StatAccumulator()
            enable_epoch_logging(
                model,
                stat_accumulator_instance,
                log_input=logging_config.get("log_input", False),
                log_grad=logging_config.get("log_grad", False)
            )

        return model, pretrained_path, save_path, train_logger, val_logger, grad_logger, stat_accumulator_instance

    def _post_process(self, data):
        return data

    def handle(self, data: dict):
        model, pretrained_path, save_path, train_logger, val_logger, grad_logger, stat_accumulator_instance = self.process(data)

        data.update({
            "model": model,
            "save_path": save_path,
            "pretrained_path": pretrained_path,
            "pretrained_mode": "finetune",  # Always finetune mode
            "is_train": True,
            "train_logger": train_logger,
            "val_logger": val_logger,
            "grad_logger": grad_logger,
            "stat_accumulator_instance": stat_accumulator_instance,
            "freeze_layers_config": self.config.get("freeze_layers", {})
        })

        return super().handle(data)
