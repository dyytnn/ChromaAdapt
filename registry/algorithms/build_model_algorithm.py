
import os
from datetime import datetime

from coolname import generate_slug

from abs_interface import AbsAlgorithmProcessHandler
from configs.config import configurable
from registry import (
    domain_fusing_training_segmentation_handler_register,
    evaluate_segmentation_handler_register,
    training_segmentation_handler_register,
)
from registry.architecture import ArchitectureBuilder
from registry.log import StatAccumulator, enable_epoch_logging, setup_logger
from utils import create_save_folder
from utils.color import c_, sr_


@evaluate_segmentation_handler_register.register("ModelBuilding")
@training_segmentation_handler_register.register("ModelBuilding")
@domain_fusing_training_segmentation_handler_register.register("ModelBuilding")
class ModelBuilding(AbsAlgorithmProcessHandler):
    @configurable
    def __init__(self, *, config: dict):
        super(ModelBuilding, self).__init__()
        self.config = config
    
    @classmethod
    def from_config(cls, config: dict, is_evaluation: bool = False):
        if is_evaluation:
            config.update({
                "logging": {
                    "log_input": False,
                    "log_grad": False
                },
                "pretrained": {
                    "mode": "debug",
                    "path": None
                }
            })
        
        return {
            "config": config
        }
    
    def _pre_process(self, data):
        pipeline_config = data.get("pipeline_config")
        builder = ArchitectureBuilder(self.config.get("architecture_config"))
        builder.validate() if self.config.get("is_validate") else None
        model = builder.model

        if self.config.get("instance_aware", {}).get("enabled", False):
            print(f"\n{c_}Wrapping model with instance-aware components...{sr_}")
            from source.models.build_instance_aware_model import build_instance_aware_model
            model = build_instance_aware_model(model, self.config)

        model.train()
        return model, pipeline_config

    def _main_process(self, data):
        """
            RULES: pretrain [None] + mode [Train] => create save folder
            RULES: pretrain [None] + mode [DEBUG] => create folder with prefix "debug_"

            RULES: pretrain [PATH] + mode [TRAIN] => RETURN PATH AND APPEND THE LOG (WARNING IT COULD BE OVERWRITTEN)
            RULES: pretrain [PATH] + mode [DEBUG] => RETURN PATH AND APPEND THE LOG (WARNING IT COULD BE OVERWRITTEN)
            RULES: pretrain [PATH] + mode [FINETUNE] => CREATE FINETUNE SUBFOLDER, LOAD FROM best_model.pth
        """
        model, pipeline_config = data
        self.config.update({
            "pipeline_config": pipeline_config
        }) if pipeline_config is not None else None

        is_train = False
        save_path = None
        pretrained_path = None

        stat_accumulator_instance = None
        train_logger = None
        val_logger = None
        grad_logger = None

        if self.config.get("pretrained")["path"] is None:
            if self.config.get("pretrained")["mode"] == "train":
                save_path = create_save_folder(prefix="TRAIN", **self.config)
                pretrained_path = None
                is_train = True
            else:
                save_path = create_save_folder(prefix="DEBUG", **self.config)
                pretrained_path = None
                is_train = False
        else:
            if self.config.get("pretrained")["mode"] == "train":
                pretrained_path = self.config.get("pretrained")["path"]
                save_path = pretrained_path
                is_train = True
            else:
                pretrained_path = self.config.get("pretrained")["path"]
                save_path = pretrained_path
                is_train = False
        
        if self.config.get("logging") and save_path is not None and is_train:
            train_logger = setup_logger(log_dir = save_path, logger_name="TrainLogger", log_prefix="train")
            val_logger = setup_logger(log_dir = save_path, logger_name="ValLogger", log_prefix="val")
            grad_logger = setup_logger(log_dir = save_path, logger_name="GradLogger", log_prefix="grad")
            
            stat_accumulator_instance = StatAccumulator()
            enable_epoch_logging(model, stat_accumulator_instance, log_input=self.config.get("logging")["log_input"], log_grad=self.config.get("logging")["log_grad"])
        
        return model, is_train, save_path, pretrained_path, train_logger, val_logger, grad_logger, stat_accumulator_instance

    def _post_process(self, data):
        return data
    
    def handle(self, data: dict):
        model, is_train, save_path, pretrained_path, train_logger, val_logger, grad_logger, stat_accumulator_instance = self.process(data)
        data.update(
            {
                "model": model,
                "save_path": save_path,
                "pretrained_path": pretrained_path,
                "pretrained_mode": self.config.get("pretrained")["mode"],
                "is_train": is_train,
                "train_logger": train_logger, 
                "val_logger": val_logger, 
                "grad_logger": grad_logger,
                "stat_accumulator_instance": stat_accumulator_instance
            }
        )
        return super().handle(data)
