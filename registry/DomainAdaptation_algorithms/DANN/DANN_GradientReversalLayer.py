
import torch
import torch.nn as nn

from registry import domain_fusing_training_segmentation_handler_register
from registry.algorithms.build_model_algorithm import ModelBuilding
from registry.DomainAdaptation_algorithms.DANN.utils.GradientReversal_Builder import (
    GradientRerversalBuilder,
)
from registry.log import StatAccumulator, enable_epoch_logging, setup_logger
from utils import create_save_folder


@domain_fusing_training_segmentation_handler_register.register("DANN_GradientReversalLayer_Building")
class DANN_GradientReversalLayer_Building(ModelBuilding):
    def __init__(self, config: dict):
        super(DANN_GradientReversalLayer_Building, self).__init__(config)
        
    def _pre_process(self, data):        
        pipeline_config = data.get("pipeline_config")
        
        builder = GradientRerversalBuilder(self.config.get("architecture_config"))
        builder.validate() if self.config.get("validate") else None
        gradient_reversal_module = builder.gradient_reversal_module

        return gradient_reversal_module, pipeline_config

    def _main_process(self, data):
        return super(DANN_GradientReversalLayer_Building, self)._main_process(data)

    def _post_process(self, data):
        return super(DANN_GradientReversalLayer_Building, self)._post_process(data)
    
    def handle(self, data: dict):
        # Unpack: (model, is_train, save_path, pretrained_path, train_logger, val_logger, grad_logger, stat_accumulator_instance)
        gradient_reversal_module, _, gradient_reversal_save_path, gradient_reversal_pretrained_path, _, _, _, _ = self.process(data)
        data.update(
            {
                "gradient_reversal_module": gradient_reversal_module,
                "gradient_reversal_save_path": gradient_reversal_save_path,  # For saving checkpoints
                "gradient_reversal_pretrained_path": gradient_reversal_pretrained_path  # For loading (None when fresh training)
            }
        )
        return super(ModelBuilding, self).handle(data)
