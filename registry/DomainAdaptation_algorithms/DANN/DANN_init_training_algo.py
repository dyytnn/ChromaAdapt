import os

import torch
import torch.nn as nn

from abs_interface import AbsAlgorithmProcessHandler
from registry import domain_fusing_training_segmentation_handler_register
from registry.DomainAdaptation_algorithms.DANN.utils.DANN_model_wrapper import (
    DANNModelWrapper,
)
from registry.DomainAdaptation_algorithms.DANN.utils.utils import (
    get_dann_params_groups_with_head_lr,
)
from utils import r_, restart_from_checkpoint, sr_


@domain_fusing_training_segmentation_handler_register.register("DANN_initTrainingHandler")
class DANN_init_training_algo(AbsAlgorithmProcessHandler):
    def __init__(self):
        super(DANN_init_training_algo, self).__init__()
        
    def _pre_process(self, data):
        return data
    
    def _main_process(self, data: dict) -> tuple[DANNModelWrapper, nn.BCEWithLogitsLoss]:
        """
        Initialize DANN model wrapper with domain classification capabilities.

        Args:
            data: Dictionary containing model and gradient reversal module

        Returns:
            Tuple of (wrapped_model, domain_classification_loss)
        """
        base_model: nn.Module = data.get("model", None)
        gradient_reversal_module: nn.Module = data.get("gradient_reversal_module", None)
        gradient_reversal_pretrained_path = data.get("gradient_reversal_pretrained_path", None)

        # Initialize domain classification loss
        domain_classification_loss = nn.BCEWithLogitsLoss()

        # Load pretrained weights if provided
        gradient_reversal_module = self._load_pretrained_weights(
            gradient_reversal_module,
            gradient_reversal_pretrained_path
        )

        # Wrap the base model with DANN capabilities
        # Use multi-head feature extractor for richer domain representations
        dann_model = DANNModelWrapper(
            base_model=base_model,
            domain_classifier=gradient_reversal_module,
            use_multihead_extractor=True,  # ✅ ENABLED: Provides LayerNorm for gradient stability
            feature_channels=288  # ✅ MassNet bottleneck channels (outputs 576 after GAP+GMP)
        )

        return dann_model, domain_classification_loss

    def _load_pretrained_weights(
        self,
        domain_classifier: nn.Module,
        pretrained_path: str
    ) -> nn.Module:
        """
        Load pretrained weights for gradient reversal module.

        Args:
            domain_classifier: The domain classifier module
            pretrained_path: Path to pretrained weights

        Returns:
            Module with loaded weights
        """
        if not pretrained_path:
            return domain_classifier

        checkpoint_path = os.path.join(pretrained_path, "last_save_dict.pth")

        if not os.path.exists(checkpoint_path):
            print(f"{r_}[WARNING] Checkpoint not found at {checkpoint_path}{sr_}")
            return domain_classifier

        try:
            to_restore = {"domain_classifier": domain_classifier}
            restart_from_checkpoint(
                checkpoint_path,
                run_variables=to_restore,
                domain_classifier=domain_classifier,
            )

            loaded_module = to_restore["domain_classifier"]
            print(f"{r_}[INFO] Gradient Reversal Module loaded from {checkpoint_path}{sr_}") # todo here  is some logic problem
            return loaded_module 

        except Exception as e:
            print(f"{r_}[ERROR] Failed to load checkpoint: {e}{sr_}")
            return domain_classifier
    
    def _post_process(self, data):
        return data
    
    def handle(self, data: dict):
        model, domain_classification_loss = self.process(data)
        data.update(
            {
                "model": model,
                "domain_classification_loss": domain_classification_loss,
                "custom_param_groups_fn": get_dann_params_groups_with_head_lr
            }
        )
        return super().handle(data)