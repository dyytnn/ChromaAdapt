from .registers import (
    activation_function_register,
    backbone_register,
    domain_fusing_training_segmentation_handler_register,
    evaluate_segmentation_builder_register,
    evaluate_segmentation_handler_register,
    head_register,
    model_register,
    module_register,
    neck_register,
    norm_register,
    optimizer_register,
    pooling_register,
    scheduler_register,
    segmentation_loss_register,
    training_segmentation_builder_register,
    training_segmentation_handler_register,
    transformation_register,
)
from .registry import Registry


def _register_all_modules():
    """Import all modules to trigger decorators for registration."""
    # from .DomainAdaptation_algorithms import Adversarial, calc_coeff
    from .evaluate_algorithms import EvaluateMetaphaseSegmentation
    from .process_builder import EvaluateSegmentation, TrainingSegmentation

_register_all_modules()

__all__: list[str] = [
    "Registry",
    "activation_function_register",
    "backbone_register",
    "head_register",
    "model_register",
    "module_register",
    "neck_register",
    "norm_register",
    "optimizer_register",
    "pooling_register",
    "scheduler_register",
    "segmentation_loss_register",
    "training_segmentation_builder_register",
    "training_segmentation_handler_register",
    "evaluate_segmentation_builder_register",
    "evaluate_segmentation_handler_register",
    "transformation_register",
    "domain_fusing_training_segmentation_handler_register",
    # "Adversarial",
    # "calc_coeff",
    "EvaluateMetaphaseSegmentation",
    "EvaluateSegmentation",
    "TrainingSegmentation"
]
