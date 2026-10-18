from .registry import Registry

norm_register = Registry(name = "norm_register")
activation_function_register = Registry(name = "activation_function_register")
pooling_register = Registry(name = "pooling_register")

backbone_register = Registry(name = "backbone_register")
neck_register = Registry(name = "neck_register")
head_register = Registry(name = "head_register")
module_register = Registry(name = "module_register")
model_register = Registry(name = "model_register")

optimizer_register = Registry(name = "optimizer_register")
scheduler_register = Registry(name = "scheduler_register")

transformation_register = Registry(name = "transformation_register")

segmentation_loss_register = Registry(name = "segmentation_loss_register")

training_segmentation_builder_register = Registry(name = "training_segmentation_builder_register")
training_segmentation_handler_register = Registry(name = "training_segmentation_handler_register")
domain_fusing_training_segmentation_handler_register = Registry(name = "domain_fusing_training_segmentation_handler_register")

evaluate_segmentation_builder_register = Registry(name = "evaluate_segmentation_builder_register")
evaluate_segmentation_handler_register = Registry(name = "evaluate_segmentation_handler_register")

