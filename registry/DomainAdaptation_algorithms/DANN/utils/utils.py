"""
Utility functions for DANN (Domain-Adversarial Neural Network) training.
"""
import torch.nn as nn


def get_dann_params_groups_with_head_lr(model: nn.Module, base_lr: float, head_lr_multiplier: float = 10.0):
    backbone_regularized = []
    backbone_not_regularized = []
    head_regularized = []
    head_not_regularized = []
    domain_classifier_regularized = []
    domain_classifier_not_regularized = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Identify parameter type
        is_head_param = ('ChromosomeSegmentation' in name or
                        'OverlapChromosomeSegmentation' in name)
        is_domain_classifier = 'domain_classifier' in name

        # We do not regularize biases nor Norm parameters (batch norm, layer norm, etc.)
        is_not_regularized = name.endswith(".bias") or len(param.shape) == 1

        # Categorize parameters
        if is_head_param:
            if is_not_regularized:
                head_not_regularized.append(param)
            else:
                head_regularized.append(param)
        elif is_domain_classifier:
            if is_not_regularized:
                domain_classifier_not_regularized.append(param)
            else:
                domain_classifier_regularized.append(param)
        else:
            # Backbone and neck parameters
            if is_not_regularized:
                backbone_not_regularized.append(param)
            else:
                backbone_regularized.append(param)

    # Calculate learning rates
    head_lr = base_lr * head_lr_multiplier

    # Build parameter groups
    param_groups = [
        # Backbone parameters (lower LR)
        {'params': backbone_regularized, 'lr': base_lr},
        {'params': backbone_not_regularized, 'lr': base_lr, 'weight_decay': 0.},

        # Domain classifier (same as backbone)
        {'params': domain_classifier_regularized, 'lr': base_lr},
        {'params': domain_classifier_not_regularized, 'lr': base_lr, 'weight_decay': 0.},

        # Segmentation heads (higher LR to overcome BatchNorm suppression)
        {'params': head_regularized, 'lr': head_lr},
        {'params': head_not_regularized, 'lr': head_lr, 'weight_decay': 0.},
    ]

    # Filter out empty groups
    param_groups = [pg for pg in param_groups if len(pg['params']) > 0]

    # Print summary
    total_params = sum(len(pg['params']) for pg in param_groups)
    print(f"[DANN Param Groups] Total trainable parameter tensors: {total_params}")
    print(f"  - Backbone params: {len(backbone_regularized) + len(backbone_not_regularized)} tensors @ LR={base_lr:.2e}")
    print(f"  - Domain classifier params: {len(domain_classifier_regularized) + len(domain_classifier_not_regularized)} tensors @ LR={base_lr:.2e}")
    print(f"  - Segmentation head params: {len(head_regularized) + len(head_not_regularized)} tensors @ LR={head_lr:.2e} ({head_lr_multiplier}x)")

    return param_groups
