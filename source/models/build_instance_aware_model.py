"""
Model building utilities for instance-aware chromosome segmentation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path

from source.models.instance_aware_model_wrapper import (
    InstanceAwareModelWrapper,
    wrap_model_for_instance_aware_training
)


def build_instance_aware_model(
    base_model: nn.Module,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Build instance-aware model from configuration.

    Args:
        base_model: Base segmentation model (e.g., MassNet)
        config: Configuration dictionary with instance_aware settings

    Returns:
        Wrapped model ready for instance-aware training
    """
    instance_config = config.get('instance_aware', {})

    if not instance_config.get('enabled', False):
        return base_model

    print("="*70)
    print("Building Instance-Aware Model for Chromosome Segmentation")
    print("="*70)

    enable_center = instance_config.get('enable_center_head', True)
    enable_embedding = instance_config.get('enable_embedding_head', False)
    embedding_dim = instance_config.get('embedding_dim', 32)
    feature_layer = instance_config.get('feature_layer_name', None)

    print(f"  Enable Center Head: {enable_center}")
    print(f"  Enable Embedding Head: {enable_embedding}")
    if enable_embedding:
        print(f"  Embedding Dimension: {embedding_dim}")

    wrapped_model = wrap_model_for_instance_aware_training(
        model=base_model,
        enable_center_head=enable_center,
        enable_embedding_head=enable_embedding,
        embedding_dim=embedding_dim,
        feature_layer_name=feature_layer,
        num_classes=47
    )

    print("✓ Instance-aware model built successfully!")
    print("="*70)

    return wrapped_model


def modify_model_forward_for_features(model: nn.Module) -> nn.Module:
    """
    Modify an existing model to return features alongside predictions.

    This function adds a simple wrapper that extracts intermediate features.

    Args:
        model: Existing segmentation model

    Returns:
        Modified model that returns (features, predictions)
    """
    test_input = torch.randn(1, 1, 64, 64)
    try:
        with torch.no_grad():
            output = model(test_input)

        if isinstance(output, dict):
            return model

        if isinstance(output, tuple) and len(output) == 2:
            return model

    except Exception as e:
        print(f"Warning: Could not test model output: {e}")

    class FeatureReturningWrapper(nn.Module):
        """Wrapper that makes model return dict with features and predictions"""

        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

            self.feature_hook = None
            self.features = None

            self._register_feature_hook()

        def _register_feature_hook(self):
            """Register forward hook to extract features"""
            feature_layers = []

            for name, module in self.base_model.named_modules():
                if any(keyword in name.lower() for keyword in ['neck', 'decoder', 'head.conv_block_0', 'features']):
                    feature_layers.append((name, module))

            if not feature_layers:
                for name, module in self.base_model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        feature_layers.append((name, module))

            if feature_layers:
                layer_name, layer_module = feature_layers[-1]
                print(f"  Extracting features from: {layer_name}")

                def hook_fn(module, input, output):
                    self.features = output

                self.feature_hook = layer_module.register_forward_hook(hook_fn)
            else:
                print("  Warning: Could not find suitable layer for feature extraction")

        def forward(self, x):
            """Forward pass returning dict with features and predictions"""
            predictions = self.base_model(x)

            # Extract predictions tensor if model returns tuple
            if isinstance(predictions, tuple):
                # Model returns (features, preds) or similar
                if len(predictions) == 2:
                    features, preds = predictions
                    return {
                        'predictions': preds,
                        'features': features
                    }
                else:
                    predictions = predictions[-1]

            if self.features is not None:
                return {
                    'predictions': predictions,
                    'features': self.features
                }

            return {
                'predictions': predictions
            }

        def inference(self, x):
            """Inference mode: predictions only"""
            output = self.forward(x)
            return output['predictions']

    wrapped = FeatureReturningWrapper(model)
    return wrapped


def example_usage():
    """
    Example of how to use instance-aware model building.
    """
    from source.models import build_model

    base_model = build_model(model_config)

    config = {
        'instance_aware': {
            'enabled': True,
            'enable_center_head': True,
            'enable_embedding_head': False,
            'embedding_dim': 32
        }
    }

    model = build_instance_aware_model(base_model, config)

    x = torch.randn(2, 1, 512, 512)
    output = model(x)

    print(f"Output keys: {output.keys()}")
    print(f"Predictions shape: {output['predictions'].shape}")
    if 'features' in output:
        print(f"Features shape: {output['features'].shape}")
    if 'center_preds' in output:
        print(f"Center predictions shape: {output['center_preds'].shape}")


if __name__ == "__main__":
    example_usage()