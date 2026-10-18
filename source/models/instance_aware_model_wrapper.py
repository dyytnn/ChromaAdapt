"""
Instance-Aware Model Wrapper for Chromosome Segmentation
Wraps existing segmentation model to output additional features for instance discrimination
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class InstanceAwareModelWrapper(nn.Module):
    """
    Wraps segmentation model to provide:
    1. Segmentation predictions (as usual)
    2. Feature maps for instance discrimination
    3. Optional: Center prediction head
    4. Optional: Embedding head
    """
    def __init__(
        self,
        base_model: nn.Module,
        enable_center_head: bool = True,
        enable_embedding_head: bool = False,
        embedding_dim: int = 32,
        num_classes: int = 47
    ):
        """
        Args:
            base_model: Your existing segmentation model (e.g., MassNet)
            enable_center_head: Add center prediction head
            enable_embedding_head: Add embedding head for instance discrimination
            embedding_dim: Dimension of embedding space
            num_classes: Number of classes (47 for chromosome segmentation)
        """
        super().__init__()
        self.base_model = base_model
        self.enable_center_head = enable_center_head
        self.enable_embedding_head = enable_embedding_head
        self.num_classes = num_classes

        # Detect feature dimension from base model
        # We'll extract features from the layer before the final classification head
        self.feature_dim = self._detect_feature_dim()

        # Additional heads
        if self.enable_center_head:
            self.center_head = nn.Sequential(
                nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feature_dim // 2, num_classes, kernel_size=1),
                nn.Sigmoid()  # Output as probability heatmap
            )

        if self.enable_embedding_head:
            self.embedding_head = nn.Sequential(
                nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feature_dim // 2, embedding_dim, kernel_size=1)
            )

    def _detect_feature_dim(self) -> int:
        """
        Automatically detect feature dimension from base model.
        Tries to find the last feature map before classification head.
        """
        # Default feature dimensions for common architectures
        default_dims = {
            'Chromosome_MassNet_Metaphase': 288,
            'ResNet': 512,
            'UNet': 64,
        }

        # Try to detect from model name
        model_name = self.base_model.__class__.__name__
        for key, dim in default_dims.items():
            if key.lower() in model_name.lower():
                return dim

        # Fallback: try to infer from model structure
        # Look for the last convolutional layer before the classification head
        try:
            # Common pattern: model has a 'head' module
            if hasattr(self.base_model, 'head') or hasattr(self.base_model, 'HEAD'):
                head = getattr(self.base_model, 'head', None) or getattr(self.base_model, 'HEAD', None)
                if hasattr(head, 'Conv_Block_0'):
                    # MassNet pattern
                    return 32

            # Another common pattern: check last conv layer
            for name, module in reversed(list(self.base_model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    # Check if this is not the final classification layer
                    if module.out_channels != self.num_classes:
                        return module.out_channels
        except:
            pass

        # Ultimate fallback
        print(f"Warning: Could not detect feature dimension automatically. Using default: 32")
        return 32

    def _extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from base model before final classification.

        Returns:
            features: Feature maps for instance discrimination
            predictions: Final segmentation predictions
        """
        # Strategy 1: If base_model has separate feature extraction
        if hasattr(self.base_model, 'extract_features'):
            features = self.base_model.extract_features(x)
            predictions = self.base_model.head(features)
            return features, predictions

        # Strategy 2: If base_model returns tuple (features, predictions)
        output = self.base_model(x)
        if isinstance(output, tuple) and len(output) == 2:
            features, predictions = output
            return features, predictions

        # Strategy 3: Hook-based feature extraction (for models that don't expose features)
        # This is a fallback that will be implemented if needed
        features = None
        predictions = output

        # Try to get features from intermediate layer
        # This requires modifying base_model forward pass or using hooks
        if hasattr(self.base_model, 'get_features'):
            features = self.base_model.get_features(x)

        return features, predictions

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with instance-aware outputs.

        Args:
            x: Input images (B, C, H, W)
            return_all: Return all outputs (features, centers, embeddings)

        Returns:
            Dictionary containing:
            - 'predictions': (B, 47, H, W) segmentation logits
            - 'features': (B, C, H, W) feature maps for contrastive loss
            - 'center_preds': (B, 47, H, W) center heatmaps [optional]
            - 'embeddings': (B, D, H, W) instance embeddings [optional]
        """
        # Extract features and predictions
        features, predictions = self._extract_features(x)

        output_dict = {
            'predictions': predictions
        }

        # Add features if available
        if features is not None and return_all:
            output_dict['features'] = features

            # Generate center predictions
            if self.enable_center_head and return_all:
                center_preds = self.center_head(features)
                output_dict['center_preds'] = center_preds

            # Generate embeddings
            if self.enable_embedding_head and return_all:
                embeddings = self.embedding_head(features)
                output_dict['embeddings'] = embeddings

        return output_dict

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference mode: only return predictions (no extra heads).

        Args:
            x: Input images

        Returns:
            predictions: Segmentation predictions only
        """
        output = self.forward(x, return_all=False)
        return output['predictions']


class FeatureExtractionHook:
    """
    Helper class to extract intermediate features using forward hooks.
    Use this if your base model doesn't expose features directly.
    """
    def __init__(self, model: nn.Module, layer_name: str):
        """
        Args:
            model: The model to hook
            layer_name: Name of layer to extract features from
        """
        self.features = None
        self.hook = None
        self.layer_name = layer_name

        # Register hook
        for name, module in model.named_modules():
            if name == layer_name:
                self.hook = module.register_forward_hook(self._hook_fn)
                break

        if self.hook is None:
            raise ValueError(f"Layer {layer_name} not found in model")

    def _hook_fn(self, module, input, output):
        """Hook function to capture features"""
        self.features = output

    def remove(self):
        """Remove the hook"""
        if self.hook is not None:
            self.hook.remove()


def add_feature_extraction_to_model(
    model: nn.Module,
    feature_layer_name: Optional[str] = None
) -> nn.Module:
    """
    Modify an existing model to return features.

    This function adds a simple wrapper that extracts features from
    a specified layer and returns both features and predictions.

    Args:
        model: Existing segmentation model
        feature_layer_name: Name of layer to extract features from.
                           If None, tries to auto-detect.

    Returns:
        Modified model that returns (features, predictions)
    """
    # Auto-detect feature layer if not specified
    if feature_layer_name is None:
        # Common patterns for feature extraction
        candidates = [
            'neck', 'decoder', 'head.Conv_Block_0',
            'encoder.layer4', 'features'
        ]

        for candidate in candidates:
            found = False
            for name, _ in model.named_modules():
                if candidate in name:
                    feature_layer_name = name
                    found = True
                    break
            if found:
                break

        if feature_layer_name is None:
            raise ValueError(
                "Could not auto-detect feature layer. "
                "Please specify feature_layer_name manually."
            )

    print(f"Extracting features from layer: {feature_layer_name}")

    # Create wrapper
    class FeatureExtractingModel(nn.Module):
        def __init__(self, base_model, layer_name):
            super().__init__()
            self.base_model = base_model
            self.hook = FeatureExtractionHook(base_model, layer_name)

        def forward(self, x):
            predictions = self.base_model(x)
            features = self.hook.features
            return features, predictions

        def extract_features(self, x):
            """Standalone feature extraction"""
            _ = self.base_model(x)
            return self.hook.features

    return FeatureExtractingModel(model, feature_layer_name)


# Utility function to wrap existing models
def wrap_model_for_instance_aware_training(
    model: nn.Module,
    enable_center_head: bool = True,
    enable_embedding_head: bool = False,
    feature_layer_name: Optional[str] = None,
    **kwargs
) -> InstanceAwareModelWrapper:
    """
    Convenience function to wrap any segmentation model for instance-aware training.

    Args:
        model: Your existing segmentation model
        enable_center_head: Add center prediction head
        enable_embedding_head: Add embedding head
        feature_layer_name: Layer to extract features from (auto-detect if None)
        **kwargs: Additional arguments for InstanceAwareModelWrapper

    Returns:
        Wrapped model ready for instance-aware training
    """
    device = model.device
    test_input = torch.randn(1, 1, 64, 64).to(device)  # Grayscale test input
    try:
        output = model(test_input)
        if not isinstance(output, tuple):
            # Model doesn't return features, add extraction
            print("Model doesn't return features. Adding feature extraction...")
            model = add_feature_extraction_to_model(model, feature_layer_name)
    except Exception as e:
        print(f"Warning: Could not test model output: {e}")

    # Wrap with instance-aware wrapper
    wrapped_model = InstanceAwareModelWrapper(
        base_model=model,
        enable_center_head=enable_center_head,
        enable_embedding_head=enable_embedding_head,
        **kwargs
    )

    return wrapped_model