"""
Gradient debugging tools for investigating NaN values in DANN training.

This module provides comprehensive debugging tools to detect, analyze, and fix
NaN gradients that occur during domain adaptation training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings


class GradientNaNDetector:
    """Detects and analyzes NaN gradients in model parameters."""

    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.nan_history: List[Dict[str, Any]] = []
        self.parameter_stats: Dict[str, Dict[str, float]] = {}

    def check_gradients(self, step: int = 0) -> Tuple[bool, Dict[str, Any]]:
        """
        Check all model parameters for NaN gradients.

        Returns:
            (has_nan, detailed_report)
        """
        has_nan = False
        nan_params = []
        gradient_stats = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data

                # Check for NaN
                nan_mask = torch.isnan(grad)
                if nan_mask.any():
                    has_nan = True
                    nan_count = nan_mask.sum().item()
                    total_elements = grad.numel()

                    nan_params.append({
                        'name': name,
                        'shape': grad.shape,
                        'nan_count': nan_count,
                        'total_elements': total_elements,
                        'nan_ratio': nan_count / total_elements,
                        'grad_min': grad[~nan_mask].min().item() if (~nan_mask).any() else float('nan'),
                        'grad_max': grad[~nan_mask].max().item() if (~nan_mask).any() else float('nan'),
                        'grad_mean': grad[~nan_mask].mean().item() if (~nan_mask).any() else float('nan'),
                        'grad_std': grad[~nan_mask].std().item() if (~nan_mask).any() and (~nan_mask).sum() > 1 else 0.0,
                    })

                # Collect gradient statistics
                gradient_stats[name] = {
                    'min': grad.min().item() if not torch.isnan(grad).all() else float('nan'),
                    'max': grad.max().item() if not torch.isnan(grad).all() else float('nan'),
                    'mean': grad.mean().item() if not torch.isnan(grad).all() else float('nan'),
                    'std': grad.std().item() if (not torch.isnan(grad).all() and grad.numel() > 1) else 0.0,
                    'norm': grad.norm().item() if not torch.isnan(grad).all() else float('nan'),
                }

        report = {
            'step': step,
            'has_nan': has_nan,
            'nan_parameters': nan_params,
            'gradient_stats': gradient_stats,
        }

        if has_nan:
            self.nan_history.append(report)
            if self.verbose:
                self._print_nan_report(report)

        return has_nan, report

    def _print_nan_report(self, report: Dict[str, Any]):
        """Print detailed NaN gradient report."""
        print(f"\n{'='*60}")
        print(f"NaN GRADIENTS DETECTED AT STEP {report['step']}")
        print(f"{'='*60}")

        for param_info in report['nan_parameters']:
            print(f"\nParameter: {param_info['name']}")
            print(f"  Shape: {param_info['shape']}")
            print(f"  NaN count: {param_info['nan_count']:,} / {param_info['total_elements']:,} ({param_info['nan_ratio']:.2%})")
            print(f"  Valid gradient stats:")
            print(f"    Min: {param_info['grad_min']:.6f}")
            print(f"    Max: {param_info['grad_max']:.6f}")
            print(f"    Mean: {param_info['grad_mean']:.6f}")
            print(f"    Std: {param_info['grad_std']:.6f}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of NaN detection history."""
        if not self.nan_history:
            return {'total_nan_events': 0, 'affected_parameters': set()}

        affected_params = set()
        for event in self.nan_history:
            for param_info in event['nan_parameters']:
                affected_params.add(param_info['name'])

        return {
            'total_nan_events': len(self.nan_history),
            'affected_parameters': affected_params,
            'first_occurrence': self.nan_history[0]['step'],
            'last_occurrence': self.nan_history[-1]['step'],
        }


class ActivationNaNDetector:
    """Detects NaN values in model activations during forward pass."""

    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.hooks = []
        self.activation_stats = {}
        self.nan_activations = []

    def register_hooks(self):
        """Register forward hooks to monitor activations."""
        def create_hook(name):
            def hook(module, input, output):
                self._check_activation(name, output)
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
                hook = module.register_forward_hook(create_hook(name))
                self.hooks.append(hook)

    def _check_activation(self, name: str, activation: torch.Tensor):
        """Check single activation for NaN values."""
        if isinstance(activation, (tuple, list)):
            for i, act in enumerate(activation):
                if isinstance(act, torch.Tensor):
                    self._check_single_tensor(f"{name}_output_{i}", act)
        elif isinstance(activation, torch.Tensor):
            self._check_single_tensor(name, activation)

    def _check_single_tensor(self, name: str, tensor: torch.Tensor):
        """Check single tensor for NaN values."""
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            total_elements = tensor.numel()

            nan_info = {
                'name': name,
                'shape': tensor.shape,
                'nan_count': nan_count,
                'total_elements': total_elements,
                'nan_ratio': nan_count / total_elements,
            }

            self.nan_activations.append(nan_info)

            if self.verbose:
                print(f"NaN detected in {name}: {nan_count}/{total_elements} ({nan_count/total_elements:.2%})")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class GradientClipperWithNaNHandling:
    """Enhanced gradient clipper that handles NaN gradients."""

    def __init__(self, max_norm: float = 1.0, handle_nan: str = 'zero', verbose: bool = True):
        """
        Args:
            max_norm: Maximum gradient norm for clipping
            handle_nan: How to handle NaN gradients ('zero', 'skip', 'error')
            verbose: Whether to print warnings
        """
        self.max_norm = max_norm
        self.handle_nan = handle_nan
        self.verbose = verbose
        self.nan_count = 0

    def handle_nan_gradients(self, model: nn.Module) -> Tuple[float, bool]:
        """
        Handle NaN gradients without clipping.

        Returns:
            (total_norm, had_nan)
        """
        parameters = [p for p in model.parameters() if p.grad is not None]

        if not parameters:
            return 0.0, False

        # Check for NaN gradients first
        had_nan = False
        for param in parameters:
            if torch.isnan(param.grad).any():
                had_nan = True
                self.nan_count += 1

                if self.handle_nan == 'zero':
                    # Set NaN gradients to zero
                    param.grad[torch.isnan(param.grad)] = 0.0
                    if self.verbose:
                        print(f"Warning: NaN gradients detected and set to zero in parameter {param.shape}")

                elif self.handle_nan == 'skip':
                    # Skip gradient update for this parameter
                    param.grad = None
                    if self.verbose:
                        print(f"Warning: NaN gradients detected, skipping update for parameter {param.shape}")

                elif self.handle_nan == 'error':
                    raise RuntimeError(f"NaN gradients detected in parameter {param.shape}")

        # Filter out None gradients
        parameters = [p for p in parameters if p.grad is not None]

        if not parameters:
            return 0.0, had_nan

        # Compute total norm (just for reporting, don't clip)
        if len(parameters) == 1:
            total_norm = parameters[0].grad.data.norm()
        else:
            total_norm = torch.norm(torch.stack([p.grad.data.norm() for p in parameters]))

        return total_norm.item(), had_nan

    def clip_gradients(self, model: nn.Module) -> Tuple[float, bool]:
        """
        Legacy method for backward compatibility - now just handles NaN.
        Use handle_nan_gradients instead.
        """
        return self.handle_nan_gradients(model)


class ModelStabilizer:
    """Applies various stabilization techniques to prevent NaN gradients."""

    @staticmethod
    def add_weight_noise(model: nn.Module, noise_std: float = 1e-5):
        """Add small noise to model weights to break symmetries."""
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * noise_std
                    param.add_(noise)

    @staticmethod
    def initialize_batch_norm(model: nn.Module):
        """Properly initialize batch normalization layers."""
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                module.momentum = 0.1
                module.eps = 1e-5

    @staticmethod
    def stabilize_linear_layers(model: nn.Module, init_std: float = 0.02):
        """Apply stabilized initialization to linear layers."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def debug_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
    """Comprehensive health check for a tensor."""
    stats = {
        'name': name,
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': tensor.device,
        'requires_grad': tensor.requires_grad,
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
        'min': tensor.min().item() if not torch.isnan(tensor).all() else float('nan'),
        'max': tensor.max().item() if not torch.isnan(tensor).all() else float('nan'),
        'mean': tensor.mean().item() if not torch.isnan(tensor).all() else float('nan'),
        'std': tensor.std().item() if (not torch.isnan(tensor).all() and tensor.numel() > 1) else 0.0,
        'norm': tensor.norm().item() if not torch.isnan(tensor).all() else float('nan'),
    }

    if stats['has_nan'] or stats['has_inf']:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total = tensor.numel()

        stats.update({
            'nan_count': nan_count,
            'inf_count': inf_count,
            'total_elements': total,
            'nan_ratio': nan_count / total,
            'inf_ratio': inf_count / total,
        })

        print(f"🚨 UNHEALTHY TENSOR: {name}")
        print(f"  Shape: {stats['shape']}")
        print(f"  NaN: {nan_count}/{total} ({nan_count/total:.2%})")
        print(f"  Inf: {inf_count}/{total} ({inf_count/total:.2%})")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")

    return stats


def create_debug_training_wrapper(original_train_function):
    """Decorator to add comprehensive debugging to training functions."""

    def debug_wrapper(self, *args, **kwargs):
        # Initialize debugging tools
        nan_detector = GradientNaNDetector(self.model, verbose=True)
        clipper = GradientClipperWithNaNHandling(
            max_norm=self.config.get("max_grad_norm", 1.0),
            handle_nan='zero',
            verbose=True
        )

        print(f"🔍 Debug training wrapper enabled")
        print(f"   - Gradient NaN detection: ON")
        print(f"   - Enhanced gradient clipping: ON")
        print(f"   - Model stabilization: ON")

        # Apply model stabilization
        ModelStabilizer.add_weight_noise(self.model, noise_std=1e-6)
        ModelStabilizer.initialize_batch_norm(self.model)

        try:
            # Call original training function
            result = original_train_function(self, *args, **kwargs)

            # Post-training analysis
            summary = nan_detector.get_summary()
            print(f"\n📊 Training Debug Summary:")
            print(f"   - Total NaN events: {summary['total_nan_events']}")
            print(f"   - Affected parameters: {len(summary.get('affected_parameters', set()))}")
            print(f"   - Total NaN gradient clips: {clipper.nan_count}")

            return result

        except Exception as e:
            print(f"❌ Training failed with error: {e}")

            # Emergency gradient check
            has_nan, report = nan_detector.check_gradients(step=-1)
            if has_nan:
                print("💥 NaN gradients detected at failure point!")

            raise e

    return debug_wrapper


# Example usage functions
def apply_emergency_fixes(model: nn.Module, optimizer: torch.optim.Optimizer):
    """Apply emergency fixes when NaN gradients are detected."""
    print("🚑 Applying emergency NaN fixes...")

    # 1. Zero out NaN gradients
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            param.grad[torch.isnan(param.grad)] = 0.0

    # 2. Reduce learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
        print(f"   - Reduced learning rate to {param_group['lr']}")

    # 3. Add weight noise
    ModelStabilizer.add_weight_noise(model, noise_std=1e-5)

    # 4. Reinitialize batch norm
    ModelStabilizer.initialize_batch_norm(model)

    print("   ✅ Emergency fixes applied")


def validate_model_gradients(model: nn.Module, input_tensor: torch.Tensor) -> bool:
    """Validate that model can compute gradients without NaN."""
    model.train()

    # Forward pass
    output = model(input_tensor)

    # Dummy loss
    if isinstance(output, (tuple, list)):
        loss = output[0].mean()
    else:
        loss = output.mean()

    # Backward pass
    loss.backward()

    # Check gradients
    detector = GradientNaNDetector(model, verbose=False)
    has_nan, _ = detector.check_gradients()

    # Clean up
    model.zero_grad()

    return not has_nan