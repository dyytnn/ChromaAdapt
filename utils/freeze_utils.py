"""
Utility functions for freezing/unfreezing model layers during fine-tuning.

This module provides flexible layer freezing strategies for domain adaptation and transfer learning.
"""

import torch.nn as nn
from typing import Optional, List


def freeze_module(module: nn.Module, verbose: bool = True):
    """
    Freeze all parameters in a module (stop gradient updates).

    Args:
        module: PyTorch module to freeze
        verbose: Print freezing information
    """
    if module is None:
        return

    frozen_count = 0
    for param in module.parameters():
        param.requires_grad = False
        frozen_count += 1

    if verbose and frozen_count > 0:
        print(f"  Frozen {frozen_count} parameters in {module.__class__.__name__}")


def unfreeze_module(module: nn.Module, verbose: bool = True):
    """
    Unfreeze all parameters in a module (enable gradient updates).

    Args:
        module: PyTorch module to unfreeze
        verbose: Print unfreezing information
    """
    if module is None:
        return

    unfrozen_count = 0
    for param in module.parameters():
        param.requires_grad = True
        unfrozen_count += 1

    if verbose and unfrozen_count > 0:
        print(f"  Unfrozen {unfrozen_count} parameters in {module.__class__.__name__}")


def freeze_layers(model, strategy: str = "freeze_backbone", verbose: bool = True):
    """
    Freeze layers based on specified strategy.

    Args:
        model: Abstract_SegModel with backbone, neck, and head attributes
        strategy: Freezing strategy to use
        verbose: Print detailed freezing information

    Strategies:
        - "none": Don't freeze anything (default fine-tuning)
        - "freeze_backbone": Freeze entire backbone, train neck + head
        - "freeze_early_layers": Freeze early backbone layers (input_conv, scale_8)
        - "heads_only": Freeze backbone + neck, train only heads
        - "progressive_unfreeze": Start with everything frozen except heads

    Returns:
        Number of trainable parameters after freezing
    """

    if strategy == "none":
        if verbose:
            print("No freezing applied - all parameters trainable")
        return count_trainable_params(model, verbose=verbose)

    if verbose:
        print(f"\nApplying freezing strategy: '{strategy}'")
        print("=" * 60)

    # First unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    if strategy == "freeze_backbone":
        """
        Freeze: backbone
        Train: neck + head

        Use when: Features transfer well, need to adapt high-level representations
        Your case: 0.98 → 0.70 shows good transfer, try this SECOND
        """
        if verbose:
            print("Strategy: Freeze backbone, train neck + head")
            print("-" * 60)

        freeze_module(model.backbone, verbose=verbose)

    elif strategy == "freeze_early_layers":
        """
        Freeze: backbone[0:2] (input_conv + scale_8_blocks)
        Train: backbone[2:] + neck + head

        Use when: Low-level features are similar, high-level need adaptation
        Your case: Good middle-ground approach
        """
        if verbose:
            print("Strategy: Freeze early backbone layers")
            print("-" * 60)

        # MassNet backbone structure: [input_conv, scale_8, scale_4, scale_2]
        if hasattr(model.backbone, '__getitem__'):
            # Freeze input_conv and scale_8_blocks
            freeze_module(model.backbone[0], verbose=verbose)  # input_conv
            freeze_module(model.backbone[1], verbose=verbose)  # scale_8_blocks
            if verbose:
                print("  Keeping trainable: scale_4_blocks, scale_2_blocks, neck, head")
        else:
            print("Warning: Backbone structure not as expected, freezing entire backbone")
            freeze_module(model.backbone, verbose=verbose)

    elif strategy == "heads_only":
        """
        Freeze: backbone + neck
        Train: head only

        Use when: Very small target dataset or very similar domains
        Your case: Try this if freeze_backbone works but overfits
        """
        if verbose:
            print("Strategy: Train heads only")
            print("-" * 60)

        freeze_module(model.backbone, verbose=verbose)
        freeze_module(model.neck, verbose=verbose)

    elif strategy == "progressive_unfreeze":
        """
        Freeze: everything except heads (to be unfrozen progressively)

        Use when: Need gradual adaptation (call unfreeze_progressive() during training)
        """
        if verbose:
            print("Strategy: Progressive unfreezing (start with heads only)")
            print("-" * 60)

        freeze_module(model.backbone, verbose=verbose)
        freeze_module(model.neck, verbose=verbose)
        if verbose:
            print("  Note: Unfreeze layers progressively using unfreeze_progressive()")

    else:
        raise ValueError(f"Unknown freezing strategy: {strategy}")

    if verbose:
        print("=" * 60)

    return count_trainable_params(model, verbose=verbose)


def unfreeze_progressive(model, epoch: int, schedule: Optional[dict] = None, verbose: bool = True):
    """
    Progressively unfreeze layers based on epoch schedule.

    Args:
        model: Model with backbone, neck, head
        epoch: Current training epoch
        schedule: Custom unfreezing schedule {epoch: modules_to_unfreeze}
        verbose: Print unfreezing information

    Default schedule:
        epoch 0-9: heads only
        epoch 10-19: + neck
        epoch 20-29: + backbone[3] (scale_2_blocks)
        epoch 30-39: + backbone[2] (scale_4_blocks)
        epoch 40+: all layers
    """

    if schedule is None:
        # Default progressive schedule
        schedule = {
            10: ["neck"],
            20: ["backbone.3"],  # scale_2_blocks
            30: ["backbone.2"],  # scale_4_blocks
            40: ["backbone.1", "backbone.0"]  # scale_8 + input_conv
        }

    for trigger_epoch, modules in schedule.items():
        if epoch == trigger_epoch:
            if verbose:
                print(f"\nEpoch {epoch}: Unfreezing {modules}")

            for module_path in modules:
                module = get_module_by_path(model, module_path)
                if module is not None:
                    unfreeze_module(module, verbose=verbose)


def get_module_by_path(model, path: str):
    """
    Get a module by its attribute path (e.g., "backbone.2" or "neck").

    Args:
        model: Root model
        path: Dot-separated path to module

    Returns:
        Module at the specified path or None
    """
    parts = path.split('.')
    current = model

    for part in parts:
        if part.isdigit():
            # Handle indexed access (e.g., backbone[2])
            try:
                current = current[int(part)]
            except (IndexError, TypeError):
                print(f"Warning: Could not access index {part} in {current}")
                return None
        else:
            # Handle attribute access
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                print(f"Warning: Module has no attribute '{part}'")
                return None

    return current


def count_trainable_params(model, verbose: bool = True):
    """
    Count and display trainable vs frozen parameters.

    Args:
        model: PyTorch model
        verbose: Print detailed breakdown

    Returns:
        Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    if verbose:
        print(f"\nParameter Summary:")
        print(f"  Trainable:   {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen:      {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"  Total:       {total_params:,}")
        print()

    return trainable_params


def print_layer_status(model, verbose: bool = True):
    """
    Print the frozen/trainable status of each layer.

    Args:
        model: Model to inspect
        verbose: Print detailed status
    """
    if not verbose:
        return

    print("\nLayer-wise Training Status:")
    print("=" * 60)

    for name, module in model.named_children():
        if not list(module.parameters()):
            continue

        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        status = "TRAINABLE" if trainable > 0 else "FROZEN"
        percentage = 100 * trainable / total if total > 0 else 0

        print(f"  {name:20s}: {status:10s} ({trainable:,} / {total:,} = {percentage:.1f}%)")

    print("=" * 60)


# Example usage:
if __name__ == "__main__":
    # This demonstrates how to use the freezing utilities
    print("Freezing Utilities - Example Usage\n")

    print("Strategy 1: freeze_backbone")
    print("  → Freeze backbone, train neck + head")
    print("  → Use when features transfer well\n")

    print("Strategy 2: freeze_early_layers")
    print("  → Freeze input_conv + scale_8_blocks")
    print("  → Train scale_4, scale_2, neck, head\n")

    print("Strategy 3: heads_only")
    print("  → Freeze backbone + neck")
    print("  → Train only segmentation heads\n")

    print("Strategy 4: progressive_unfreeze")
    print("  → Start with heads only")
    print("  → Gradually unfreeze deeper layers\n")