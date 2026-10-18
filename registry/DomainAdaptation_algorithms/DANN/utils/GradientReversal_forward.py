from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function


class GradReversalLayer(Function):
    DEBUG_ASSERTIONS = True

    @staticmethod
    def forward(ctx: Any, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_outputs: Tensor):
        outputs = grad_outputs.neg() * ctx.alpha

        if GradReversalLayer.DEBUG_ASSERTIONS and ctx.alpha != 0:
            if abs(ctx.alpha - 1.0) < 1e-6:
                reversed_plus_original = outputs + grad_outputs
                assert torch.allclose(reversed_plus_original, torch.zeros_like(reversed_plus_original), atol=1e-6), \
                    f"Gradient reversal failed: reversed + original = {reversed_plus_original.abs().max().item()}"

            expected_magnitude = grad_outputs.abs() * abs(ctx.alpha)
            actual_magnitude = outputs.abs()
            assert torch.allclose(actual_magnitude, expected_magnitude, atol=1e-6), \
                f"Magnitude preservation failed: expected {expected_magnitude.max().item()}, got {actual_magnitude.max().item()}"

            if ctx.alpha > 0:
                sign_product = torch.sign(grad_outputs) * torch.sign(outputs)
                non_zero_mask = (grad_outputs.abs() > 1e-8) & (outputs.abs() > 1e-8)
                if non_zero_mask.any():
                    assert torch.all(sign_product[non_zero_mask] <= 0), \
                        "Signs should be opposite for gradient reversal"

        return outputs, None

class ConditionalGradReversal:
    @staticmethod
    def apply(x: Tensor, epsilon: float, apply_reversal: bool = True) -> Tensor:
        """
        Conditionally apply gradient reversal.
        
        Args:
            x: Input tensor
            alpha: Gradient reversal strength
            apply_reversal: Whether to apply gradient reversal (default: True)
            
        Returns:
            Tensor with or without gradient reversal applied
        """
        if apply_reversal:
            return GradReversalLayer.apply(x, epsilon)
        else:
            return x