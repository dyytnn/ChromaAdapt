from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from registry.DomainAdaptation_algorithms.DANN.utils.GradientReversal_forward import (
    GradReversalLayer,
)

PoolType = Literal[
    "cls",
    "max",
    "mean",
    "sum",
    "cls_max",
    "mean_max",
    "attention",
]

class FeatureNormalizer(nn.Module):
    """
    Apply a consistent normalization on flattened features [B, C] before GRL.
    modes:
      - 'none'          : no-op
      - 'layernorm'     : per-sample LayerNorm
      - 'l2'            : per-sample L2 normalization
      - 'standardize'   : per-sample (x - mean)/std
      - 'batchnorm'     : LazyBatchNorm1d (batch statistics)
    """
    def __init__(self, mode: str = "layernorm", eps: float = 1e-6):
        super().__init__()
        self.mode = mode
        self.eps = eps
        if mode == "batchnorm":
            self.bn = nn.LazyBatchNorm1d()
        else:
            self.bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        elif self.mode == "layernorm":
            return F.layer_norm(x, x.shape[1:], eps=self.eps)
        elif self.mode == "l2":
            return F.normalize(x, p=2, dim=1, eps=self.eps)
        elif self.mode == "standardize":
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True).clamp_min(self.eps)
            return (x - mu) / sigma
        elif self.mode == "batchnorm":
            return self.bn(x)
        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

class AdvancedPooling(nn.Module):
    """
    Advanced pooling over token/patch embeddings with multiple strategies.

    Expected input:
        hidden_states: Float tensor of shape [B, N, D]
            B = batch size
            N = sequence length (#tokens/patches)
            D = hidden dimension
        mask (optional): Bool tensor of shape [B, N]
            True = valid token, False = padded/invalid token.
            If provided, mean/sum/max/attention will ignore masked tokens.

    Pooling types:
        - "cls"       : returns hidden_states[:, 0, :]              -> [B, D]
        - "max"       : (mask-aware) max over tokens                -> [B, D]
        - "mean"      : (mask-aware) mean over tokens               -> [B, D]
        - "sum"       : (mask-aware) sum over tokens                -> [B, D]
        - "cls_max"   : concat(cls, max)                            -> [B, 2D]
        - "mean_max"  : concat(mean, max)                           -> [B, 2D]
        - "attention" : softmax(q·x_i) weighted sum over tokens     -> [B, D]
                        with a learnable query vector q.

    Attention controls:
        - attn_scale   : if True, scales dot-product by 1/sqrt(D)
        - temperature  : softmax temperature (divide scores by T); if set,
                         combines with attn_scale multiplicatively.
        - q_norm       : L2-normalize the learnable query vector each forward
        - attn_dropout : dropout on attention weights after softmax
        - return_attn  : if True, also return attention weights [B, N]

    Post-pool normalization:
        - post_pool_norm in {"none", "layernorm", "l2", "standardize"}
          applies to the final pooled vector (per-sample).

    Notes:
        * Works with fp32/fp16/bf16.
        * Mask handling is robust for all pooling types that aggregate tokens.
        * For "cls" pooling, the mask is ignored (assumes a real CLS token).

    Example:
        pool = AdvancedPooling(
            pooling_type="attention",
            hidden_dim=768,
            attn_scale=True,
            temperature=1.0,
            q_norm=True,
            attn_dropout=0.1,
            post_pool_norm="layernorm",
        )
        y = pool(x, mask=mask)         # x: [B, N, D], mask: [B, N]
        # If return_attn=True: y, attn = pool(x, mask=mask)
    """

    def __init__(
        self,
        pooling_type: PoolType = "attention",
        hidden_dim: Optional[int] = None,
        *,
        attn_scale: bool = True,
        temperature: Optional[float] = None,
        q_norm: bool = True,
        attn_dropout: float = 0.0,
        return_attn: bool = False,
        post_pool_norm: Literal["none", "layernorm", "l2", "standardize"] = "none",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pooling_type: PoolType = pooling_type
        self.hidden_dim = hidden_dim  # used by attention; if None, inferred at first forward
        self.attn_scale = attn_scale
        self.temperature = float(temperature) if temperature is not None else None
        self.q_norm = q_norm
        self.return_attn = return_attn
        self.post_pool_norm = post_pool_norm
        self.eps = eps

        if self.pooling_type == "attention":
            if self.hidden_dim is not None:
                self.query_vector = nn.Parameter(torch.randn(self.hidden_dim))
                inv_sqrt_d = (self.hidden_dim ** -0.5) if self.hidden_dim > 0 else 1.0
                self.register_buffer("_inv_sqrt_d", torch.tensor(inv_sqrt_d), persistent=False)
            else:
                self.query_vector = None
                self.register_buffer("_inv_sqrt_d", torch.tensor(1.0), persistent=False)

            self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout and attn_dropout > 0 else nn.Identity()
        else:
            self.query_vector = None
            self.register_buffer("_inv_sqrt_d", torch.tensor(1.0), persistent=False)
            self.attn_drop = nn.Identity()

        self._output_dim: Optional[int] = None

    @staticmethod
    def _masked_sum(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.sum(dim=1)
        mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * mask_f).sum(dim=1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], eps: float) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
        s = (x * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(eps)
        return s / denom

    @staticmethod
    def _masked_max(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.amax(dim=1)
        finfo = torch.finfo(x.dtype) if x.dtype.is_floating_point else None
        very_low = finfo.min if finfo is not None else -1e9
        x_masked = x.masked_fill(~mask.unsqueeze(-1), very_low)
        return x_masked.amax(dim=1)

    @staticmethod
    def _normalize_vector(
        x: torch.Tensor,
        mode: Literal["none", "layernorm", "l2", "standardize"],
        eps: float,
    ) -> torch.Tensor:
        if mode == "none":
            return x
        if mode == "layernorm":
            return F.layer_norm(x, normalized_shape=(x.shape[-1],), eps=eps)
        if mode == "l2":
            return F.normalize(x, p=2, dim=-1, eps=eps)
        if mode == "standardize":
            mu = x.mean(dim=-1, keepdim=True)
            sigma = x.std(dim=-1, keepdim=True).clamp_min(eps)
            return (x - mu) / sigma
        raise ValueError(f"Unknown post_pool_norm: {mode}")

    def _maybe_init_attention(self, D: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.query_vector is None:
            self.hidden_dim = D
            q = torch.randn(D, device=device, dtype=dtype)
            self.query_vector = nn.Parameter(q)
            inv_sqrt_d = (D ** -0.5) if D > 0 else 1.0
            self._inv_sqrt_d = torch.tensor(inv_sqrt_d, device=device, dtype=dtype)

    def get_output_dim(self, input_dim: int) -> int:
        """
        Compute the output feature dimension for a given input_dim D.
        Useful for building downstream heads.
        """
        if self.pooling_type in ("cls", "max", "mean", "sum", "attention"):
            return input_dim
        elif self.pooling_type in ("cls_max", "mean_max"):
            return 2 * input_dim
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states: [B, N, D] float tensor
            mask: Optional[BoolTensor] [B, N] where True=valid

        Returns:
            pooled: [B, D] or [B, 2D] depending on pooling_type
            (attn_weights): [B, N] if pooling_type == "attention" and return_attn=True
        """
        if hidden_states.ndim != 3:
            raise ValueError(f"`hidden_states` must be [B, N, D], got {hidden_states.shape}")
        B, N, D = hidden_states.shape

        if mask is not None and (mask.ndim != 2 or mask.shape != (B, N)):
            raise ValueError(f"`mask` must be [B, N] bool, got {None if mask is None else mask.shape}")

        device = hidden_states.device
        dtype = hidden_states.dtype

        if self.pooling_type == "cls":
            pooled = hidden_states[:, 0, :]

        elif self.pooling_type == "max":
            pooled = self._masked_max(hidden_states, mask)

        elif self.pooling_type == "mean":
            pooled = self._masked_mean(hidden_states, mask, eps=self.eps)

        elif self.pooling_type == "sum":
            pooled = self._masked_sum(hidden_states, mask)

        elif self.pooling_type == "cls_max":
            cls_vec = hidden_states[:, 0, :]
            max_vec = self._masked_max(hidden_states, mask)
            pooled = torch.cat([cls_vec, max_vec], dim=-1)

        elif self.pooling_type == "mean_max":
            mean_vec = self._masked_mean(hidden_states, mask, eps=self.eps)
            max_vec = self._masked_max(hidden_states, mask)
            pooled = torch.cat([mean_vec, max_vec], dim=-1)

        elif self.pooling_type == "attention":
            self._maybe_init_attention(D=D, device=device, dtype=dtype)

            q = self.query_vector
            if self.q_norm:
                q = F.normalize(q, p=2, dim=0, eps=self.eps)

            scores = torch.matmul(hidden_states, q)

            if self.attn_scale:
                scale = self._inv_sqrt_d
                if scale.dtype != scores.dtype or scale.device != scores.device:
                    scale = scale.to(dtype=scores.dtype, device=scores.device)
                scores = scores * scale

            if self.temperature is not None:
                T = max(self.temperature, 1e-4)
                scores = scores / T

            if mask is not None:
                finfo = torch.finfo(scores.dtype) if scores.dtype.is_floating_point else None
                very_neg = finfo.min if finfo is not None else -1e9
                scores = scores.masked_fill(~mask, very_neg)

            attn = torch.softmax(scores, dim=1)
            attn = self.attn_drop(attn)
            pooled = torch.sum(hidden_states * attn.unsqueeze(-1), dim=1)

            if self.return_attn:
                pooled = self._normalize_vector(pooled, self.post_pool_norm, self.eps)
                return pooled, attn

        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        pooled = self._normalize_vector(pooled, self.post_pool_norm, self.eps)

        if self._output_dim is None:
            self._output_dim = self.get_output_dim(input_dim=D)

        return pooled


class MultiHeadDomainFeatureExtractor(nn.Module):
    """
    Simplified domain feature extractor with channel attention.
    Uses Global Average Pooling (GAP) and Global Max Pooling (GMP) for stability.

    Output: in_channels * 2 features (mean + max)
    """

    def __init__(self, in_channels=288, use_attention=True, use_multiscale=False):
        super().__init__()
        self.in_channels = in_channels
        self.use_attention = use_attention

        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, in_channels, 1),
                nn.Sigmoid()
            )

        self.output_dim = in_channels * 2

        self.feature_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] feature tensor (e.g., [32, 288, 14, 14])

        Returns:
            [B, output_dim] pooled features (e.g., [32, 576])
        """
        if self.use_attention:
            attn_weights = self.channel_attention(x)
            x = x * attn_weights

        avg_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        avg_pooled = avg_pooled.flatten(1)

        max_pooled = torch.nn.functional.adaptive_max_pool2d(x, (1, 1))
        max_pooled = max_pooled.flatten(1)

        output = torch.cat([avg_pooled, max_pooled], dim=1)

        output = self.feature_norm(output)

        return output


class DANNModelWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        domain_classifier: nn.Module,
        pooling_type: str = "adaptive_avg",
        hidden_dim: Optional[int] = None,
        pooling_output_size: Union[int, Tuple[int, int]] = (1, 1),
        use_multihead_extractor: bool = False,
        feature_channels: int = 288,
        feature_normalization: str = "layernorm",
        prepool_instance_norm: bool = False,
        norm_eps: float = 1e-6
    ):
        super().__init__()
        self.base_model = base_model
        self.domain_classifier = domain_classifier
        self.pooling_type = pooling_type
        self.use_multihead_extractor = use_multihead_extractor

        if isinstance(pooling_output_size, int):
            self.pooling_output_size = (pooling_output_size, pooling_output_size)
        else:
            self.pooling_output_size = pooling_output_size

        self.prepool_norm2d = nn.InstanceNorm2d(feature_channels, affine=False, eps=1e-5) \
                              if prepool_instance_norm else None

        if use_multihead_extractor or pooling_type == "multihead":
            self.multihead_extractor = MultiHeadDomainFeatureExtractor(
                in_channels=feature_channels,
                use_attention=True,
                use_multiscale=True
            )
            self.advanced_pooling = None
        elif pooling_type != "adaptive_avg":
            self.advanced_pooling = AdvancedPooling(pooling_type=pooling_type, hidden_dim=hidden_dim)
            self.multihead_extractor = None
        else:
            self.advanced_pooling = None
            self.multihead_extractor = None

        self.feature_normalizer = FeatureNormalizer(mode=feature_normalization, eps=norm_eps)


    def forward(self, x: torch.Tensor, epsilon: Optional[float] = None, return_features: bool = False, **kwargs):
        features, segmentation_output = self.base_model(x, **kwargs)
        if epsilon is None:
            return segmentation_output

        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features is None:
            return segmentation_output

        if features.ndim == 2:
            flattened_features = features  # [B, C]
        elif features.ndim == 4:
            if self.prepool_norm2d is not None:
                features = self.prepool_norm2d(features)

            if self.multihead_extractor is not None:
                flattened_features = self.multihead_extractor(features)
            elif self.advanced_pooling is not None:
                B, C, H, W = features.shape
                pooling_input = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
                pooled = self.advanced_pooling(pooling_input)
                flattened_features = pooled.view(pooled.size(0), -1)
            else:
                flattened_features = self._extract_fallback(features)
        else:
            raise ValueError(f"Unexpected features shape: {features.shape}")

        normalized_features = self.feature_normalizer(flattened_features)

        grl_feats = GradReversalLayer.apply(normalized_features, epsilon)

        first_weight = next(self.domain_classifier.parameters())
        grl_feats = grl_feats.to(dtype=first_weight.dtype)

        domain_output = self.domain_classifier(grl_feats)

        if return_features:
            return segmentation_output, domain_output, normalized_features
        return segmentation_output, domain_output

    def _extract_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback feature extraction using configurable adaptive average pooling"""
        return torch.nn.functional.adaptive_avg_pool2d(x, self.pooling_output_size).flatten(1, -1)

    def __getattr__(self, name: str):
        """Delegate attribute access to base model when not found in wrapper"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)