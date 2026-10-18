import torch
import torch.nn as nn
from torch import Tensor

from registry import (
    activation_function_register,
    backbone_register,
    head_register,
    module_register,
    neck_register,
    norm_register,
    pooling_register,
)
from registry.registry import NormalRegistry

from .convolutions import (
    Conv_Block,
    Convolution,
    DropPath,
    Interpolation_Upsample,
    ResidualConv,
    Transpose_Conv_Block,
    TransposeConv,
    depthwise_separable_conv,
)

for module, name in [
    (nn.BatchNorm1d, 'BN1d'),
    (nn.BatchNorm2d, 'BN2d'),
    (nn.BatchNorm3d, 'BN3d'),
    (nn.SyncBatchNorm, 'SyncBN'),
    (nn.GroupNorm, 'GN'),
    (nn.LayerNorm, 'LN'),
    (nn.InstanceNorm2d, 'IN'),
    (nn.InstanceNorm1d, 'IN1d'),
    (nn.InstanceNorm2d, 'IN2d'),
    (nn.InstanceNorm3d, 'IN3d'),
]:
    register_norm = NormalRegistry(norm_register, module)
    register_norm(name)


for module, name in [
    (nn.MaxPool1d, 'MaxPool1d'),
    (nn.MaxPool2d, 'MaxPool2d'),
    (nn.MaxPool3d, 'MaxPool3d'),
    (nn.AvgPool1d, 'AvgPool1d'),
    (nn.AvgPool2d, 'AvgPool2d'),
    (nn.AvgPool3d, 'AvgPool3d'),
    (nn.AdaptiveMaxPool1d, 'AdaptiveMaxPool1d'),
    (nn.AdaptiveMaxPool2d, 'AdaptiveMaxPool2d'),
    (nn.AdaptiveMaxPool3d, 'AdaptiveMaxPool3d'),
    (nn.AdaptiveAvgPool1d, 'AdaptiveAvgPool1d'),
    (nn.AdaptiveAvgPool2d, 'AdaptiveAvgPool2d'),
    (nn.AdaptiveAvgPool3d, 'AdaptiveAvgPool3d'),
]:
    register_pool = NormalRegistry(pooling_register, module)
    register_pool(name)


for module, name in [
    (nn.ReLU, 'ReLU'),
    (nn.LeakyReLU, 'LeakyReLU'),
    (nn.ELU, 'ELU'),
    (nn.SELU, 'SELU'),
    (nn.GELU, 'GELU'),
    (nn.Sigmoid, 'Sigmoid'),
    (nn.Tanh, 'Tanh'),
    (nn.Softplus, 'Softplus'),
    (nn.Softsign, 'Softsign'),
    (nn.ReLU6, 'ReLU6'),
    (nn.Hardtanh, 'Hardtanh'),
    (nn.PReLU, 'PReLU'),
    (nn.Hardswish, 'Hardswish'),
    (nn.Hardshrink, 'Hardshrink'),
    (nn.LogSigmoid, 'LogSigmoid'),
    (nn.LogSoftmax, 'LogSoftmax'),
    (nn.Mish, 'Mish'),
]:
    register_activation = NormalRegistry(activation_function_register, module)
    register_activation(name)


for module, name in [
    (nn.Dropout, 'Dropout'),
    (nn.Dropout1d, 'Dropout1d'),
    (nn.Dropout2d, 'Dropout2d'),
    (nn.Dropout3d, 'Dropout3d'),
    (nn.AlphaDropout, 'AlphaDropout'),
]:
    register_module = NormalRegistry(module_register, module)
    register_module(name)


from .norm import CycleBatchNormList, FrozenBatchNorm2d, NaiveSyncBatchNorm


@module_register.register("Upsample")
class Upsample(nn.Module):
    def __init__(self,size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        super().__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)
        
    def forward(self, input):
        return self.upsample(input)

@activation_function_register.register('Swish')
class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x) 


__all__ = [
    'Conv_Block',
    'Convolution',
    'TransposeConv',
    'Transpose_Conv_Block',
    'ResidualConv',
    'DropPath',
    'Interpolation_Upsample',
    'depthwise_separable_conv',
    'CycleBatchNormList',
    'FrozenBatchNorm2d',
    'NaiveSyncBatchNorm',
    'Upsample',
    'Swish',
]
