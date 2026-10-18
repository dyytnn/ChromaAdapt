from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from registry import activation_function_register, module_register


@module_register.register("Dilated_Conv_Blocks")
class Dilated_Conv_Block(nn.Module):
    def __init__(self, in_chans, out_chans, stride, dilation, activation: Literal['GeLU', 'ReLU', 'Swish'], **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_chans, out_channels = out_chans, stride = stride, dilation=dilation, **kwargs)
        self.batch_norm = nn.BatchNorm2d(num_features=out_chans)
        self.act = activation_function_register.get(activation)

    def forward(self, x):
        return self.act(self.batch_norm(self.conv(x)))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  (conv): {repr(self.conv)}\n  (batch_norm): {repr(self.batch_norm)}\n  (act): {repr(self.act)}\n)"

@module_register.register("interpolation_upsample")
class Interpolation_Upsample(nn.Module):
    def __init__(self, scale=2):
        super().__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)

@module_register.register("ResidualConv")
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, kernel_size = 3):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )        
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        residual = self.conv_skip(x)
        x = self.conv_block(x)
        x = x + residual

        return self.final_relu(x)

@module_register.register("Convolution")
class Convolution(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(**kwargs)
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
    
@module_register.register("Conv_Block")
class Conv_Block(nn.Module):
    def __init__(self, in_chans, out_chans, stride = 1, activation: Literal['GeLU', 'ReLU', 'Swish'] = 'ReLU', **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_chans, out_channels = out_chans, stride = stride, **kwargs)
        self.batch_norm = nn.BatchNorm2d(num_features=out_chans)
        self.activation_name = activation
        self.act = activation_function_register.get(activation, inplace=False) if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.act(x)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  (conv): {repr(self.conv)}\n  (batch_norm): {repr(self.batch_norm)}\n  (act): {self.activation_name}()\n)"
    
@module_register.register("Depthwise_Separable_Conv_Blocks")
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False, stride=1, activation: Literal['GeLU', 'ReLU', 'Swish'] = 'ReLU' , **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=nin,
            out_channels=nin,
            kernel_size=kernel_size,
            padding=padding,
            groups=nin,
            bias=bias,
            stride=stride,
            **kwargs  
        )
        self.pointwise = nn.Conv2d(
            in_channels=nin,
            out_channels=nout,
            kernel_size=1,
            bias=bias
        )
        self.BatchNorm = nn.BatchNorm2d(num_features=nout)
        self.activation = activation_function_register.get(activation)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.BatchNorm(out)
        out = self.activation(out)
        return out
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  (depthwise): {repr(self.depthwise)}\n  (pointwise): {repr(self.pointwise)}\n  (BatchNorm): {repr(self.BatchNorm)}\n  (activation): {repr(self.activation)}\n)"
    
@module_register.register("Transpose_Conv_Blocks")
class Transpose_Conv_Block(nn.Module):
    def __init__(self, in_chans, out_chans, stride, activation: Literal['GeLU', 'ReLU', 'Swish'], **kwargs) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels = in_chans, out_channels = out_chans, stride = stride, **kwargs)
        self.batch_norm = nn.BatchNorm2d(num_features=out_chans, eps=1e-3)
        self.activation_name = activation
        self.act = activation_function_register.get(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.act(x)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  (conv): {repr(self.conv)}\n  (batch_norm): {repr(self.batch_norm)}\n  (act): {self.activation_name}()\n)"

@module_register.register("TransposeConv")
class TransposeConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, **kwargs):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel_size, stride=stride, **kwargs
        )

    def forward(self, x):
        return self.upsample(x)
    
@module_register.register('DropPath')
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor
    
@module_register.register("LinearModule")
class LinearModule(nn.Sequential):
    def __init__(
        self, output_size, in_channels, classes, pooling="avg", dropout=0.2, activation=None
    ):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling)
            )
        pool = nn.AdaptiveAvgPool2d(output_size) if pooling == "avg" else nn.AdaptiveMaxPool2d(output_size)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation_func = activation_function_register._registry[activation]()
        super().__init__(pool, flatten, dropout, linear, activation_func)

class DepthwiseSeparableConvModule(nn.Module):
    """Standard depthwise separable convolution: Depthwise → Pointwise → BN → Activation"""
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                activation: Literal['GeLU', 'ReLU', 'Swish'] = 'ReLU'):
        super(DepthwiseSeparableConvModule, self).__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation_function_register.get(activation)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
