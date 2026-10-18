# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

"""
    when use nn.DataParallel, to wrap the network during training, Pytorch's implementation normalizes the tensor on each device using the statistics only on that device, which accelerated the computation and is also easy to implement, but the statistics might be inaccurate. Instead in this synchronized version, the statistics will be computed over all training samples distributed on multiple devices.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class BatchNorm2dReimpl(nn.Module):
    def __init__(self, num_features, eps = 1e-5, momentum = 0.1) -> None:
        super(BatchNorm2dReimpl, self).__init__()
        
        self.num_features = num_features
        self.eps = eps  
        self.momentum = momentum
        self.weight = nn.Parameter(torch.empty(num_features))
        self.bias = nn.Parameter(torch.empty(num_features))
        
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.reset_parameters()
        
    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.fill_(1)
        
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        init.uniform_(self.weight)
        init.zeros_(self.bias)
        
    def forward(self, input_: Tensor) -> Tensor:
        batchsize, channels, height, width = input_.size()
        numel = batchsize * height * width
        
        input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
        # When you call .permute(1, 0, 2, 3), PyTorch doesn't physically rearrange the data in memory - it just changes how the tensor indices map to memory locations. This creates a "view" that may not be stored contiguously in memory.
        # However, .view() requires the tensor to be contiguous in memory. So .contiguous() creates a new tensor with the same data but arranged contiguously in memory, allowing the subsequent .view() operation to work.
        
        sum_ = input_.sum(1)
        sum_of_square = input_.pow(2).sum(1)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean
        
        self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * mean.detach())
        unbias_var = sumvar / (numel - 1)
        self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach())
        
        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = ((input_ - mean.unsqueeze(1)) * inv_std.unsqueeze(1) * self.weight.unsqueeze(1) + self.bias.unsqueeze(1))

        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous()