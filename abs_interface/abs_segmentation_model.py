from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from configs.config import configurable


class Abstract_SegModel(nn.Module):
    # @configurable
    def __init__(self, backbone, neck, head, auxiliary = None, decode_head = None):
        super(Abstract_SegModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.auxiliary = auxiliary
        self.decode_head = decode_head

    @property
    def device(self):
        """
        Get the device of the model.
        Returns the device of the first parameter found.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # If model has no parameters, return CPU
            return torch.device('cpu')

    @classmethod
    def from_config(cls, cfg):
        """Convert config to kwargs for __init__"""
        from registry.architecture import ArchitectureBuilder

        # Parse backbone config
        backbone_cfg = cfg.get("BACKBONE", {})
        backbone = ArchitectureBuilder.build_component_from_config(backbone_cfg)

        # Parse neck config
        neck_cfg = cfg.get("NECK", {})
        neck = ArchitectureBuilder.build_component_from_config(neck_cfg) if neck_cfg else None

        # Parse head config
        head_cfg = cfg.get("HEAD", {})
        head = ArchitectureBuilder.build_component_from_config(head_cfg) if head_cfg else nn.Identity()

        # Parse auxiliary config
        auxiliary_cfg = cfg.get("AUXILIARY", {})
        auxiliary = ArchitectureBuilder.build_component_from_config(auxiliary_cfg) if auxiliary_cfg else None

        # Parse decode_head config
        decode_head_cfg = cfg.get("DECODE_HEAD", {})
        decode_head = ArchitectureBuilder.build_component_from_config(decode_head_cfg) if decode_head_cfg else None

        return {
            "backbone": backbone,
            "neck": neck,
            "head": head,
            "auxiliary": auxiliary,
            "decode_head": decode_head
        }
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.running_mean, 0)
                nn.init.constant_(m.running_var, 1)
                
class Full_Abstract_SegModel(nn.Module):
    def __init__(self, cfg, build_function):
        super(Full_Abstract_SegModel, self).__init__()
        self.model: nn.Module = build_function(cfg)
        self._init_weights(self.model)

    @property
    def device(self):
        """
        Get the device of the model.
        Returns the device of the first parameter found.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # If model has no parameters, return CPU
            return torch.device('cpu')

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def _init_weights(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)