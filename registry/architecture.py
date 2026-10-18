import types
from abc import abstractmethod

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from abs_interface import Abstract_SegModel
from registry.registers import (
    activation_function_register,
    backbone_register,
    head_register,
    module_register,
    norm_register,
    pooling_register,
)
from source.modules import *

from . import forward


class ArchitectureBuilder:
    def __init__(self, config_dir: str):
        with open(config_dir, "r") as f:
            config = yaml.safe_load(f)

        config_name = config_dir.split('/')[-1].split('.')[0]

        # Try using @configurable approach first
        if hasattr(self, '_build_configurable'):
            self.model = self._build_configurable(config, config_name)
        else:
            # Fallback to legacy method
            self.model = getattr(self, f"build_{config_name}", None)(config)

        self.model._init_weights()
        forward_func = getattr(forward, f"forward_{config_name}", None)
        if forward_func:
            self.model.forward = types.MethodType(forward_func, self.model)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.__class__.__name__ = config_name

    def _build_configurable(self, config, config_name):
        """Build model using @configurable pattern"""
        from abs_interface import Abstract_SegModel

        # Try direct config instantiation
        try:
            return Abstract_SegModel(config)
        except Exception:
            # Fallback to legacy method if configurable fails
            return getattr(self, f"build_{config_name}", None)(config)
    @abstractmethod
    def validate(self, is_return_feature: bool = False, **kwargs) -> str:
        """
        Validate the architecture compatible
        """
        self.model.train()
        for channels in [3, 1]:
            try:
                tmp_input = torch.rand(8, channels, 224, 224)
                # Move input to same device as model
                device = next(self.model.parameters()).device
                tmp_input = tmp_input.to(device)
                tmp_output_chromsome = self.model(tmp_input, **kwargs)[-1][0] if is_return_feature else self.model(tmp_input, **kwargs)[-1]

                # Check spatial shape
                assert tmp_input.shape[2:] == tmp_output_chromsome.shape[2:], \
                    f"Model output spatial shape {tmp_output_chromsome.shape[2:]} does not match input {tmp_input.shape[2:]}"

                # If successful, break the loop
                print(f"✓ Model validation passed with {channels}-channel input")
                break
            except Exception as e:
                print(f"✗ Validation failed for {channels}-channel input: {type(e).__name__}: {e}")
                last_exception = e
                import traceback
                traceback.print_exc()
        else:
            raise RuntimeError(
                f"Model failed to process both 3-channel and 1-channel inputs. Last error: {last_exception}"
            )
    
    @staticmethod
    def build_from_yaml(modules: nn.Sequential, config: dict[dict]):
        for layer_name, layer in config.items():
            for block_name, block_params in layer.items():
                register = None
                if block_name in pooling_register._registry.keys():
                    register = pooling_register
                elif block_name in backbone_register._registry.keys():
                    register = backbone_register
                elif block_name in head_register._registry.keys():
                    register = head_register
                elif block_name in activation_function_register._registry.keys():
                    register = activation_function_register
                elif block_name in norm_register._registry.keys():
                    register = norm_register
                elif block_name in module_register._registry.keys():
                    register = module_register
                elif "pytorch" in block_name:
                    modules.add_module(layer_name, getattr(torch.nn, block_name.replace("pytorch_", ""))(**block_params)) if block_params else modules.add_module(layer_name, getattr(torch.nn, block_name.replace("pytorch_", ""))())
                    continue
                assert register is not None, "Unknown block name: " + block_name
                modules.add_module(layer_name, register.get(block_name, **block_params))
        return modules

    @staticmethod
    def build_component_from_config(config: dict) -> nn.Module:
        """
        Build any component (backbone, neck, head) from config recursively.
        Supports both simple and complex configurations.
        """
        if not config:
            return None

        # Handle direct registry-based component
        if len(config) == 1:
            component_name, component_config = list(config.items())[0]

            # Check if it's a registered component
            for register in [backbone_register, head_register, pooling_register,
                           activation_function_register, norm_register, module_register]:
                if component_name in register._registry.keys():
                    if isinstance(component_config, dict) and "config" in component_config:
                        return register.get(component_name, **component_config["config"])
                    elif isinstance(component_config, dict):
                        return register.get(component_name, **component_config)
                    else:
                        return register.get(component_name)

            # Handle pytorch modules
            if "pytorch" in component_name:
                pytorch_name = component_name.replace("pytorch_", "")
                if isinstance(component_config, dict):
                    return getattr(torch.nn, pytorch_name)(**component_config)
                else:
                    return getattr(torch.nn, pytorch_name)()

        # Handle complex nested configuration (like multi-layer architectures)
        module_seq = nn.Sequential()
        for layer_name, layer_config in config.items():
            if isinstance(layer_config, dict):
                # Recursively build sub-components
                sub_component = ArchitectureBuilder.build_component_from_config({layer_name: layer_config})
                if sub_component is not None:
                    module_seq.add_module(layer_name, sub_component)
            else:
                # Handle direct module instantiation
                ArchitectureBuilder.build_from_yaml(module_seq, {layer_name: layer_config})

        return module_seq if len(module_seq) > 0 else None

    def build_Chromosome_MassNet(self, config) -> Abstract_SegModel:
        input_conv = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Input_convolution"])
        scale_8_blocks = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Scale_8_Block"])
        scale_4_blocks = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Scale_4_Block"])
        scale_2_blocks = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Scale_2_Block"])
        
        backbone = nn.Sequential(*[input_conv, scale_8_blocks, scale_4_blocks, scale_2_blocks])
        neck = self.build_from_yaml(nn.Sequential(), config["NECK"])
        head = self.build_from_yaml(nn.Sequential(), config["HEAD"]) if "HEAD" in config.keys() else nn.Identity()
        
        model = Abstract_SegModel(backbone=backbone, neck=neck, head=head)
        return model
    
    def build_Chromosome_MassNet_Metaphase(self, config) -> Abstract_SegModel:
        """Builder for Metaphase 47-class segmentation architecture"""
        input_conv = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Input_convolution"])
        scale_8_blocks = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Scale_8_Block"])
        scale_4_blocks = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Scale_4_Block"])
        scale_2_blocks = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Scale_2_Block"])

        backbone = nn.Sequential(*[input_conv, scale_8_blocks, scale_4_blocks, scale_2_blocks])
        neck = self.build_from_yaml(nn.Sequential(), config["NECK"])
        head = self.build_from_yaml(nn.Sequential(), config["HEAD"]) if "HEAD" in config.keys() else nn.Identity()

        model = Abstract_SegModel(backbone=backbone, neck=neck, head=head)
        return model

    def build_Massnet_DANN(self, config: dict):
        """Builder for MassNet with Domain Adversarial Neural Network (DANN)"""
        input_conv = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Input_convolution"])
        scale_8_blocks = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Scale_8_Block"])
        scale_4_blocks = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Scale_4_Block"])
        scale_2_blocks = self.build_from_yaml(nn.Sequential(), config["BACKBONE"]["Scale_2_Block"])

        backbone = nn.Sequential(*[input_conv, scale_8_blocks, scale_4_blocks, scale_2_blocks])
        neck = self.build_from_yaml(nn.Sequential(), config["NECK"])
        head = self.build_from_yaml(nn.Sequential(), config["HEAD"]) if "HEAD" in config.keys() else nn.Identity()

        model = Abstract_SegModel(backbone=backbone, neck=neck, head=head)
        return model