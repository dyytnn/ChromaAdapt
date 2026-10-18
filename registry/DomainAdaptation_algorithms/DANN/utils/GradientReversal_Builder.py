import types
from abc import abstractmethod

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from abs_interface import Abstract_SegModel
from registry.DomainAdaptation_algorithms.DANN.utils import GradientReversal_forward
from registry.registers import (
    activation_function_register,
    backbone_register,
    head_register,
    module_register,
    norm_register,
    pooling_register,
)


class GradientRerversalBuilder:
    def __init__(self, config_dir: str):
        with open(config_dir, "r") as f:
            config = yaml.safe_load(f)

        config_name = config_dir.split('/')[-1].split('.')[0]
        self.gradient_reversal_module: nn.Module = getattr(self, config_name, None)(config)

        if torch.cuda.is_available():
            self.gradient_reversal_module = self.gradient_reversal_module.cuda()
            
    @abstractmethod
    def validate(self, is_return_feature: bool = False, **kwargs) -> str:
        """
        Validate the architecture compatible
        """

        for channels in [3, 1]:
            try:
                tmp_input = torch.rand(8, channels, 224, 224)
                # Move input to same device as gradient_reversal_module
                device = next(self.gradient_reversal_module.parameters()).device
                tmp_input = tmp_input.to(device)
                tmp_output_chromsome = self.gradient_reversal_module(tmp_input, **kwargs)[-1][0] if is_return_feature else self.gradient_reversal_module(tmp_input, **kwargs)[-1]

                # Check spatial shape
                assert tmp_input.shape[2:] == tmp_output_chromsome.shape[2:], \
                    f"Model output spatial shape {tmp_output_chromsome.shape[2:]} does not match input {tmp_input.shape[2:]}"

                # If successful, break the loop
                break
            except Exception as e:
                last_exception = e
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
    
    def MassNet(self, config) -> Abstract_SegModel:
        domain_classifier_module: nn.Sequential = self.build_from_yaml(nn.Sequential(), config["DOMAIN_CLASSIFIER"])

        return domain_classifier_module

    def MassNet_MultiHead(self, config) -> Abstract_SegModel:
        """Multi-Head domain classifier for advanced feature extraction (1728D input)"""
        domain_classifier_module: nn.Sequential = self.build_from_yaml(nn.Sequential(), config["DOMAIN_CLASSIFIER"])

        return domain_classifier_module


    def DeepLabv3(self) -> Abstract_SegModel:
        ...
        
    def segformer_mit_b5(self, config) -> Abstract_SegModel:
        ...
    
    def LRFormer(self, config) -> Abstract_SegModel:
        ...
        
    def ResUnet(self, config) -> Abstract_SegModel:
        ...
        
    def ResUnetPlus(self, config) -> Abstract_SegModel:
        ...
    
    def FPN(self, config) -> Abstract_SegModel:
        ...

    def DNANet_baseline(self, config: dict):
        ...

    def PSFNet(self, config: dict):
        domain_classifier_module: nn.Sequential = self.build_from_yaml(nn.Sequential(), config["DOMAIN_CLASSIFIER"])
        
        return domain_classifier_module
    
    def Xception_Adversarial(self, config: dict):
        ...
    
    def Adaptation_PSFNet(self, config: dict):
        ...
    
    def maskformer384_UDA4inst(self, config: dict):
        ...
