from typing import Union

from ...configs import Config, ConfigDict


def build_model_from_config(config: Union[dict, Config, ConfigDict]):
    from ..registers import (
        backbone_register,
        head_register,
        module_register,
        neck_register,
        norm_register,
        segmentation_loss_register,
    )
    
    args = config.copy()
    
    