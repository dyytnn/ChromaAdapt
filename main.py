import argparse
import atexit

import torch
import torch.distributed as dist
import yaml

from manager.manager import SegmentationManager
from registry import training_segmentation_builder_register
from source import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training segmentation model with custom configurations.")
    
    parser.add_argument('--pipeline_config',  type=str,  help="Path to the pipeline configuration YAML file.")

    return parser.parse_args()

def main(args: dict):
    with open(args.get("pipeline_config"), "r") as f:
        training_pipeline_config = yaml.safe_load(f)
        
    segmentation_manager = SegmentationManager()
    builder_cls = training_pipeline_config.get("builder", None)
    if builder_cls is not None:
        if training_segmentation_builder_register.check(builder_cls):
            builder = training_segmentation_builder_register.get(builder_cls, cfg = training_pipeline_config['cfg'])
            segmentation_manager.builder = builder
        else:
            raise ValueError(f"No builder class {builder_cls} found!")
    else:
        raise ValueError("No builder class specified in config file")

    segmentation_manager.build_full_product()
    pipeline = segmentation_manager.builder.product
    
    estimation_output_data = pipeline.process(
        data = {
            "pipeline_config": args.get("pipeline_config")
        }
    )
    print("Successfull")

def cleanup():
    """Clean up distributed processes"""
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    # Register cleanup function
    atexit.register(cleanup)
    
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    
    try:
        main(vars(args))
    except Exception as e:
        cleanup()
        raise e