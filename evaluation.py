import argparse

import yaml

from manager.manager import SegmentationManager
from registry import evaluate_segmentation_builder_register
from registry.evaluate_algorithms import EvaluateSegmentation
from registry.inference_algorithm import Inference
from source import *


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model with custom configurations.")
    
    parser.add_argument('--pipeline_config', 
                       type=str, 
                       help="Path to the pipeline configuration YAML file.")
    return parser.parse_args()


def main(args: dict):
    with open(args.get("pipeline_config"), "r") as f:
        evaluate_pipeline_config = yaml.safe_load(f)
    segmentation_manager = SegmentationManager()

    builder_cls = evaluate_pipeline_config.get("builder", None)
    if builder_cls is not None:
        if evaluate_segmentation_builder_register.check(builder_cls):
            builder = evaluate_segmentation_builder_register.get(builder_cls, cfg = evaluate_pipeline_config['cfg'])
            segmentation_manager.builder = builder
        else:
            raise ValueError(f"No builder class {builder_cls} found!")
    else:
        raise ValueError("No builder class specified in config file")

    segmentation_manager.build_full_product()
    pipeline = segmentation_manager.builder.product
    
    estimation_output_data = pipeline.process(
        data = args
    )
    
    
    print(estimation_output_data)
    
if __name__ == "__main__":
    args = parse_args()
    main(vars(args))