#!/usr/bin/env python3
"""Export and validate the four paper checkpoints with ONNX Runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_to_onnx import export_to_onnx, load_model_from_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHITECTURE = (
    PROJECT_ROOT / "configs/yamlconfig/MassNet/Chromosome_MassNet.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--susy-checkpoint", type=Path, required=True)
    parser.add_argument("--sure-checkpoint", type=Path, required=True)
    parser.add_argument("--susy-sure-checkpoint", type=Path, required=True)
    parser.add_argument("--susy-sedann-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-config", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "onnx_models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = (
        ("SuSy", args.susy_checkpoint, "massnet_susy.onnx"),
        ("SuRe", args.sure_checkpoint, "massnet_sure.onnx"),
        ("SuSy-to-SuRe", args.susy_sure_checkpoint, "massnet_susy_to_sure.onnx"),
        (
            "SuSy-to-SeDANN",
            args.susy_sedann_checkpoint,
            "massnet_susy_to_sedann.onnx",
        ),
    )

    for model_name, checkpoint_path, filename in models:
        print(f"\n[{model_name}]")
        model = load_model_from_checkpoint(checkpoint_path, args.architecture_config)
        export_to_onnx(
            model=model,
            output_path=args.output_dir / filename,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
        )


if __name__ == "__main__":
    main()
