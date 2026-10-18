"""Export a trained dual-head MASS-Net checkpoint to ONNX.

The exported graph exposes only the two inference outputs:

* ``chromosome_logits``: ``[N, 3, 224, 224]``
* ``overlap_logits``: ``[N, 1, 224, 224]``

Both regular checkpoints and the DANN wrapper checkpoint are supported. The
loader is intentionally strict so an incompatible checkpoint cannot silently
produce an ONNX model with randomly initialized parameters.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_architecture_builder():
    """Import model-building modules without loading training dependencies.

    ``registry.__init__`` eagerly imports the distributed training pipeline,
    which is unnecessary for export and requires a newer PyTorch distributed
    checkpoint API. A small package shell lets us load only the registry
    primitives and architecture code required for inference.
    """
    # ``source.modules.norm`` only needs this helper. Avoid importing the full
    # ``utils`` package, whose training-only FSDP helpers are version-specific.
    utils_package = types.ModuleType("utils")
    utils_package.__path__ = [str(PROJECT_ROOT / "utils")]
    utils_package.__package__ = "utils"
    dist_utils = types.ModuleType("utils.dist_utils")
    dist_utils.get_world_size = lambda: 1
    sys.modules["utils"] = utils_package
    sys.modules["utils.dist_utils"] = dist_utils

    registry_path = PROJECT_ROOT / "registry"
    package = types.ModuleType("registry")
    package.__path__ = [str(registry_path)]
    package.__package__ = "registry"
    sys.modules["registry"] = package

    registers = importlib.import_module("registry.registers")
    for name in (
        "activation_function_register",
        "backbone_register",
        "head_register",
        "module_register",
        "neck_register",
        "norm_register",
        "pooling_register",
    ):
        setattr(package, name, getattr(registers, name))

    registry_module = importlib.import_module("registry.registry")
    package.Registry = registry_module.Registry
    return importlib.import_module("registry.architecture").ArchitectureBuilder


ArchitectureBuilder = _import_architecture_builder()


DEFAULT_ARCHITECTURE = PROJECT_ROOT / "configs/yamlconfig/MassNet/Chromosome_MassNet.yaml"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationExportWrapper(nn.Module):
    """Remove training-only features from the native MASS-Net output."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        output = self.model(image)

        # Native MASS-Net: (feature_aggregation, (chromosome, overlap)).
        if (
            isinstance(output, (tuple, list))
            and len(output) == 2
            and isinstance(output[1], (tuple, list))
            and len(output[1]) == 2
        ):
            chromosome_logits, overlap_logits = output[1]
        # Also accept a model that already returns the two heads directly.
        elif isinstance(output, (tuple, list)) and len(output) == 2:
            chromosome_logits, overlap_logits = output
        else:
            raise RuntimeError(
                "Expected MASS-Net to return (features, (chromosome, overlap)) "
                "or (chromosome, overlap)."
            )

        return chromosome_logits, overlap_logits


def _extract_state_dict(checkpoint: object) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    state_dict: object = checkpoint
    for key in ("state_dict", "model", "last_model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state_dict = candidate
            break

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint does not contain a non-empty state dict")
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("Checkpoint state-dict keys must be strings")

    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        cleaned[clean_key] = value

    # The DANN best checkpoint contains both base_model.* and
    # domain_classifier.*. Only the segmentation model is exported.
    base_model_items = {
        key[len("base_model.") :]: value
        for key, value in cleaned.items()
        if key.startswith("base_model.")
    }
    return base_model_items or cleaned


def load_model_from_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    architecture_config: str | os.PathLike[str] = DEFAULT_ARCHITECTURE,
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path).resolve()
    architecture_config = Path(architecture_config).resolve()

    print(f"Loading architecture: {architecture_config}")
    model = ArchitectureBuilder(str(architecture_config)).model.cpu()

    print(f"Loading checkpoint:   {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with PyTorch versions that predate weights_only.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = _extract_state_dict(checkpoint)
    model_keys = set(model.state_dict())
    checkpoint_keys = set(state_dict)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={len(missing)} (first: {missing[:5]})")
        if unexpected:
            details.append(f"unexpected={len(unexpected)} (first: {unexpected[:5]})")
        raise RuntimeError("Checkpoint does not match MASS-Net: " + "; ".join(details))

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded {len(state_dict)} tensors with strict key matching")
    return model


def _add_metadata(
    output_path: Path,
    model_name: str,
    input_size: Sequence[int],
) -> None:
    import onnx

    model = onnx.load(str(output_path))
    metadata = {
        "model_name": model_name,
        "architecture": "dual-head MASS-Net",
        "input_layout": "NCHW",
        "input_size": f"3x{input_size[0]}x{input_size[1]}",
        "input_color_order": "BGR (matching the reported evaluation pipeline)",
        "input_value_range": "0..255 before normalization",
        "normalization_mean": ",".join(map(str, IMAGENET_MEAN)),
        "normalization_std": ",".join(map(str, IMAGENET_STD)),
        "chromosome_classes": "0=background,1=chromosome_1,2=chromosome_2",
        "overlap_postprocess": "sigmoid(logits) > 0.85",
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def export_to_onnx(
    model: nn.Module,
    output_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    model_name: str,
    input_size: tuple[int, int] = (224, 224),
    batch_size: int = 1,
    opset_version: int = 16,
    dynamic_batch: bool = True,
) -> dict[str, float]:
    """Export, validate, run, and numerically compare an ONNX graph."""
    import onnx
    import onnxruntime as ort

    output_path = Path(output_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapped_model = SegmentationExportWrapper(model.cpu().eval())
    height, width = input_size
    torch.manual_seed(42)
    dummy_input = torch.randn(batch_size, 3, height, width, dtype=torch.float32)

    with torch.inference_mode():
        torch_outputs = wrapped_model(dummy_input)
    expected_shapes = (
        (batch_size, 3, height, width),
        (batch_size, 1, height, width),
    )
    actual_shapes = tuple(tuple(output.shape) for output in torch_outputs)
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"Unexpected model outputs: {actual_shapes}; expected {expected_shapes}")

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "chromosome_logits": {0: "batch_size"},
            "overlap_logits": {0: "batch_size"},
        }

    print(f"Exporting ONNX:       {output_path}")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["chromosome_logits", "overlap_logits"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    _add_metadata(output_path, model_name, input_size)
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"input": dummy_input.numpy()})

    max_differences = {}
    for name, torch_output, ort_output in zip(
        ("chromosome_logits", "overlap_logits"), torch_outputs, ort_outputs
    ):
        difference = float(np.max(np.abs(torch_output.numpy() - ort_output)))
        max_differences[name] = difference
        print(f"  {name}: shape={ort_output.shape}, max_abs_diff={difference:.3e}")
        if difference > 1e-4:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"PyTorch/ONNX mismatch for {name}: max diff {difference:.3e}")

    # Verify that the dynamic batch annotation is functional, not only present.
    if dynamic_batch:
        batch_two = np.repeat(dummy_input.numpy(), 2, axis=0)
        batch_two_outputs = session.run(None, {"input": batch_two})
        if batch_two_outputs[0].shape[0] != 2 or batch_two_outputs[1].shape[0] != 2:
            output_path.unlink(missing_ok=True)
            raise RuntimeError("ONNX Runtime failed the dynamic-batch smoke test")

    print(f"Validated ONNX model ({output_path.stat().st_size / 1024**2:.2f} MiB)")
    return max_differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a MASS-Net checkpoint to ONNX")
    parser.add_argument("--checkpoint-path", "--checkpoint_path", required=True)
    parser.add_argument("--output-path", "--output_path")
    parser.add_argument(
        "--architecture-config",
        "--architecture_config",
        default=str(DEFAULT_ARCHITECTURE),
    )
    parser.add_argument("--model-name", default="MASS-Net")
    parser.add_argument("--input-size", "--input_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--opset-version", "--opset_version", type=int, default=16)
    parser.add_argument("--fixed-batch", "--no_dynamic_axes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    architecture_config = Path(args.architecture_config)
    if not architecture_config.is_file():
        raise FileNotFoundError(f"Architecture config not found: {architecture_config}")

    output_path = (
        Path(args.output_path)
        if args.output_path
        else checkpoint_path.with_name("model.onnx")
    )
    model = load_model_from_checkpoint(checkpoint_path, architecture_config)
    export_to_onnx(
        model=model,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        model_name=args.model_name,
        input_size=tuple(args.input_size),
        batch_size=args.batch_size,
        opset_version=args.opset_version,
        dynamic_batch=not args.fixed_batch,
    )


if __name__ == "__main__":
    main()
