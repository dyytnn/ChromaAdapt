"""Export a torchvision Mask R-CNN (ResNet-50 FPN) checkpoint to ONNX.

Used for the T4-07 full-metaphase chromosome instance-segmentation model
(2 classes: background + chromosome). Unlike the MASS-Net exports, this graph
takes a whole metaphase image of any size and contains torchvision's resize,
RPN, ROI heads and mask pasting, so a consumer only needs ONNX Runtime.

Graph interface:

* ``image``:  ``float32 [3, H, W]``, RGB in ``[0, 1]`` (dynamic H and W)
* ``boxes``:  ``float32 [K, 4]``, xyxy in input-image pixels
* ``labels``: ``int64 [K]``
* ``scores``: ``float32 [K]``
* ``masks``:  ``float32 [K, 1, H, W]``, per-instance probabilities pasted
  into the input frame (threshold at 0.5)

Loading is strict: missing or unexpected parameter keys abort the export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn
from torchvision.models.detection import maskrcnn_resnet50_fpn


class MaskRcnnExportWrapper(nn.Module):
    """Flatten torchvision's ``list[dict]`` output into plain tensors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        detections = self.model([image])[0]
        return detections["boxes"], detections["labels"], detections["scores"], detections["masks"]


def load_model(checkpoint: Path, num_classes: int) -> nn.Module:
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    model.load_state_dict(state, strict=True)
    return model.eval()


def export(checkpoint: Path, output: Path, num_classes: int, opset: int, size: tuple[int, int]) -> None:
    model = MaskRcnnExportWrapper(load_model(checkpoint, num_classes)).eval()
    height, width = size
    # Fixed seed so the trace input (and therefore the exported file) is reproducible.
    generator = torch.Generator().manual_seed(0)
    dummy = torch.rand(3, height, width, generator=generator)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(output),
        opset_version=opset,
        dynamo=False,
        input_names=["image"],
        output_names=["boxes", "labels", "scores", "masks"],
        dynamic_axes={
            "image": {1: "height", 2: "width"},
            "boxes": {0: "detections"},
            "labels": {0: "detections"},
            "scores": {0: "detections"},
            "masks": {0: "detections", 2: "height", 3: "width"},
        },
    )
    onnx.checker.check_model(onnx.load(str(output)))


def compare(checkpoint: Path, output: Path, num_classes: int, image: Path) -> None:
    """Compare PyTorch and ONNX Runtime detections on one image (kept local, never copied)."""
    import cv2

    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot decode {image}")
    tensor = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).permute(2, 0, 1).float() / 255.0
    model = load_model(checkpoint, num_classes)
    with torch.no_grad():
        reference = model([tensor])[0]
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    boxes, _, scores, masks = session.run(None, {"image": tensor.numpy()})
    keep_ref = reference["scores"] >= 0.5
    keep_onnx = scores >= 0.5
    print(f"detections >=0.5: torch {int(keep_ref.sum())} | onnx {int(keep_onnx.sum())}")
    count = min(len(reference["scores"]), len(scores))
    if count:
        print(f"max |score diff| over top {count}: "
              f"{float(np.abs(reference['scores'][:count].numpy() - scores[:count]).max()):.3e}")
        print(f"max |box diff|: {float(np.abs(reference['boxes'][:count].numpy() - boxes[:count]).max()):.3e} px")
        mask_diff = np.abs(reference["masks"][:count].numpy() - masks[:count])
        print(f"max |mask prob diff|: {float(mask_diff.max()):.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="state dict (or {'model': state dict}) .pt/.pth")
    parser.add_argument("--output", type=Path, default=Path("onnx_models/maskrcnn_t407_real.onnx"))
    parser.add_argument("--num-classes", type=int, default=2, help="including background")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--trace-size", type=int, nargs=2, default=(1024, 1280), metavar=("H", "W"))
    parser.add_argument("--compare-image", type=Path, help="optional local image for a PyTorch-vs-ONNX check")
    args = parser.parse_args()

    export(args.checkpoint, args.output, args.num_classes, args.opset, tuple(args.trace_size))
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    if args.compare_image:
        compare(args.checkpoint, args.output, args.num_classes, args.compare_image)


if __name__ == "__main__":
    main()
