"""ONNX Runtime adapter for the T4-07 full-metaphase Mask R-CNN (ResNet-50 FPN).

This is a different model family from the dual-head MASS-Net models in
``adapters.py`` and the pair-crop YOLO model in ``yolo_overlap_touch_adapter.py``:
it takes a WHOLE metaphase image of any size and returns one instance per
chromosome (typically ~46), with resize, RPN, ROI heads and mask pasting all
inside the ONNX graph. It keeps its own pre/postprocessing and does not subclass
``OnnxMassNetAdapter``.

The graph is produced by ``utils/export_maskrcnn_onnx.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_FILE = "maskrcnn_t407_real.onnx"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class MaskRcnnPrediction:
    """Instances kept above the score threshold, in the input image's coordinates."""

    boxes_xyxy: np.ndarray  # [K, 4] float32
    scores: np.ndarray  # [K] float32
    masks: np.ndarray  # [K, H, W] bool


class MaskRcnnT407OnnxAdapter:
    """Load and run the T4-07 Mask R-CNN ONNX graph on CPU."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        providers: list[str] | None = None,
        intra_op_num_threads: int | None = None,
    ) -> None:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self.model_path = Path(model_path) if model_path else MODEL_DIR / DEFAULT_MODEL_FILE
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads is not None:
            if intra_op_num_threads < 1:
                raise ValueError("intra_op_num_threads must be positive")
            options.intra_op_num_threads = intra_op_num_threads
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = [output.name for output in self.session.get_outputs()]
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, found {[x.name for x in inputs]}")
        if outputs != ["boxes", "labels", "scores", "masks"]:
            raise ValueError(f"Unexpected ONNX outputs: {outputs}")
        self.input_name = inputs[0].name

    def predict_image(self, image_bgr: np.ndarray) -> MaskRcnnPrediction:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected HWC three-channel image, got {image_bgr.shape}")
        rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0, 1]
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)

        boxes, _, scores, masks = self.session.run(None, {self.input_name: tensor})
        keep = scores >= self.score_threshold
        return MaskRcnnPrediction(
            boxes_xyxy=boxes[keep].astype(np.float32),
            scores=scores[keep].astype(np.float32),
            masks=masks[keep][:, 0] > self.mask_threshold,
        )

    def predict_path(self, image_path: str | Path) -> tuple[np.ndarray, MaskRcnnPrediction]:
        path = Path(image_path)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode: {path}")
        return image, self.predict_image(image)


__all__ = ["MaskRcnnT407OnnxAdapter", "MaskRcnnPrediction"]
