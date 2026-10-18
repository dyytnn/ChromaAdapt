"""ONNX Runtime adapters for the released dual-head MASS-Net models.

All four released checkpoints share one graph contract.  The named subclasses
make the paper regimes explicit while keeping preprocessing and inference
behavior identical and easy to maintain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import onnxruntime as ort


MODEL_FILES = {
    "susy": "massnet_susy.onnx",
    "sure": "massnet_sure.onnx",
    "susy_to_sure": "massnet_susy_to_sure.onnx",
    "susy_to_sedann": "massnet_susy_to_sedann.onnx",
}
MODEL_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class MassNetPrediction:
    """Raw logits and decoded masks for a batch of pair crops."""

    chromosome_logits: np.ndarray  # [N, 3, H, W]
    overlap_logits: np.ndarray  # [N, 1, H, W]
    chromosome_labels: np.ndarray  # [N, H, W], IDs 0/1/2
    overlap_masks: np.ndarray  # [N, H, W], uint8 values 0/1


class OnnxMassNetAdapter:
    """Load and run one released dual-head MASS-Net ONNX graph on CPU.

    ``infer_tensor`` accepts already-normalized float32 NCHW data. Use
    ``preprocess_image`` / ``predict_image`` for OpenCV BGR images and paths.
    The model is trained for 224x224 pair crops, not full metaphase images.
    """

    model_key: ClassVar[str]

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        color_order: str = "bgr",
        overlap_threshold: float = 0.85,
        providers: list[str] | None = None,
        intra_op_num_threads: int | None = None,
    ) -> None:
        if color_order not in {"bgr", "rgb"}:
            raise ValueError("color_order must be 'bgr' or 'rgb'")
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in [0, 1]")
        self.color_order = color_order
        self.overlap_threshold = float(overlap_threshold)
        self.model_path = Path(model_path) if model_path else MODEL_DIR / MODEL_FILES[self.model_key]
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
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, found {[x.name for x in inputs]}")
        output_names = [output.name for output in outputs]
        expected = ["chromosome_logits", "overlap_logits"]
        if output_names != expected:
            raise ValueError(f"Unexpected ONNX outputs {output_names}; expected {expected}")
        self.input_name = inputs[0].name
        self.output_names = output_names

    @classmethod
    def from_model(cls, model_key: str, **kwargs) -> "OnnxMassNetAdapter":
        """Construct the adapter for one of the four bundled model regimes."""
        try:
            adapter_type = ADAPTERS[model_key.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown model {model_key!r}; choose from {sorted(ADAPTERS)}") from exc
        return adapter_type(**kwargs)

    def preprocess_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected HWC three-channel image, got {image_bgr.shape}")
        resized = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if self.color_order == "rgb" else resized
        tensor = image.astype(np.float32) / 255.0
        tensor = (tensor - MEAN) / STD
        return np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[None], dtype=np.float32)

    def infer_tensor(self, tensor: np.ndarray) -> MassNetPrediction:
        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError(f"Expected NCHW input with 3 channels, got {tensor.shape}")
        if not np.isfinite(tensor).all():
            raise ValueError("Input tensor contains NaN or infinity")
        chromosome, overlap = self.session.run(
            self.output_names, {self.input_name: np.ascontiguousarray(tensor)}
        )
        if chromosome.ndim != 4 or chromosome.shape[1] != 3:
            raise RuntimeError(f"Unexpected chromosome logits shape: {chromosome.shape}")
        if overlap.ndim != 4 or overlap.shape[1] != 1:
            raise RuntimeError(f"Unexpected overlap logits shape: {overlap.shape}")
        probability = 1.0 / (1.0 + np.exp(-np.clip(overlap[:, 0], -80.0, 80.0)))
        return MassNetPrediction(
            chromosome_logits=chromosome,
            overlap_logits=overlap,
            chromosome_labels=np.argmax(chromosome, axis=1).astype(np.uint8),
            overlap_masks=(probability > self.overlap_threshold).astype(np.uint8),
        )

    def predict_image(self, image: np.ndarray) -> MassNetPrediction:
        return self.infer_tensor(self.preprocess_image(image))

    def predict_path(self, image_path: str | Path) -> tuple[np.ndarray, MassNetPrediction]:
        path = Path(image_path)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode: {path}")
        return image, self.predict_image(image)


class SuSyOnnxAdapter(OnnxMassNetAdapter):
    model_key = "susy"


class SuReOnnxAdapter(OnnxMassNetAdapter):
    model_key = "sure"


class SuSyToSuReOnnxAdapter(OnnxMassNetAdapter):
    model_key = "susy_to_sure"


class SuSyToSeDANNOnnxAdapter(OnnxMassNetAdapter):
    model_key = "susy_to_sedann"


ADAPTERS = {
    "susy": SuSyOnnxAdapter,
    "sure": SuReOnnxAdapter,
    "susy_to_sure": SuSyToSuReOnnxAdapter,
    "susy_to_sedann": SuSyToSeDANNOnnxAdapter,
}


__all__ = [
    "ADAPTERS",
    "MODEL_FILES",
    "MassNetPrediction",
    "OnnxMassNetAdapter",
    "SuSyOnnxAdapter",
    "SuReOnnxAdapter",
    "SuSyToSuReOnnxAdapter",
    "SuSyToSeDANNOnnxAdapter",
]
