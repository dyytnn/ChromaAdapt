"""ONNX Runtime adapter for the single-class ("chromosome") YOLO26n-seg
instance-segmentation model, trained on core2 (exactly-two-chromosome)
touch/overlap crops mined from manually-labeled metaphase archives.

Unlike the dual-head MASS-Net models in ``adapters.py`` (dense per-pixel
3-class segmentation at a fixed 224x224 input), this model is an anchor-free
instance detector: it predicts a variable number of (box, score, mask)
instances per crop -- typically two, one per chromosome in the pair -- at a
letterboxed 320x320 input. It is a different architecture family released
under a different project; this module intentionally keeps its own
preprocessing/postprocessing and does not subclass ``OnnxMassNetAdapter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_FILE = "yolo26n_seg_overlap_touch_core2.onnx"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class YoloOverlapTouchPrediction:
    """Decoded instance detections for one chromosome-pair crop, in the
    ORIGINAL (pre-letterbox) image's coordinate system."""

    boxes_xyxy: np.ndarray  # [K, 4] float32, K = number of detected instances
    scores: np.ndarray  # [K] float32
    masks: np.ndarray  # [K, H, W] bool, H/W = original crop size


class YoloOverlapTouchOnnxAdapter:
    """Load and run the released YOLO26n-seg overlap/touch ONNX graph on CPU.

    ``infer_tensor`` is not exposed here (unlike ``OnnxMassNetAdapter``)
    because this model's postprocessing needs the letterbox scale/padding
    computed during preprocessing to map boxes and masks back to the
    original image size -- use ``predict_image`` / ``predict_path`` instead.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        imgsz: int = 320,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        providers: list[str] | None = None,
        intra_op_num_threads: int | None = None,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be in [0, 1]")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.model_path = Path(model_path) if model_path else MODEL_DIR / DEFAULT_MODEL_FILE
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads is not None:
            if intra_op_num_threads < 1:
                raise ValueError("intra_op_num_threads must be positive")
            options.intra_op_num_threads = intra_op_num_threads
        # Note: on machines where CUDAExecutionProvider reports "available"
        # but cuDNN is not actually installed, onnxruntime raises at run()
        # time, not at session-creation time -- default to CPU, this model
        # (nano, 320x320) is fast enough there. Pass providers=[...] to opt
        # into GPU once cuDNN is confirmed working.
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, found {[x.name for x in inputs]}")
        if len(outputs) != 2:
            raise ValueError(f"Expected two ONNX outputs, found {[x.name for x in outputs]}")
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in outputs]

    def _letterbox(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize with unchanged aspect ratio, pad to (imgsz, imgsz).
        Returns (padded_image, scale, (pad_x, pad_y))."""
        h, w = image_bgr.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        nh, nw = round(h * scale), round(w * scale)
        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return padded, scale, (pad_x, pad_y)

    @staticmethod
    def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
        areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_threshold]
        return keep

    def predict_image(self, image_bgr: np.ndarray) -> YoloOverlapTouchPrediction:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected HWC three-channel image, got {image_bgr.shape}")
        orig_h, orig_w = image_bgr.shape[:2]
        padded, scale, (pad_x, pad_y) = self._letterbox(image_bgr)

        tensor = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=np.float32)

        output0, output1 = self.session.run(self.output_names, {self.input_name: tensor})
        if output0.ndim != 3 or output0.shape[1] < 6:
            raise RuntimeError(f"Unexpected detection head shape: {output0.shape}")
        if output1.ndim != 4:
            raise RuntimeError(f"Unexpected mask-prototype shape: {output1.shape}")

        preds = output0[0].T  # (num_anchors, 4 + nc + nm); nc=1 here
        protos = output1[0]  # (nm, mh, mw)
        nc = preds.shape[1] - 4 - protos.shape[0]
        if nc != 1:
            raise RuntimeError(f"Expected a single-class head (nc=1), inferred nc={nc}")

        scores = preds[:, 4]
        above_thr = scores >= self.conf_threshold
        if not above_thr.any():
            return YoloOverlapTouchPrediction(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                masks=np.zeros((0, orig_h, orig_w), dtype=bool),
            )
        preds, scores = preds[above_thr], scores[above_thr]

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        boxes_letterboxed = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        mask_coeffs = preds[:, 5:]

        keep = self._nms(boxes_letterboxed, scores, self.iou_threshold)
        boxes_letterboxed, scores, mask_coeffs = boxes_letterboxed[keep], scores[keep], mask_coeffs[keep]

        boxes_orig = boxes_letterboxed.copy()
        boxes_orig[:, [0, 2]] = ((boxes_orig[:, [0, 2]] - pad_x) / scale).clip(0, orig_w)
        boxes_orig[:, [1, 3]] = ((boxes_orig[:, [1, 3]] - pad_y) / scale).clip(0, orig_h)

        nm, mh, mw = protos.shape
        proto_flat = protos.reshape(nm, mh * mw)
        resized_h, resized_w = round(orig_h * scale), round(orig_w * scale)

        masks = np.zeros((len(scores), orig_h, orig_w), dtype=bool)
        for i in range(len(scores)):
            mask_logits = mask_coeffs[i] @ proto_flat
            mask_prob = 1.0 / (1.0 + np.exp(-mask_logits.reshape(mh, mw)))
            mask_full = cv2.resize(mask_prob, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            mask_cropped = mask_full[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w]
            masks[i] = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) > 0.5

        return YoloOverlapTouchPrediction(
            boxes_xyxy=boxes_orig.astype(np.float32),
            scores=scores.astype(np.float32),
            masks=masks,
        )

    def predict_path(self, image_path: str | Path) -> tuple[np.ndarray, YoloOverlapTouchPrediction]:
        path = Path(image_path)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode: {path}")
        return image, self.predict_image(image)


__all__ = ["YoloOverlapTouchOnnxAdapter", "YoloOverlapTouchPrediction"]
