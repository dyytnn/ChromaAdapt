#!/usr/bin/env python3
"""Run CPU inference with an exported dual-head MASS-Net ONNX model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
CHROMOSOME_PALETTE_BGR = np.asarray(
    [[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8
)


def find_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        images = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No supported images found in: {input_path}")
        return images
    raise FileNotFoundError(f"Input does not exist: {input_path}")


def preprocess_image(
    image_bgr: np.ndarray, color_order: str
) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    model_image = resized
    if color_order == "rgb":
        model_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = model_image.astype(np.float32) / 255.0
    tensor = (tensor - MEAN) / STD
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return image_bgr, np.ascontiguousarray(tensor, dtype=np.float32)


def load_and_preprocess(
    image_path: Path, color_order: str
) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not decode: {image_path}")
    return preprocess_image(image_bgr, color_order)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def save_outputs(
    output_dir: Path,
    image_path: Path,
    original_bgr: np.ndarray,
    chromosome_logits: np.ndarray,
    overlap_logits: np.ndarray,
    overlap_threshold: float,
) -> tuple[Path, Path, Path]:
    height, width = original_bgr.shape[:2]
    chromosome_labels = np.argmax(chromosome_logits[0], axis=0).astype(np.uint8)
    overlap_mask = (sigmoid(overlap_logits[0, 0]) > overlap_threshold).astype(np.uint8)

    chromosome_labels = cv2.resize(
        chromosome_labels, (width, height), interpolation=cv2.INTER_NEAREST
    )
    overlap_mask = cv2.resize(
        overlap_mask, (width, height), interpolation=cv2.INTER_NEAREST
    )

    chromosome_color = CHROMOSOME_PALETTE_BGR[chromosome_labels]
    overlay = cv2.addWeighted(original_bgr, 0.55, chromosome_color, 0.45, 0.0)
    overlay[overlap_mask == 1] = (0, 0, 255)

    sample_dir = output_dir / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    labels_path = sample_dir / "chromosome_labels.png"
    overlap_path = sample_dir / "overlap_mask.png"
    overlay_path = sample_dir / "overlay.png"

    cv2.imwrite(str(labels_path), chromosome_labels)
    cv2.imwrite(str(overlap_path), overlap_mask * 255)
    cv2.imwrite(str(overlay_path), overlay)
    return labels_path, overlap_path, overlay_path


def benchmark(
    session: ort.InferenceSession, tensor: np.ndarray, runs: int
) -> None:
    for _ in range(3):
        session.run(None, {"input": tensor})
    started = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"input": tensor})
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / runs
    print(f"CPU benchmark: {elapsed_ms:.2f} ms/image ({runs} measured runs)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="ONNX model path")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", type=Path, help="Image or flat image directory"
    )
    input_group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on an in-memory neutral image; no dataset is required",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Output directory"
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.85,
        help="Sigmoid threshold for overlap logits (paper evaluation: 0.85)",
    )
    parser.add_argument(
        "--color-order",
        choices=("bgr", "rgb"),
        default="bgr",
        help="BGR reproduces the reported evaluation pipeline",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=0,
        help="Optional number of timed CPU runs on the first image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not 0.0 <= args.overlap_threshold <= 1.0:
        raise ValueError("--overlap-threshold must be between 0 and 1")
    if args.benchmark_runs < 0:
        raise ValueError("--benchmark-runs cannot be negative")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in session.get_outputs()]
    expected_outputs = ["chromosome_logits", "overlap_logits"]
    if output_names != expected_outputs:
        raise RuntimeError(f"Unexpected ONNX outputs: {output_names}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_tensor = None
    if args.smoke_test:
        samples = [(Path("smoke_test.png"), np.full((224, 224, 3), 127, np.uint8))]
    else:
        samples = [(path, None) for path in find_images(args.input)]

    for image_path, generated_image in samples:
        if generated_image is None:
            original_bgr, tensor = load_and_preprocess(image_path, args.color_order)
        else:
            original_bgr, tensor = preprocess_image(generated_image, args.color_order)
        chromosome_logits, overlap_logits = session.run(
            expected_outputs, {"input": tensor}
        )
        paths = save_outputs(
            args.output_dir,
            image_path,
            original_bgr,
            chromosome_logits,
            overlap_logits,
            args.overlap_threshold,
        )
        print(f"{image_path.name}: saved {', '.join(path.name for path in paths)}")
        if first_tensor is None:
            first_tensor = tensor

    if args.benchmark_runs:
        benchmark(session, first_tensor, args.benchmark_runs)


if __name__ == "__main__":
    main()
