"""Generate labeled touching/overlapping chromosome pairs.

The implementation preserves the paper pipeline's core operations: independent
rotation, constrained translation, controlled overlap, distance-transform
blending, and exact two-instance masks.  It reads the pool produced by
``generation.preprocessing`` and writes the layout expected by this repository's
synthetic data loader.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
from scipy import ndimage as ndi


Interaction = Literal["TOUCHING", "OVERLAP"]


@dataclass(frozen=True)
class ChromosomeSample:
    image: np.ndarray
    mask: np.ndarray
    category: int
    karyotype_id: str
    relative_path: str


@dataclass(frozen=True)
class Placement:
    first_y: int
    first_x: int
    second_y: int
    second_x: int
    canvas_height: int
    canvas_width: int
    overlap_fraction: float


@dataclass(frozen=True)
class GenerationRecord:
    index: int
    interaction: str
    image: str
    mask: str
    first_source: str
    second_source: str
    first_category: int
    second_category: int
    first_rotation_degrees: float
    second_rotation_degrees: float
    overlap_fraction: float


def _read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def _category_from_name(path: Path) -> int:
    match = re.search(r"chromosome_category_(\d+)$", path.stem)
    if not match:
        raise ValueError(
            f"expected filename chromosome_category_<number>.png, got {path.name}"
        )
    return int(match.group(1))


def load_chromosome_pool(root: Path) -> list[ChromosomeSample]:
    """Load a mirrored ``images/`` and ``masks/`` chromosome pool."""
    mask_root = root / "masks"
    image_root = root / "images"
    if not mask_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError(
            f"{root} must contain both images/ and masks/ directories"
        )

    samples: list[ChromosomeSample] = []
    for mask_path in sorted(mask_root.rglob("*.png")):
        relative = mask_path.relative_to(mask_root)
        image_path = image_root / relative
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"missing image corresponding to {mask_path}: {image_path}"
            )
        mask = _read_binary_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"image/mask size mismatch for {relative}: {image.shape[:2]} vs {mask.shape}"
            )
        if not mask.any():
            raise ValueError(f"empty mask: {mask_path}")
        samples.append(
            ChromosomeSample(
                image=image,
                mask=mask,
                category=_category_from_name(mask_path),
                karyotype_id=relative.parent.as_posix(),
                relative_path=relative.as_posix(),
            )
        )
    if len(samples) < 2:
        raise ValueError(f"at least two chromosome samples are required under {root}")
    return samples


def _rotate_sample(
    sample: ChromosomeSample,
    angle: float,
) -> ChromosomeSample:
    height, width = sample.mask.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    output_width = max(1, int(math.ceil(height * sine + width * cosine)))
    output_height = max(1, int(math.ceil(height * cosine + width * sine)))
    matrix[0, 2] += output_width / 2.0 - center[0]
    matrix[1, 2] += output_height / 2.0 - center[1]

    rotated_mask = cv2.warpAffine(
        sample.mask,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_image = cv2.warpAffine(
        sample.image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    coordinates = np.argwhere(rotated_mask > 0)
    if coordinates.size == 0:
        raise ValueError("rotation produced an empty mask")
    y1, x1 = coordinates.min(axis=0)
    y2, x2 = coordinates.max(axis=0) + 1
    return ChromosomeSample(
        image=rotated_image[y1:y2, x1:x2],
        mask=(rotated_mask[y1:y2, x1:x2] > 0).astype(np.uint8),
        category=sample.category,
        karyotype_id=sample.karyotype_id,
        relative_path=sample.relative_path,
    )


def _opposite_orientations(first: np.ndarray, second: np.ndarray) -> bool:
    first_ratio = first.shape[0] / max(first.shape[1], 1)
    second_ratio = second.shape[0] / max(second.shape[1], 1)
    return (first_ratio > 1.2 and second_ratio < 0.83) or (
        first_ratio < 0.83 and second_ratio > 1.2
    )


def rotate_to_distinct_orientations(
    first: ChromosomeSample,
    second: ChromosomeSample,
    rng: random.Random,
    *,
    attempts: int = 32,
) -> tuple[ChromosomeSample, ChromosomeSample, float, float]:
    """Independently rotate a pair until their orientations are sufficiently distinct."""
    for _ in range(attempts):
        first_angle = rng.uniform(-180.0, 180.0)
        second_angle = rng.uniform(-180.0, 180.0)
        first_rotated = _rotate_sample(first, first_angle)
        second_rotated = _rotate_sample(second, second_angle)
        if _opposite_orientations(first_rotated.mask, second_rotated.mask):
            return first_rotated, second_rotated, first_angle, second_angle
    raise RuntimeError("could not obtain sufficiently distinct chromosome orientations")


def overlap_fraction(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    denominator = min(int(first_mask.sum()), int(second_mask.sum()))
    if denominator == 0:
        return 0.0
    return float(np.logical_and(first_mask > 0, second_mask > 0).sum() / denominator)


def _placement_canvas(
    first: ChromosomeSample,
    second: ChromosomeSample,
) -> tuple[int, int, int, int]:
    height = 3 * (first.mask.shape[0] + second.mask.shape[0])
    width = 3 * (first.mask.shape[1] + second.mask.shape[1])
    first_y = (height - first.mask.shape[0]) // 2
    first_x = (width - first.mask.shape[1]) // 2
    return height, width, first_y, first_x


def _paste_mask(canvas: np.ndarray, mask: np.ndarray, y: int, x: int) -> None:
    height, width = mask.shape
    canvas[y : y + height, x : x + width] = mask


def find_touching_placement(
    first: ChromosomeSample,
    second: ChromosomeSample,
    rng: random.Random,
    *,
    attempts: int = 64,
) -> Placement:
    """Translate the second chromosome until masks touch without overlapping."""
    height, width, first_y, first_x = _placement_canvas(first, second)
    first_canvas = np.zeros((height, width), dtype=np.uint8)
    _paste_mask(first_canvas, first.mask, first_y, first_x)
    first_center_y = first_y + first.mask.shape[0] / 2.0
    first_center_x = first_x + first.mask.shape[1] / 2.0
    max_distance = (
        first.mask.shape[0]
        + first.mask.shape[1]
        + second.mask.shape[0]
        + second.mask.shape[1]
    )
    contact_zone = ndi.binary_dilation(
        first_canvas > 0, structure=np.ones((3, 3), dtype=bool)
    )

    for _ in range(attempts):
        direction = rng.uniform(0.0, 2.0 * math.pi)
        dy, dx = math.sin(direction), math.cos(direction)
        previous: tuple[int, int, np.ndarray] | None = None

        for distance in range(max_distance, -1, -1):
            second_center_y = first_center_y + dy * distance
            second_center_x = first_center_x + dx * distance
            second_y = int(round(second_center_y - second.mask.shape[0] / 2.0))
            second_x = int(round(second_center_x - second.mask.shape[1] / 2.0))
            if (
                second_y < 0
                or second_x < 0
                or second_y + second.mask.shape[0] > height
                or second_x + second.mask.shape[1] > width
            ):
                continue

            second_canvas = np.zeros_like(first_canvas)
            _paste_mask(second_canvas, second.mask, second_y, second_x)
            if np.logical_and(first_canvas, second_canvas).any():
                if previous is not None:
                    previous_y, previous_x, previous_canvas = previous
                    if np.logical_and(contact_zone, previous_canvas).any():
                        return Placement(
                            first_y,
                            first_x,
                            previous_y,
                            previous_x,
                            height,
                            width,
                            0.0,
                        )
                break
            previous = (second_y, second_x, second_canvas)

    raise RuntimeError("could not find a non-overlapping touching placement")


def find_overlap_placement(
    first: ChromosomeSample,
    second: ChromosomeSample,
    rng: random.Random,
    *,
    minimum: float = 0.2,
    maximum: float = 0.5,
    attempts: int = 1000,
) -> Placement:
    """Find a translation whose intersection occupies a controlled mask fraction."""
    if not 0.0 < minimum <= maximum < 1.0:
        raise ValueError("overlap range must satisfy 0 < minimum <= maximum < 1")
    height, width, first_y, first_x = _placement_canvas(first, second)
    first_canvas = np.zeros((height, width), dtype=np.uint8)
    _paste_mask(first_canvas, first.mask, first_y, first_x)
    target = rng.uniform(minimum, maximum)
    best: tuple[float, int, int, float] | None = None

    min_y = first_y - second.mask.shape[0] + 1
    max_y = first_y + first.mask.shape[0] - 1
    min_x = first_x - second.mask.shape[1] + 1
    max_x = first_x + first.mask.shape[1] - 1

    for _ in range(attempts):
        second_y = rng.randint(min_y, max_y)
        second_x = rng.randint(min_x, max_x)
        second_canvas = np.zeros_like(first_canvas)
        _paste_mask(second_canvas, second.mask, second_y, second_x)
        fraction = overlap_fraction(first_canvas, second_canvas)
        error = abs(fraction - target)
        if fraction > 0 and (best is None or error < best[0]):
            best = (error, second_y, second_x, fraction)
        if minimum <= fraction <= maximum:
            return Placement(
                first_y,
                first_x,
                second_y,
                second_x,
                height,
                width,
                fraction,
            )

    if best is not None and minimum <= best[3] <= maximum:
        return Placement(first_y, first_x, best[1], best[2], height, width, best[3])
    raise RuntimeError("could not find a placement in the requested overlap range")


def _place_image(canvas: np.ndarray, image: np.ndarray, y: int, x: int) -> None:
    height, width = image.shape[:2]
    canvas[y : y + height, x : x + width] = image


def distance_transform_blend(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    *,
    feather_size: int = 5,
) -> np.ndarray:
    """Composite two canvases using distance-derived weights in their intersection."""
    first_binary = (first_mask > 0).astype(np.uint8)
    second_binary = (second_mask > 0).astype(np.uint8)
    output = np.full_like(first_image, 255)
    first_only = (first_binary > 0) & (second_binary == 0)
    second_only = (second_binary > 0) & (first_binary == 0)
    intersection = (first_binary > 0) & (second_binary > 0)
    output[first_only] = first_image[first_only]
    output[second_only] = second_image[second_only]

    if intersection.any():
        first_distance = cv2.distanceTransform(first_binary, cv2.DIST_L2, 5)
        second_distance = cv2.distanceTransform(second_binary, cv2.DIST_L2, 5)
        denominator = first_distance + second_distance + 1e-6
        first_weight = first_distance / denominator
        if feather_size > 0:
            kernel_size = feather_size * 2 + 1
            first_weight = cv2.GaussianBlur(
                first_weight,
                (kernel_size, kernel_size),
                feather_size / 2.0,
            )
        first_weight = np.clip(first_weight, 0.0, 1.0)
        second_weight = 1.0 - first_weight
        output[intersection] = (
            first_weight[intersection, None] * first_image[intersection]
            + second_weight[intersection, None] * second_image[intersection]
        ).astype(np.uint8)
    return output


def compose_pair(
    first: ChromosomeSample,
    second: ChromosomeSample,
    placement: Placement,
    *,
    feather_size: int = 5,
    crop_padding: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a placement and return its image and 3-channel 0/1 instance mask."""
    first_mask = np.zeros(
        (placement.canvas_height, placement.canvas_width), dtype=np.uint8
    )
    second_mask = np.zeros_like(first_mask)
    _paste_mask(first_mask, first.mask, placement.first_y, placement.first_x)
    _paste_mask(second_mask, second.mask, placement.second_y, placement.second_x)

    first_image = np.full((*first_mask.shape, 3), 255, dtype=np.uint8)
    second_image = np.full_like(first_image, 255)
    _place_image(first_image, first.image, placement.first_y, placement.first_x)
    _place_image(second_image, second.image, placement.second_y, placement.second_x)

    first_pixels = first_image[first_mask > 0]
    second_pixels = second_image[second_mask > 0]
    if first_pixels.size and second_pixels.size:
        brightness_offset = float(second_pixels.mean()) - float(first_pixels.mean())
        if abs(brightness_offset) > 5.0:
            adjusted = np.clip(
                first_image.astype(np.float32) + brightness_offset,
                0,
                255,
            ).astype(np.uint8)
            adjusted[first_mask == 0] = 255
            first_image = adjusted

    image = distance_transform_blend(
        first_image,
        second_image,
        first_mask,
        second_mask,
        feather_size=feather_size,
    )
    combined_mask = np.zeros((*first_mask.shape, 3), dtype=np.uint8)
    combined_mask[..., 1] = first_mask
    combined_mask[..., 2] = second_mask

    coordinates = np.argwhere((first_mask | second_mask) > 0)
    y1, x1 = coordinates.min(axis=0)
    y2, x2 = coordinates.max(axis=0) + 1
    y1 = max(int(y1) - crop_padding, 0)
    x1 = max(int(x1) - crop_padding, 0)
    y2 = min(int(y2) + crop_padding, image.shape[0])
    x2 = min(int(x2) + crop_padding, image.shape[1])
    return image[y1:y2, x1:x2], combined_mask[y1:y2, x1:x2]


def _sample_pair(
    samples: Sequence[ChromosomeSample],
    strategy: str,
    rng: random.Random,
) -> tuple[ChromosomeSample, ChromosomeSample]:
    if strategy == "random":
        return tuple(rng.sample(list(samples), 2))  # type: ignore[return-value]

    attribute = "karyotype_id" if strategy == "same-karyotype" else "category"
    groups: dict[object, list[ChromosomeSample]] = {}
    for sample in samples:
        groups.setdefault(getattr(sample, attribute), []).append(sample)
    eligible = [group for group in groups.values() if len(group) >= 2]
    if not eligible:
        raise ValueError(f"no group contains two samples for strategy {strategy}")
    return tuple(rng.sample(rng.choice(eligible), 2))  # type: ignore[return-value]


def generate_dataset(
    input_root: Path,
    output_root: Path,
    *,
    count: int = 10_000,
    mode: Literal["both", "touching", "overlap"] = "both",
    touching_probability: float = 0.8,
    sampling: Literal["random", "same-karyotype", "same-class"] = "same-karyotype",
    minimum_overlap: float = 0.2,
    maximum_overlap: float = 0.5,
    feather_size: int = 5,
    seed: int = 42,
    overwrite: bool = False,
) -> list[GenerationRecord]:
    """Generate an exact number of synthetic pairs and return their metadata."""
    if count < 1:
        raise ValueError("count must be positive")
    if not 0.0 <= touching_probability <= 1.0:
        raise ValueError("touching_probability must be in [0, 1]")
    manifest_path = output_root / "manifest.jsonl"
    if not overwrite and manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite {manifest_path}; choose a new output or pass --overwrite"
        )
    samples = load_chromosome_pool(input_root)
    rng = random.Random(seed)
    records: list[GenerationRecord] = []
    target_touching = (
        round(count * touching_probability)
        if mode == "both"
        else count
        if mode == "touching"
        else 0
    )
    target_overlap = count - target_touching
    touching_done = 0
    overlap_done = 0
    maximum_attempts = max(count * 50, 100)
    attempts = 0

    while len(records) < count and attempts < maximum_attempts:
        attempts += 1
        try:
            original_first, original_second = _sample_pair(samples, sampling, rng)
            first, second, first_angle, second_angle = rotate_to_distinct_orientations(
                original_first,
                original_second,
                rng,
            )
            if mode == "touching":
                interaction: Interaction = "TOUCHING"
            elif mode == "overlap":
                interaction = "OVERLAP"
            else:
                touching_left = target_touching - touching_done
                overlap_left = target_overlap - overlap_done
                if touching_left == 0:
                    interaction = "OVERLAP"
                elif overlap_left == 0:
                    interaction = "TOUCHING"
                else:
                    interaction = (
                        "TOUCHING"
                        if rng.randrange(touching_left + overlap_left) < touching_left
                        else "OVERLAP"
                    )

            if interaction == "TOUCHING":
                placement = find_touching_placement(first, second, rng)
            else:
                placement = find_overlap_placement(
                    first,
                    second,
                    rng,
                    minimum=minimum_overlap,
                    maximum=maximum_overlap,
                )
            image, mask = compose_pair(
                first,
                second,
                placement,
                feather_size=feather_size,
            )
        except RuntimeError:
            continue

        index = len(records)
        filename = f"{index:06d}_{interaction}_{first.category}_{second.category}.png"
        image_path = output_root / "images" / interaction / filename
        mask_path = output_root / "masks" / interaction / filename
        if not overwrite and (image_path.exists() or mask_path.exists()):
            raise FileExistsError(
                f"refusing to overwrite {filename}; choose a new output or pass --overwrite"
            )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"could not write {image_path}")
        # Values intentionally remain 0/1 for Instance_segmentation_touch_overlap.
        if not cv2.imwrite(str(mask_path), mask):
            raise OSError(f"could not write {mask_path}")

        records.append(
            GenerationRecord(
                index=index,
                interaction=interaction,
                image=image_path.relative_to(output_root).as_posix(),
                mask=mask_path.relative_to(output_root).as_posix(),
                first_source=original_first.relative_path,
                second_source=original_second.relative_path,
                first_category=first.category,
                second_category=second.category,
                first_rotation_degrees=first_angle,
                second_rotation_degrees=second_angle,
                overlap_fraction=placement.overlap_fraction,
            )
        )
        if interaction == "TOUCHING":
            touching_done += 1
        else:
            overlap_done += 1
        if len(records) % 100 == 0 or len(records) == count:
            print(f"Generated {len(records)}/{count} pairs")

    if len(records) != count:
        raise RuntimeError(
            f"generated only {len(records)}/{count} pairs after {attempts} attempts"
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Extracted chromosome pool"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Generated dataset directory"
    )
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument(
        "--mode", choices=("both", "touching", "overlap"), default="both"
    )
    parser.add_argument("--touching-probability", type=float, default=0.8)
    parser.add_argument(
        "--sampling",
        choices=("random", "same-karyotype", "same-class"),
        default="same-karyotype",
    )
    parser.add_argument("--minimum-overlap", type=float, default=0.2)
    parser.add_argument("--maximum-overlap", type=float, default=0.5)
    parser.add_argument("--feather-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    records = generate_dataset(
        args.input,
        args.output,
        count=args.count,
        mode=args.mode,
        touching_probability=args.touching_probability,
        sampling=args.sampling,
        minimum_overlap=args.minimum_overlap,
        maximum_overlap=args.maximum_overlap,
        feather_size=args.feather_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    touching = sum(record.interaction == "TOUCHING" for record in records)
    overlap = len(records) - touching
    print(
        f"Saved {len(records)} pairs ({touching} touching, {overlap} overlap) to {args.output}"
    )


if __name__ == "__main__":
    main()
