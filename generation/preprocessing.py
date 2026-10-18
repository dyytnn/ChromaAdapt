"""Karyotype binarization, weak-bridge splitting, and gap/hole repair.

This module is a standalone extraction of the preprocessing used to build the
single-chromosome pool for the paper. It intentionally has no dependency on
the original application registry, application data structures, visualizers,
or data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, erosion


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass
class ChromosomeInstance:
    """One extracted chromosome crop and its binary mask."""

    image: np.ndarray
    mask: np.ndarray
    bounding_box: tuple[int, int, int, int]
    category: int = 0


def binarize_karyotype(image: np.ndarray, threshold: int = 225) -> np.ndarray:
    """Return a 0/1 mask in which pixels darker than ``threshold`` are foreground."""
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty array")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in [0, 255]")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {image.shape}")
    return (gray < threshold).astype(np.uint8)


def fill_gap_holes(
    mask: np.ndarray,
    max_area_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill small background components in a chromosome mask.

    The area limit is relative to the crop area, matching the original
    ``remove_gap_hole`` implementation.  The returned arrays are the repaired
    0/1 mask and a boolean map of the pixels that were filled.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if not 0.0 <= max_area_ratio <= 1.0:
        raise ValueError("max_area_ratio must be in [0, 1]")

    foreground = mask.astype(bool)
    background = np.logical_not(foreground)
    structure = ndi.generate_binary_structure(2, 1)
    components, _ = ndi.label(background, structure=structure)
    component_sizes = np.bincount(components.ravel())
    area_limit = int(max_area_ratio * mask.shape[0] * mask.shape[1])

    filled = np.zeros_like(foreground)
    if area_limit > 0 and len(component_sizes) > 1:
        small = component_sizes < area_limit
        small[0] = False
        filled = small[components]

    repaired = foreground | filled
    return repaired.astype(np.uint8), filled


def _stable_under_erosion(mask: np.ndarray) -> bool:
    """Return whether progressive erosion preserves the component count."""
    original_count = int(label(mask > 0).max())
    for radius in (1, 2, 3):
        eroded_count = int(label(erosion(mask > 0, disk(radius))).max())
        if eroded_count != original_count:
            return False
    return True


def split_weak_connections(
    mask: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    *,
    erosion_radius: int = 2,
    max_depth: int = 3,
    _depth: int = 0,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Recursively split components connected by a thin morphological bridge."""
    binary = (mask > 0).astype(np.uint8)
    if _depth >= max_depth or _stable_under_erosion(binary):
        return [(binary, bounding_box)]

    separated = label(erosion(binary, disk(erosion_radius)))
    if separated.max() <= 1:
        return [(binary, bounding_box)]

    x1, y1, _, _ = bounding_box
    results: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for component_id in range(1, int(separated.max()) + 1):
        component = separated == component_id
        recovered = dilation(component, disk(erosion_radius)) & (binary > 0)
        coordinates = np.argwhere(recovered)
        if coordinates.size == 0:
            continue

        local_y1, local_x1 = coordinates.min(axis=0)
        local_y2, local_x2 = coordinates.max(axis=0) + 1
        child_mask = recovered[local_y1:local_y2, local_x1:local_x2].astype(np.uint8)
        child_box = (
            int(x1 + local_x1),
            int(y1 + local_y1),
            int(x1 + local_x2),
            int(y1 + local_y2),
        )
        results.extend(
            split_weak_connections(
                child_mask,
                child_box,
                erosion_radius=erosion_radius,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        )

    return results or [(binary, bounding_box)]


def _sudden_drop_index(instances: Sequence[ChromosomeInstance], factor: float) -> int:
    for index in range(1, len(instances)):
        previous_area = int(instances[index - 1].mask.sum())
        current_area = int(instances[index].mask.sum())
        if current_area < previous_area * factor:
            return index
    return len(instances)


def _masked_crop(
    image: np.ndarray,
    mask: np.ndarray,
    bounding_box: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = bounding_box
    crop = image[y1:y2, x1:x2].copy()
    if crop.shape[:2] != mask.shape:
        raise ValueError(
            f"crop shape {crop.shape[:2]} does not match mask shape {mask.shape}"
        )
    output = np.full_like(crop, 255)
    output[mask > 0] = crop[mask > 0]
    return output


def extract_chromosomes(
    image: np.ndarray,
    *,
    threshold: int = 225,
    bbox_padding: int = 8,
    num_rows: int = 4,
    drop_factor: float = 0.5,
    max_hole_area_ratio: float = 0.1,
    split_bridges: bool = True,
    split_max_depth: int = 3,
) -> list[ChromosomeInstance]:
    """Extract, clean, and spatially order chromosome instances from a karyotype."""
    if bbox_padding < 0:
        raise ValueError("bbox_padding must be non-negative")
    if num_rows < 1:
        raise ValueError("num_rows must be positive")
    if not 0.0 < drop_factor <= 1.0:
        raise ValueError("drop_factor must be in (0, 1]")

    binary = binarize_karyotype(image, threshold)
    component_map = label(binary)
    height, width = binary.shape
    instances: list[ChromosomeInstance] = []

    for region in regionprops(component_map):
        region_y1, region_x1, region_y2, region_x2 = region.bbox
        x1 = max(int(region_x1) - bbox_padding, 0)
        y1 = max(int(region_y1) - bbox_padding, 0)
        x2 = min(int(region_x2) + bbox_padding, width)
        y2 = min(int(region_y2) + bbox_padding, height)
        box = (x1, y1, x2, y2)
        component_mask = (component_map[y1:y2, x1:x2] == region.label).astype(np.uint8)

        if split_bridges:
            candidates = split_weak_connections(
                component_mask,
                box,
                max_depth=split_max_depth,
            )
        else:
            candidates = [(component_mask, box)]

        for candidate_mask, candidate_box in candidates:
            repaired_mask, _ = fill_gap_holes(candidate_mask, max_hole_area_ratio)
            if repaired_mask.any():
                instances.append(
                    ChromosomeInstance(
                        image=_masked_crop(image, repaired_mask, candidate_box),
                        mask=repaired_mask,
                        bounding_box=candidate_box,
                    )
                )

    instances.sort(key=lambda item: int(item.mask.sum()), reverse=True)
    instances = instances[: _sudden_drop_index(instances, drop_factor)]

    row_height = height / num_rows
    rows: list[list[ChromosomeInstance]] = [[] for _ in range(num_rows)]
    for instance in instances:
        x1, y1, x2, y2 = instance.bounding_box
        center_y = (y1 + y2) / 2.0
        row_index = min(int(center_y // row_height), num_rows - 1)
        rows[row_index].append(instance)
    for row in rows:
        row.sort(key=lambda item: (item.bounding_box[0] + item.bounding_box[2]) / 2.0)

    ordered = [item for row in rows for item in row]
    for category, instance in enumerate(ordered, start=1):
        instance.category = category
    return ordered


def _iter_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image extension: {input_path.suffix}")
        yield input_path
        return
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    yield from sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def extract_path(input_path: Path, output_path: Path, **kwargs: object) -> int:
    """Extract every karyotype at ``input_path`` and save an image/mask pool."""
    total = 0
    for image_path in _iter_images(input_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {image_path}")
        instances = extract_chromosomes(image, **kwargs)
        sample_id = image_path.stem
        image_dir = output_path / "images" / sample_id
        mask_dir = output_path / "masks" / sample_id
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for instance in instances:
            filename = f"chromosome_category_{instance.category}.png"
            if not cv2.imwrite(str(image_dir / filename), instance.image):
                raise OSError(f"could not write {image_dir / filename}")
            if not cv2.imwrite(str(mask_dir / filename), instance.mask):
                raise OSError(f"could not write {mask_dir / filename}")
        print(f"{image_path.name}: extracted {len(instances)} chromosomes")
        total += len(instances)
    return total


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Karyotype image or directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output pool directory"
    )
    parser.add_argument("--threshold", type=int, default=225)
    parser.add_argument("--bbox-padding", type=int, default=8)
    parser.add_argument("--num-rows", type=int, default=4)
    parser.add_argument("--drop-factor", type=float, default=0.5)
    parser.add_argument("--max-hole-area-ratio", type=float, default=0.1)
    parser.add_argument("--no-split-bridges", action="store_true")
    parser.add_argument("--split-max-depth", type=int, default=3)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    total = extract_path(
        args.input,
        args.output,
        threshold=args.threshold,
        bbox_padding=args.bbox_padding,
        num_rows=args.num_rows,
        drop_factor=args.drop_factor,
        max_hole_area_ratio=args.max_hole_area_ratio,
        split_bridges=not args.no_split_bridges,
        split_max_depth=args.split_max_depth,
    )
    print(f"Saved {total} chromosome crops under {args.output}")


if __name__ == "__main__":
    main()
