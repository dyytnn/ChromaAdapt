"""Data-free smoke tests for chromosome extraction and pair generation."""

import random
import unittest

import cv2
import numpy as np
from scipy import ndimage as ndi

from generation.preprocessing import (
    extract_chromosomes,
    fill_gap_holes,
    split_weak_connections,
)
from generation.synthetic_pairs import (
    ChromosomeSample,
    compose_pair,
    find_overlap_placement,
    find_touching_placement,
    overlap_fraction,
)


def _sample(vertical: bool) -> ChromosomeSample:
    height, width = (60, 20) if vertical else (20, 60)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        (max(2, width // 3), max(2, height // 2 - 2)),
        0,
        0,
        360,
        1,
        -1,
    )
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[mask > 0] = (80, 100, 120)
    return ChromosomeSample(
        image=image,
        mask=mask,
        category=1 if vertical else 2,
        karyotype_id="in-memory",
        relative_path="in-memory.png",
    )


class PreprocessingTests(unittest.TestCase):
    def test_gap_hole_fill(self) -> None:
        mask = np.ones((21, 21), dtype=np.uint8)
        mask[9:12, 9:12] = 0
        repaired, filled = fill_gap_holes(mask, max_area_ratio=0.05)
        self.assertTrue(repaired.all())
        self.assertEqual(int(filled.sum()), 9)

    def test_connected_component_extraction(self) -> None:
        image = np.full((240, 360, 3), 255, dtype=np.uint8)
        centers = [
            (45, 45),
            (120, 45),
            (195, 45),
            (270, 45),
            (45, 150),
            (120, 150),
            (195, 150),
            (270, 150),
        ]
        for center in centers:
            cv2.ellipse(image, center, (9, 28), 12, 0, 360, (70, 70, 70), -1)
        instances = extract_chromosomes(
            image,
            bbox_padding=4,
            num_rows=2,
            split_bridges=False,
        )
        self.assertEqual(len(instances), len(centers))
        self.assertEqual(
            [instance.category for instance in instances],
            list(range(1, len(centers) + 1)),
        )

    def test_weak_bridge_is_split(self) -> None:
        mask = np.zeros((50, 100), dtype=np.uint8)
        cv2.circle(mask, (25, 25), 14, 1, -1)
        cv2.circle(mask, (75, 25), 14, 1, -1)
        mask[25, 39:62] = 1
        parts = split_weak_connections(mask, (0, 0, 100, 50))
        self.assertEqual(len(parts), 2)


class SyntheticPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = _sample(vertical=True)
        self.second = _sample(vertical=False)
        self.rng = random.Random(42)

    def test_touching_pair_has_contact_without_overlap(self) -> None:
        placement = find_touching_placement(self.first, self.second, self.rng)
        _, mask = compose_pair(self.first, self.second, placement)
        first_mask = mask[..., 1] > 0
        second_mask = mask[..., 2] > 0
        self.assertFalse(np.logical_and(first_mask, second_mask).any())
        contact = np.logical_and(
            ndi.binary_dilation(first_mask, structure=np.ones((3, 3), dtype=bool)),
            second_mask,
        )
        self.assertTrue(contact.any())

    def test_overlap_pair_respects_requested_range(self) -> None:
        placement = find_overlap_placement(
            self.first,
            self.second,
            self.rng,
            minimum=0.1,
            maximum=0.4,
        )
        _, mask = compose_pair(self.first, self.second, placement)
        fraction = overlap_fraction(mask[..., 1], mask[..., 2])
        self.assertGreaterEqual(fraction, 0.1)
        self.assertLessEqual(fraction, 0.4)
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
