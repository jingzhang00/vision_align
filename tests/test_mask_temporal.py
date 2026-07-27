from __future__ import annotations

import unittest

import numpy as np

from human_to_robot.segment_sam3 import repair_temporal_mask_dropouts


def _rectangle(
    *,
    shape: tuple[int, int] = (64, 96),
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


class TemporalMaskRepairTests(unittest.TestCase):
    def test_repairs_single_empty_frame_between_consistent_neighbors(
        self,
    ) -> None:
        previous = _rectangle(x0=20, y0=16, x1=48, y1=52)
        following = _rectangle(x0=21, y0=16, x1=49, y1=52)
        missing = np.zeros_like(previous)

        repaired, repaired_frames = repair_temporal_mask_dropouts(
            [previous, missing, following]
        )

        self.assertEqual(repaired_frames, [1])
        self.assertGreater(np.count_nonzero(repaired[1]), 0)
        np.testing.assert_array_equal(
            repaired[1] != 0,
            (previous != 0) | (following != 0),
        )
        np.testing.assert_array_equal(repaired[0], previous)
        np.testing.assert_array_equal(repaired[2], following)

    def test_repairs_short_partial_component_collapse(self) -> None:
        left = _rectangle(x0=8, y0=18, x1=28, y1=50)
        right = _rectangle(x0=54, y0=14, x1=78, y1=50)
        complete = np.bitwise_or(left, right)
        partial = left.copy()

        repaired, repaired_frames = repair_temporal_mask_dropouts(
            [complete, partial, partial, complete]
        )

        self.assertEqual(repaired_frames, [1, 2])
        for index in repaired_frames:
            np.testing.assert_array_equal(repaired[index], complete)

    def test_does_not_extend_a_sustained_disappearance(self) -> None:
        present = _rectangle(x0=20, y0=16, x1=48, y1=52)
        missing = np.zeros_like(present)

        repaired, repaired_frames = repair_temporal_mask_dropouts(
            [present, present, missing, missing, missing]
        )

        self.assertEqual(repaired_frames, [])
        for index in (2, 3, 4):
            self.assertEqual(np.count_nonzero(repaired[index]), 0)

    def test_does_not_bridge_spatially_inconsistent_neighbors(self) -> None:
        previous = _rectangle(x0=2, y0=16, x1=26, y1=52)
        following = _rectangle(x0=68, y0=16, x1=92, y1=52)
        missing = np.zeros_like(previous)

        repaired, repaired_frames = repair_temporal_mask_dropouts(
            [previous, missing, following]
        )

        self.assertEqual(repaired_frames, [])
        self.assertEqual(np.count_nonzero(repaired[1]), 0)


if __name__ == "__main__":
    unittest.main()
