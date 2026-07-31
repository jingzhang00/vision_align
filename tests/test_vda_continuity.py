from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from repaint.vda_continuity import (
    analyze_prepared_episode,
    depth_median_transition,
    flow_aligned_abs_relative_from_flows,
    frame_depth_statistics,
    symmetric_nearest_neighbor_statistics,
    world_points_from_depth,
    write_report,
)


class VdaContinuityMetricTests(unittest.TestCase):
    def test_depth_median_jump_ratio_and_triangle_count(self) -> None:
        previous = np.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        current = np.where(previous > 0.0, 1.1, 0.0)

        transition = depth_median_transition(previous, current)
        self.assertAlmostEqual(transition["signed_jump_m"], 0.1)
        self.assertAlmostEqual(transition["absolute_jump_m"], 0.1)
        self.assertAlmostEqual(
            transition["ratio_current_over_previous"],
            1.1,
        )
        self.assertAlmostEqual(transition["symmetric_ratio"], 1.1)

        statistics = frame_depth_statistics(
            np.ones((3, 4), dtype=np.float64)
        )
        self.assertEqual(statistics["valid_pixel_count"], 12)
        self.assertEqual(statistics["valid_rate"], 1.0)
        self.assertEqual(statistics["triangle_count"], 12)

    def test_known_backward_flow_perfectly_aligns_depth(self) -> None:
        previous = np.tile(
            np.arange(1.0, 6.0, dtype=np.float64),
            (4, 1),
        )
        current = np.zeros_like(previous)
        current[:, 1:] = previous[:, :-1]
        backward_flow = np.zeros((4, 5, 2), dtype=np.float32)
        backward_flow[..., 0] = -1.0

        statistics = flow_aligned_abs_relative_from_flows(
            previous,
            current,
            backward_flow,
        )

        self.assertEqual(statistics["valid_pixel_count"], 16)
        self.assertEqual(statistics["coverage_of_current_valid"], 1.0)
        self.assertEqual(statistics["p50"], 0.0)
        self.assertEqual(statistics["p95"], 0.0)

    def test_world_projection_and_symmetric_nn(self) -> None:
        intrinsic = np.eye(3, dtype=np.float64)
        pose = np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        points = world_points_from_depth(
            np.asarray([[1.0]], dtype=np.float64),
            intrinsic,
            pose,
            point_stride=1,
            max_points=None,
        )
        np.testing.assert_allclose(points, [[1.0, 2.0, 4.0]])

        statistics = symmetric_nearest_neighbor_statistics(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
        )
        self.assertEqual(statistics["distance_sample_count"], 2)
        self.assertEqual(statistics["p50_m"], 1.0)
        self.assertEqual(statistics["p95_m"], 1.0)

    def test_analyze_prepared_episode_and_write_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb_root = root / "rgb"
            depth_root = root / "depth_mm"
            rgb_root.mkdir()
            depth_root.mkdir()

            random = np.random.default_rng(7)
            rgb = random.integers(
                0,
                256,
                size=(24, 32, 3),
                dtype=np.uint8,
            )
            depth_mm = np.full((6, 8), 1000, dtype=np.uint16)
            for frame_index in range(2):
                self.assertTrue(
                    cv2.imwrite(
                        str(rgb_root / f"{frame_index:05d}.png"),
                        rgb,
                    )
                )
                self.assertTrue(
                    cv2.imwrite(
                        str(depth_root / f"{frame_index:05d}.png"),
                        depth_mm,
                    )
                )

            intrinsic = np.asarray(
                [
                    [4.0, 0.0, 3.5],
                    [0.0, 4.0, 2.5],
                    [0.0, 0.0, 1.0],
                ]
            )
            poses = np.repeat(
                np.asarray(
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
                ),
                2,
                axis=0,
            )
            np.savez_compressed(
                root / "episode_data.npz",
                episode_index=np.asarray(3, dtype=np.int64),
                fps=np.asarray(30.0, dtype=np.float64),
                source_size=np.asarray([32, 24], dtype=np.int64),
                mesh_size=np.asarray([8, 6], dtype=np.int64),
                frame_indices=np.asarray([10, 11], dtype=np.int64),
                mesh_intrinsic=intrinsic,
                camera_pose_cv_xyzw=poses,
            )

            report = analyze_prepared_episode(
                root,
                point_stride=1,
                max_points=None,
                flow_forward_backward_threshold_px=None,
            )

            self.assertEqual(report["frame_count"], 2)
            self.assertEqual(report["adjacent_pair_count"], 1)
            self.assertEqual(report["frames"][0]["valid_rate"], 1.0)
            self.assertEqual(report["frames"][0]["triangle_count"], 70)
            pair = report["adjacent_pairs"][0]
            self.assertAlmostEqual(
                pair["depth_median"]["absolute_jump_m"],
                0.0,
            )
            self.assertAlmostEqual(
                pair["flow_aligned_abs_relative"]["p95"],
                0.0,
                places=7,
            )
            self.assertAlmostEqual(
                pair["world_symmetric_nearest_neighbor"]["p95_m"],
                0.0,
            )
            self.assertEqual(
                pair["triangle_count"]["absolute_jump"],
                0,
            )

            output = write_report(report, root / "report.json")
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["episode_index"], 3)
            self.assertEqual(loaded["summary"]["frame_valid_rate"]["p50"], 1.0)


if __name__ == "__main__":
    unittest.main()
