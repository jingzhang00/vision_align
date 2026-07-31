from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from repaint.robot_camera import load_source_hand_points_camera_cv
from repaint.scene_hand_alignment import (
    align_scene_camera_poses_to_initial_hands,
    estimate_initial_hand_midpoint_translation,
)
from repaint.vda_scene_isaac import build_parser


class SceneHandAlignmentTests(unittest.TestCase):
    def test_recovers_world_translation_despite_morphology_and_outlier(
        self,
    ) -> None:
        frame_count = 15
        angles = np.linspace(-0.3, 0.4, frame_count)
        rotations = Rotation.from_euler(
            "z",
            angles,
        )
        camera_pose = np.zeros((frame_count, 7), dtype=np.float64)
        camera_pose[:, 3:7] = rotations.as_quat()
        source = np.repeat(
            np.asarray(
                [[[-0.2, 0.0, 0.8], [0.2, 0.0, 0.8]]],
                dtype=np.float64,
            ),
            frame_count,
            axis=0,
        )
        scene_scale = 0.8
        expected = np.asarray([-0.01, 0.03, -0.11])
        expected_camera = rotations.inv().apply(
            np.repeat(expected[None], frame_count, axis=0)
        )
        robot = scene_scale * source
        robot += expected_camera[:, None, :]
        # Opposite per-hand offsets model a human/G1 arm-span mismatch. Their
        # midpoint remains a valid scene-position anchor.
        robot[:, 0, 0] -= 0.07
        robot[:, 1, 0] += 0.07
        # One gross temporal outlier must not move the robust translation.
        robot[-1] += np.asarray([0.4, -0.2, 0.3])
        confidence = np.ones((frame_count, 2), dtype=np.float64)

        translation, report = estimate_initial_hand_midpoint_translation(
            source_hand_points_camera_cv=source,
            source_hand_confidence=confidence,
            robot_hand_points_camera_cv=robot,
            scene_camera_pose_cv_xyzw=camera_pose,
            scene_scale=scene_scale,
        )

        np.testing.assert_allclose(translation, expected, atol=1e-10)
        self.assertEqual(report["robust_inlier_count"], frame_count - 1)
        self.assertLess(
            report["midpoint_inlier_residual_after_m"]["p95"],
            1e-10,
        )
        self.assertGreater(
            report["individual_hand_residual_after_m"]["p50"],
            0.05,
        )

    def test_rejects_too_few_paired_confident_frames(self) -> None:
        frame_count = 15
        source = np.ones((frame_count, 2, 3), dtype=np.float64)
        robot = source.copy()
        confidence = np.zeros((frame_count, 2), dtype=np.float64)
        confidence[:3] = 1.0
        camera_pose = np.zeros((frame_count, 7), dtype=np.float64)
        camera_pose[:, 6] = 1.0

        with self.assertRaisesRegex(ValueError, "Too few paired"):
            estimate_initial_hand_midpoint_translation(
                source_hand_points_camera_cv=source,
                source_hand_confidence=confidence,
                robot_hand_points_camera_cv=robot,
                scene_camera_pose_cv_xyzw=camera_pose,
                scene_scale=1.0,
            )

    def test_rejects_implausibly_large_translation(self) -> None:
        frame_count = 15
        source = np.repeat(
            np.asarray([[[-0.2, 0.0, 0.8], [0.2, 0.0, 0.8]]]),
            frame_count,
            axis=0,
        )
        robot = source + np.asarray([0.31, 0.0, 0.0])
        confidence = np.ones((frame_count, 2), dtype=np.float64)
        camera_pose = np.zeros((frame_count, 7), dtype=np.float64)
        camera_pose[:, 6] = 1.0

        with self.assertRaisesRegex(ValueError, "implausibly large"):
            estimate_initial_hand_midpoint_translation(
                source_hand_points_camera_cv=source,
                source_hand_confidence=confidence,
                robot_hand_points_camera_cv=robot,
                scene_camera_pose_cv_xyzw=camera_pose,
                scene_scale=1.0,
            )

    def test_rejects_unstable_single_translation(self) -> None:
        frame_count = 15
        source = np.repeat(
            np.asarray([[[-0.2, 0.0, 0.8], [0.2, 0.0, 0.8]]]),
            frame_count,
            axis=0,
        )
        offsets = np.linspace(-0.04, 0.04, frame_count)
        robot = source.copy()
        robot[:, :, 0] += offsets[:, None]
        confidence = np.ones((frame_count, 2), dtype=np.float64)
        camera_pose = np.zeros((frame_count, 7), dtype=np.float64)
        camera_pose[:, 6] = 1.0

        with self.assertRaisesRegex(ValueError, "not stable enough"):
            estimate_initial_hand_midpoint_translation(
                source_hand_points_camera_cv=source,
                source_hand_confidence=confidence,
                robot_hand_points_camera_cv=robot,
                scene_camera_pose_cv_xyzw=camera_pose,
                scene_scale=1.0,
            )

    def test_wrapper_applies_one_constant_translation_to_all_scene_poses(
        self,
    ) -> None:
        frame_count = 5
        source = np.repeat(
            np.asarray([[[-0.2, 0.0, 0.8], [0.2, 0.0, 0.8]]]),
            3,
            axis=0,
        )
        expected = np.asarray([0.02, -0.01, 0.08])
        robot = source + expected
        confidence = np.ones((3, 2), dtype=np.float64)
        camera_pose = np.zeros((frame_count, 7), dtype=np.float64)
        camera_pose[:, :3] = np.arange(frame_count)[:, None] * 0.01
        camera_pose[:, 6] = 1.0

        with tempfile.TemporaryDirectory() as temporary:
            source_hdf5 = Path(temporary) / "source.hdf5"
            with h5py.File(source_hdf5, "w"):
                pass
            with (
                patch(
                    "repaint.scene_hand_alignment."
                    "load_source_hand_points_camera_cv",
                    return_value=(source, confidence),
                ),
                patch(
                    "repaint.scene_hand_alignment."
                    "robot_frame_points_in_camera_cv",
                    return_value=robot,
                ),
            ):
                alignment = align_scene_camera_poses_to_initial_hands(
                    source_hdf5=source_hdf5,
                    frame_indices=np.arange(frame_count),
                    scene_scale=1.0,
                    scene_camera_pose_cv_xyzw=camera_pose,
                    urdf_path=Path(temporary) / "g1.urdf",
                    joint_names=("joint",),
                    joint_positions=np.zeros((frame_count, 1)),
                    root_pose_xyzw=np.zeros((frame_count, 7)),
                    window_frames=3,
                )

        np.testing.assert_allclose(
            alignment.camera_pose_cv_xyzw[:, :3],
            camera_pose[:, :3] + expected,
        )
        np.testing.assert_allclose(
            alignment.camera_pose_cv_xyzw[:, 3:],
            camera_pose[:, 3:],
        )
        self.assertEqual(alignment.report["application_count"], 1)
        self.assertEqual(
            alignment.report["applied_to"],
            "scene mesh camera-pose translations only",
        )

    def test_default_replay_mode_is_initial_hand_alignment(self) -> None:
        args = build_parser().parse_args(["--prepared", "prepared"])
        self.assertEqual(args.scene_alignment, "initial-hands")
        self.assertEqual(args.scene_alignment_window_frames, 15)
        self.assertEqual(args.scene_hand_confidence, 0.5)


class MissingSourceConfidenceTests(unittest.TestCase):
    def test_missing_confidence_defaults_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.hdf5"
            camera = np.repeat(
                np.eye(4, dtype=np.float64)[None],
                3,
                axis=0,
            )
            left = camera.copy()
            right = camera.copy()
            left[:, 0, 3] = -0.2
            right[:, 0, 3] = 0.2
            left[:, 2, 3] = 0.8
            right[:, 2, 3] = 0.8
            with h5py.File(path, "w") as handle:
                transforms = handle.create_group("transforms")
                transforms.create_dataset("camera", data=camera)
                transforms.create_dataset("leftHand", data=left)
                transforms.create_dataset("rightHand", data=right)

            points, confidence = load_source_hand_points_camera_cv(
                path,
                np.asarray([1, 2], dtype=np.int64),
            )

        self.assertEqual(points.shape, (2, 2, 3))
        np.testing.assert_allclose(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
