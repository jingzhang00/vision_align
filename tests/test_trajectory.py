from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from human_to_robot.trajectory import (
    HAND_KEYPOINT_NAMES,
    HAND_LANDMARK_NAMES,
    hdf5_action_keypoint_availability,
    load_hand_trajectory,
)


def _pose(
    translation: np.ndarray,
    rotation_vector: np.ndarray | None = None,
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = translation
    if rotation_vector is not None:
        result[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    return result


def _landmark_local_pose(
    name: str,
    *,
    side: str,
    frame_id: int,
) -> np.ndarray:
    if name == "Hand":
        return np.eye(4, dtype=np.float64)

    side_sign = -1.0 if side == "left" else 1.0
    special_positions = {
        "ThumbTip": np.array([0.04, 0.10, 0.0]),
        "IndexFingerTip": np.array([0.0, 0.10, 0.0]),
        "MiddleFingerTip": np.array([0.0, 0.10, 0.0]),
    }
    if name in special_positions:
        position = special_positions[name].copy()
    else:
        landmark_id = HAND_LANDMARK_NAMES.index(name)
        position = np.array(
            [
                0.002 * landmark_id,
                0.025 + 0.003 * landmark_id,
                side_sign * 0.001 * landmark_id,
            ],
            dtype=np.float64,
        )
    rotation_vector = np.array(
        [
            0.002 * frame_id,
            side_sign * 0.005 * HAND_LANDMARK_NAMES.index(name),
            0.001 * frame_id,
        ],
        dtype=np.float64,
    )
    return _pose(position, rotation_vector)


class HandLandmarkTrajectoryTests(unittest.TestCase):
    def test_loader_defaults_missing_confidences_to_one(self) -> None:
        frame_count = 5
        positions = {
            "Hand": np.array([0.0, 0.0, 1.0]),
            "ThumbTip": np.array([0.04, 0.10, 1.0]),
            "IndexFingerTip": np.array([0.0, 0.10, 1.0]),
            "MiddleFingerTip": np.array([0.0, 0.10, 1.0]),
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.hdf5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("camera/intrinsic", data=np.eye(3))
                handle.create_dataset(
                    "transforms/camera",
                    data=np.repeat(np.eye(4)[None], frame_count, axis=0),
                )
                for side in ("left", "right"):
                    for name in HAND_LANDMARK_NAMES:
                        transforms = np.repeat(
                            np.eye(4)[None],
                            frame_count,
                            axis=0,
                        )
                        transforms[:, :3, 3] = positions.get(
                            name,
                            positions["Hand"],
                        )
                        handle.create_dataset(
                            f"transforms/{side}{name}",
                            data=transforms,
                        )

            availability = hdf5_action_keypoint_availability(path)
            trajectory = load_hand_trajectory(path)

        self.assertEqual(availability, {"left": True, "right": True})
        np.testing.assert_array_equal(
            trajectory.left_confidence,
            np.ones(frame_count),
        )
        np.testing.assert_array_equal(
            trajectory.right_confidence,
            np.ones(frame_count),
        )
        for side in ("left", "right"):
            articulated = getattr(trajectory, f"{side}_articulated_hand")
            self.assertIsNotNone(articulated)
            assert articulated is not None
            np.testing.assert_array_equal(
                articulated.confidence,
                np.ones((frame_count, len(HAND_LANDMARK_NAMES))),
            )

    def test_loader_preserves_complete_hand_tracks_in_reference_and_palm_frames(
        self,
    ) -> None:
        frame_count = 7
        landmark_count = len(HAND_LANDMARK_NAMES)
        camera_to_world = _pose(
            np.array([0.8, -0.3, 0.2]),
            np.array([0.0, 0.0, 0.35]),
        )
        expected_reference: dict[str, np.ndarray] = {}
        expected_palm: dict[str, np.ndarray] = {}
        expected_confidence: dict[str, np.ndarray] = {}

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.hdf5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("camera/intrinsic", data=np.eye(3))
                handle.create_dataset(
                    "transforms/camera",
                    data=np.repeat(
                        camera_to_world[None],
                        frame_count,
                        axis=0,
                    ),
                )
                for side, x_offset in (("left", -0.2), ("right", 0.2)):
                    reference_transforms = np.empty(
                        (frame_count, landmark_count, 4, 4),
                        dtype=np.float64,
                    )
                    palm_transforms = np.empty_like(reference_transforms)
                    confidence = np.empty(
                        (frame_count, landmark_count),
                        dtype=np.float64,
                    )
                    for frame_id in range(frame_count):
                        hand_reference = _pose(
                            np.array(
                                [
                                    x_offset,
                                    0.01 * frame_id,
                                    1.0,
                                ]
                            ),
                            np.array([0.0, 0.0, 0.02 * frame_id]),
                        )
                        for landmark_id, name in enumerate(
                            HAND_LANDMARK_NAMES
                        ):
                            local = _landmark_local_pose(
                                name,
                                side=side,
                                frame_id=frame_id,
                            )
                            reference = hand_reference @ local
                            world = camera_to_world @ reference
                            reference_transforms[
                                frame_id, landmark_id
                            ] = reference
                            palm_transforms[frame_id, landmark_id] = local
                            confidence[frame_id, landmark_id] = (
                                0.5
                                + 0.01 * frame_id
                                + 0.001 * landmark_id
                            )

                            handle.require_group("transforms")
                            handle.require_group("confidences")
                            transform_key = f"transforms/{side}{name}"
                            confidence_key = f"confidences/{side}{name}"
                            if transform_key not in handle:
                                handle.create_dataset(
                                    transform_key,
                                    shape=(frame_count, 4, 4),
                                    dtype=np.float64,
                                )
                                handle.create_dataset(
                                    confidence_key,
                                    shape=(frame_count,),
                                    dtype=np.float64,
                                )
                            handle[transform_key][frame_id] = world
                            handle[confidence_key][frame_id] = confidence[
                                frame_id, landmark_id
                            ]

                    expected_reference[side] = reference_transforms
                    expected_palm[side] = palm_transforms
                    expected_confidence[side] = confidence

            trajectory = load_hand_trajectory(path)

        for side in ("left", "right"):
            with self.subTest(side=side):
                landmarks = getattr(
                    trajectory,
                    f"{side}_articulated_hand",
                )
                self.assertIsNotNone(landmarks)
                assert landmarks is not None
                self.assertEqual(landmarks.names, HAND_LANDMARK_NAMES)
                self.assertEqual(landmarks.frame_count, frame_count)
                self.assertEqual(
                    landmarks.transforms_reference_camera.shape,
                    (frame_count, landmark_count, 4, 4),
                )
                np.testing.assert_allclose(
                    landmarks.transforms_reference_camera,
                    expected_reference[side],
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    landmarks.transforms_palm,
                    expected_palm[side],
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    landmarks.confidence,
                    expected_confidence[side],
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    landmarks.points_reference_camera,
                    expected_reference[side][..., :3, 3],
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    landmarks.rotations_palm,
                    expected_palm[side][..., :3, :3],
                    atol=1e-12,
                )
                hand_id = HAND_LANDMARK_NAMES.index("Hand")
                np.testing.assert_allclose(
                    landmarks.transforms_palm[:, hand_id],
                    np.repeat(
                        np.eye(4)[None],
                        frame_count,
                        axis=0,
                    ),
                    atol=1e-12,
                )

    def test_four_point_inputs_keep_existing_loader_contract(self) -> None:
        frame_count = 5
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.hdf5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("camera/intrinsic", data=np.eye(3))
                handle.create_dataset(
                    "transforms/camera",
                    data=np.repeat(np.eye(4)[None], frame_count, axis=0),
                )
                positions = {
                    "Hand": np.array([0.0, 0.0, 1.0]),
                    "ThumbTip": np.array([0.04, 0.10, 1.0]),
                    "IndexFingerTip": np.array([0.0, 0.10, 1.0]),
                    "MiddleFingerTip": np.array([0.0, 0.10, 1.0]),
                }
                for side in ("left", "right"):
                    for name in HAND_KEYPOINT_NAMES:
                        transforms = np.repeat(
                            np.eye(4)[None],
                            frame_count,
                            axis=0,
                        )
                        transforms[:, :3, 3] = positions[name]
                        handle.create_dataset(
                            f"transforms/{side}{name}",
                            data=transforms,
                        )
                        handle.create_dataset(
                            f"confidences/{side}{name}",
                            data=np.ones(frame_count),
                        )
                body_positions = {
                    "hip": np.array([0.0, 0.0, 0.3]),
                    "neck4": np.array([0.0, 0.0, 0.9]),
                    "leftArm": np.array([0.0, 0.2, 0.7]),
                    "leftForearm": np.array([0.2, 0.2, 0.7]),
                    "rightArm": np.array([0.0, -0.2, 0.7]),
                    "rightForearm": np.array([0.2, -0.2, 0.7]),
                }
                for name, position in body_positions.items():
                    transforms = np.repeat(
                        np.eye(4)[None],
                        frame_count,
                        axis=0,
                    )
                    transforms[:, :3, 3] = position
                    handle.create_dataset(
                        f"transforms/{name}",
                        data=transforms,
                    )

            trajectory = load_hand_trajectory(path)

        self.assertIsNone(trajectory.left_articulated_hand)
        self.assertIsNone(trajectory.right_articulated_hand)
        self.assertEqual(trajectory.frame_count, frame_count)
        self.assertTrue(np.isfinite(trajectory.left_widths).all())
        self.assertTrue(np.isfinite(trajectory.right_widths).all())
        self.assertIsNotNone(trajectory.body_alignment)
        assert trajectory.body_alignment is not None
        np.testing.assert_array_equal(
            trajectory.body_alignment.world_up,
            [0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            trajectory.body_alignment.shoulder_center_reference_camera,
            np.repeat(
                np.asarray([[0.0, 0.0, 0.7]]),
                frame_count,
                axis=0,
            ),
        )
        self.assertTrue(
            np.all(trajectory.body_alignment.arm_lengths > 0.0)
        )


if __name__ == "__main__":
    unittest.main()
