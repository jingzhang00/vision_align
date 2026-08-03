from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from repaint.hand_trajectory import (
    HAND_TARGET_MODE_ATTRIBUTE,
    RIGID_PALM_TARGET,
    align_hand_action,
    load_hand_trajectory,
)
from repaint.rgbd_episode import load_rgbd_episode


class NymeriaRobotFormatTest(unittest.TestCase):
    def test_hand_action_returns_only_smoothed_width(self) -> None:
        keypoints = np.zeros((5, 4, 3), dtype=np.float64)
        keypoints[:, 1, 0] = 0.08
        keypoints[:, 2, 0] = 0.02
        keypoints[:, 3, 0] = 0.03

        widths = align_hand_action(keypoints, np.ones((5, 4)))

        self.assertEqual(widths.shape, (5,))
        self.assertTrue(np.isfinite(widths).all())

    def test_rigid_hand_loader_needs_only_consumed_data(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hands.hdf5"
            transforms = np.repeat(np.eye(4)[None], 2, axis=0)
            with h5py.File(path, "w") as handle:
                handle.attrs[HAND_TARGET_MODE_ATTRIBUTE] = RIGID_PALM_TARGET
                handle.create_dataset("transforms/camera", data=transforms)
                handle.create_dataset("transforms/leftHand", data=transforms)
                handle.create_dataset("transforms/rightHand", data=transforms)

            trajectory = load_hand_trajectory(path)

            self.assertEqual(trajectory.frame_count, 2)
            self.assertEqual(trajectory.target_mode, RIGID_PALM_TARGET)
            self.assertTrue(np.isnan(trajectory.left_widths).all())

    def test_loads_native_trajectory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "robot"
            removed = root / "handless"
            episode = Path("episodes/episode_000000")
            (dataset / episode).mkdir(parents=True)
            (removed / episode).mkdir(parents=True)
            source_video = root / "source.mp4"
            adapter = root / "adapter.hdf5"
            source_video.touch()
            (removed / episode / "source.mp4").touch()

            intrinsic = np.eye(3)
            source_camera = np.repeat(np.eye(4)[None], 2, axis=0)
            source_camera[1, 0, 3] = 0.1
            with h5py.File(adapter, "w") as handle:
                handle.create_dataset("camera/intrinsic", data=intrinsic)
                handle.create_dataset("transforms/camera", data=source_camera)

            camera_pose = np.array(
                [
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                    [1.2, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
            np.savez_compressed(
                dataset / episode / "trajectory.npz",
                joint_positions=np.zeros((2, 1)),
                root_pose_xyzw=np.tile([0, 0, 0, 0, 0, 0, 1], (2, 1)),
                camera_pose_cv_xyzw=camera_pose,
                camera_intrinsic=intrinsic,
                frame_indices=np.arange(2),
            )
            (dataset / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "nymeria-g1-trajectory-v1",
                        "joint_names": ["joint"],
                        "episodes": [
                            {
                                "episode_index": 0,
                                "source_stem": "clip",
                                "source_video": str(source_video),
                                "video_path": str(episode / "source.mp4"),
                                "trajectory_path": str(
                                    episode / "trajectory.npz"
                                ),
                                "frame_count": 2,
                                "camera_alignment": {
                                    "status": "validated",
                                    "adapter_hdf5": str(adapter),
                                    "source_to_robot_camera_translation_scale": 2.0,
                                    "source_to_robot_rotation": np.eye(3).tolist(),
                                    "source_to_robot_translation": [1, 2, 3],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "repaint.rgbd_episode._video_metadata",
                return_value=(640, 480, 2, 30.0),
            ):
                loaded = load_rgbd_episode(
                    dataset_root=dataset,
                    source_root=root,
                    removed_root=removed,
                    episode_index=0,
                )

            np.testing.assert_allclose(loaded.camera_pose_cv_xyzw, camera_pose)
            self.assertEqual(loaded.joint_names, ("joint",))
            self.assertEqual(loaded.frame_count, 2)


if __name__ == "__main__":
    unittest.main()
