from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np

from human_to_robot.composite import compose_video
from human_to_robot.retarget_g1_pinocchio import build_pinocchio_g1
from human_to_robot.video_io import video_metadata


REPOSITORY = Path(__file__).resolve().parents[1]


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 12.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV mp4v encoder is unavailable")
    for frame in frames:
        writer.write(frame)
    writer.release()


class PinocchioRetargetingTests(unittest.TestCase):
    def test_g1_urdf_has_all_actuated_joints_and_operational_palms(self) -> None:
        model = build_pinocchio_g1(
            REPOSITORY
            / "assets/unitree_description/urdf/g1/main.urdf"
        )

        self.assertEqual(model.nq, 29)
        self.assertEqual(model.nv, 29)
        self.assertEqual(len(model.names[1:]), 29)
        for frame_name in ("left_palm", "right_palm"):
            self.assertLess(model.getFrameId(frame_name), model.nframes)
        for joint_name in (
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ):
            self.assertGreater(model.getJointId(joint_name), 0)


class StreamingCompositeTests(unittest.TestCase):
    def test_non_debug_streams_final_video_and_consumes_render_frames(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            inpaint = root / "inpaint.mp4"
            output = root / "output"
            g1_root = root / "g1"
            trajectory_path = root / "trajectory" / "g1_trajectory.npz"
            frame_count = 4
            width, height = 48, 32
            source_frames = [
                np.full((height, width, 3), 40 + index * 5, np.uint8)
                for index in range(frame_count)
            ]
            background_frames = [
                np.full((height, width, 3), 60 + index * 5, np.uint8)
                for index in range(frame_count)
            ]
            _write_video(source, source_frames)
            _write_video(inpaint, background_frames)

            for directory_name in (
                "robot_rgb",
                "robot_masks",
                "robot_depth_mm",
            ):
                (g1_root / directory_name).mkdir(parents=True)
            for index in range(frame_count):
                robot = np.zeros((height, width, 3), np.uint8)
                robot[8:24, 18:30] = (180, 120, 80)
                mask = np.zeros((height, width), np.uint8)
                mask[8:24, 18:30] = 255
                depth = np.zeros((height, width), np.uint16)
                depth[8:24, 18:30] = 1000
                cv2.imwrite(
                    str(g1_root / "robot_rgb" / f"{index:05d}.png"),
                    robot,
                )
                cv2.imwrite(
                    str(g1_root / "robot_masks" / f"{index:05d}.png"),
                    mask,
                )
                cv2.imwrite(
                    str(
                        g1_root
                        / "robot_depth_mm"
                        / f"{index:05d}.png"
                    ),
                    depth,
                )
            trajectory_path.parent.mkdir()
            np.savez_compressed(trajectory_path, scale=0.84)

            hdf5_path = root / "episode.hdf5"
            transforms = np.repeat(
                np.eye(4, dtype=np.float64)[None],
                frame_count,
                axis=0,
            )
            left = transforms.copy()
            right = transforms.copy()
            left[:, :3, 3] = (-0.2, 0.0, 1.0)
            right[:, :3, 3] = (0.2, 0.0, 1.0)
            with h5py.File(hdf5_path, "w") as handle:
                handle.create_dataset(
                    "camera/intrinsic",
                    data=np.array(
                        [
                            [40.0, 0.0, width / 2],
                            [0.0, 40.0, height / 2],
                            [0.0, 0.0, 1.0],
                        ]
                    ),
                )
                handle.create_dataset("transforms/camera", data=transforms)
                handle.create_dataset("transforms/leftHand", data=left)
                handle.create_dataset("transforms/rightHand", data=right)
                handle.create_dataset(
                    "confidences/leftHand", data=np.ones(frame_count)
                )
                handle.create_dataset(
                    "confidences/rightHand", data=np.ones(frame_count)
                )

            report = compose_video(
                source_video=source,
                inpaint_video=inpaint,
                hdf5_path=hdf5_path,
                g1_root=g1_root,
                trajectory_path=trajectory_path,
                object_mask_root=None,
                output_root=output,
                delete_input_frames=True,
            )

            self.assertEqual(report["frame_count"], frame_count)
            self.assertEqual(report["trajectory"], str(trajectory_path))
            self.assertEqual(
                video_metadata(output / "synthetic_g1.mp4")[:3],
                (width, height, frame_count),
            )
            self.assertFalse(
                (output / "comparison_original_inpaint_g1.mp4").exists()
            )
            self.assertFalse((output / "synthetic_contact.jpg").exists())
            for directory_name in (
                "robot_rgb",
                "robot_masks",
                "robot_depth_mm",
            ):
                self.assertFalse((g1_root / directory_name).exists())


if __name__ == "__main__":
    unittest.main()
