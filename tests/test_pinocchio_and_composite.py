from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from human_to_robot.composite import (
    InMemoryVideoCompositor,
    compose_video,
)
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
            / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
        )

        self.assertEqual(model.nq, 51)
        self.assertEqual(model.nv, 51)
        self.assertEqual(len(model.names[1:]), 51)
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
    def test_in_memory_compositor_writes_final_video_without_png_frames(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            inpaint = root / "inpaint.mp4"
            output = root / "output"
            render_root = root / "render"
            trajectory_path = root / "g1_trajectory.npz"
            frame_count = 4
            width, height = 48, 32
            _write_video(
                source,
                [
                    np.full(
                        (height, width, 3),
                        40 + index * 5,
                        np.uint8,
                    )
                    for index in range(frame_count)
                ],
            )
            _write_video(
                inpaint,
                [
                    np.full(
                        (height, width, 3),
                        60 + index * 5,
                        np.uint8,
                    )
                    for index in range(frame_count)
                ],
            )
            np.savez_compressed(trajectory_path, scale=0.84)

            compositor = InMemoryVideoCompositor(
                source_video=source,
                inpaint_video=inpaint,
                trajectory_path=trajectory_path,
                object_mask_root=None,
                output_root=output,
                g1_root=render_root,
                hardware_encoding=False,
            )
            with compositor:
                for _ in range(frame_count):
                    robot = np.zeros((height, width, 3), np.uint8)
                    robot[8:24, 18:30] = (180, 120, 80)
                    mask = np.zeros((height, width), np.uint8)
                    mask[8:24, 18:30] = 255
                    depth = np.full(
                        (height, width),
                        np.inf,
                        dtype=np.float32,
                    )
                    depth[8:24, 18:30] = 1.0
                    compositor.write(robot, mask, depth)

            self.assertTrue(compositor.report["streaming_in_memory"])
            self.assertEqual(
                video_metadata(output / "synthetic_g1.mp4")[:3],
                (width, height, frame_count),
            )
            self.assertTrue((output / "composite_report.json").is_file())
            for directory_name in (
                "robot_rgb",
                "robot_masks",
                "robot_depth_mm",
            ):
                self.assertFalse((render_root / directory_name).exists())

    def test_in_memory_compositor_does_not_publish_partial_video(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            inpaint = root / "inpaint.mp4"
            output = root / "output"
            trajectory_path = root / "g1_trajectory.npz"
            frame = np.full((32, 48, 3), 40, np.uint8)
            _write_video(source, [frame, frame])
            _write_video(inpaint, [frame, frame])
            np.savez_compressed(trajectory_path, scale=0.84)
            robot = np.zeros_like(frame)
            mask = np.zeros(frame.shape[:2], np.uint8)
            depth = np.full(frame.shape[:2], np.inf, np.float32)

            with self.assertRaisesRegex(
                RuntimeError,
                "received 1/2 frames",
            ):
                with InMemoryVideoCompositor(
                    source_video=source,
                    inpaint_video=inpaint,
                    trajectory_path=trajectory_path,
                    object_mask_root=None,
                    output_root=output,
                    g1_root=root / "render",
                    hardware_encoding=False,
                ) as compositor:
                    compositor.write(robot, mask, depth)

            self.assertFalse((output / "synthetic_g1.mp4").exists())
            self.assertFalse((output / "composite_report.json").exists())

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

            report = compose_video(
                source_video=source,
                inpaint_video=inpaint,
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
