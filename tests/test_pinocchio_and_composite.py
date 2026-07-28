from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from human_to_robot.composite import (
    InMemoryVideoCompositor,
    _composite_frame,
    _robot_visibility_mask,
    compose_video,
)
from human_to_robot.retarget_g1_pinocchio import (
    _base_candidates,
    _representative_keyframes,
    _robot_maximum_reach,
    _solve_frame,
    _trajectory_centroid,
    _active_velocity_indices,
    build_pinocchio_g1,
)
from human_to_robot.render_isaacsim import (
    _camera_projection_from_intrinsic,
)
import pinocchio as pin
from scipy.spatial.transform import Rotation
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


def _write_scene_depths(
    root: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    depth_mm: int = 2000,
) -> None:
    root.mkdir(parents=True)
    for frame_id in range(frame_count):
        cv2.imwrite(
            str(root / f"{frame_id:05d}.png"),
            np.full((height, width), depth_mm, dtype=np.uint16),
        )


class PinocchioRetargetingTests(unittest.TestCase):
    def test_isaac_projection_reconstructs_supported_intrinsic(self) -> None:
        width, height = 1920, 1080
        intrinsic = np.asarray(
            [
                [736.6339111328125, 0.0, width / 2],
                [0.0, 736.6339111328125, height / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        horizontal, vertical, focal = (
            _camera_projection_from_intrinsic(
                intrinsic,
                width=width,
                height=height,
            )
        )

        self.assertAlmostEqual(horizontal, 2.0955)
        self.assertAlmostEqual(width * focal / horizontal, intrinsic[0, 0])
        self.assertAlmostEqual(height * focal / vertical, intrinsic[1, 1])

    def test_isaac_projection_rejects_unsupported_principal_point(self) -> None:
        intrinsic = np.asarray(
            [[700.0, 0.0, 639.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]]
        )
        with self.assertRaisesRegex(ValueError, "centered principal point"):
            _camera_projection_from_intrinsic(
                intrinsic,
                width=1280,
                height=720,
            )

    def test_isaac_projection_rejects_skew(self) -> None:
        intrinsic = np.asarray(
            [[700.0, 0.1, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]]
        )
        with self.assertRaisesRegex(ValueError, "zero skew"):
            _camera_projection_from_intrinsic(
                intrinsic,
                width=1280,
                height=720,
            )

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

    def test_base_search_uses_spatial_extremes_and_full_trajectory_centroid(
        self,
    ) -> None:
        left = np.asarray(
            [
                [0.0, 4.0, 2.0],
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 1.0],
                [3.0, 1.0, 4.0],
            ],
            dtype=np.float64,
        )
        right = np.asarray(
            [
                [8.0, 2.0, 6.0],
                [6.0, 3.0, 8.0],
                [7.0, 4.0, 5.0],
                [5.0, 1.0, 7.0],
            ],
            dtype=np.float64,
        )
        left_tracked = np.asarray([True, True, True, False])
        right_tracked = np.asarray([False, True, True, True])

        keyframes = _representative_keyframes(
            left,
            right,
            left_tracked,
            right_tracked,
        )
        centroid = _trajectory_centroid(
            left,
            right,
            left_tracked,
            right_tracked,
        )

        np.testing.assert_array_equal(keyframes, [0, 1, 2, 3])
        expected_centers = np.asarray(
            [
                left[0],
                0.5 * (left[1] + right[1]),
                0.5 * (left[2] + right[2]),
                right[3],
            ]
        )
        np.testing.assert_allclose(centroid, expected_centers.mean(axis=0))

    def test_base_grid_is_trajectory_centered_and_reach_constrained(
        self,
    ) -> None:
        keyframe_positions = np.asarray(
            [
                [0.3, -0.1, 1.0],
                [0.5, 0.1, 1.2],
            ],
            dtype=np.float64,
        )
        centroid = np.asarray([0.4, 0.0, 1.1], dtype=np.float64)
        candidates = list(
            _base_candidates(
                keyframe_positions,
                trajectory_centroid=centroid,
                r_max=0.7,
            )
        )

        self.assertGreater(len(candidates), 0)
        for position, rotation in candidates:
            self.assertTrue(
                np.all(
                    np.linalg.norm(
                        keyframe_positions - position,
                        axis=1,
                    )
                    <= 0.7 + 1e-9
                )
            )
            np.testing.assert_allclose(
                rotation.T @ rotation,
                np.eye(3),
                atol=1e-12,
            )

    def test_model_driven_g1_reach_and_rotated_base_transform(self) -> None:
        model = build_pinocchio_g1(
            REPOSITORY
            / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
        )
        self.assertAlmostEqual(_robot_maximum_reach(model), 0.801882, places=5)

        data = model.createData()
        reference_q = np.clip(
            pin.neutral(model),
            model.lowerPositionLimit + 0.02,
            model.upperPositionLimit - 0.02,
        )
        pin.forwardKinematics(model, data, reference_q)
        pin.updateFramePlacements(model, data)
        left = data.oMf[model.getFrameId("left_palm")]
        right = data.oMf[model.getFrameId("right_palm")]
        base_position = np.asarray([0.2, -0.3, 0.8])
        base_rotation = Rotation.from_euler("z", 0.6).as_matrix()
        solved, metrics = _solve_frame(
            model,
            data,
            reference_q.copy(),
            base_position=base_position,
            base_rotation=base_rotation,
            left_target=base_position
            + base_rotation @ left.translation,
            right_target=base_position
            + base_rotation @ right.translation,
            left_rotation=base_rotation @ left.rotation,
            right_rotation=base_rotation @ right.rotation,
            left_active=True,
            right_active=True,
            active_indices=_active_velocity_indices(
                model,
                position_only=False,
            ),
            iterations=5,
        )

        np.testing.assert_allclose(solved, reference_q, atol=1e-10)
        np.testing.assert_allclose(metrics, 0.0, atol=1e-10)


class StreamingCompositeTests(unittest.TestCase):
    def test_depth_visibility_matches_paper_inequality(self) -> None:
        robot_mask = np.full((2, 4), 255, dtype=np.uint8)
        robot_depth_mm = np.asarray(
            [[800, 800, 800, 0], [800, 800, 800, 800]],
            dtype=np.uint16,
        )
        scene_depth_mm = np.asarray(
            [[1200, 600, 800, 1200], [0, 0, 0, 900]],
            dtype=np.uint16,
        )

        visible, valid_scene = _robot_visibility_mask(
            robot_mask=robot_mask,
            robot_depth_mm=robot_depth_mm,
            scene_depth_mm=scene_depth_mm,
        )

        np.testing.assert_array_equal(
            visible,
            [
                [True, False, True, False],
                [False, False, False, True],
            ],
        )
        np.testing.assert_array_equal(
            valid_scene,
            [
                [True, True, True, True],
                [False, False, False, True],
            ],
        )

    def test_occluded_robot_keeps_clean_inpaint_background(self) -> None:
        background = np.full((2, 3, 3), (0, 200, 0), dtype=np.uint8)
        robot = np.full((2, 3, 3), (200, 0, 0), dtype=np.uint8)
        mask = np.full((2, 3), 255, dtype=np.uint8)
        robot_depth = np.full((2, 3), 0.8, dtype=np.float32)
        scene_depth_mm = np.asarray(
            [[1200, 600, 800], [0, 900, 700]],
            dtype=np.uint16,
        )

        synthetic, visible_fraction, occluded_fraction, valid_fraction = (
            _composite_frame(
                background=background,
                robot=robot,
                robot_mask=mask,
                robot_depth=robot_depth,
                scene_depth_mm=scene_depth_mm,
                robot_exposure=0.0,
                robot_rgb_gain=(1.0, 1.0, 1.0),
                robot_contrast=1.0,
                robot_saturation=1.0,
            )
        )

        visible = np.asarray(
            [[True, False, True], [False, True, False]]
        )
        np.testing.assert_array_equal(
            synthetic[visible],
            robot[visible],
        )
        np.testing.assert_array_equal(
            synthetic[~visible],
            background[~visible],
        )
        self.assertAlmostEqual(visible_fraction, 0.5)
        self.assertAlmostEqual(occluded_fraction, 0.5)
        self.assertAlmostEqual(valid_fraction, 5 / 6)

    def test_in_memory_compositor_writes_final_video_without_png_frames(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            inpaint = root / "inpaint.mp4"
            output = root / "output"
            render_root = root / "render"
            scene_depths = root / "scene_depths"
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
            _write_scene_depths(
                scene_depths,
                frame_count=frame_count,
                width=width,
                height=height,
            )

            compositor = InMemoryVideoCompositor(
                source_video=source,
                inpaint_video=inpaint,
                trajectory_path=trajectory_path,
                scene_depth_root=scene_depths,
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
            scene_depths = root / "scene_depths"
            frame = np.full((32, 48, 3), 40, np.uint8)
            _write_video(source, [frame, frame])
            _write_video(inpaint, [frame, frame])
            np.savez_compressed(trajectory_path, scale=0.84)
            _write_scene_depths(
                scene_depths,
                frame_count=2,
                width=48,
                height=32,
            )
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
                    scene_depth_root=scene_depths,
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
            scene_depths = root / "scene_depths"
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
            _write_scene_depths(
                scene_depths,
                frame_count=frame_count,
                width=width,
                height=height,
            )

            report = compose_video(
                source_video=source,
                inpaint_video=inpaint,
                g1_root=g1_root,
                trajectory_path=trajectory_path,
                scene_depth_root=scene_depths,
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
