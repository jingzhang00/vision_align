from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from human_to_robot.composite import (
    InMemoryVideoCompositor,
    _composite_frame,
    _robot_visibility_mask,
    compose_video,
)
from human_to_robot.retarget_g1_pinocchio import (
    CV_TO_G1_WORLD,
    G1_BASE_FRAME,
    NoCommonBasePlacementError,
    _base_candidates,
    _derive_spatial_alignment,
    _representative_keyframes,
    _robot_arm_lengths,
    _robot_maximum_reach,
    _robot_nominal_pelvis_height,
    _robot_shoulder_center,
    _shoulder_aligned_base_candidates,
    _solve_frame,
    _trajectory_centroid,
    _active_velocity_indices,
    build_pinocchio_g1,
    main as retarget_main,
)
from human_to_robot.trajectory import SourceBodyAlignment
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
                base_height=0.8,
            )
        )

        self.assertGreater(len(candidates), 0)
        for position, rotation in candidates:
            self.assertAlmostEqual(position[2], 0.8)
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

    def test_missing_common_base_range_has_a_distinct_skip_error(self) -> None:
        with self.assertRaises(NoCommonBasePlacementError):
            list(
                _base_candidates(
                    np.asarray(
                        [[0.0, 0.0, 0.8], [2.0, 0.0, 0.8]],
                        dtype=np.float64,
                    ),
                    trajectory_centroid=np.asarray(
                        [1.0, 0.0, 0.8],
                        dtype=np.float64,
                    ),
                    r_max=0.5,
                    base_height=0.8,
                )
            )

    def test_retarget_cli_writes_episode_skip_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "task" / "12.mp4"
            hdf5 = root / "task" / "12.hdf5"
            output = root / "output"
            trajectory = mock.Mock(
                frame_count=4,
                target_mode="rigid_palm",
            )
            argv = [
                "retarget_g1_pinocchio",
                "--video",
                str(video),
                "--hdf5",
                str(hdf5),
                "--urdf",
                str(root / "robot.urdf"),
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch(
                    "human_to_robot.retarget_g1_pinocchio.video_metadata",
                    return_value=(48, 32, 4, 12.0),
                ),
                mock.patch(
                    "human_to_robot.retarget_g1_pinocchio."
                    "load_hand_trajectory",
                    return_value=trajectory,
                ),
                mock.patch(
                    "human_to_robot.retarget_g1_pinocchio."
                    "build_pinocchio_g1",
                    return_value=mock.sentinel.model,
                ),
                mock.patch(
                    "human_to_robot.retarget_g1_pinocchio."
                    "retarget_with_pinocchio",
                    side_effect=NoCommonBasePlacementError(
                        "no common grounded placement"
                    ),
                ),
            ):
                retarget_main()

            report = json.loads(
                (output / "episode_skip.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "skipped")
            self.assertEqual(
                report["reason_code"],
                "no_common_base_search_region",
            )
            self.assertEqual(report["episode_path"], str(hdf5.resolve()))

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

    def test_g1_base_is_pelvis_and_waist_participates_in_ik(self) -> None:
        model = build_pinocchio_g1(
            REPOSITORY
            / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
        )

        self.assertEqual(G1_BASE_FRAME, "pelvis")
        self.assertLess(model.getFrameId(G1_BASE_FRAME), model.nframes)
        self.assertAlmostEqual(
            _robot_nominal_pelvis_height(model),
            0.793863752,
            places=8,
        )
        active = set(
            _active_velocity_indices(
                model,
                position_only=False,
            ).tolist()
        )
        for joint_name in (
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ):
            self.assertIn(
                model.idx_vs[model.getJointId(joint_name)],
                active,
            )
        for joint_name in (
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
        ):
            self.assertNotIn(
                model.idx_vs[model.getJointId(joint_name)],
                active,
            )

    def test_episode_body_alignment_derives_scale_and_camera_origin(
        self,
    ) -> None:
        model = build_pinocchio_g1(
            REPOSITORY
            / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
        )
        pelvis_samples = np.asarray(
            [
                [0.10, 0.60, 0.20],
                [0.12, 0.58, 0.21],
                [0.08, 0.62, 0.19],
            ],
            dtype=np.float64,
        )
        shoulder_samples = np.asarray(
            [
                [0.12, 0.18, 0.82],
                [0.14, 0.16, 0.83],
                [0.10, 0.20, 0.81],
            ],
            dtype=np.float64,
        )
        trajectory = type(
            "Trajectory",
            (),
            {
                "camera_to_world": np.repeat(
                    np.block(
                        [
                            [CV_TO_G1_WORLD, np.zeros((3, 1))],
                            [np.zeros((1, 3)), np.ones((1, 1))],
                        ]
                    )[None],
                    3,
                    axis=0,
                ),
                "body_alignment": SourceBodyAlignment(
                    pelvis_reference_camera=pelvis_samples,
                    shoulder_center_reference_camera=shoulder_samples,
                    arm_lengths=np.asarray([0.57, 0.57]),
                    world_up=np.asarray([0.0, 0.0, 1.0]),
                    source="synthetic_test",
                ),
            },
        )()
        pelvis_height = _robot_nominal_pelvis_height(model)

        alignment = _derive_spatial_alignment(
            model,
            trajectory,
            nominal_pelvis_height=pelvis_height,
            scale=None,
            camera_origin=None,
        )

        robot_lengths = _robot_arm_lengths(model)
        expected_scale = float(np.median(robot_lengths / 0.57))
        np.testing.assert_allclose(alignment.scale, expected_scale)
        np.testing.assert_allclose(
            alignment.world_from_reference_camera,
            CV_TO_G1_WORLD,
            atol=1e-12,
        )
        mapped_pelvis = alignment.camera_origin + (
            alignment.world_from_reference_camera
            @ (
                alignment.scale
                * np.median(pelvis_samples, axis=0)
            )
        )
        np.testing.assert_allclose(
            mapped_pelvis,
            [0.0, 0.0, pelvis_height],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            alignment.shoulder_anchor_reference_camera,
            np.median(shoulder_samples, axis=0),
        )
        np.testing.assert_allclose(
            alignment.robot_shoulder_center_from_pelvis,
            _robot_shoulder_center(model),
        )
        self.assertIn("synthetic_test", alignment.scale_source)

    def test_shoulder_alignment_derives_pelvis_for_each_yaw(self) -> None:
        source_shoulder = np.asarray([0.4, -0.2, 1.15])
        robot_shoulder = np.asarray([0.01, 0.0, 0.29])
        centroid = np.asarray([0.8, 0.1, 1.05])

        candidates = list(
            _shoulder_aligned_base_candidates(
                source_shoulder,
                robot_shoulder,
                trajectory_centroid=centroid,
            )
        )

        self.assertEqual(len(candidates), 9)
        for pelvis_position, pelvis_rotation in candidates:
            np.testing.assert_allclose(
                pelvis_position + pelvis_rotation @ robot_shoulder,
                source_shoulder,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                pelvis_rotation.T @ pelvis_rotation,
                np.eye(3),
                atol=1e-12,
            )

    def test_spatial_alignment_has_no_silent_scale_or_camera_fallback(
        self,
    ) -> None:
        model = build_pinocchio_g1(
            REPOSITORY
            / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
        )
        trajectory = type(
            "Trajectory",
            (),
            {"body_alignment": None},
        )()

        with self.assertRaisesRegex(ValueError, "upper-body anchors"):
            _derive_spatial_alignment(
                model,
                trajectory,
                nominal_pelvis_height=_robot_nominal_pelvis_height(model),
                scale=None,
                camera_origin=None,
            )


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
