from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from repaint.rgbd_isaac import (
    _resolve_d435_calibration,
    _resolve_interactive_command,
)
from repaint.rgbd_geometry import (
    backproject_z_depth,
    camera_cv_pose_to_usd_wxyz,
    center_patch_median_depth,
    depth_grid_triangles,
    image_plane_corners_camera_cv,
    minimum_image_plane_distance_for_foreground,
    scale_intrinsic,
    temporal_median_depth,
)
from repaint.robot_camera import (
    CAMERA_LINK_FROM_CV_OPTICAL,
    camera_transforms_to_poses_cv,
    camera_pose_delta_report,
    co_centered_source_camera_poses_cv,
    d435_camera_poses_cv,
    fit_co_centered_source_scale,
    rigid_source_scene_camera_poses_cv,
    robot_frame_points_in_camera_cv,
)
from repaint.trajectory_io import DEFAULT_G1_URDF


class RgbdGeometryTests(unittest.TestCase):
    def test_center_patch_depth_rejects_outlier(self) -> None:
        depth = np.full((9, 9), 0.42, dtype=np.float64)
        depth[4, 4] = 8.0
        self.assertAlmostEqual(
            center_patch_median_depth(
                depth,
                half_fraction=0.2,
            ),
            0.42,
        )

    def test_temporal_depth_filter_rejects_one_frame_spike(self) -> None:
        filtered = temporal_median_depth(
            np.asarray([0.4, 0.41, 1.5, 0.42, 0.43]),
            window=3,
        )
        np.testing.assert_allclose(
            filtered,
            [0.4, 0.41, 0.42, 0.43, 0.43],
        )

    def test_fixed_beingbeyond_intrinsic_scales_with_render_size(self) -> None:
        target, size, source = _resolve_d435_calibration(
            render_scale=0.5,
        )
        self.assertEqual(size, (320, 240))
        self.assertIn("fixed BeingBeyond", source)
        np.testing.assert_allclose(
            target,
            [
                [303.6060791015625, 0.0, 155.9999542236328],
                [0.0, 303.5042419433594, 130.5760498046875],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_d435_intrinsic_is_always_fixed_beingbeyond_k(self) -> None:
        target, size, source = _resolve_d435_calibration(
            render_scale=1.0,
        )
        self.assertEqual(size, (640, 480))
        self.assertIn("BeingBeyond", source)
        np.testing.assert_array_equal(
            target,
            [
                [607.212158203125, 0.0, 311.9999084472656],
                [0.0, 607.0084838867188, 261.152099609375],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_minimum_plane_distance_keeps_plane_behind_foreground(self) -> None:
        intrinsic = np.asarray(
            [
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        identity_pose = np.asarray(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        foreground = np.asarray([[[0.0, 0.0, 0.4]]])
        distance = minimum_image_plane_distance_for_foreground(
            intrinsic=intrinsic,
            image_size=(2, 2),
            plane_camera_pose_cv_xyzw=identity_pose,
            render_camera_pose_cv_xyzw=identity_pose,
            foreground_points_render_camera=foreground,
            clearance_m=0.05,
        )
        self.assertAlmostEqual(distance, 0.45)

    def test_image_plane_edges_reproject_to_image_edges(self) -> None:
        intrinsic = np.asarray(
            [
                [800.0, 0.0, 960.0],
                [0.0, 820.0, 540.0],
                [0.0, 0.0, 1.0],
            ]
        )
        corners = image_plane_corners_camera_cv(
            intrinsic,
            image_size=(1920, 1080),
            distance_m=0.7,
        )
        pixels = np.empty((4, 2), dtype=np.float64)
        pixels[:, 0] = (
            intrinsic[0, 0] * corners[:, 0] / corners[:, 2]
            + intrinsic[0, 2]
        )
        pixels[:, 1] = (
            intrinsic[1, 1] * corners[:, 1] / corners[:, 2]
            + intrinsic[1, 2]
        )
        np.testing.assert_allclose(
            pixels,
            [[0.0, 0.0], [1920.0, 0.0], [0.0, 1080.0], [1920.0, 1080.0]],
            atol=1e-10,
        )

    def test_rigid_source_scene_preserves_extrinsic_rotation(self) -> None:
        source_rotation = Rotation.from_euler(
            "xyz",
            [[10.0, -20.0, 30.0], [12.0, -18.0, 35.0]],
            degrees=True,
        )
        source_pose = np.concatenate(
            [
                np.asarray([[4.0, 5.0, 6.0], [4.2, 5.1, 6.3]]),
                source_rotation.as_quat(),
            ],
            axis=1,
        )
        source_to_robot = Rotation.from_euler(
            "z",
            90.0,
            degrees=True,
        ).as_matrix()
        robot_anchor = np.asarray(
            [0.5, -0.2, 1.3, 0.0, 0.0, 0.0, 1.0]
        )
        mapped = rigid_source_scene_camera_poses_cv(
            source_camera_pose_cv_xyzw=source_pose,
            source_to_robot_rotation=source_to_robot,
            robot_camera_anchor_pose_cv_xyzw=robot_anchor,
            scene_scale=0.8,
        )
        np.testing.assert_allclose(mapped[0, :3], robot_anchor[:3])
        np.testing.assert_allclose(
            mapped[1, :3],
            robot_anchor[:3]
            + 0.8
            * source_to_robot
            @ (source_pose[1, :3] - source_pose[0, :3]),
        )
        np.testing.assert_allclose(
            Rotation.from_quat(mapped[:, 3:7]).as_matrix(),
            np.einsum(
                "ij,njk->nik",
                source_to_robot,
                source_rotation.as_matrix(),
            ),
        )

    def test_co_centered_scene_uses_d435_centers_and_source_rotation(
        self,
    ) -> None:
        source_rotation = Rotation.from_euler(
            "xyz",
            [[5.0, 10.0, 15.0], [7.0, 12.0, 18.0]],
            degrees=True,
        )
        source_pose = np.concatenate(
            [
                np.asarray([[4.0, 5.0, 6.0], [4.2, 5.1, 6.3]]),
                source_rotation.as_quat(),
            ],
            axis=1,
        )
        robot_pose = np.asarray(
            [
                [0.2, -0.1, 1.4, 0.0, 0.0, 0.0, 1.0],
                [0.3, -0.1, 1.5, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        source_to_robot = Rotation.from_euler(
            "z",
            90.0,
            degrees=True,
        ).as_matrix()
        mapped = co_centered_source_camera_poses_cv(
            source_camera_pose_cv_xyzw=source_pose,
            source_to_robot_rotation=source_to_robot,
            robot_camera_pose_cv_xyzw=robot_pose,
        )
        np.testing.assert_allclose(mapped[:, :3], robot_pose[:, :3])
        np.testing.assert_allclose(
            Rotation.from_quat(mapped[:, 3:7]).as_matrix(),
            np.einsum(
                "ij,njk->nik",
                source_to_robot,
                source_rotation.as_matrix(),
            ),
        )

    def test_co_centered_scale_recovers_vector_scale(self) -> None:
        source_pose = np.asarray(
            [
                [2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0],
                [2.1, 3.2, 4.2, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        robot_pose = np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        source_hand = np.asarray(
            [
                [[0.1, 0.0, 0.3], [-0.1, 0.0, 0.4]],
                [[0.2, 0.1, 0.5], [-0.2, 0.1, 0.6]],
            ]
        )
        robot_hand = 1.7 * source_hand
        scale, report = fit_co_centered_source_scale(
            source_hand_points_camera_cv=source_hand,
            source_hand_confidence=np.ones((2, 2)),
            source_camera_pose_cv_xyzw=source_pose,
            source_to_robot_rotation=np.eye(3),
            robot_camera_pose_cv_xyzw=robot_pose,
            robot_hand_points_camera_cv=robot_hand,
        )
        self.assertAlmostEqual(scale, 1.7)
        self.assertAlmostEqual(report["rmse_m"], 0.0)

    def test_camera_transform_conversion(self) -> None:
        transforms = np.repeat(np.eye(4)[None], 2, axis=0)
        transforms[0, :3, 3] = [1.0, 2.0, 3.0]
        transforms[1, :3, :3] = Rotation.from_euler(
            "y",
            25.0,
            degrees=True,
        ).as_matrix()
        poses = camera_transforms_to_poses_cv(transforms)
        np.testing.assert_allclose(poses[0, :3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            Rotation.from_quat(poses[:, 3:7]).as_matrix(),
            transforms[:, :3, :3],
        )

    def test_interactive_commands_resolve_frames(self) -> None:
        frame_indices = np.asarray([10, 11, 12, 15])
        self.assertEqual(
            _resolve_interactive_command(
                "n",
                current_frame=1,
                frame_indices=frame_indices,
            ),
            (2, False, None),
        )
        self.assertEqual(
            _resolve_interactive_command(
                "p",
                current_frame=1,
                frame_indices=frame_indices,
            ),
            (0, False, None),
        )
        self.assertEqual(
            _resolve_interactive_command(
                "g 15",
                current_frame=1,
                frame_indices=frame_indices,
            ),
            (3, False, None),
        )
        self.assertEqual(
            _resolve_interactive_command(
                "q",
                current_frame=1,
                frame_indices=frame_indices,
            ),
            (1, True, None),
        )
        frame, should_quit, message = _resolve_interactive_command(
            "g 14",
            current_frame=1,
            frame_indices=frame_indices,
        )
        self.assertEqual(frame, 1)
        self.assertFalse(should_quit)
        self.assertIn("not in this prepared sequence", message)

    def test_scale_intrinsic_for_quarter_resolution(self) -> None:
        intrinsic = np.asarray(
            [
                [736.6339, 0.0, 960.0],
                [0.0, 736.6339, 540.0],
                [0.0, 0.0, 1.0],
            ]
        )
        scaled = scale_intrinsic(
            intrinsic,
            source_size=(1920, 1080),
            target_size=(480, 270),
        )
        np.testing.assert_allclose(
            scaled,
            [
                [184.158475, 0.0, 240.0],
                [0.0, 184.158475, 135.0],
                [0.0, 0.0, 1.0],
            ],
        )

    def test_backproject_center_pixel(self) -> None:
        depth = np.ones((3, 3), dtype=np.float64)
        intrinsic = np.asarray(
            [
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        points = backproject_z_depth(depth, intrinsic)
        np.testing.assert_allclose(points[1, 1], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(points[1, 2], [0.5, 0.0, 1.0])

    def test_depth_discontinuity_is_not_bridged(self) -> None:
        depth = np.ones((2, 3), dtype=np.float64)
        depth[:, 2] = 3.0
        triangles = depth_grid_triangles(
            depth,
            absolute_threshold_m=0.01,
            relative_threshold=0.01,
        )
        self.assertEqual(triangles.shape, (2, 3))
        self.assertTrue(np.all(triangles < 5))

    def test_cv_to_usd_camera_axes(self) -> None:
        pose = np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        position, quaternion_wxyz = camera_cv_pose_to_usd_wxyz(pose)
        np.testing.assert_allclose(position, pose[:3])
        rotation = Rotation.from_quat(
            quaternion_wxyz[[1, 2, 3, 0]]
        ).as_matrix()
        np.testing.assert_allclose(
            rotation,
            np.diag([1.0, -1.0, -1.0]),
            atol=1e-7,
        )

    def test_camera_link_to_cv_optical_axes(self) -> None:
        # CV right/down/forward become ROS-link -Y/-Z/+X.
        np.testing.assert_allclose(
            CAMERA_LINK_FROM_CV_OPTICAL,
            np.column_stack(
                [
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
        )
        np.testing.assert_allclose(
            CAMERA_LINK_FROM_CV_OPTICAL.T
            @ CAMERA_LINK_FROM_CV_OPTICAL,
            np.eye(3),
        )
        self.assertAlmostEqual(
            float(np.linalg.det(CAMERA_LINK_FROM_CV_OPTICAL)),
            1.0,
        )

    def test_d435_pose_uses_urdf_fk_root_and_optical_axes(self) -> None:
        import pinocchio as pin

        model = pin.buildModelFromUrdf(str(DEFAULT_G1_URDF.resolve()))
        joint_names = tuple(model.names[1:])
        joint_positions = np.zeros((1, model.nq), dtype=np.float64)
        root_rotation = Rotation.from_euler("z", 35.0, degrees=True)
        root_pose = np.concatenate(
            [
                np.asarray([[0.4, -0.2, 0.8]]),
                root_rotation.as_quat()[None],
            ],
            axis=1,
        )
        optical_offset = np.asarray([0.01, -0.02, 0.03])
        pose = d435_camera_poses_cv(
            urdf_path=DEFAULT_G1_URDF,
            joint_names=joint_names,
            joint_positions=joint_positions,
            root_pose_xyzw=root_pose,
            optical_center_in_link_xyz=optical_offset,
        )

        data = model.createData()
        pin.forwardKinematics(model, data, joint_positions[0])
        pin.updateFramePlacements(model, data)
        model_from_link = data.oMf[model.getFrameId("d435_link")]
        world_from_link_rotation = (
            root_rotation.as_matrix()
            @ np.asarray(model_from_link.rotation)
        )
        expected_position = (
            root_rotation.as_matrix()
            @ np.asarray(model_from_link.translation)
            + root_pose[0, :3]
            + world_from_link_rotation @ optical_offset
        )
        expected_rotation = (
            world_from_link_rotation @ CAMERA_LINK_FROM_CV_OPTICAL
        )
        np.testing.assert_allclose(pose[0, :3], expected_position)
        np.testing.assert_allclose(
            Rotation.from_quat(pose[0, 3:7]).as_matrix(),
            expected_rotation,
            atol=1e-12,
        )

    def test_camera_pose_delta_report(self) -> None:
        reference = np.asarray(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        actual = np.concatenate(
            [
                np.asarray([[1.0, 2.0, 3.0]]),
                Rotation.from_euler("z", 30.0, degrees=True).as_quat()[None],
            ],
            axis=1,
        )
        report = camera_pose_delta_report(reference, actual)
        np.testing.assert_allclose(
            report["translation_actual_minus_reference_world_m"]["median"],
            [1.0, 2.0, 3.0],
        )
        self.assertAlmostEqual(
            report["rotation_angle_degrees"]["median"],
            30.0,
        )


if __name__ == "__main__":
    unittest.main()
