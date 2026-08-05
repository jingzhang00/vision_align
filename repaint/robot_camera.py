from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


# ROS camera links use X forward, Y left, Z up. OpenCV optical frames use
# X right, Y down, Z forward. The columns below are the OpenCV axes expressed
# in the ROS camera-link frame.
CAMERA_LINK_FROM_CV_OPTICAL = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

HEAD_CAMERA_PARENT_FRAME = "torso_link"
TORSO_ANCHORED_CAMERA_MAPPING = (
    "g1_torso_pose_anchor_with_source_spine3_camera_delta"
)
HEAD_CAM_OFFSET_POS_BASE = np.asarray(
    [0.0576235, 0.01753, 0.42987],
    dtype=np.float64,
)
HEAD_CAM_OFFSET_ROT_WXYZ = np.asarray(
    [0.91495966, 0.0, 0.4035453, 0.0],
    dtype=np.float64,
)
HEAD_CAM_RESET_GUARD_X_M = 0.005
HEAD_CAM_OFFSET_POS_ACTIVE = (
    HEAD_CAM_OFFSET_POS_BASE
    + np.asarray(
        [HEAD_CAM_RESET_GUARD_X_M, 0.0, 0.0],
        dtype=np.float64,
    )
)

def _continuous_xyzw(quaternions: np.ndarray) -> np.ndarray:
    """Return an equivalent quaternion sequence without sign flips."""

    continuous = np.asarray(quaternions, dtype=np.float64).copy()
    for frame_id in range(1, len(continuous)):
        if np.dot(continuous[frame_id - 1], continuous[frame_id]) < 0.0:
            continuous[frame_id] *= -1.0
    return continuous


def robot_frame_poses_world(
    *,
    urdf_path: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
    frame_name: str,
) -> np.ndarray:
    """Return one URDF frame pose in robot world for every trajectory frame."""

    urdf_path = urdf_path.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    joints = np.asarray(joint_positions, dtype=np.float64)
    roots = np.asarray(root_pose_xyzw, dtype=np.float64)
    if joints.ndim != 2:
        raise ValueError("joint_positions must have shape [frames, joints]")
    frame_count = len(joints)
    if roots.shape != (frame_count, 7):
        raise ValueError("root_pose_xyzw must have shape [frames, 7]")
    if not np.isfinite(joints).all() or not np.isfinite(roots).all():
        raise ValueError("joint_positions and root_pose_xyzw must be finite")
    quaternion_norms = np.linalg.norm(roots[:, 3:7], axis=1)
    if np.any(quaternion_norms <= 1e-8):
        raise ValueError("root_pose_xyzw contains a zero quaternion")

    model = pin.buildModelFromUrdf(str(urdf_path))
    if tuple(model.names[1:]) != joint_names:
        raise ValueError(
            "Prepared joint names do not match the G1 URDF joint order"
        )
    if joints.shape[1] != model.nq:
        raise ValueError(
            f"Expected {model.nq} joint positions, got {joints.shape[1]}"
        )
    robot_frame_id = model.getFrameId(frame_name)
    if robot_frame_id >= model.nframes:
        raise KeyError(f"G1 URDF has no frame named {frame_name!r}")

    root_rotations = Rotation.from_quat(
        roots[:, 3:7] / quaternion_norms[:, None]
    ).as_matrix()
    poses = np.empty((frame_count, 7), dtype=np.float64)
    data = model.createData()
    for frame_id in range(frame_count):
        pin.forwardKinematics(model, data, joints[frame_id])
        pin.updateFramePlacements(model, data)
        model_from_frame = data.oMf[robot_frame_id]
        world_from_frame_rotation = (
            root_rotations[frame_id]
            @ np.asarray(model_from_frame.rotation, dtype=np.float64)
        )
        poses[frame_id, :3] = (
            root_rotations[frame_id]
            @ np.asarray(model_from_frame.translation, dtype=np.float64)
            + roots[frame_id, :3]
        )
        poses[frame_id, 3:7] = Rotation.from_matrix(
            world_from_frame_rotation
        ).as_quat()
    poses[:, 3:7] = _continuous_xyzw(poses[:, 3:7])
    return poses


def source_camera_poses_in_spine3_cv(
    *,
    source_camera_to_world: np.ndarray,
    source_spine3_to_world: np.ndarray,
) -> np.ndarray:
    """Express each source CV camera pose in its source spine3 frame."""

    camera = np.asarray(source_camera_to_world, dtype=np.float64)
    spine3 = np.asarray(source_spine3_to_world, dtype=np.float64)
    if (
        camera.ndim != 3
        or camera.shape[1:] != (4, 4)
        or spine3.shape != camera.shape
        or not np.isfinite(camera).all()
        or not np.isfinite(spine3).all()
        or not np.allclose(camera[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-5)
        or not np.allclose(spine3[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-5)
    ):
        raise ValueError("Source camera and spine3 transforms are invalid")
    relative = np.einsum("nij,njk->nik", np.linalg.inv(spine3), camera)
    poses = np.concatenate(
        [
            relative[:, :3, 3],
            Rotation.from_matrix(relative[:, :3, :3]).as_quat(),
        ],
        axis=1,
    )
    poses[:, 3:7] = _continuous_xyzw(poses[:, 3:7])
    return poses


def torso_anchored_camera_poses_cv(
    *,
    source_camera_pose_in_spine3_xyzw: np.ndarray,
    robot_torso_pose_world_xyzw: np.ndarray,
    source_spine3_to_robot_torso_rotation: np.ndarray,
    scene_scale: float,
) -> np.ndarray:
    """Apply source spine3-to-camera motion to the retargeted G1 torso."""

    camera = np.asarray(source_camera_pose_in_spine3_xyzw, dtype=np.float64)
    torso = np.asarray(robot_torso_pose_world_xyzw, dtype=np.float64)
    basis = np.asarray(
        source_spine3_to_robot_torso_rotation,
        dtype=np.float64,
    )
    frame_count = len(camera)
    if (
        camera.shape != (frame_count, 7)
        or torso.shape != (frame_count, 7)
        or basis.shape != (3, 3)
        or not np.isfinite(camera).all()
        or not np.isfinite(torso).all()
        or not np.isfinite(basis).all()
        or np.any(np.linalg.norm(camera[:, 3:7], axis=1) <= 1e-8)
        or np.any(np.linalg.norm(torso[:, 3:7], axis=1) <= 1e-8)
        or not np.isfinite(scene_scale)
        or scene_scale <= 0.0
        or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-5)
        or np.linalg.det(basis) < 0.999
    ):
        raise ValueError("Torso-anchored camera inputs are invalid")
    camera_rotations = Rotation.from_quat(camera[:, 3:7]).as_matrix()
    torso_rotations = Rotation.from_quat(torso[:, 3:7]).as_matrix()
    torso_from_source_spine3 = np.einsum(
        "nij,jk->nik",
        torso_rotations,
        basis,
    )
    positions = torso[:, :3] + scene_scale * np.einsum(
        "nij,nj->ni",
        torso_from_source_spine3,
        camera[:, :3],
    )
    rotations = np.einsum(
        "nij,njk->nik",
        torso_from_source_spine3,
        camera_rotations,
    )
    quaternions = _continuous_xyzw(
        Rotation.from_matrix(rotations).as_quat()
    )
    return np.concatenate([positions, quaternions], axis=1)


def head_camera_poses_world(
    *,
    urdf_path: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
) -> np.ndarray:
    """
    The returned XYZ + XYZW pose describes the camera's ``world`` convention:
    local +X points forward and local +Z points up.  It is therefore suitable
    for ``Camera.set_world_pose(..., camera_axes="world")`` without another
    camera-axis conversion.
    """

    parent_poses = robot_frame_poses_world(
        urdf_path=urdf_path,
        joint_names=joint_names,
        joint_positions=joint_positions,
        root_pose_xyzw=root_pose_xyzw,
        frame_name=HEAD_CAMERA_PARENT_FRAME,
    )
    parent_rotations = Rotation.from_quat(
        parent_poses[:, 3:7]
    ).as_matrix()
    camera_from_parent_rotation = Rotation.from_quat(
        HEAD_CAM_OFFSET_ROT_WXYZ[[1, 2, 3, 0]]
    ).as_matrix()
    poses = parent_poses.copy()
    poses[:, :3] += np.einsum(
        "nij,j->ni", parent_rotations, HEAD_CAM_OFFSET_POS_ACTIVE
    )
    poses[:, 3:7] = Rotation.from_matrix(
        np.einsum(
            "nij,jk->nik",
            parent_rotations,
            camera_from_parent_rotation,
        )
    ).as_quat()
    poses[:, 3:7] = _continuous_xyzw(poses[:, 3:7])
    return poses


def robot_frame_points_in_camera_cv(
    *,
    urdf_path: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
    camera_pose_cv_xyzw: np.ndarray,
    frame_names: tuple[str, ...],
    frame_local_offsets_m: np.ndarray | None = None,
) -> np.ndarray:
    """Return selected robot frame origins in each CV camera frame."""

    urdf_path = urdf_path.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    root_pose_xyzw = np.asarray(root_pose_xyzw, dtype=np.float64)
    camera_pose = np.asarray(camera_pose_cv_xyzw, dtype=np.float64)
    if joint_positions.ndim != 2:
        raise ValueError("joint_positions must have shape [frames, joints]")
    frame_count = len(joint_positions)
    if root_pose_xyzw.shape != (frame_count, 7):
        raise ValueError("root_pose_xyzw must have shape [frames, 7]")
    if camera_pose.shape != (frame_count, 7):
        raise ValueError("camera_pose_cv_xyzw must have shape [frames, 7]")
    if not frame_names:
        raise ValueError("frame_names must not be empty")
    if frame_local_offsets_m is None:
        local_offsets = np.zeros(
            (len(frame_names), 3),
            dtype=np.float64,
        )
    else:
        local_offsets = np.asarray(
            frame_local_offsets_m,
            dtype=np.float64,
        )
        if (
            local_offsets.shape != (len(frame_names), 3)
            or not np.isfinite(local_offsets).all()
        ):
            raise ValueError(
                "frame_local_offsets_m must be finite [frame_count,3]"
            )
    if (
        not np.isfinite(joint_positions).all()
        or not np.isfinite(root_pose_xyzw).all()
        or not np.isfinite(camera_pose).all()
    ):
        raise ValueError("trajectory and camera poses must be finite")

    model = pin.buildModelFromUrdf(str(urdf_path))
    if tuple(model.names[1:]) != joint_names:
        raise ValueError(
            "Prepared joint names do not match the G1 URDF joint order"
        )
    if joint_positions.shape[1] != model.nq:
        raise ValueError(
            f"Expected {model.nq} joint positions, got "
            f"{joint_positions.shape[1]}"
        )
    robot_frame_ids = [model.getFrameId(name) for name in frame_names]
    missing = [
        name
        for name, frame_id in zip(
            frame_names,
            robot_frame_ids,
            strict=True,
        )
        if frame_id >= model.nframes
    ]
    if missing:
        raise KeyError(f"G1 URDF is missing frames: {missing}")

    root_rotation = Rotation.from_quat(
        root_pose_xyzw[:, 3:7]
    ).as_matrix()
    camera_rotation = Rotation.from_quat(
        camera_pose[:, 3:7]
    ).as_matrix()
    points_camera = np.empty(
        (frame_count, len(robot_frame_ids), 3),
        dtype=np.float64,
    )
    data = model.createData()
    for frame_id in range(frame_count):
        pin.forwardKinematics(model, data, joint_positions[frame_id])
        pin.updateFramePlacements(model, data)
        for point_id, robot_frame_id in enumerate(robot_frame_ids):
            model_from_frame = data.oMf[robot_frame_id]
            point_world = (
                root_rotation[frame_id]
                @ (
                    np.asarray(
                        model_from_frame.rotation,
                        dtype=np.float64,
                    )
                    @ local_offsets[point_id]
                    + np.asarray(
                        model_from_frame.translation,
                        dtype=np.float64,
                    )
                )
                + root_pose_xyzw[frame_id, :3]
            )
            points_camera[frame_id, point_id] = (
                camera_rotation[frame_id].T
                @ (point_world - camera_pose[frame_id, :3])
            )
    return points_camera
