"""Exact BeingBeyond G1 head-camera and source/G1 hand-anchor helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from human_to_robot.retarget_g1_pinocchio import (
    HAND_TIP_LOCAL_OFFSETS,
    PALM_OFFSET_FROM_WRIST,
)


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

# Exact active head-camera profile in BeingBeyond's
# ``G1_with_hand_head_CylinderEnv.py``.  The CameraCfg is parented to
# ``torso_link`` because the fixed d435 link is merged during URDF import.
# Its zero-noise reset still triggers the occlusion reject guard, which moves
# the camera forward by exactly 5 mm before ``set_world_poses(...,
# convention="world")`` is called.
BEINGBEYOND_HEAD_CAMERA_PROFILE = (
    "beingbeyond_torso_head_camera_exact_v1"
)
BEINGBEYOND_HEAD_CAMERA_PARENT_FRAME = "torso_link"
BEINGBEYOND_HEAD_CAM_OFFSET_POS_BASE = np.asarray(
    [0.0576235, 0.01753, 0.42987],
    dtype=np.float64,
)
BEINGBEYOND_HEAD_CAM_OFFSET_ROT_WXYZ = np.asarray(
    [0.91495966, 0.0, 0.4035453, 0.0],
    dtype=np.float64,
)
BEINGBEYOND_HEAD_CAM_RESET_GUARD_X_M = 0.005
BEINGBEYOND_HEAD_CAM_OFFSET_POS_ACTIVE = (
    BEINGBEYOND_HEAD_CAM_OFFSET_POS_BASE
    + np.asarray(
        [BEINGBEYOND_HEAD_CAM_RESET_GUARD_X_M, 0.0, 0.0],
        dtype=np.float64,
    )
)

# Exact semantic TCP used by the canonical EgoDex-to-G1 retargeter:
# 0.5*thumb + 0.35*index + 0.15*middle.  Source confidence uses the same
# contribution weights, so a low-confidence middle tip does not veto a TCP
# whose position is dominated by reliable thumb/index observations.
HAND_TCP_DIGITS = ("thumb", "index", "middle")
HAND_TCP_POSITION_WEIGHTS = np.asarray(
    [0.5, 0.35, 0.15],
    dtype=np.float64,
)
SOURCE_HAND_TCP_TRANSFORMS = tuple(
    f"{side}{source_digit}Tip"
    for side in ("left", "right")
    for source_digit in ("Thumb", "IndexFinger", "MiddleFinger")
)
ROBOT_HAND_TCP_FRAMES = tuple(
    f"{side}_{digit}_distal"
    for side in ("left", "right")
    for digit in HAND_TCP_DIGITS
)
ROBOT_HAND_TCP_OFFSETS_LOCAL_M = np.asarray(
    [
        HAND_TIP_LOCAL_OFFSETS[digit]
        for _side in ("left", "right")
        for digit in HAND_TCP_DIGITS
    ],
    dtype=np.float64,
)
SOURCE_PALM_TRANSFORMS = ("leftHand", "rightHand")
ROBOT_WRIST_FRAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
ROBOT_PALM_OFFSETS_LOCAL_M = np.repeat(
    PALM_OFFSET_FROM_WRIST[None],
    len(ROBOT_WRIST_FRAMES),
    axis=0,
)


def _continuous_xyzw(quaternions: np.ndarray) -> np.ndarray:
    """Return an equivalent quaternion sequence without sign flips."""

    continuous = np.asarray(quaternions, dtype=np.float64).copy()
    for frame_id in range(1, len(continuous)):
        if np.dot(continuous[frame_id - 1], continuous[frame_id]) < 0.0:
            continuous[frame_id] *= -1.0
    return continuous


def beingbeyond_head_camera_poses_world(
    *,
    urdf_path: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
) -> np.ndarray:
    """Return exact BeingBeyond head-camera poses in Isaac world axes.

    The returned XYZ + XYZW pose describes the camera's ``world`` convention:
    local +X points forward and local +Z points up.  It is therefore suitable
    for ``Camera.set_world_pose(..., camera_axes="world")`` without another
    camera-axis conversion.
    """

    urdf_path = urdf_path.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    root_pose_xyzw = np.asarray(root_pose_xyzw, dtype=np.float64)
    if joint_positions.ndim != 2:
        raise ValueError("joint_positions must have shape [frames, joints]")
    frame_count = joint_positions.shape[0]
    if root_pose_xyzw.shape != (frame_count, 7):
        raise ValueError("root_pose_xyzw must have shape [frames, 7]")
    if (
        not np.isfinite(joint_positions).all()
        or not np.isfinite(root_pose_xyzw).all()
    ):
        raise ValueError("joint_positions and root_pose_xyzw must be finite")
    quaternion_norm = np.linalg.norm(root_pose_xyzw[:, 3:7], axis=1)
    if np.any(quaternion_norm <= 1e-8):
        raise ValueError("root_pose_xyzw contains a zero quaternion")

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
    parent_frame_id = model.getFrameId(
        BEINGBEYOND_HEAD_CAMERA_PARENT_FRAME
    )
    if parent_frame_id >= model.nframes:
        raise KeyError(
            "G1 URDF has no frame named "
            f"{BEINGBEYOND_HEAD_CAMERA_PARENT_FRAME!r}"
        )

    root_rotations = Rotation.from_quat(
        root_pose_xyzw[:, 3:7] / quaternion_norm[:, None]
    ).as_matrix()
    camera_from_parent_rotation = Rotation.from_quat(
        BEINGBEYOND_HEAD_CAM_OFFSET_ROT_WXYZ[[1, 2, 3, 0]]
    ).as_matrix()
    poses = np.empty((frame_count, 7), dtype=np.float64)
    data = model.createData()
    for frame_id in range(frame_count):
        pin.forwardKinematics(model, data, joint_positions[frame_id])
        pin.updateFramePlacements(model, data)
        model_from_parent = data.oMf[parent_frame_id]
        world_from_parent_rotation = (
            root_rotations[frame_id]
            @ np.asarray(model_from_parent.rotation, dtype=np.float64)
        )
        world_from_parent_translation = (
            root_rotations[frame_id]
            @ np.asarray(model_from_parent.translation, dtype=np.float64)
            + root_pose_xyzw[frame_id, :3]
        )
        world_from_camera_rotation = (
            world_from_parent_rotation @ camera_from_parent_rotation
        )
        poses[frame_id, :3] = (
            world_from_parent_translation
            + world_from_parent_rotation
            @ BEINGBEYOND_HEAD_CAM_OFFSET_POS_ACTIVE
        )
        poses[frame_id, 3:7] = Rotation.from_matrix(
            world_from_camera_rotation
        ).as_quat()
    poses[:, 3:7] = _continuous_xyzw(poses[:, 3:7])
    return poses


def beingbeyond_head_camera_poses_cv(
    *,
    urdf_path: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
) -> np.ndarray:
    """Return the same exact camera poses with OpenCV optical axes.

    This companion representation is for projection and hand-anchor
    diagnostics only.  Isaac rendering uses the world-axis poses directly.
    """

    world_poses = beingbeyond_head_camera_poses_world(
        urdf_path=urdf_path,
        joint_names=joint_names,
        joint_positions=joint_positions,
        root_pose_xyzw=root_pose_xyzw,
    )
    world_rotations = Rotation.from_quat(
        world_poses[:, 3:7]
    ).as_matrix()
    cv_poses = world_poses.copy()
    cv_poses[:, 3:7] = Rotation.from_matrix(
        world_rotations @ CAMERA_LINK_FROM_CV_OPTICAL
    ).as_quat()
    cv_poses[:, 3:7] = _continuous_xyzw(cv_poses[:, 3:7])
    return cv_poses


def load_source_hand_tcp_points_camera_cv(
    source_hdf5: Path,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the bilateral semantic TCP used by retarget arm IK.

    For each hand ``virtual_finger = 0.7*index + 0.3*middle`` and
    ``TCP = 0.5*(thumb + virtual_finger)``.
    """

    source_hdf5 = source_hdf5.resolve()
    if not source_hdf5.is_file():
        raise FileNotFoundError(source_hdf5)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if frame_indices.ndim != 1 or len(frame_indices) == 0:
        raise ValueError("frame_indices must be a non-empty vector")
    if np.any(frame_indices < 0):
        raise ValueError("frame_indices must be non-negative")

    with h5py.File(source_hdf5, "r") as handle:
        camera = np.asarray(
            handle["transforms/camera"],
            dtype=np.float64,
        )
        if camera.ndim != 3 or camera.shape[1:] != (4, 4):
            raise ValueError(
                "transforms/camera must have shape [frames,4,4]"
            )
        fingertip_values: list[np.ndarray] = []
        confidence_values: list[np.ndarray] = []
        for name in SOURCE_HAND_TCP_TRANSFORMS:
            key = f"transforms/{name}"
            transforms = np.asarray(handle[key], dtype=np.float64)
            if transforms.shape != camera.shape:
                raise ValueError(
                    f"{key} must have shape {camera.shape}, "
                    f"got {transforms.shape}"
                )
            fingertip_values.append(transforms[:, :3, 3])
            confidence_key = f"confidences/{name}"
            if confidence_key in handle:
                confidence = np.asarray(
                    handle[confidence_key],
                    dtype=np.float64,
                )
                if confidence.shape != (len(camera),):
                    raise ValueError(
                        f"{confidence_key} must have shape {(len(camera),)}, "
                        f"got {confidence.shape}"
                    )
            else:
                confidence = np.ones(len(camera), dtype=np.float64)
            confidence_values.append(confidence)

    if int(frame_indices.max()) >= len(camera):
        raise IndexError("source hand-TCP frame index is out of range")
    fingertips_world = np.stack(
        fingertip_values,
        axis=1,
    ).reshape(len(camera), 2, 3, 3)[frame_indices]
    fingertip_confidence = np.stack(
        confidence_values,
        axis=1,
    ).reshape(len(camera), 2, 3)[frame_indices]
    camera = camera[frame_indices]
    relative_world = (
        fingertips_world - camera[:, None, None, :3, 3]
    )
    fingertips_camera = np.einsum(
        "nji,nsdj->nsdi",
        camera[:, :3, :3],
        relative_world,
    )
    tcp = np.einsum(
        "d,nsdj->nsj",
        HAND_TCP_POSITION_WEIGHTS,
        fingertips_camera,
    )
    tcp_confidence = np.einsum(
        "d,nsd->ns",
        HAND_TCP_POSITION_WEIGHTS,
        fingertip_confidence,
    )
    return tcp, tcp_confidence


def load_source_palm_points_camera_cv(
    source_hdf5: Path,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Return ``leftHand/rightHand`` under the rigid-palm action contract."""

    source_hdf5 = source_hdf5.resolve()
    if not source_hdf5.is_file():
        raise FileNotFoundError(source_hdf5)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if frame_indices.ndim != 1 or len(frame_indices) == 0:
        raise ValueError("frame_indices must be a non-empty vector")
    if np.any(frame_indices < 0):
        raise ValueError("frame_indices must be non-negative")

    with h5py.File(source_hdf5, "r") as handle:
        camera = np.asarray(handle["transforms/camera"], dtype=np.float64)
        if camera.ndim != 3 or camera.shape[1:] != (4, 4):
            raise ValueError(
                "transforms/camera must have shape [frames,4,4]"
            )
        palm_values: list[np.ndarray] = []
        for name in SOURCE_PALM_TRANSFORMS:
            key = f"transforms/{name}"
            transforms = np.asarray(handle[key], dtype=np.float64)
            if transforms.shape != camera.shape:
                raise ValueError(
                    f"{key} must have shape {camera.shape}, "
                    f"got {transforms.shape}"
                )
            palm_values.append(transforms[:, :3, 3])

    if int(frame_indices.max()) >= len(camera):
        raise IndexError("source palm frame index is out of range")
    camera = camera[frame_indices]
    palms_world = np.stack(palm_values, axis=1)[frame_indices]
    return np.einsum(
        "nji,nsj->nsi",
        camera[:, :3, :3],
        palms_world - camera[:, None, :3, 3],
    )


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


def robot_hand_tcp_points_in_camera_cv(
    *,
    urdf_path: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
    camera_pose_cv_xyzw: np.ndarray,
) -> np.ndarray:
    """Return the bilateral semantic retarget TCP as ``[frames,2,3]``."""

    fingertips = robot_frame_points_in_camera_cv(
        urdf_path=urdf_path,
        joint_names=joint_names,
        joint_positions=joint_positions,
        root_pose_xyzw=root_pose_xyzw,
        camera_pose_cv_xyzw=camera_pose_cv_xyzw,
        frame_names=ROBOT_HAND_TCP_FRAMES,
        frame_local_offsets_m=ROBOT_HAND_TCP_OFFSETS_LOCAL_M,
    ).reshape(
        len(joint_positions),
        2,
        3,
        3,
    )
    return np.einsum(
        "d,nsdj->nsj",
        HAND_TCP_POSITION_WEIGHTS,
        fingertips,
    )
