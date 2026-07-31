"""Robot-mounted camera poses for the independent RGB-D renderer."""

from __future__ import annotations

from pathlib import Path

import h5py
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


def camera_transforms_to_poses_cv(transforms: np.ndarray) -> np.ndarray:
    """Convert world-from-CV-camera matrices to XYZ + XYZW poses."""

    transforms = np.asarray(transforms, dtype=np.float64)
    if (
        transforms.ndim != 3
        or transforms.shape[1:] != (4, 4)
        or not np.isfinite(transforms).all()
    ):
        raise ValueError("camera transforms must be finite [frames, 4, 4]")
    if not np.allclose(
        transforms[:, 3, :],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
    ):
        raise ValueError("camera transforms have invalid homogeneous rows")
    rotations = transforms[:, :3, :3]
    if not np.allclose(
        np.einsum("nji,njk->nik", rotations, rotations),
        np.eye(3)[None],
        atol=1e-5,
    ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5):
        raise ValueError("camera transform rotations must lie in SO(3)")
    quaternions = Rotation.from_matrix(rotations).as_quat()
    for frame_id in range(1, len(quaternions)):
        if np.dot(quaternions[frame_id - 1], quaternions[frame_id]) < 0.0:
            quaternions[frame_id] *= -1.0
    return np.concatenate([transforms[:, :3, 3], quaternions], axis=1)


def load_source_camera_poses_cv(
    source_hdf5: Path,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Load selected EgoDex world-from-camera extrinsics."""

    source_hdf5 = source_hdf5.resolve()
    if not source_hdf5.is_file():
        raise FileNotFoundError(source_hdf5)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if frame_indices.ndim != 1 or len(frame_indices) == 0:
        raise ValueError("frame_indices must be a non-empty vector")
    if np.any(frame_indices < 0):
        raise ValueError("frame_indices must be non-negative")
    with h5py.File(source_hdf5, "r") as handle:
        transforms = np.asarray(
            handle["transforms/camera"],
            dtype=np.float64,
        )
    if int(frame_indices.max()) >= len(transforms):
        raise IndexError("source camera frame index is out of range")
    return camera_transforms_to_poses_cv(transforms[frame_indices])


def load_source_hand_points_camera_cv(
    source_hdf5: Path,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load left/right human-hand origins in each source CV camera frame."""

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
        hands = np.stack(
            [
                np.asarray(
                    handle["transforms/leftHand"],
                    dtype=np.float64,
                ),
                np.asarray(
                    handle["transforms/rightHand"],
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )
        confidence_values: list[np.ndarray] = []
        for side in ("left", "right"):
            key = f"confidences/{side}Hand"
            if key in handle:
                values = np.asarray(handle[key], dtype=np.float64)
                if values.shape != (len(camera),):
                    raise ValueError(
                        f"{key} must have shape {(len(camera),)}, "
                        f"got {values.shape}"
                    )
            else:
                # Nymeria/pour clips have valid 3-D transforms but omit the
                # optional confidence group. This matches the trajectory
                # loader contract: absent confidence means one, not invalid.
                values = np.ones(len(camera), dtype=np.float64)
            confidence_values.append(values)
        confidence = np.stack(confidence_values, axis=1)
    if int(frame_indices.max()) >= len(camera):
        raise IndexError("source hand frame index is out of range")
    camera = camera[frame_indices]
    hands = hands[frame_indices]
    confidence = confidence[frame_indices]
    relative_world = hands[:, :, :3, 3] - camera[:, None, :3, 3]
    points_camera = np.einsum(
        "nji,nsj->nsi",
        camera[:, :3, :3],
        relative_world,
    )
    return points_camera, confidence


def co_centered_source_camera_poses_cv(
    *,
    source_camera_pose_cv_xyzw: np.ndarray,
    source_to_robot_rotation: np.ndarray,
    robot_camera_pose_cv_xyzw: np.ndarray,
) -> np.ndarray:
    """Put mapped source-camera centers on D435 while retaining rotation."""

    source_pose = np.asarray(
        source_camera_pose_cv_xyzw,
        dtype=np.float64,
    )
    robot_pose = np.asarray(
        robot_camera_pose_cv_xyzw,
        dtype=np.float64,
    )
    source_to_robot = np.asarray(
        source_to_robot_rotation,
        dtype=np.float64,
    )
    if (
        source_pose.ndim != 2
        or source_pose.shape[1] != 7
        or robot_pose.shape != source_pose.shape
        or not np.isfinite(source_pose).all()
        or not np.isfinite(robot_pose).all()
    ):
        raise ValueError("source and robot camera poses must be finite [N, 7]")
    if (
        source_to_robot.shape != (3, 3)
        or not np.isfinite(source_to_robot).all()
        or not np.allclose(
            source_to_robot.T @ source_to_robot,
            np.eye(3),
            atol=1e-6,
        )
        or not np.isclose(np.linalg.det(source_to_robot), 1.0, atol=1e-6)
    ):
        raise ValueError("source_to_robot_rotation must lie in SO(3)")

    mapped_rotation = np.einsum(
        "ij,njk->nik",
        source_to_robot,
        Rotation.from_quat(source_pose[:, 3:7]).as_matrix(),
    )
    mapped_pose = np.empty_like(source_pose)
    mapped_pose[:, :3] = robot_pose[:, :3]
    mapped_pose[:, 3:7] = Rotation.from_matrix(
        mapped_rotation
    ).as_quat()
    for frame_id in range(1, len(mapped_pose)):
        if (
            np.dot(
                mapped_pose[frame_id - 1, 3:7],
                mapped_pose[frame_id, 3:7],
            )
            < 0.0
        ):
            mapped_pose[frame_id, 3:7] *= -1.0
    return mapped_pose


def fit_co_centered_source_scale(
    *,
    source_hand_points_camera_cv: np.ndarray,
    source_hand_confidence: np.ndarray,
    source_camera_pose_cv_xyzw: np.ndarray,
    source_to_robot_rotation: np.ndarray,
    robot_camera_pose_cv_xyzw: np.ndarray,
    robot_hand_points_camera_cv: np.ndarray,
    confidence_threshold: float = 0.5,
) -> tuple[float, dict[str, object]]:
    """Fit one scale after forcing source and D435 optical centers equal."""

    source_hand = np.asarray(
        source_hand_points_camera_cv,
        dtype=np.float64,
    )
    confidence = np.asarray(
        source_hand_confidence,
        dtype=np.float64,
    )
    source_pose = np.asarray(
        source_camera_pose_cv_xyzw,
        dtype=np.float64,
    )
    robot_pose = np.asarray(
        robot_camera_pose_cv_xyzw,
        dtype=np.float64,
    )
    robot_hand = np.asarray(
        robot_hand_points_camera_cv,
        dtype=np.float64,
    )
    source_to_robot = np.asarray(
        source_to_robot_rotation,
        dtype=np.float64,
    )
    frame_count = len(source_pose)
    if (
        source_pose.shape != (frame_count, 7)
        or robot_pose.shape != source_pose.shape
        or source_hand.ndim != 3
        or source_hand.shape[0] != frame_count
        or source_hand.shape[2] != 3
        or robot_hand.shape != source_hand.shape
        or confidence.shape != source_hand.shape[:2]
    ):
        raise ValueError("co-centered scale inputs have incompatible shapes")
    if (
        not np.isfinite(source_pose).all()
        or not np.isfinite(robot_pose).all()
        or not np.isfinite(source_hand).all()
        or not np.isfinite(robot_hand).all()
        or not np.isfinite(confidence).all()
    ):
        raise ValueError("co-centered scale inputs must be finite")
    if source_to_robot.shape != (3, 3):
        raise ValueError("source_to_robot_rotation must be 3x3")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in [0, 1]")

    source_rotation = Rotation.from_quat(
        source_pose[:, 3:7]
    ).as_matrix()
    robot_rotation = Rotation.from_quat(
        robot_pose[:, 3:7]
    ).as_matrix()
    robot_camera_from_source_camera = np.einsum(
        "nji,jk,nkl->nil",
        robot_rotation,
        source_to_robot,
        source_rotation,
    )
    mapped_source_hand = np.einsum(
        "nij,nsj->nsi",
        robot_camera_from_source_camera,
        source_hand,
    )
    valid = (
        (confidence >= confidence_threshold)
        & (source_hand[:, :, 2] > 0.0)
        & (robot_hand[:, :, 2] > 0.0)
    )
    if int(valid.sum()) < 4:
        raise ValueError(
            "Too few confident hand correspondences for co-centered scale"
        )
    source_vector = mapped_source_hand[valid]
    robot_vector = robot_hand[valid]
    denominator = float(np.sum(np.square(source_vector)))
    if denominator <= 1e-12:
        raise ValueError("Source hand vectors cannot determine a scale")
    scale = float(np.sum(source_vector * robot_vector) / denominator)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Fitted co-centered scale is not positive")
    error = np.linalg.norm(
        scale * source_vector - robot_vector,
        axis=1,
    )
    report: dict[str, object] = {
        "method": "co-centered source/G1 hand-vector least squares",
        "fitted_scale": scale,
        "correspondence_count": int(valid.sum()),
        "confidence_threshold": float(confidence_threshold),
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "median_error_m": float(np.median(error)),
        "p95_error_m": float(np.quantile(error, 0.95)),
    }
    return scale, report


def rigid_source_scene_camera_poses_cv(
    *,
    source_camera_pose_cv_xyzw: np.ndarray,
    source_to_robot_rotation: np.ndarray,
    robot_camera_anchor_pose_cv_xyzw: np.ndarray,
    scene_scale: float,
) -> np.ndarray:
    """Map source-camera extrinsics into one rigid robot-world scene.

    Source world orientation is preserved through the episode-level
    ``source_to_robot_rotation``. The first selected source-camera center is
    translated onto the first G1 camera center, but its rotation is not forced
    to the fixed G1 camera rotation. This keeps source-world planes and gravity
    correctly oriented instead of rotating the whole scene with the D435.
    """

    source_pose = np.asarray(
        source_camera_pose_cv_xyzw,
        dtype=np.float64,
    )
    anchor_pose = np.asarray(
        robot_camera_anchor_pose_cv_xyzw,
        dtype=np.float64,
    )
    source_to_robot = np.asarray(
        source_to_robot_rotation,
        dtype=np.float64,
    )
    if (
        source_pose.ndim != 2
        or source_pose.shape[1] != 7
        or len(source_pose) == 0
        or not np.isfinite(source_pose).all()
    ):
        raise ValueError("source camera poses must be finite [frames, 7]")
    if anchor_pose.shape != (7,) or not np.isfinite(anchor_pose).all():
        raise ValueError("robot camera anchor pose must be a finite pose")
    if (
        source_to_robot.shape != (3, 3)
        or not np.isfinite(source_to_robot).all()
        or not np.allclose(
            source_to_robot.T @ source_to_robot,
            np.eye(3),
            atol=1e-6,
        )
        or not np.isclose(np.linalg.det(source_to_robot), 1.0, atol=1e-6)
    ):
        raise ValueError("source_to_robot_rotation must lie in SO(3)")
    if not np.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError("scene_scale must be positive and finite")

    source_rotation = Rotation.from_quat(
        source_pose[:, 3:7]
    ).as_matrix()
    mapped_rotation = np.einsum(
        "ij,njk->nik",
        source_to_robot,
        source_rotation,
    )
    source_motion = source_pose[:, :3] - source_pose[0, :3]
    mapped_translation = (
        anchor_pose[:3]
        + scene_scale * (source_to_robot @ source_motion.T).T
    )
    mapped_quaternion = Rotation.from_matrix(mapped_rotation).as_quat()
    for frame_id in range(1, len(mapped_quaternion)):
        if (
            np.dot(
                mapped_quaternion[frame_id - 1],
                mapped_quaternion[frame_id],
            )
            < 0.0
        ):
            mapped_quaternion[frame_id] *= -1.0
    return np.concatenate(
        [mapped_translation, mapped_quaternion],
        axis=1,
    )


def d435_camera_poses_cv(
    *,
    urdf_path: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
    frame_name: str = "d435_link",
    optical_center_in_link_xyz: np.ndarray | tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    ),
) -> np.ndarray:
    """Return world-from-D435-optical poses for a G1 trajectory.

    The URDF frame is treated as a standard ROS camera-link frame. The
    returned quaternion rotates OpenCV optical coordinates into world
    coordinates, matching RGB-D backprojection in ``rgbd_geometry``.
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
    if not np.isfinite(joint_positions).all():
        raise ValueError("joint_positions must be finite")
    if not np.isfinite(root_pose_xyzw).all():
        raise ValueError("root_pose_xyzw must be finite")
    quaternion_norm = np.linalg.norm(root_pose_xyzw[:, 3:7], axis=1)
    if np.any(quaternion_norm <= 1e-8):
        raise ValueError("root_pose_xyzw contains a zero quaternion")

    optical_center = np.asarray(
        optical_center_in_link_xyz,
        dtype=np.float64,
    )
    if optical_center.shape != (3,) or not np.isfinite(optical_center).all():
        raise ValueError("optical_center_in_link_xyz must be a finite XYZ")

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
    camera_frame_id = model.getFrameId(frame_name)
    if camera_frame_id >= model.nframes:
        raise KeyError(f"G1 URDF has no frame named {frame_name!r}")

    data = model.createData()
    root_rotations = Rotation.from_quat(
        root_pose_xyzw[:, 3:7]
    ).as_matrix()
    poses = np.empty((frame_count, 7), dtype=np.float64)
    for frame_id in range(frame_count):
        pin.forwardKinematics(model, data, joint_positions[frame_id])
        pin.updateFramePlacements(model, data)
        model_from_link = data.oMf[camera_frame_id]
        world_from_link_rotation = (
            root_rotations[frame_id]
            @ np.asarray(model_from_link.rotation, dtype=np.float64)
        )
        world_from_link_translation = (
            root_rotations[frame_id]
            @ np.asarray(model_from_link.translation, dtype=np.float64)
            + root_pose_xyzw[frame_id, :3]
        )
        poses[frame_id, :3] = (
            world_from_link_translation
            + world_from_link_rotation @ optical_center
        )
        poses[frame_id, 3:7] = Rotation.from_matrix(
            world_from_link_rotation @ CAMERA_LINK_FROM_CV_OPTICAL
        ).as_quat()

    # Avoid quaternion sign flips in reports/interpolation even though q and
    # -q encode the same rotation.
    for frame_id in range(1, frame_count):
        if np.dot(poses[frame_id - 1, 3:7], poses[frame_id, 3:7]) < 0.0:
            poses[frame_id, 3:7] *= -1.0
    return poses


def robot_frame_points_in_camera_cv(
    *,
    urdf_path: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
    camera_pose_cv_xyzw: np.ndarray,
    frame_names: tuple[str, ...],
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
            point_world = (
                root_rotation[frame_id]
                @ np.asarray(
                    data.oMf[robot_frame_id].translation,
                    dtype=np.float64,
                )
                + root_pose_xyzw[frame_id, :3]
            )
            points_camera[frame_id, point_id] = (
                camera_rotation[frame_id].T
                @ (point_world - camera_pose[frame_id, :3])
            )
    return points_camera


def camera_pose_delta_report(
    reference_pose_xyzw: np.ndarray,
    actual_pose_xyzw: np.ndarray,
) -> dict[str, object]:
    """Summarize actual-minus-reference camera translation and rotation."""

    reference = np.asarray(reference_pose_xyzw, dtype=np.float64)
    actual = np.asarray(actual_pose_xyzw, dtype=np.float64)
    if (
        reference.ndim != 2
        or reference.shape[1] != 7
        or actual.shape != reference.shape
    ):
        raise ValueError("camera poses must have matching [frames, 7] shapes")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("camera poses must be finite")

    translation = actual[:, :3] - reference[:, :3]
    reference_rotation = Rotation.from_quat(reference[:, 3:7])
    actual_rotation = Rotation.from_quat(actual[:, 3:7])
    angle_degrees = np.rad2deg(
        (reference_rotation.inv() * actual_rotation).magnitude()
    )
    return {
        "translation_actual_minus_reference_world_m": {
            "first": translation[0].tolist(),
            "median": np.median(translation, axis=0).tolist(),
            "mean": np.mean(translation, axis=0).tolist(),
            "std": np.std(translation, axis=0).tolist(),
        },
        "rotation_angle_degrees": {
            "first": float(angle_degrees[0]),
            "median": float(np.median(angle_degrees)),
            "min": float(np.min(angle_degrees)),
            "max": float(np.max(angle_degrees)),
        },
    }
