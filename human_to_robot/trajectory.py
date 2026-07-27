from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


HAND_KEYPOINT_NAMES = (
    "Hand",
    "ThumbTip",
    "IndexFingerTip",
    "MiddleFingerTip",
)


@dataclass(frozen=True)
class HandTrajectory:
    """Wrist IK targets plus continuous hand widths in the first camera frame.

    The end-effector position, orientation, visibility, and confidence come
    from the stable ``*Hand`` rigid-body tracks. Finger keypoints are used only
    to estimate the paper's continuous jaw width ``w_t``. This separation
    ensures that enabling articulated hands cannot change the arm IK targets.
    """

    left_reference_camera: np.ndarray
    right_reference_camera: np.ndarray
    left_reference_rotations: np.ndarray
    right_reference_rotations: np.ndarray
    left_widths: np.ndarray
    right_widths: np.ndarray
    left_normalized_widths: np.ndarray
    right_normalized_widths: np.ndarray
    hand_sources: tuple[str, str]
    left_observation_camera: np.ndarray
    right_observation_camera: np.ndarray
    left_pixels: np.ndarray
    right_pixels: np.ndarray
    left_confidence: np.ndarray
    right_confidence: np.ndarray
    intrinsic: np.ndarray
    camera_to_world: np.ndarray
    camera_motion: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.left_reference_camera.shape[0])


@dataclass(frozen=True)
class _AlignedHand:
    position: np.ndarray
    rotation: np.ndarray
    width: np.ndarray
    normalized_width: np.ndarray
    confidence: np.ndarray


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim == 2:
        homogeneous = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=points.dtype)],
            axis=1,
        )
        return np.einsum("nij,nj->ni", transform, homogeneous)[:, :3]
    if points.ndim == 3:
        homogeneous = np.concatenate(
            [
                points,
                np.ones((*points.shape[:2], 1), dtype=points.dtype),
            ],
            axis=2,
        )
        return np.einsum("nij,nkj->nki", transform, homogeneous)[..., :3]
    raise ValueError(f"Expected points with rank 2 or 3, got {points.shape}")


def _actual_savgol_window(
    length: int,
    requested: int,
    polynomial_order: int,
) -> int | None:
    if length < 5:
        return None
    window = min(requested, length if length % 2 else length - 1)
    minimum = polynomial_order + 2
    if minimum % 2 == 0:
        minimum += 1
    window = max(window, minimum)
    if window > length:
        window = length if length % 2 else length - 1
    return window if window > polynomial_order else None


def _interpolate_valid(
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.shape[0] != valid.shape[0]:
        raise ValueError("Values and validity mask have different lengths")
    if valid.sum() < 2:
        raise ValueError("The hand trajectory contains fewer than two valid frames")
    result = values.copy()
    frame_ids = np.arange(len(result), dtype=np.float64)
    flattened = result.reshape(len(result), -1)
    for column in range(flattened.shape[1]):
        finite = valid & np.isfinite(flattened[:, column])
        if finite.sum() < 2:
            raise ValueError(
                "The hand trajectory contains fewer than two finite samples"
            )
        flattened[:, column] = np.interp(
            frame_ids,
            frame_ids[finite],
            flattened[finite, column],
        )
    return flattened.reshape(result.shape)


def _smooth_values(
    values: np.ndarray,
    valid: np.ndarray,
    window: int,
    polynomial_order: int,
) -> np.ndarray:
    result = _interpolate_valid(values, valid)
    actual_window = _actual_savgol_window(
        len(result), window, polynomial_order
    )
    if actual_window is None:
        return result
    return savgol_filter(
        result,
        window_length=actual_window,
        polyorder=min(polynomial_order, actual_window - 1),
        axis=0,
        mode="interp",
    )


def _interpolate_rotations(
    matrices: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size < 2:
        raise ValueError("The hand orientation contains fewer than two valid frames")
    frames = np.arange(len(matrices), dtype=np.float64)
    clipped_frames = np.clip(frames, valid_ids[0], valid_ids[-1])
    interpolation = Slerp(
        valid_ids.astype(np.float64),
        Rotation.from_matrix(matrices[valid_ids]),
    )
    return interpolation(clipped_frames).as_matrix()


def _smooth_rotations_gaussian(
    matrices: np.ndarray,
    valid: np.ndarray,
    window: int,
) -> np.ndarray:
    """Apply a local Gaussian-weighted mean on SO(3).

    Neighboring rotations are mapped into the tangent space around the center
    rotation, averaged with Gaussian weights, then mapped back. This is the
    stable multi-sample equivalent of Gaussian-weighted SLERP.
    """

    interpolated = _interpolate_rotations(matrices, valid)
    radius = max(1, window // 2)
    sigma = max(1.0, window / 4.0)
    result = np.empty_like(interpolated)
    for frame_id, center_matrix in enumerate(interpolated):
        start = max(0, frame_id - radius)
        stop = min(len(interpolated), frame_id + radius + 1)
        offsets = np.arange(start, stop, dtype=np.float64) - frame_id
        weights = np.exp(-0.5 * np.square(offsets / sigma))
        center = Rotation.from_matrix(center_matrix)
        relative = (
            center.inv() * Rotation.from_matrix(interpolated[start:stop])
        ).as_rotvec()
        mean_relative = np.average(relative, axis=0, weights=weights)
        result[frame_id] = (
            center * Rotation.from_rotvec(mean_relative)
        ).as_matrix()
    return result


def _smooth_rigid_body_rotations(
    matrices: np.ndarray,
    valid: np.ndarray,
    window: int,
    polynomial_order: int,
) -> np.ndarray:
    """Smooth tracked rigid-body rotations without changing their frame."""

    matrices = np.asarray(matrices, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(matrices).all(
        axis=(1, 2)
    )
    if valid.sum() < 2:
        raise ValueError(
            "The hand orientation contains fewer than two valid frames"
        )
    anchor_id = int(np.flatnonzero(valid)[0])
    anchor = Rotation.from_matrix(matrices[anchor_id])
    relative_vectors = np.full((len(matrices), 3), np.nan, dtype=np.float64)
    relative_vectors[valid] = (
        anchor.inv() * Rotation.from_matrix(matrices[valid])
    ).as_rotvec()
    smoothed_vectors = _smooth_values(
        relative_vectors,
        valid,
        window,
        polynomial_order,
    )
    return (anchor * Rotation.from_rotvec(smoothed_vectors)).as_matrix()


def align_hand_action(
    keypoints_reference_camera: np.ndarray,
    keypoint_confidence: np.ndarray,
    *,
    side: str,
    smooth_window: int = 11,
    orientation_smooth_window: int = 15,
    polynomial_order: int = 2,
) -> _AlignedHand:
    """Construct the paper's continuous ``(p, R, w)`` hand action."""

    if side not in {"left", "right"}:
        raise ValueError(f"Unknown hand side: {side}")
    points = np.asarray(keypoints_reference_camera, dtype=np.float64)
    confidence = np.asarray(keypoint_confidence, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (4, 3):
        raise ValueError(f"Expected hand keypoints [N, 4, 3], got {points.shape}")
    if confidence.shape != points.shape[:2]:
        raise ValueError(
            f"Expected hand confidence {points.shape[:2]}, got {confidence.shape}"
        )

    wrist, thumb, index, middle = np.moveaxis(points, 1, 0)
    virtual_finger = 0.7 * index + 0.3 * middle
    jaw = thumb - virtual_finger
    width = np.linalg.norm(jaw, axis=1)
    wrist_to_finger = virtual_finger - wrist
    palm_length = np.linalg.norm(wrist_to_finger, axis=1)
    sign = 1.0 if side == "right" else -1.0
    z_axis = sign * jaw / np.maximum(width[:, None], 1e-9)
    y_unnormalized = np.cross(z_axis, wrist_to_finger)
    y_norm = np.linalg.norm(y_unnormalized, axis=1)
    y_axis = y_unnormalized / np.maximum(y_norm[:, None], 1e-9)
    x_axis = np.cross(y_axis, z_axis)
    raw_rotations = np.stack([x_axis, y_axis, z_axis], axis=2)
    frame_confidence = np.min(confidence, axis=1)
    usable_keypoints = (
        np.isfinite(points).all(axis=2)
        & np.isfinite(confidence)
        & (confidence > 0.0)
    )
    raw_position = 0.5 * (thumb + virtual_finger)
    finger_valid = usable_keypoints[:, 1:].all(axis=1)
    position_valid = (
        finger_valid
        & np.isfinite(raw_position).all(axis=1)
        & np.isfinite(width)
        & (width > 1e-6)
    )
    width_valid = (
        finger_valid
        & np.isfinite(width)
        & (width > 1e-6)
    )
    palm_valid = (
        usable_keypoints[:, [0, 2, 3]].all(axis=1)
        & np.isfinite(palm_length)
        & (palm_length > 1e-6)
    )
    orientation_valid = (
        usable_keypoints.all(axis=1)
        & np.isfinite(raw_rotations).all(axis=(1, 2))
        & np.isfinite(width)
        & np.isfinite(palm_length)
        & np.isfinite(y_norm)
        & (width > 1e-6)
        & (palm_length > 1e-6)
        & (y_norm > 1e-6)
    )
    position = _smooth_values(
        raw_position,
        position_valid,
        smooth_window,
        polynomial_order,
    )
    smoothed_width = np.maximum(
        _smooth_values(
            width[:, None],
            width_valid,
            smooth_window,
            polynomial_order,
        )[:, 0],
        0.0,
    )
    smoothed_palm_length = np.maximum(
        _smooth_values(
            palm_length[:, None],
            palm_valid,
            smooth_window,
            polynomial_order,
        )[:, 0],
        0.0,
    )
    normalized_width = np.clip(
        smoothed_width / np.maximum(smoothed_palm_length, 1e-6),
        0.0,
        2.0,
    )
    rotations = _smooth_rotations_gaussian(
        raw_rotations,
        orientation_valid,
        orientation_smooth_window,
    )
    return _AlignedHand(
        position=position,
        rotation=rotations,
        width=smoothed_width,
        normalized_width=normalized_width,
        confidence=frame_confidence,
    )


def _hdf5_hand_keys(side: str) -> tuple[str, ...]:
    return tuple(
        f"{root}/{side}{name}"
        for name in HAND_KEYPOINT_NAMES
        for root in ("transforms", "confidences")
    )


def _hdf5_hand_usable(handle: h5py.File, *, side: str) -> bool:
    if not all(key in handle for key in _hdf5_hand_keys(side)):
        return False
    frame_count = int(handle["transforms/camera"].shape[0])
    valid = np.ones(frame_count, dtype=bool)
    for name in HAND_KEYPOINT_NAMES:
        transform = handle[f"transforms/{side}{name}"]
        confidence = handle[f"confidences/{side}{name}"]
        if transform.shape != (frame_count, 4, 4) or confidence.shape != (
            frame_count,
        ):
            return False
        confidence_values = np.asarray(confidence, dtype=np.float64)
        valid &= np.isfinite(transform[:, :3, 3]).all(axis=1)
        valid &= np.isfinite(confidence_values)
        valid &= confidence_values > 0.0
    return int(valid.sum()) >= 2


def hdf5_action_keypoint_availability(
    hdf5_path: str | Path,
) -> dict[str, bool]:
    with h5py.File(hdf5_path, "r") as handle:
        return {
            side: _hdf5_hand_usable(handle, side=side)
            for side in ("left", "right")
        }


def _hdf5_keypoints_reference_camera(
    handle: h5py.File,
    *,
    side: str,
    reference_world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transforms = np.stack(
        [
            np.asarray(handle[f"transforms/{side}{name}"], dtype=np.float64)
            for name in HAND_KEYPOINT_NAMES
        ],
        axis=1,
    )
    confidence = np.stack(
        [
            np.asarray(handle[f"confidences/{side}{name}"], dtype=np.float64)
            for name in HAND_KEYPOINT_NAMES
        ],
        axis=1,
    )
    points_world = transforms[..., :3, 3]
    repeated = np.repeat(
        reference_world_to_camera[None],
        points_world.shape[0],
        axis=0,
    )
    return _transform_points(repeated, points_world), confidence


def _project(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = points_camera[:, 2]
    valid = z > 1e-6
    u = np.full_like(z, np.nan)
    v = np.full_like(z, np.nan)
    u[valid] = intrinsic[0, 0] * points_camera[valid, 0] / z[valid] + intrinsic[0, 2]
    v[valid] = intrinsic[1, 1] * points_camera[valid, 1] / z[valid] + intrinsic[1, 2]
    return np.stack([u, v], axis=1)


def load_hand_trajectory(
    hdf5_path: str | Path,
    *,
    smooth_window: int = 11,
    orientation_smooth_window: int = 15,
    polynomial_order: int = 2,
) -> HandTrajectory:
    """Load wrist IK targets and independently infer continuous hand widths."""

    path = Path(hdf5_path)
    with h5py.File(path, "r") as handle:
        required = ("camera/intrinsic", "transforms/camera")
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"{path} is missing datasets: {', '.join(missing)}")
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        camera_to_world = np.asarray(
            handle["transforms/camera"], dtype=np.float64
        )
        if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (4, 4):
            raise ValueError(
                "transforms/camera must have shape [frame_count, 4, 4]"
            )
        frame_count = len(camera_to_world)
        reference_world_to_camera = np.linalg.inv(camera_to_world[0])
        camera_motion = np.einsum(
            "ij,njk->nik", reference_world_to_camera, camera_to_world
        )
        availability = {
            side: _hdf5_hand_usable(handle, side=side)
            for side in ("left", "right")
        }
        missing_sides = [
            side for side, usable in availability.items() if not usable
        ]
        if missing_sides:
            required_names = ", ".join(HAND_KEYPOINT_NAMES)
            raise ValueError(
                f"{path} lacks usable annotated keypoints for "
                f"{', '.join(missing_sides)} hand(s). Required transforms "
                f"and confidences: {required_names}, with at least two "
                "valid frames."
            )
        keypoints = {
            side: _hdf5_keypoints_reference_camera(
                handle,
                side=side,
                reference_world_to_camera=reference_world_to_camera,
            )
            for side in ("left", "right")
        }
        rigid_body_tracks = {
            side: (
                np.asarray(
                    handle[f"transforms/{side}Hand"],
                    dtype=np.float64,
                ),
                np.asarray(
                    handle[f"confidences/{side}Hand"],
                    dtype=np.float64,
                ),
            )
            for side in ("left", "right")
        }

    aligned = {
        side: align_hand_action(
            *keypoints[side],
            side=side,
            smooth_window=smooth_window,
            orientation_smooth_window=orientation_smooth_window,
            polynomial_order=polynomial_order,
        )
        for side in ("left", "right")
    }
    left = aligned["left"]
    right = aligned["right"]
    ik_targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    repeated_reference_transform = np.repeat(
        reference_world_to_camera[None],
        frame_count,
        axis=0,
    )
    for side in ("left", "right"):
        transforms, confidence = rigid_body_tracks[side]
        position_valid = (
            np.isfinite(transforms[:, :3, 3]).all(axis=1)
            & np.isfinite(confidence)
            & (confidence > 0.2)
        )
        reference_position = _smooth_values(
            _transform_points(
                repeated_reference_transform,
                transforms[:, :3, 3],
            ),
            position_valid,
            smooth_window,
            polynomial_order,
        )
        rotation_valid = (
            np.isfinite(confidence)
            & (confidence > 0.2)
            & np.isfinite(transforms[:, :3, :3]).all(axis=(1, 2))
        )
        world_rotation = _smooth_rigid_body_rotations(
            transforms[:, :3, :3],
            rotation_valid,
            orientation_smooth_window,
            polynomial_order,
        )
        reference_rotation = np.einsum(
            "ij,njk->nik",
            reference_world_to_camera[:3, :3],
            world_rotation,
        )
        ik_targets[side] = (
            reference_position,
            reference_rotation,
            confidence,
        )

    left_position, left_rotation, left_confidence = ik_targets["left"]
    right_position, right_rotation, right_confidence = ik_targets["right"]
    current_from_reference = np.linalg.inv(camera_motion)
    left_observation_camera = _transform_points(
        current_from_reference, left_position
    )
    right_observation_camera = _transform_points(
        current_from_reference, right_position
    )
    return HandTrajectory(
        left_reference_camera=left_position,
        right_reference_camera=right_position,
        left_reference_rotations=left_rotation,
        right_reference_rotations=right_rotation,
        left_widths=left.width,
        right_widths=right.width,
        left_normalized_widths=left.normalized_width,
        right_normalized_widths=right.normalized_width,
        hand_sources=("hdf5", "hdf5"),
        left_observation_camera=left_observation_camera,
        right_observation_camera=right_observation_camera,
        left_pixels=_project(left_observation_camera, intrinsic),
        right_pixels=_project(right_observation_camera, intrinsic),
        left_confidence=left_confidence,
        right_confidence=right_confidence,
        intrinsic=intrinsic,
        camera_to_world=camera_to_world,
        camera_motion=camera_motion,
    )
