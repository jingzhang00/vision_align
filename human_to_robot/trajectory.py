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
MANO_KEYPOINT_INDICES = {
    "wrist": 0,
    "thumb": 4,
    "index": 8,
    "middle": 12,
}


@dataclass(frozen=True)
class HandTrajectory:
    """Human hand actions expressed in the first camera frame.

    End-effector position, orientation, and width follow the action-alignment
    construction from the paper. ``binary_actions`` has shape ``[N, 2]`` in
    left/right order, where 0 is open and 1 is closed.
    """

    left_reference_camera: np.ndarray
    right_reference_camera: np.ndarray
    left_reference_rotations: np.ndarray
    right_reference_rotations: np.ndarray
    left_widths: np.ndarray
    right_widths: np.ndarray
    left_normalized_widths: np.ndarray
    right_normalized_widths: np.ndarray
    binary_actions: np.ndarray
    action_thresholds: np.ndarray
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
    binary_action: np.ndarray
    action_threshold: float


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


def _binary_close_action(
    normalized_width: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, float]:
    samples = np.asarray(normalized_width, dtype=np.float64)[valid]
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        raise ValueError("Cannot infer hand open/closed state from fewer than 2 frames")

    low, high = np.percentile(samples, [20.0, 80.0])
    for _ in range(20):
        threshold = 0.5 * (low + high)
        low_group = samples[samples <= threshold]
        high_group = samples[samples > threshold]
        if low_group.size == 0 or high_group.size == 0:
            break
        updated = float(low_group.mean()), float(high_group.mean())
        if np.allclose(updated, (low, high), atol=1e-6):
            low, high = updated
            break
        low, high = updated

    separation = float(high - low)
    threshold = (
        0.5 * float(low + high)
        if separation >= 0.04
        else 0.35
    )
    hysteresis = max(0.01, min(0.04, 0.15 * max(separation, 0.04)))
    values = np.asarray(normalized_width, dtype=np.float64)
    actions = np.empty(len(values), dtype=np.uint8)
    closed = bool(values[0] < threshold)
    for frame_id, value in enumerate(values):
        if closed and value >= threshold + hysteresis:
            closed = False
        elif not closed and value <= threshold - hysteresis:
            closed = True
        actions[frame_id] = int(closed)
    return actions, threshold


def align_hand_action(
    keypoints_reference_camera: np.ndarray,
    keypoint_confidence: np.ndarray,
    *,
    side: str,
    smooth_window: int = 11,
    orientation_smooth_window: int = 15,
    polynomial_order: int = 2,
) -> _AlignedHand:
    """Construct ``(p, R, w)`` and a binary grasp action from four keypoints."""

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
    valid = (
        np.isfinite(points).all(axis=(1, 2))
        & np.isfinite(raw_rotations).all(axis=(1, 2))
        & (frame_confidence > 0.2)
        & (width > 1e-6)
        & (palm_length > 1e-6)
        & (y_norm > 1e-6)
    )
    position = _smooth_values(
        0.5 * (thumb + virtual_finger),
        valid,
        smooth_window,
        polynomial_order,
    )
    smoothed_width = _smooth_values(
        width[:, None],
        valid,
        smooth_window,
        polynomial_order,
    )[:, 0]
    smoothed_palm_length = _smooth_values(
        palm_length[:, None],
        valid,
        smooth_window,
        polynomial_order,
    )[:, 0]
    normalized_width = np.clip(
        smoothed_width / np.maximum(smoothed_palm_length, 1e-6),
        0.0,
        2.0,
    )
    rotations = _smooth_rotations_gaussian(
        raw_rotations,
        valid,
        orientation_smooth_window,
    )
    binary_action, action_threshold = _binary_close_action(
        normalized_width,
        valid,
    )
    return _AlignedHand(
        position=position,
        rotation=rotations,
        width=smoothed_width,
        normalized_width=normalized_width,
        confidence=frame_confidence,
        binary_action=binary_action,
        action_threshold=action_threshold,
    )


def _hdf5_hand_keys(side: str) -> tuple[str, ...]:
    return tuple(
        f"{root}/{side}{name}"
        for name in HAND_KEYPOINT_NAMES
        for root in ("transforms", "confidences")
    )


def hdf5_action_keypoint_availability(
    hdf5_path: str | Path,
) -> dict[str, bool]:
    with h5py.File(hdf5_path, "r") as handle:
        return {
            side: all(key in handle for key in _hdf5_hand_keys(side))
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


def _mano_keypoints_reference_camera(
    cache: np.lib.npyio.NpzFile,
    *,
    side: str,
    camera_motion: np.ndarray,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    points_key = f"{side}_keypoints_camera"
    confidence_key = f"{side}_confidence"
    if points_key not in cache or confidence_key not in cache:
        raise KeyError(
            f"MANO cache is missing {points_key!r} or {confidence_key!r}"
        )
    points = np.asarray(cache[points_key], dtype=np.float64)
    confidence = np.asarray(cache[confidence_key], dtype=np.float64)
    if points.shape != (frame_count, 21, 3):
        raise ValueError(
            f"{points_key} must have shape {(frame_count, 21, 3)}, "
            f"got {points.shape}"
        )
    if confidence.shape == (frame_count,):
        confidence = np.repeat(confidence[:, None], 21, axis=1)
    if confidence.shape != (frame_count, 21):
        raise ValueError(
            f"{confidence_key} must have shape {(frame_count, 21)}, "
            f"got {confidence.shape}"
        )
    indices = [
        MANO_KEYPOINT_INDICES[name]
        for name in ("wrist", "thumb", "index", "middle")
    ]
    return (
        _transform_points(camera_motion, points[:, indices]),
        confidence[:, indices],
    )


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
    mano_keypoints_path: str | Path | None = None,
    smooth_window: int = 11,
    orientation_smooth_window: int = 15,
    polynomial_order: int = 2,
) -> HandTrajectory:
    """Load action-aligned hands from HDF5, with MANO as a per-hand fallback."""

    path = Path(hdf5_path)
    mano_path = (
        Path(mano_keypoints_path)
        if mano_keypoints_path is not None
        else None
    )
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
            side: all(key in handle for key in _hdf5_hand_keys(side))
            for side in ("left", "right")
        }
        cache = np.load(mano_path) if mano_path is not None else None
        try:
            keypoints: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            sources: list[str] = []
            for side in ("left", "right"):
                if availability[side]:
                    keypoints[side] = _hdf5_keypoints_reference_camera(
                        handle,
                        side=side,
                        reference_world_to_camera=reference_world_to_camera,
                    )
                    sources.append("hdf5")
                elif cache is not None:
                    keypoints[side] = _mano_keypoints_reference_camera(
                        cache,
                        side=side,
                        camera_motion=camera_motion,
                        frame_count=frame_count,
                    )
                    sources.append("mano")
                else:
                    raise KeyError(
                        f"{path} has no complete {side} hand keypoints; "
                        "provide --mano-keypoints generated from the video"
                    )
        finally:
            if cache is not None:
                cache.close()

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
    current_from_reference = np.linalg.inv(camera_motion)
    left_observation_camera = _transform_points(
        current_from_reference, left.position
    )
    right_observation_camera = _transform_points(
        current_from_reference, right.position
    )
    return HandTrajectory(
        left_reference_camera=left.position,
        right_reference_camera=right.position,
        left_reference_rotations=left.rotation,
        right_reference_rotations=right.rotation,
        left_widths=left.width,
        right_widths=right.width,
        left_normalized_widths=left.normalized_width,
        right_normalized_widths=right.normalized_width,
        binary_actions=np.stack(
            [left.binary_action, right.binary_action], axis=1
        ),
        action_thresholds=np.asarray(
            [left.action_threshold, right.action_threshold],
            dtype=np.float64,
        ),
        hand_sources=(sources[0], sources[1]),
        left_observation_camera=left_observation_camera,
        right_observation_camera=right_observation_camera,
        left_pixels=_project(left_observation_camera, intrinsic),
        right_pixels=_project(right_observation_camera, intrinsic),
        left_confidence=left.confidence,
        right_confidence=right.confidence,
        intrinsic=intrinsic,
        camera_to_world=camera_to_world,
        camera_motion=camera_motion,
    )


# Kept as a narrow compatibility alias for callers outside this repository.
load_single_point_trajectory = load_hand_trajectory
