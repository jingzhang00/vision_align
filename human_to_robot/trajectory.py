from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class HandTrajectory:
    """Two single-point hand trajectories and the source camera motion.

    Reference points and rotations are expressed in the *first* camera frame.
    They are suitable IK inputs because head motion has been factored out.
    ``camera_motion`` is ``T_reference_camera_current`` and restores the source
    head motion during rendering. The camera convention is +x image-right,
    +y image-down and +z forward.
    """

    left_reference_camera: np.ndarray
    right_reference_camera: np.ndarray
    left_reference_rotations: np.ndarray
    right_reference_rotations: np.ndarray
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


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1
    )
    return np.einsum("nij,nj->ni", transform, homogeneous)[:, :3]


def _smooth_valid(
    points: np.ndarray,
    confidence: np.ndarray,
    window: int,
    polynomial_order: int,
) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).copy()
    valid = np.isfinite(result).all(axis=1) & (confidence > 0.2)
    frame_ids = np.arange(len(result), dtype=np.float64)
    if valid.sum() < 2:
        raise ValueError("The hand trajectory contains fewer than two valid frames")
    for axis in range(3):
        result[:, axis] = np.interp(
            frame_ids, frame_ids[valid], result[valid, axis]
        )

    if len(result) >= 5:
        actual_window = min(window, len(result) if len(result) % 2 else len(result) - 1)
        actual_window = max(actual_window, polynomial_order + 2)
        if actual_window % 2 == 0:
            actual_window -= 1
        result = savgol_filter(
            result,
            window_length=actual_window,
            polyorder=min(polynomial_order, actual_window - 1),
            axis=0,
            mode="interp",
        )
    return result


def _smooth_rotations(
    matrices: np.ndarray,
    confidence: np.ndarray,
    window: int,
    polynomial_order: int,
) -> np.ndarray:
    """Smooth a short manipulation trajectory in the tangent space of SO(3)."""

    finite = np.isfinite(matrices).all(axis=(1, 2))
    valid = finite & (confidence > 0.2)
    if valid.sum() < 2:
        raise ValueError("The hand orientation contains fewer than two valid frames")
    anchor_id = int(np.flatnonzero(valid)[0])
    anchor = Rotation.from_matrix(matrices[anchor_id])
    relative_vectors = np.full((len(matrices), 3), np.nan, dtype=np.float64)
    relative_vectors[valid] = (
        anchor.inv() * Rotation.from_matrix(matrices[valid])
    ).as_rotvec()
    smoothed_vectors = _smooth_valid(
        relative_vectors,
        confidence,
        window,
        polynomial_order,
    )
    return (anchor * Rotation.from_rotvec(smoothed_vectors)).as_matrix()


def _project(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = points_camera[:, 2]
    valid = z > 1e-6
    u = np.full_like(z, np.nan)
    v = np.full_like(z, np.nan)
    u[valid] = intrinsic[0, 0] * points_camera[valid, 0] / z[valid] + intrinsic[0, 2]
    v[valid] = intrinsic[1, 1] * points_camera[valid, 1] / z[valid] + intrinsic[1, 2]
    return np.stack([u, v], axis=1)


def load_single_point_trajectory(
    hdf5_path: str | Path,
    *,
    smooth_window: int = 11,
    orientation_smooth_window: int = 15,
    polynomial_order: int = 2,
) -> HandTrajectory:
    """Load each ``*Hand`` rigid-body origin as one end-effector point.

    Position uses only the rigid-body origin. Wrist orientation comes from the
    same HDF5 rigid transform; MANO, fingertips, gripper width, and finger
    opening/closing remain unused.
    """

    path = Path(hdf5_path)
    with h5py.File(path, "r") as handle:
        required = (
            "camera/intrinsic",
            "transforms/camera",
            "transforms/leftHand",
            "transforms/rightHand",
            "confidences/leftHand",
            "confidences/rightHand",
        )
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"{path} is missing datasets: {', '.join(missing)}")

        intrinsic = handle["camera/intrinsic"][...].astype(np.float64)
        camera_to_world = handle["transforms/camera"][...].astype(np.float64)
        left_world = handle["transforms/leftHand"][:, :3, 3].astype(np.float64)
        right_world = handle["transforms/rightHand"][:, :3, 3].astype(np.float64)
        left_world_rotations = handle["transforms/leftHand"][:, :3, :3].astype(
            np.float64
        )
        right_world_rotations = handle["transforms/rightHand"][:, :3, :3].astype(
            np.float64
        )
        left_confidence = handle["confidences/leftHand"][...].astype(np.float64)
        right_confidence = handle["confidences/rightHand"][...].astype(np.float64)

    counts = {
        len(camera_to_world),
        len(left_world),
        len(right_world),
        len(left_confidence),
        len(right_confidence),
    }
    if len(counts) != 1:
        raise ValueError(f"HDF5 frame counts disagree: {sorted(counts)}")

    reference_world_to_camera = np.linalg.inv(camera_to_world[0])
    left_reference_camera = _transform_points(
        np.repeat(reference_world_to_camera[None], len(left_world), axis=0),
        left_world,
    )
    right_reference_camera = _transform_points(
        np.repeat(reference_world_to_camera[None], len(right_world), axis=0),
        right_world,
    )
    left_reference_camera = _smooth_valid(
        left_reference_camera, left_confidence, smooth_window, polynomial_order
    )
    right_reference_camera = _smooth_valid(
        right_reference_camera, right_confidence, smooth_window, polynomial_order
    )
    reference_world_to_camera_rotation = reference_world_to_camera[:3, :3]
    left_reference_rotations = np.einsum(
        "ij,njk->nik",
        reference_world_to_camera_rotation,
        _smooth_rotations(
            left_world_rotations,
            left_confidence,
            orientation_smooth_window,
            polynomial_order,
        ),
    )
    right_reference_rotations = np.einsum(
        "ij,njk->nik",
        reference_world_to_camera_rotation,
        _smooth_rotations(
            right_world_rotations,
            right_confidence,
            orientation_smooth_window,
            polynomial_order,
        ),
    )

    camera_motion = np.einsum(
        "ij,njk->nik", reference_world_to_camera, camera_to_world
    )
    current_from_reference = np.linalg.inv(camera_motion)
    left_observation_camera = _transform_points(
        current_from_reference, left_reference_camera
    )
    right_observation_camera = _transform_points(
        current_from_reference, right_reference_camera
    )

    return HandTrajectory(
        left_reference_camera=left_reference_camera,
        right_reference_camera=right_reference_camera,
        left_reference_rotations=left_reference_rotations,
        right_reference_rotations=right_reference_rotations,
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
