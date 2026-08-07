"""Shared RGB-D geometry helpers for the canonical VDA-to-Isaac route."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def scale_intrinsic(
    intrinsic: np.ndarray,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale an image-edge OpenCV intrinsic between uncropped resolutions."""

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target dimensions must be positive")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled = intrinsic.copy()
    scaled[0] *= scale_x
    scaled[1] *= scale_y
    scaled[2] = intrinsic[2]
    return scaled


def backproject_z_depth(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Backproject OpenCV camera-Z depth to an ``[H, W, 3]`` point grid."""

    depth_m = np.asarray(depth_m, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("focal lengths must be positive")
    height, width = depth_m.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    points = np.empty((height, width, 3), dtype=np.float64)
    points[..., 0] = (u - cx) * depth_m / fx
    points[..., 1] = (v - cy) * depth_m / fy
    points[..., 2] = depth_m
    invalid = ~np.isfinite(depth_m) | (depth_m <= 0.0)
    points[invalid] = np.nan
    return points


def camera_points_to_world(
    points_camera_cv: np.ndarray,
    camera_pose_xyzw: np.ndarray,
) -> np.ndarray:
    """Transform OpenCV-camera points with a world-from-camera pose."""

    points_camera_cv = np.asarray(points_camera_cv, dtype=np.float64)
    camera_pose_xyzw = np.asarray(camera_pose_xyzw, dtype=np.float64)
    if points_camera_cv.shape[-1] != 3:
        raise ValueError("points_camera_cv must end in XYZ coordinates")
    if camera_pose_xyzw.shape != (7,):
        raise ValueError("camera_pose_xyzw must have shape (7,)")
    if not np.isfinite(camera_pose_xyzw).all():
        raise ValueError("camera_pose_xyzw must be finite")
    rotation = Rotation.from_quat(camera_pose_xyzw[3:7]).as_matrix()
    return (
        points_camera_cv @ rotation.T
        + camera_pose_xyzw[:3]
    )


def grid_uvs(width: int, height: int) -> np.ndarray:
    """Return vertex UVs with image top mapped to texture top."""

    if width <= 1 or height <= 1:
        raise ValueError("UV grid dimensions must exceed one pixel")
    u, v = np.meshgrid(
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        np.linspace(1.0, 0.0, height, dtype=np.float32),
    )
    return np.stack([u, v], axis=-1).reshape(-1, 2)


def depth_grid_triangles(
    depth_m: np.ndarray,
    *,
    absolute_threshold_m: float = 0.05,
    relative_threshold: float = 0.05,
) -> np.ndarray:
    """Triangulate valid quads without bridging large depth discontinuities."""

    depth_m = np.asarray(depth_m, dtype=np.float64)
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    if absolute_threshold_m < 0.0 or relative_threshold < 0.0:
        raise ValueError("depth thresholds must be non-negative")
    height, width = depth_m.shape
    if width < 2 or height < 2:
        return np.empty((0, 3), dtype=np.int32)

    d00 = depth_m[:-1, :-1]
    d01 = depth_m[:-1, 1:]
    d10 = depth_m[1:, :-1]
    d11 = depth_m[1:, 1:]
    depths = np.stack([d00, d01, d10, d11], axis=-1)
    valid = np.isfinite(depths).all(axis=-1) & (depths > 0.0).all(axis=-1)
    minimum = np.min(depths, axis=-1)
    maximum = np.max(depths, axis=-1)
    threshold = absolute_threshold_m + relative_threshold * minimum
    valid &= (maximum - minimum) <= threshold

    rows, columns = np.nonzero(valid)
    top_left = rows * width + columns
    top_right = top_left + 1
    bottom_left = top_left + width
    bottom_right = bottom_left + 1
    first = np.stack(
        [top_left, bottom_left, bottom_right],
        axis=1,
    )
    second = np.stack(
        [top_left, bottom_right, top_right],
        axis=1,
    )
    return np.concatenate([first, second], axis=0).astype(np.int32)
