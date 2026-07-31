"""Geometry helpers for the independent RGB-D Isaac repaint experiment."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


CV_TO_USD_CAMERA_AXES = np.diag([1.0, -1.0, -1.0])


def center_patch_median_depth(
    depth_m: np.ndarray,
    *,
    half_fraction: float,
) -> float:
    """Return robust positive depth around the geometric image center."""

    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or min(depth.shape) <= 0:
        raise ValueError("depth_m must be a non-empty 2D image")
    if (
        not np.isfinite(half_fraction)
        or half_fraction <= 0.0
        or half_fraction > 0.5
    ):
        raise ValueError("half_fraction must lie in (0, 0.5]")
    height, width = depth.shape
    radius_x = max(1, int(round(width * half_fraction)))
    radius_y = max(1, int(round(height * half_fraction)))
    center_x = width // 2
    center_y = height // 2
    patch = depth[
        max(0, center_y - radius_y) : min(
            height,
            center_y + radius_y + 1,
        ),
        max(0, center_x - radius_x) : min(
            width,
            center_x + radius_x + 1,
        ),
    ]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise ValueError("center depth patch has no positive finite values")
    return float(np.median(valid))


def temporal_median_depth(
    depth_m: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Median-filter a finite positive depth series with edge replication."""

    depth = np.asarray(depth_m, dtype=np.float64)
    if (
        depth.ndim != 1
        or len(depth) == 0
        or not np.isfinite(depth).all()
        or np.any(depth <= 0.0)
    ):
        raise ValueError("depth_m must be a positive finite vector")
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    radius = window // 2
    padded = np.pad(depth, (radius, radius), mode="edge")
    filtered = np.empty_like(depth)
    for frame_id in range(len(depth)):
        filtered[frame_id] = np.median(
            padded[frame_id : frame_id + window]
        )
    return filtered


def image_plane_corners_camera_cv(
    intrinsic: np.ndarray,
    *,
    image_size: tuple[int, int],
    distance_m: float,
) -> np.ndarray:
    """Return TL, TR, BL, BR image-edge corners on a CV camera plane."""

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("distance_m must be positive and finite")
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("focal lengths must be positive")
    pixels = np.asarray(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [0.0, float(height)],
            [float(width), float(height)],
        ],
        dtype=np.float64,
    )
    corners = np.empty((4, 3), dtype=np.float64)
    corners[:, 0] = (pixels[:, 0] - cx) * distance_m / fx
    corners[:, 1] = (pixels[:, 1] - cy) * distance_m / fy
    corners[:, 2] = distance_m
    return corners


def minimum_image_plane_distance_for_foreground(
    *,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    plane_camera_pose_cv_xyzw: np.ndarray,
    render_camera_pose_cv_xyzw: np.ndarray,
    foreground_points_render_camera: np.ndarray,
    clearance_m: float,
) -> float:
    """Find the plane distance that keeps every corner behind foreground."""

    plane_pose = np.asarray(
        plane_camera_pose_cv_xyzw,
        dtype=np.float64,
    )
    render_pose = np.asarray(
        render_camera_pose_cv_xyzw,
        dtype=np.float64,
    )
    foreground = np.asarray(
        foreground_points_render_camera,
        dtype=np.float64,
    )
    if (
        plane_pose.ndim != 2
        or plane_pose.shape[1] != 7
        or render_pose.shape != plane_pose.shape
    ):
        raise ValueError("camera poses must have matching [frames, 7] shapes")
    if (
        foreground.ndim != 3
        or foreground.shape[0] != len(plane_pose)
        or foreground.shape[2] != 3
    ):
        raise ValueError(
            "foreground points must have shape [frames, points, 3]"
        )
    if (
        not np.isfinite(plane_pose).all()
        or not np.isfinite(render_pose).all()
        or not np.isfinite(foreground).all()
    ):
        raise ValueError("camera poses and foreground points must be finite")
    if not np.isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError("clearance_m must be finite and non-negative")

    unit_corners = image_plane_corners_camera_cv(
        intrinsic,
        image_size=image_size,
        distance_m=1.0,
    )
    plane_rotation = Rotation.from_quat(
        plane_pose[:, 3:7]
    ).as_matrix()
    render_rotation = Rotation.from_quat(
        render_pose[:, 3:7]
    ).as_matrix()
    required_distances: list[float] = []
    for frame_id in range(len(plane_pose)):
        render_from_world = render_rotation[frame_id].T
        center_offset_camera = (
            render_from_world
            @ (plane_pose[frame_id, :3] - render_pose[frame_id, :3])
        )
        corner_directions_camera = (
            unit_corners
            @ plane_rotation[frame_id].T
            @ render_rotation[frame_id]
        )
        minimum_corner_z_per_meter = float(
            corner_directions_camera[:, 2].min()
        )
        if minimum_corner_z_per_meter <= 1e-8:
            raise ValueError(
                "Part of the source video plane faces behind the render "
                "camera; no positive plane distance can cover this view"
            )
        positive_foreground_z = foreground[frame_id, :, 2]
        positive_foreground_z = positive_foreground_z[
            positive_foreground_z > 0.0
        ]
        if positive_foreground_z.size == 0:
            continue
        target_z = float(positive_foreground_z.max() + clearance_m)
        required_distances.append(
            max(
                0.0,
                (target_z - center_offset_camera[2])
                / minimum_corner_z_per_meter,
            )
        )
    if not required_distances or max(required_distances) <= 0.0:
        raise ValueError("No positive foreground depth was available")
    return float(max(required_distances))


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


def camera_cv_pose_to_usd_wxyz(
    camera_pose_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a CV-axis camera pose to an Isaac/USD-axis camera pose."""

    camera_pose_xyzw = np.asarray(camera_pose_xyzw, dtype=np.float64)
    if camera_pose_xyzw.shape != (7,):
        raise ValueError("camera_pose_xyzw must have shape (7,)")
    rotation_cv = Rotation.from_quat(
        camera_pose_xyzw[3:7]
    ).as_matrix()
    rotation_usd = rotation_cv @ CV_TO_USD_CAMERA_AXES
    quaternion_xyzw = Rotation.from_matrix(rotation_usd).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    return camera_pose_xyzw[:3].copy(), quaternion_wxyz


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
