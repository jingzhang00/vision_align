#!/usr/bin/env python3
"""Measure temporal continuity of an already prepared RGB-D episode.

This module deliberately performs no model inference.  It consumes only the
``rgb/``, ``depth_mm/``, and ``episode_data.npz`` artifacts written by the
RGB-D preparation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.rgbd_geometry import (  # noqa: E402
    backproject_z_depth,
    camera_points_to_world,
    depth_grid_triangles,
)


DEFAULT_FLOW_FB_THRESHOLD_PX = 1.5
DEFAULT_POINT_STRIDE = 4
DEFAULT_MAX_POINTS = 50_000
DEFAULT_DEPTH_EDGE_ABSOLUTE_M = 0.05
DEFAULT_DEPTH_EDGE_RELATIVE = 0.05


def distribution_summary(
    values: Iterable[float | int | None],
) -> dict[str, float | int | None]:
    """Return JSON-safe count, p50, p95, and maximum statistics."""

    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and np.isfinite(value)
        ],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "count": 0,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def frame_depth_statistics(
    depth_m: np.ndarray,
    *,
    absolute_threshold_m: float = DEFAULT_DEPTH_EDGE_ABSOLUTE_M,
    relative_threshold: float = DEFAULT_DEPTH_EDGE_RELATIVE,
) -> dict[str, float | int | None]:
    """Summarize valid depth and Isaac-compatible mesh topology."""

    depth_m = np.asarray(depth_m, dtype=np.float64)
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    valid_count = int(np.count_nonzero(valid))
    median_m = (
        float(np.median(depth_m[valid]))
        if valid_count
        else None
    )
    triangles = depth_grid_triangles(
        depth_m,
        absolute_threshold_m=absolute_threshold_m,
        relative_threshold=relative_threshold,
    )
    return {
        "valid_pixel_count": valid_count,
        "pixel_count": int(depth_m.size),
        "valid_rate": (
            float(valid_count / depth_m.size)
            if depth_m.size
            else None
        ),
        "median_depth_m": median_m,
        "triangle_count": int(len(triangles)),
    }


def depth_median_transition(
    previous_depth_m: np.ndarray,
    current_depth_m: np.ndarray,
) -> dict[str, float | None]:
    """Compare the positive finite depth medians of two frames."""

    previous_depth_m = np.asarray(previous_depth_m, dtype=np.float64)
    current_depth_m = np.asarray(current_depth_m, dtype=np.float64)
    if previous_depth_m.ndim != 2 or current_depth_m.ndim != 2:
        raise ValueError("depth arrays must be 2D")
    previous_valid = (
        np.isfinite(previous_depth_m) & (previous_depth_m > 0.0)
    )
    current_valid = (
        np.isfinite(current_depth_m) & (current_depth_m > 0.0)
    )
    previous_median = (
        float(np.median(previous_depth_m[previous_valid]))
        if np.any(previous_valid)
        else None
    )
    current_median = (
        float(np.median(current_depth_m[current_valid]))
        if np.any(current_valid)
        else None
    )
    if previous_median is None or current_median is None:
        return {
            "previous_median_m": previous_median,
            "current_median_m": current_median,
            "signed_jump_m": None,
            "absolute_jump_m": None,
            "ratio_current_over_previous": None,
            "symmetric_ratio": None,
        }
    signed_jump = current_median - previous_median
    ratio = current_median / previous_median
    return {
        "previous_median_m": previous_median,
        "current_median_m": current_median,
        "signed_jump_m": float(signed_jump),
        "absolute_jump_m": float(abs(signed_jump)),
        "ratio_current_over_previous": float(ratio),
        "symmetric_ratio": float(max(ratio, 1.0 / ratio)),
    }


def dense_optical_flow(
    source_gray: np.ndarray,
    destination_gray: np.ndarray,
) -> np.ndarray:
    """Estimate dense source-to-destination Farneback flow."""

    source_gray = np.asarray(source_gray)
    destination_gray = np.asarray(destination_gray)
    if (
        source_gray.ndim != 2
        or source_gray.shape != destination_gray.shape
    ):
        raise ValueError("grayscale frames must be matching 2D arrays")
    source_u8 = np.clip(source_gray, 0, 255).astype(np.uint8)
    destination_u8 = np.clip(
        destination_gray,
        0,
        255,
    ).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        source_u8,
        destination_u8,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    if flow is None or flow.shape != (*source_u8.shape, 2):
        raise RuntimeError("Farneback optical flow returned an invalid result")
    return np.asarray(flow, dtype=np.float32)


def flow_aligned_abs_relative_from_flows(
    previous_depth_m: np.ndarray,
    current_depth_m: np.ndarray,
    backward_flow: np.ndarray,
    *,
    forward_flow: np.ndarray | None = None,
    forward_backward_threshold_px: float | None = None,
) -> dict[str, float | int | None]:
    """Compare depth after warping the previous frame into the current view.

    ``backward_flow[y, x]`` maps a current-frame pixel to its previous-frame
    source coordinate.  The error is
    ``abs(current - warped_previous) / current``.  If both flows and a
    threshold are supplied, occluded or inconsistent correspondences are
    rejected using forward-backward consistency.
    """

    previous_depth_m = np.asarray(previous_depth_m, dtype=np.float64)
    current_depth_m = np.asarray(current_depth_m, dtype=np.float64)
    backward_flow = np.asarray(backward_flow, dtype=np.float32)
    if (
        previous_depth_m.ndim != 2
        or current_depth_m.shape != previous_depth_m.shape
    ):
        raise ValueError("depth arrays must be matching 2D arrays")
    height, width = previous_depth_m.shape
    if backward_flow.shape != (height, width, 2):
        raise ValueError("backward_flow must have shape [H, W, 2]")
    if forward_backward_threshold_px is not None:
        if (
            not np.isfinite(forward_backward_threshold_px)
            or forward_backward_threshold_px < 0.0
        ):
            raise ValueError(
                "forward_backward_threshold_px must be non-negative"
            )
        if forward_flow is None:
            raise ValueError(
                "forward_flow is required for forward-backward filtering"
            )
    if forward_flow is not None:
        forward_flow = np.asarray(forward_flow, dtype=np.float32)
        if forward_flow.shape != (height, width, 2):
            raise ValueError("forward_flow must have shape [H, W, 2]")

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    source_x = grid_x + backward_flow[..., 0]
    source_y = grid_y + backward_flow[..., 1]
    inside = (
        (source_x >= 0.0)
        & (source_x <= width - 1)
        & (source_y >= 0.0)
        & (source_y <= height - 1)
    )

    previous_valid = (
        np.isfinite(previous_depth_m) & (previous_depth_m > 0.0)
    )
    previous_values = np.where(
        previous_valid,
        previous_depth_m,
        0.0,
    ).astype(np.float32)
    previous_weights = previous_valid.astype(np.float32)
    warped_values = cv2.remap(
        previous_values,
        source_x,
        source_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_weights = cv2.remap(
        previous_weights,
        source_x,
        source_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_previous = np.divide(
        warped_values,
        warped_weights,
        out=np.zeros_like(warped_values),
        where=warped_weights > 1e-6,
    )
    current_valid = np.isfinite(current_depth_m) & (current_depth_m > 0.0)
    valid = (
        inside
        & current_valid
        & (warped_weights >= 1.0 - 1e-5)
        & np.isfinite(warped_previous)
        & (warped_previous > 0.0)
    )

    if (
        forward_backward_threshold_px is not None
        and forward_flow is not None
    ):
        sampled_forward_x = cv2.remap(
            forward_flow[..., 0],
            source_x,
            source_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        sampled_forward_y = cv2.remap(
            forward_flow[..., 1],
            source_x,
            source_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        consistency_error = np.hypot(
            backward_flow[..., 0] + sampled_forward_x,
            backward_flow[..., 1] + sampled_forward_y,
        )
        valid &= (
            np.isfinite(consistency_error)
            & (consistency_error <= forward_backward_threshold_px)
        )

    errors = (
        np.abs(current_depth_m[valid] - warped_previous[valid])
        / current_depth_m[valid]
    )
    summary = distribution_summary(errors)
    current_valid_count = int(np.count_nonzero(current_valid))
    valid_count = int(errors.size)
    return {
        "valid_pixel_count": valid_count,
        "pixel_count": int(current_depth_m.size),
        "current_valid_pixel_count": current_valid_count,
        "valid_rate": (
            float(valid_count / current_depth_m.size)
            if current_depth_m.size
            else None
        ),
        "coverage_of_current_valid": (
            float(valid_count / current_valid_count)
            if current_valid_count
            else None
        ),
        "p50": summary["p50"],
        "p95": summary["p95"],
    }


def _gray_at_depth_resolution(
    rgb_bgr: np.ndarray,
    depth_shape: tuple[int, int],
) -> np.ndarray:
    rgb_bgr = np.asarray(rgb_bgr)
    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
        raise ValueError("rgb_bgr must have shape [H, W, 3]")
    height, width = depth_shape
    if rgb_bgr.shape[:2] != (height, width):
        interpolation = (
            cv2.INTER_AREA
            if (
                rgb_bgr.shape[0] >= height
                and rgb_bgr.shape[1] >= width
            )
            else cv2.INTER_LINEAR
        )
        rgb_bgr = cv2.resize(
            rgb_bgr,
            (width, height),
            interpolation=interpolation,
        )
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)


def flow_aligned_abs_relative(
    previous_rgb_bgr: np.ndarray,
    current_rgb_bgr: np.ndarray,
    previous_depth_m: np.ndarray,
    current_depth_m: np.ndarray,
    *,
    forward_backward_threshold_px: float | None = (
        DEFAULT_FLOW_FB_THRESHOLD_PX
    ),
) -> dict[str, float | int | None]:
    """Estimate optical flow and return aligned depth error percentiles."""

    previous_depth_m = np.asarray(previous_depth_m, dtype=np.float64)
    current_depth_m = np.asarray(current_depth_m, dtype=np.float64)
    if (
        previous_depth_m.ndim != 2
        or current_depth_m.shape != previous_depth_m.shape
    ):
        raise ValueError("depth arrays must be matching 2D arrays")
    previous_gray = _gray_at_depth_resolution(
        previous_rgb_bgr,
        previous_depth_m.shape,
    )
    current_gray = _gray_at_depth_resolution(
        current_rgb_bgr,
        current_depth_m.shape,
    )
    backward_flow = dense_optical_flow(current_gray, previous_gray)
    forward_flow = (
        dense_optical_flow(previous_gray, current_gray)
        if forward_backward_threshold_px is not None
        else None
    )
    return flow_aligned_abs_relative_from_flows(
        previous_depth_m,
        current_depth_m,
        backward_flow,
        forward_flow=forward_flow,
        forward_backward_threshold_px=forward_backward_threshold_px,
    )


def world_points_from_depth(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    camera_pose_cv_xyzw: np.ndarray,
    *,
    point_stride: int = DEFAULT_POINT_STRIDE,
    max_points: int | None = DEFAULT_MAX_POINTS,
) -> np.ndarray:
    """Backproject deterministic depth samples into world coordinates."""

    if point_stride <= 0:
        raise ValueError("point_stride must be positive")
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive or None")
    points_camera = backproject_z_depth(depth_m, intrinsic)
    sampled_camera = points_camera[::point_stride, ::point_stride]
    sampled_world = camera_points_to_world(
        sampled_camera,
        np.asarray(camera_pose_cv_xyzw, dtype=np.float64),
    ).reshape(-1, 3)
    sampled_world = sampled_world[
        np.isfinite(sampled_world).all(axis=1)
    ]
    if max_points is not None and len(sampled_world) > max_points:
        indices = np.linspace(
            0,
            len(sampled_world) - 1,
            max_points,
            dtype=np.int64,
        )
        sampled_world = sampled_world[indices]
    return np.asarray(sampled_world, dtype=np.float64)


def symmetric_nearest_neighbor_statistics(
    previous_points_world: np.ndarray,
    current_points_world: np.ndarray,
) -> dict[str, float | int | None]:
    """Return percentiles of both directed nearest-neighbor distances."""

    previous = np.asarray(previous_points_world, dtype=np.float64)
    current = np.asarray(current_points_world, dtype=np.float64)
    if (
        previous.ndim != 2
        or current.ndim != 2
        or previous.shape[1:] != (3,)
        or current.shape[1:] != (3,)
    ):
        raise ValueError("point clouds must have shape [N, 3]")
    previous = previous[np.isfinite(previous).all(axis=1)]
    current = current[np.isfinite(current).all(axis=1)]
    if len(previous) == 0 or len(current) == 0:
        return {
            "previous_point_count": int(len(previous)),
            "current_point_count": int(len(current)),
            "distance_sample_count": 0,
            "p50_m": None,
            "p95_m": None,
        }
    previous_tree = cKDTree(previous)
    current_tree = cKDTree(current)
    previous_to_current = current_tree.query(previous, k=1)[0]
    current_to_previous = previous_tree.query(current, k=1)[0]
    distances = np.concatenate(
        [previous_to_current, current_to_previous]
    )
    summary = distribution_summary(distances)
    return {
        "previous_point_count": int(len(previous)),
        "current_point_count": int(len(current)),
        "distance_sample_count": int(len(distances)),
        "p50_m": summary["p50"],
        "p95_m": summary["p95"],
    }


def _read_rgb(path: Path) -> np.ndarray:
    rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(f"Failed to read RGB frame: {path}")
    return rgb


def _read_depth_m(
    path: Path,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise ValueError(f"Failed to read depth frame: {path}")
    if depth_mm.dtype != np.uint16:
        raise ValueError(f"Depth frame is not uint16 millimeters: {path}")
    if depth_mm.shape != expected_shape:
        raise ValueError(
            f"Depth frame {path} has shape {depth_mm.shape}, expected "
            f"{expected_shape}"
        )
    return depth_mm.astype(np.float64) / 1000.0


def _prepared_paths(
    prepared_root: Path,
    frame_count: int,
) -> tuple[list[Path], list[Path]]:
    rgb_root = prepared_root / "rgb"
    depth_root = prepared_root / "depth_mm"
    rgb_paths = [rgb_root / f"{index:05d}.png" for index in range(frame_count)]
    depth_paths = [
        depth_root / f"{index:05d}.png" for index in range(frame_count)
    ]
    missing = [
        str(path)
        for path in (*rgb_paths, *depth_paths)
        if not path.is_file()
    ]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3})"
        raise FileNotFoundError(f"Missing prepared frames: {preview}{suffix}")
    return rgb_paths, depth_paths


def _load_episode_geometry(
    archive_path: Path,
) -> dict[str, object]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {
            "fps",
            "source_size",
            "mesh_size",
            "frame_indices",
            "mesh_intrinsic",
            "camera_pose_cv_xyzw",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"Prepared archive is missing: {missing}")
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        mesh_size_array = np.asarray(archive["mesh_size"], dtype=np.int64)
        source_size_array = np.asarray(
            archive["source_size"],
            dtype=np.int64,
        )
        intrinsic = np.asarray(
            archive["mesh_intrinsic"],
            dtype=np.float64,
        )
        camera_poses = np.asarray(
            archive["camera_pose_cv_xyzw"],
            dtype=np.float64,
        )
        fps = float(archive["fps"])
        episode_index = (
            int(archive["episode_index"])
            if "episode_index" in archive.files
            else None
        )
    if frame_indices.ndim != 1 or len(frame_indices) == 0:
        raise ValueError("frame_indices must be a non-empty 1D array")
    if mesh_size_array.shape != (2,) or np.any(mesh_size_array <= 0):
        raise ValueError("mesh_size must contain positive width and height")
    if source_size_array.shape != (2,) or np.any(source_size_array <= 0):
        raise ValueError("source_size must contain positive width and height")
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise ValueError("mesh_intrinsic must be a valid finite 3x3 matrix")
    if (
        camera_poses.shape != (len(frame_indices), 7)
        or not np.isfinite(camera_poses).all()
    ):
        raise ValueError(
            "camera_pose_cv_xyzw must be finite with shape [frames, 7]"
        )
    quaternion_norms = np.linalg.norm(camera_poses[:, 3:7], axis=1)
    if np.any(quaternion_norms <= 1e-8):
        raise ValueError("camera poses contain an invalid quaternion")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    return {
        "episode_index": episode_index,
        "fps": fps,
        "source_size": tuple(int(value) for value in source_size_array),
        "mesh_size": tuple(int(value) for value in mesh_size_array),
        "frame_indices": frame_indices,
        "mesh_intrinsic": intrinsic,
        "camera_poses": camera_poses,
    }


def _pair_jump(
    previous: float | int | None,
    current: float | int | None,
) -> dict[str, float | int | None]:
    if previous is None or current is None:
        return {
            "previous": previous,
            "current": current,
            "signed_jump": None,
            "absolute_jump": None,
            "ratio_current_over_previous": None,
        }
    signed = current - previous
    ratio = (
        float(current / previous)
        if previous != 0
        else None
    )
    if isinstance(previous, int) and isinstance(current, int):
        signed_value: float | int = int(signed)
        absolute_value: float | int = int(abs(signed))
    else:
        signed_value = float(signed)
        absolute_value = float(abs(signed))
    return {
        "previous": previous,
        "current": current,
        "signed_jump": signed_value,
        "absolute_jump": absolute_value,
        "ratio_current_over_previous": ratio,
    }


def _report_summary(
    frames: list[dict[str, object]],
    pairs: list[dict[str, object]],
) -> dict[str, object]:
    def pair_values(section: str, key: str) -> list[float | int | None]:
        return [
            pair[section][key]  # type: ignore[index]
            for pair in pairs
        ]

    return {
        "frame_valid_rate": distribution_summary(
            frame["valid_rate"] for frame in frames
        ),
        "frame_triangle_count": distribution_summary(
            frame["triangle_count"] for frame in frames
        ),
        "adjacent_depth_median_absolute_jump_m": distribution_summary(
            pair_values("depth_median", "absolute_jump_m")
        ),
        "adjacent_depth_median_symmetric_ratio": distribution_summary(
            pair_values("depth_median", "symmetric_ratio")
        ),
        "adjacent_flow_abs_relative_p50": distribution_summary(
            pair_values("flow_aligned_abs_relative", "p50")
        ),
        "adjacent_flow_abs_relative_p95": distribution_summary(
            pair_values("flow_aligned_abs_relative", "p95")
        ),
        "adjacent_world_symmetric_nn_p50_m": distribution_summary(
            pair_values("world_symmetric_nearest_neighbor", "p50_m")
        ),
        "adjacent_world_symmetric_nn_p95_m": distribution_summary(
            pair_values("world_symmetric_nearest_neighbor", "p95_m")
        ),
        "adjacent_valid_rate_absolute_jump": distribution_summary(
            pair_values("valid_rate", "absolute_jump")
        ),
        "adjacent_triangle_count_absolute_jump": distribution_summary(
            pair_values("triangle_count", "absolute_jump")
        ),
    }


def analyze_prepared_episode(
    prepared_root: Path,
    *,
    point_stride: int = DEFAULT_POINT_STRIDE,
    max_points: int | None = DEFAULT_MAX_POINTS,
    flow_forward_backward_threshold_px: float | None = (
        DEFAULT_FLOW_FB_THRESHOLD_PX
    ),
    depth_edge_absolute_m: float = DEFAULT_DEPTH_EDGE_ABSOLUTE_M,
    depth_edge_relative: float = DEFAULT_DEPTH_EDGE_RELATIVE,
) -> dict[str, object]:
    """Analyze a prepared episode without loading a depth model."""

    prepared_root = Path(prepared_root).resolve()
    if point_stride <= 0:
        raise ValueError("point_stride must be positive")
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive or None")
    if depth_edge_absolute_m < 0.0 or depth_edge_relative < 0.0:
        raise ValueError("depth edge thresholds must be non-negative")

    geometry = _load_episode_geometry(
        prepared_root / "episode_data.npz"
    )
    frame_indices = geometry["frame_indices"]
    camera_poses = geometry["camera_poses"]
    intrinsic = geometry["mesh_intrinsic"]
    assert isinstance(frame_indices, np.ndarray)
    assert isinstance(camera_poses, np.ndarray)
    assert isinstance(intrinsic, np.ndarray)
    frame_count = len(frame_indices)
    mesh_width, mesh_height = geometry["mesh_size"]  # type: ignore[misc]
    rgb_paths, depth_paths = _prepared_paths(prepared_root, frame_count)

    frames: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    previous_rgb: np.ndarray | None = None
    previous_depth: np.ndarray | None = None
    previous_points: np.ndarray | None = None
    previous_frame_stats: dict[str, float | int | None] | None = None

    for local_index, (rgb_path, depth_path) in enumerate(
        zip(rgb_paths, depth_paths, strict=True)
    ):
        rgb = _read_rgb(rgb_path)
        depth = _read_depth_m(
            depth_path,
            expected_shape=(mesh_height, mesh_width),
        )
        frame_stats = frame_depth_statistics(
            depth,
            absolute_threshold_m=depth_edge_absolute_m,
            relative_threshold=depth_edge_relative,
        )
        frame_report: dict[str, object] = {
            "local_index": local_index,
            "source_frame_index": int(frame_indices[local_index]),
            **frame_stats,
        }
        frames.append(frame_report)
        points = world_points_from_depth(
            depth,
            intrinsic,
            camera_poses[local_index],
            point_stride=point_stride,
            max_points=max_points,
        )

        if (
            previous_rgb is not None
            and previous_depth is not None
            and previous_points is not None
            and previous_frame_stats is not None
        ):
            depth_transition = depth_median_transition(
                previous_depth,
                depth,
            )
            flow_stats = flow_aligned_abs_relative(
                previous_rgb,
                rgb,
                previous_depth,
                depth,
                forward_backward_threshold_px=(
                    flow_forward_backward_threshold_px
                ),
            )
            nn_stats = symmetric_nearest_neighbor_statistics(
                previous_points,
                points,
            )
            pairs.append(
                {
                    "previous_local_index": local_index - 1,
                    "current_local_index": local_index,
                    "previous_source_frame_index": int(
                        frame_indices[local_index - 1]
                    ),
                    "current_source_frame_index": int(
                        frame_indices[local_index]
                    ),
                    "depth_median": depth_transition,
                    "flow_aligned_abs_relative": flow_stats,
                    "world_symmetric_nearest_neighbor": nn_stats,
                    "valid_rate": _pair_jump(
                        previous_frame_stats["valid_rate"],
                        frame_stats["valid_rate"],
                    ),
                    "triangle_count": _pair_jump(
                        previous_frame_stats["triangle_count"],
                        frame_stats["triangle_count"],
                    ),
                }
            )

        previous_rgb = rgb
        previous_depth = depth
        previous_points = points
        previous_frame_stats = frame_stats

    report: dict[str, object] = {
        "schema_version": 1,
        "method": "prepared_rgbd_temporal_continuity",
        "prepared_root": str(prepared_root),
        "episode_index": geometry["episode_index"],
        "fps": geometry["fps"],
        "source_size": list(geometry["source_size"]),  # type: ignore[arg-type]
        "mesh_size": list(geometry["mesh_size"]),  # type: ignore[arg-type]
        "frame_count": frame_count,
        "adjacent_pair_count": len(pairs),
        "settings": {
            "point_stride": point_stride,
            "max_points_per_frame": max_points,
            "flow_method": "Farneback",
            "flow_forward_backward_threshold_px": (
                flow_forward_backward_threshold_px
            ),
            "depth_edge_absolute_m": depth_edge_absolute_m,
            "depth_edge_relative": depth_edge_relative,
        },
        "definitions": {
            "depth_median_jump": (
                "difference between positive finite frame medians"
            ),
            "depth_median_symmetric_ratio": (
                "max(current/previous, previous/current); ideal is 1"
            ),
            "flow_aligned_abs_relative": (
                "abs(current_depth - backward-warped_previous_depth) "
                "/ current_depth"
            ),
            "world_symmetric_nearest_neighbor": (
                "concatenated previous-to-current and "
                "current-to-previous world-space NN distances"
            ),
            "valid_depth": "finite depth strictly greater than zero",
            "triangle_count": (
                "two triangles per valid 2x2 depth quad passing the "
                "configured absolute-plus-relative edge threshold"
            ),
        },
        "summary": _report_summary(frames, pairs),
        "frames": frames,
        "adjacent_pairs": pairs,
    }
    return report


def write_report(report: dict[str, object], output_path: Path) -> Path:
    """Atomically write a continuity report as UTF-8 JSON."""

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure temporal continuity from an existing prepared "
            "rgb/depth_mm/episode_data.npz directory; no model is loaded."
        )
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help=(
            "Directory containing rgb/, depth_mm/, and episode_data.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "JSON destination (default: "
            "<prepared>/vda_continuity.json)"
        ),
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=DEFAULT_POINT_STRIDE,
        help="Use every Nth pixel in each direction for world NN (default: 4)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help=(
            "Deterministic per-frame point cap after striding "
            "(default: 50000)"
        ),
    )
    parser.add_argument(
        "--flow-fb-threshold-px",
        type=float,
        default=DEFAULT_FLOW_FB_THRESHOLD_PX,
        help=(
            "Reject flow pairs whose forward-backward error exceeds this "
            "many pixels (default: 1.5)"
        ),
    )
    parser.add_argument(
        "--no-flow-fb-check",
        action="store_true",
        help="Disable forward-backward flow consistency filtering",
    )
    parser.add_argument(
        "--depth-edge-absolute",
        type=float,
        default=DEFAULT_DEPTH_EDGE_ABSOLUTE_M,
        metavar="METERS",
        help="Absolute component of the mesh edge threshold (default: 0.05)",
    )
    parser.add_argument(
        "--depth-edge-relative",
        type=float,
        default=DEFAULT_DEPTH_EDGE_RELATIVE,
        help="Relative component of the mesh edge threshold (default: 0.05)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    threshold = (
        None
        if args.no_flow_fb_check
        else args.flow_fb_threshold_px
    )
    report = analyze_prepared_episode(
        args.prepared,
        point_stride=args.point_stride,
        max_points=args.max_points,
        flow_forward_backward_threshold_px=threshold,
        depth_edge_absolute_m=args.depth_edge_absolute,
        depth_edge_relative=args.depth_edge_relative,
    )
    output_path = (
        args.output
        if args.output is not None
        else args.prepared / "vda_continuity.json"
    )
    written = write_report(report, output_path)
    print(
        json.dumps(
            {
                "output": str(written),
                "frame_count": report["frame_count"],
                "adjacent_pair_count": report["adjacent_pair_count"],
                "summary": report["summary"],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
