#!/usr/bin/env python3
"""Replay a prepared Video Depth Anything scene and G1 trajectory in Isaac.

Each prepared RGB-D frame is reconstructed as a textured USD mesh using its
prepared virtual source-camera pose while the distinct capture camera follows
the frame-aligned G1 trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ISAAC_NO_LOG_ARGS = (
    "--/log/enabled=false",
    "--/log/file=/dev/null",
    "--/structuredLog/enable=false",
    "--/telemetry/enableAnonymousData=false",
    "--/telemetry/log/file=/dev/null",
    "--/app/enableStdoutOutput=0",
)
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.isaac_g1 import import_g1, isaac_joint_order  # noqa: E402
from repaint.lighting_profile import (  # noqa: E402
    DEFAULT_COMPOSITE_PROFILE,
    DEFAULT_ISAAC_PROFILE,
    load_lighting_profile,
)
from repaint.rgbd_geometry import (  # noqa: E402
    backproject_z_depth,
    camera_points_to_world,
    depth_grid_triangles,
    grid_uvs,
)
from repaint.robot_camera import (  # noqa: E402
    CAMERA_LINK_FROM_CV_OPTICAL,
    HEAD_CAMERA_PARENT_FRAME,
    HEAD_CAM_OFFSET_POS_ACTIVE,
    HEAD_CAM_OFFSET_POS_BASE,
    HEAD_CAM_OFFSET_ROT_WXYZ,
    HEAD_CAM_RESET_GUARD_X_M,
    head_camera_poses_world,
    robot_frame_points_in_camera_cv,
)
D435_REFERENCE_SIZE = (640, 480)
D435_INTRINSIC = np.asarray(
    [
        [607.212158203125, 0.0, 311.9999084472656],
        [0.0, 607.0084838867188, 261.152099609375],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
D435_DISTORTION = np.zeros(12, dtype=np.float64)
ROBOT_HAND_FRAMES = ("left_hand_base_link", "right_hand_base_link")
HAND_CAPTURE_WARNING_THRESHOLD = 0.8
JOINT_COMMIT_ATOL_RAD = 1.0e-4
ROOT_POSITION_COMMIT_ATOL_M = 1.0e-4
ROOT_ORIENTATION_COMMIT_ATOL_RAD = 1.0e-4
BATCH_REPORT_SCHEMA_VERSION = 8


@dataclass(frozen=True)
class PreparedScene:
    """Validated scene and robot data needed by the combined renderer."""

    root: Path
    source_dataset_kind: str
    rgb_paths: tuple[Path, ...]
    depth_paths: tuple[Path, ...]
    frame_indices: np.ndarray
    fps: float
    source_size: tuple[int, int]
    mesh_size: tuple[int, int]
    intrinsic: np.ndarray
    mesh_intrinsic: np.ndarray
    camera_poses_cv_xyzw: np.ndarray
    scene_translation_world_xyz_m: np.ndarray
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    root_poses_xyzw: np.ndarray
    g1_urdf: Path

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)


@dataclass(frozen=True)
class SceneScan:
    """Geometry bounds and temporal diagnostics collected before Isaac starts."""

    bounds_min: np.ndarray
    bounds_max: np.ndarray
    valid_vertex_counts: np.ndarray
    triangle_counts: np.ndarray
    median_depths_m: np.ndarray


@dataclass(frozen=True)
class RunOptions:
    """Validated batch or interactive execution settings."""

    interactive: bool
    loop: bool
    playback_fps: float
    output_video: Path | None
    report_path: Path | None


def _numbered_png_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Prepared directory does not exist: {directory}")
    paths = tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".png"
                and path.stem.isdigit()
            ),
            key=lambda path: int(path.stem),
        )
    )
    if not paths:
        raise FileNotFoundError(f"No numbered PNG files in {directory}")
    indices = [int(path.stem) for path in paths]
    if indices != list(range(len(paths))):
        raise ValueError(
            f"Numbered PNG files in {directory} must be contiguous from 00000"
        )
    return paths


def _validated_size(value: np.ndarray, *, name: str) -> tuple[int, int]:
    array = np.asarray(value)
    if array.shape != (2,):
        raise ValueError(f"{name} must contain WIDTH, HEIGHT")
    size = tuple(int(item) for item in array.tolist())
    if min(size) <= 1:
        raise ValueError(f"{name} dimensions must both exceed one pixel")
    return size


def load_prepared_scene(prepared_root: Path) -> PreparedScene:
    """Load and validate frame-aligned VDA scene and G1 trajectory data."""

    root = prepared_root.expanduser().resolve()
    archive_path = root / "episode_data.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Prepared archive does not exist: {archive_path}")

    with np.load(archive_path, allow_pickle=False) as archive:
        required = {
            "fps",
            "source_dataset_kind",
            "source_size",
            "mesh_size",
            "frame_indices",
            "camera_pose_cv_xyzw",
            "scene_translation_world_xyz_m",
            "intrinsic",
            "mesh_intrinsic",
            "joint_names",
            "joint_positions",
            "root_pose_xyzw",
            "g1_urdf",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"Prepared archive is missing: {missing}")

        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64).copy()
        if frame_indices.ndim != 1 or len(frame_indices) == 0:
            raise ValueError("frame_indices must be a non-empty vector")
        if np.any(np.diff(frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")

        fps = float(archive["fps"])
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("Prepared fps must be positive and finite")

        source_dataset_kind_value = np.asarray(
            archive["source_dataset_kind"]
        )
        if source_dataset_kind_value.shape != ():
            raise ValueError("source_dataset_kind must be a scalar")
        source_dataset_kind = str(source_dataset_kind_value.item())
        if source_dataset_kind not in {"egodex", "nymeria"}:
            raise ValueError(
                f"Unsupported prepared source dataset: {source_dataset_kind!r}"
            )
        source_size = _validated_size(archive["source_size"], name="source_size")
        mesh_size = _validated_size(archive["mesh_size"], name="mesh_size")
        intrinsic = np.asarray(archive["intrinsic"], dtype=np.float64).copy()
        mesh_intrinsic = np.asarray(
            archive["mesh_intrinsic"],
            dtype=np.float64,
        ).copy()
        for name, calibration in (
            ("intrinsic", intrinsic),
            ("mesh_intrinsic", mesh_intrinsic),
        ):
            if (
                calibration.shape != (3, 3)
                or not np.isfinite(calibration).all()
                or calibration[0, 0] <= 0.0
                or calibration[1, 1] <= 0.0
            ):
                raise ValueError(f"{name} must be a valid finite pinhole matrix")

        camera_poses = np.asarray(
            archive["camera_pose_cv_xyzw"],
            dtype=np.float64,
        ).copy()
        if camera_poses.shape != (len(frame_indices), 7):
            raise ValueError(
                "camera_pose_cv_xyzw must have shape [frame_count, 7]"
            )
        if not np.isfinite(camera_poses).all():
            raise ValueError("camera_pose_cv_xyzw must be finite")
        quaternion_norms = np.linalg.norm(camera_poses[:, 3:7], axis=1)
        if np.any(quaternion_norms <= 1e-8):
            raise ValueError("camera_pose_cv_xyzw contains a zero quaternion")
        camera_poses[:, 3:7] /= quaternion_norms[:, None]
        scene_translation_world_xyz_m = np.asarray(
            archive["scene_translation_world_xyz_m"],
            dtype=np.float64,
        ).copy()
        if (
            scene_translation_world_xyz_m.shape != (3,)
            or not np.isfinite(scene_translation_world_xyz_m).all()
        ):
            raise ValueError(
                "scene_translation_world_xyz_m must be finite XYZ"
            )

        joint_names_array = np.asarray(archive["joint_names"])
        if joint_names_array.ndim != 1 or len(joint_names_array) == 0:
            raise ValueError("joint_names must be a non-empty vector")
        joint_names = tuple(str(name) for name in joint_names_array.tolist())
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("joint_names contains duplicates")

        joint_positions = np.asarray(
            archive["joint_positions"],
            dtype=np.float64,
        ).copy()
        if joint_positions.shape != (len(frame_indices), len(joint_names)):
            raise ValueError(
                "joint_positions must have shape [frame_count, joint_count]"
            )
        if not np.isfinite(joint_positions).all():
            raise ValueError("joint_positions must be finite")

        root_poses = np.asarray(
            archive["root_pose_xyzw"],
            dtype=np.float64,
        ).copy()
        if root_poses.shape != (len(frame_indices), 7):
            raise ValueError("root_pose_xyzw must have shape [frame_count, 7]")
        if not np.isfinite(root_poses).all():
            raise ValueError("root_pose_xyzw must be finite")
        root_quaternion_norms = np.linalg.norm(root_poses[:, 3:7], axis=1)
        if np.any(root_quaternion_norms <= 1e-8):
            raise ValueError("root_pose_xyzw contains a zero quaternion")
        root_poses[:, 3:7] /= root_quaternion_norms[:, None]

        g1_urdf_value = np.asarray(archive["g1_urdf"])
        if g1_urdf_value.shape != ():
            raise ValueError("g1_urdf must be a scalar path")
        g1_urdf = Path(str(g1_urdf_value.item())).expanduser().resolve()
        if not g1_urdf.is_file():
            raise FileNotFoundError(f"G1 URDF does not exist: {g1_urdf}")

    rgb_paths = _numbered_png_paths(root / "rgb")
    depth_paths = _numbered_png_paths(root / "depth_mm")
    frame_count = len(frame_indices)
    if len(rgb_paths) != frame_count or len(depth_paths) != frame_count:
        raise ValueError(
            "Prepared RGB/depth counts must both match frame_indices: "
            f"frames={frame_count}, rgb={len(rgb_paths)}, "
            f"depth={len(depth_paths)}"
        )

    first_rgb = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise ValueError(f"Could not read prepared RGB frame: {rgb_paths[0]}")
    if first_rgb.shape[:2] != (source_size[1], source_size[0]):
        raise ValueError(
            f"Prepared RGB size {first_rgb.shape[1]}x{first_rgb.shape[0]} "
            f"does not match source_size {source_size}"
        )

    return PreparedScene(
        root=root,
        source_dataset_kind=source_dataset_kind,
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        frame_indices=frame_indices,
        fps=fps,
        source_size=source_size,
        mesh_size=mesh_size,
        intrinsic=intrinsic,
        mesh_intrinsic=mesh_intrinsic,
        camera_poses_cv_xyzw=camera_poses,
        scene_translation_world_xyz_m=scene_translation_world_xyz_m,
        joint_names=joint_names,
        joint_positions=joint_positions,
        root_poses_xyzw=root_poses,
        g1_urdf=g1_urdf,
    )


def _load_depth_m(path: Path, mesh_size: tuple[int, int]) -> np.ndarray:
    depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise ValueError(f"Could not read prepared depth frame: {path}")
    if depth_mm.dtype != np.uint16:
        raise ValueError(
            f"Prepared depth must be uint16 millimeters, got "
            f"{depth_mm.dtype} in {path}"
        )
    if depth_mm.shape != (mesh_size[1], mesh_size[0]):
        raise ValueError(
            f"Prepared depth size {depth_mm.shape[::-1]} does not match "
            f"mesh_size {mesh_size}: {path}"
        )
    return depth_mm.astype(np.float64) / 1000.0


def scan_scene(
    scene: PreparedScene,
    *,
    camera_poses_cv_xyzw: np.ndarray,
    absolute_threshold_m: float,
    relative_threshold: float,
) -> SceneScan:
    """Validate all depth frames and collect bounds/topology diagnostics."""

    if not np.isfinite(absolute_threshold_m) or absolute_threshold_m < 0.0:
        raise ValueError("--depth-edge-absolute must be finite and non-negative")
    if not np.isfinite(relative_threshold) or relative_threshold < 0.0:
        raise ValueError("--depth-edge-relative must be finite and non-negative")

    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    valid_vertex_counts = np.empty(scene.frame_count, dtype=np.int64)
    triangle_counts = np.empty(scene.frame_count, dtype=np.int64)
    median_depths_m = np.empty(scene.frame_count, dtype=np.float64)

    for frame_id, depth_path in enumerate(scene.depth_paths):
        depth_m = _load_depth_m(depth_path, scene.mesh_size)
        points_camera = backproject_z_depth(depth_m, scene.mesh_intrinsic)
        points_world = camera_points_to_world(
            points_camera,
            camera_poses_cv_xyzw[frame_id],
        )
        valid = np.isfinite(points_world).all(axis=-1)
        points_world[valid] += scene.scene_translation_world_xyz_m
        valid_points = points_world[valid]
        if len(valid_points) == 0:
            raise ValueError(f"Depth frame has no valid points: {depth_path}")
        valid_depths = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
        bounds_min = np.minimum(bounds_min, valid_points.min(axis=0))
        bounds_max = np.maximum(bounds_max, valid_points.max(axis=0))
        valid_vertex_counts[frame_id] = len(valid_points)
        triangle_counts[frame_id] = len(
            depth_grid_triangles(
                depth_m,
                absolute_threshold_m=absolute_threshold_m,
                relative_threshold=relative_threshold,
            )
        )
        median_depths_m[frame_id] = float(np.median(valid_depths))

    if not np.isfinite(bounds_min).all() or not np.isfinite(bounds_max).all():
        raise ValueError("Could not derive finite scene bounds")
    return SceneScan(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        valid_vertex_counts=valid_vertex_counts,
        triangle_counts=triangle_counts,
        median_depths_m=median_depths_m,
    )


def _camera_projection(
    intrinsic: np.ndarray,
    *,
    width: int,
    height: int,
    horizontal_aperture: float = 2.0955,
) -> tuple[float, float, float]:
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or min(fx, fy, width, height) <= 0.0
    ):
        raise ValueError("Camera intrinsic and image size must be valid")
    focal_length = fx / width * horizontal_aperture
    vertical_aperture = focal_length * height / fy
    return horizontal_aperture, vertical_aperture, focal_length


def _resolve_d435_calibration(
    source_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], str]:
    """Uniformly scale the fixed 4:3 K into the source video frame."""

    width, height = (int(source_size[0]), int(source_size[1]))
    if min(width, height) <= 1:
        raise ValueError("source_size dimensions must both exceed one pixel")
    unit_count = min(width // 4, height // 3)
    if unit_count <= 0:
        raise ValueError(
            "source_size is too small to contain an integer 4:3 capture"
        )
    content_size = (4 * unit_count, 3 * unit_count)
    uniform_scale = (
        content_size[0] / D435_REFERENCE_SIZE[0]
    )
    render_intrinsic = D435_INTRINSIC.copy()
    render_intrinsic[0, :] *= uniform_scale
    render_intrinsic[1, :] *= uniform_scale
    calibration_source = (
        f"uniformly scaled from {D435_REFERENCE_SIZE[0]}x"
        f"{D435_REFERENCE_SIZE[1]} into the largest centered "
        "4:3 capture contained by prepared source_size"
    )
    return render_intrinsic, content_size, calibration_source


def _resolve_capture_calibration(
    *,
    capture_view: str,
    source_size: tuple[int, int],
    source_intrinsic: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], str]:
    """Resolve the selected camera's intrinsic and native capture size."""

    if capture_view == "robot-camera":
        return _resolve_d435_calibration(source_size)
    if capture_view != "scene":
        raise ValueError(f"Unsupported capture view: {capture_view!r}")
    intrinsic = np.asarray(source_intrinsic, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise ValueError("source_intrinsic must be a valid pinhole matrix")
    return (
        intrinsic.copy(),
        tuple(int(value) for value in source_size),
        "prepared source intrinsic at prepared source_size",
    )


def _camera_fov_degrees(
    intrinsic: np.ndarray,
    *,
    image_size: tuple[int, int],
) -> dict[str, float]:
    """Return full pinhole FOV, respecting an off-center principal point."""

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    width, height = (int(value) for value in image_size)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or min(width, height, intrinsic[0, 0], intrinsic[1, 1]) <= 0
    ):
        raise ValueError("intrinsic and image_size must be valid")
    horizontal = np.arctan2(intrinsic[0, 2], intrinsic[0, 0])
    horizontal += np.arctan2(
        width - intrinsic[0, 2],
        intrinsic[0, 0],
    )
    vertical = np.arctan2(intrinsic[1, 2], intrinsic[1, 1])
    vertical += np.arctan2(
        height - intrinsic[1, 2],
        intrinsic[1, 1],
    )
    return {
        "horizontal": float(np.degrees(horizontal)),
        "vertical": float(np.degrees(vertical)),
    }


def _hand_projection_report(
    *,
    hand_points_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    capture_size: tuple[int, int],
    warning_threshold: float = HAND_CAPTURE_WARNING_THRESHOLD,
) -> dict[str, object]:
    """Summarize whether the two G1 hands are visible to the head camera."""

    points = np.asarray(hand_points_camera_cv, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if (
        points.ndim != 3
        or points.shape[1:] != (2, 3)
        or len(points) == 0
        or not np.isfinite(points).all()
    ):
        raise ValueError("hand_points_camera_cv must be finite [N,2,3]")
    if (
        not np.isfinite(warning_threshold)
        or not 0.0 <= warning_threshold <= 1.0
    ):
        raise ValueError("warning_threshold must lie in [0, 1]")
    width, height = (int(value) for value in capture_size)
    if intrinsic.shape != (3, 3) or min(width, height) <= 0:
        raise ValueError("intrinsic and capture_size must be valid")

    in_front = points[:, :, 2] > 0.0
    pixels = np.full((len(points), 2, 2), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels[:, :, 0] = (
            intrinsic[0, 0] * points[:, :, 0] / points[:, :, 2]
            + intrinsic[0, 2]
        )
        pixels[:, :, 1] = (
            intrinsic[1, 1] * points[:, :, 1] / points[:, :, 2]
            + intrinsic[1, 2]
        )
    in_capture = (
        in_front
        & (pixels[:, :, 0] >= 0.0)
        & (pixels[:, :, 0] < width)
        & (pixels[:, :, 1] >= 0.0)
        & (pixels[:, :, 1] < height)
    )
    distances = np.linalg.norm(points, axis=2)
    per_hand_fraction = {
        name: float(np.mean(in_capture[:, hand_id]))
        for hand_id, name in enumerate(("left", "right"))
    }
    warnings = [
        (
            f"{name} hand projected inside capture for "
            f"{fraction:.1%} of frames, below "
            f"{warning_threshold:.0%}"
        )
        for name, fraction in per_hand_fraction.items()
        if fraction < warning_threshold
    ]
    return {
        "hand_frames": list(ROBOT_HAND_FRAMES),
        "frame_count": int(len(points)),
        "capture_size": [width, height],
        "fov_degrees": _camera_fov_degrees(
            intrinsic,
            image_size=capture_size,
        ),
        "in_front_fraction": {
            name: float(np.mean(in_front[:, hand_id]))
            for hand_id, name in enumerate(("left", "right"))
        },
        "in_capture_fraction": {
            **per_hand_fraction,
            "all_hand_samples": float(np.mean(in_capture)),
            "both_hands_per_frame": float(
                np.mean(np.all(in_capture, axis=1))
            ),
        },
        "camera_to_hand_distance_m": {
            name: _distribution(distances[:, hand_id])
            for hand_id, name in enumerate(("left", "right"))
        },
        "warning_threshold": float(warning_threshold),
        "warnings": warnings,
        "status": "warning" if warnings else "ok",
    }


def _capture_camera_report(
    *,
    content_intrinsic: np.ndarray,
    output_intrinsic: np.ndarray,
    content_size: tuple[int, int],
    output_size: tuple[int, int],
    letterbox: dict[str, object],
    hand_projection: dict[str, object],
) -> dict[str, object]:
    """Build the stable, machine-checkable exact camera profile report."""

    content_intrinsic = np.asarray(content_intrinsic, dtype=np.float64)
    output_intrinsic = np.asarray(output_intrinsic, dtype=np.float64)
    if (
        content_intrinsic.shape != (3, 3)
        or output_intrinsic.shape != (3, 3)
        or not np.isfinite(content_intrinsic).all()
        or not np.isfinite(output_intrinsic).all()
    ):
        raise ValueError("camera report intrinsics must be finite 3x3")
    return {
        "parent_urdf_frame": HEAD_CAMERA_PARENT_FRAME,
        "parent_prim": (
            "/World/G1/torso_link (resolved through URDF FK)"
        ),
        "pose_source": (
            "per-frame torso_link URDF forward kinematics from prepared "
            "G1 root and joint trajectory, composed with the active "
            "Head CameraCfg offset"
        ),
        "set_world_pose_camera_axes": "world",
        "offset_position_base_xyz_m": (
            HEAD_CAM_OFFSET_POS_BASE.tolist()
        ),
        "reset_guard_forward_x_m": (
            HEAD_CAM_RESET_GUARD_X_M
        ),
        "reset_guard_applied": True,
        "reset_guard_reason": (
            "zero-noise delta_x=0 is below the active 5 mm occlusion "
            "reject threshold, so fallback clamps delta_x to +0.005 m"
        ),
        "offset_position_active_xyz_m": (
            HEAD_CAM_OFFSET_POS_ACTIVE.tolist()
        ),
        "offset_rotation_parent_wxyz": (
            HEAD_CAM_OFFSET_ROT_WXYZ.tolist()
        ),
        "native_size": list(D435_REFERENCE_SIZE),
        "reference_size": list(D435_REFERENCE_SIZE),
        "content_size": [int(value) for value in content_size],
        "output_size": [int(value) for value in output_size],
        "letterbox": letterbox,
        "reference_intrinsic": D435_INTRINSIC.tolist(),
        "content_intrinsic": content_intrinsic.tolist(),
        "output_intrinsic_after_letterbox": output_intrinsic.tolist(),
        "intrinsic_scaling": "uniform x/y scale",
        "distortion_model": "zero distortion",
        "cv_pose_usage": (
            "projection and visibility diagnostics only; rendering uses "
            "the world-axis pose"
        ),
        "hand_projection": hand_projection,
    }


def _articulation_commit_errors(
    *,
    expected_joint_positions: np.ndarray,
    actual_joint_positions: np.ndarray,
    expected_root_pose_xyzw: np.ndarray,
    actual_root_position: np.ndarray,
    actual_root_orientation_wxyz: np.ndarray,
) -> dict[str, float]:
    """Measure one Isaac articulation commit against the requested frame."""

    expected_joint = np.asarray(
        expected_joint_positions,
        dtype=np.float64,
    ).reshape(-1)
    actual_joint = np.asarray(
        actual_joint_positions,
        dtype=np.float64,
    ).reshape(-1)
    expected_root = np.asarray(
        expected_root_pose_xyzw,
        dtype=np.float64,
    ).reshape(-1)
    actual_position = np.asarray(
        actual_root_position,
        dtype=np.float64,
    ).reshape(-1)
    actual_wxyz = np.asarray(
        actual_root_orientation_wxyz,
        dtype=np.float64,
    ).reshape(-1)
    if expected_joint.shape != actual_joint.shape:
        raise ValueError("expected and actual joints must have equal shape")
    if (
        expected_root.shape != (7,)
        or actual_position.shape != (3,)
        or actual_wxyz.shape != (4,)
    ):
        raise ValueError("root pose components have invalid shapes")
    if not all(
        np.isfinite(value).all()
        for value in (
            expected_joint,
            actual_joint,
            expected_root,
            actual_position,
            actual_wxyz,
        )
    ):
        raise ValueError("articulation commit inputs must be finite")
    expected_quaternion = expected_root[3:7]
    actual_quaternion = actual_wxyz[[1, 2, 3, 0]]
    expected_norm = float(np.linalg.norm(expected_quaternion))
    actual_norm = float(np.linalg.norm(actual_quaternion))
    if min(expected_norm, actual_norm) <= 1e-8:
        raise ValueError("root orientation contains a zero quaternion")
    quaternion_dot = float(
        np.clip(
            abs(
                np.dot(
                    expected_quaternion / expected_norm,
                    actual_quaternion / actual_norm,
                )
            ),
            0.0,
            1.0,
        )
    )
    joint_error = np.abs(expected_joint - actual_joint)
    return {
        "joint_max_abs_rad": float(
            np.max(joint_error) if len(joint_error) else 0.0
        ),
        "joint_rmse_rad": float(
            np.sqrt(np.mean(np.square(joint_error)))
            if len(joint_error)
            else 0.0
        ),
        "root_position_m": float(
            np.linalg.norm(expected_root[:3] - actual_position)
        ),
        "root_orientation_rad": float(2.0 * np.arccos(quaternion_dot)),
    }


def _commit_articulation_for_render(
    *,
    world: object,
    robot: object,
    root_pose_xyzw: np.ndarray,
    joint_positions: np.ndarray,
    zero_joint_velocity: np.ndarray,
    zero_root_velocity: np.ndarray,
) -> int:
    """Teleport one articulation frame and propagate it to the renderer.

    Replicator pauses the timeline after every capture.  Isaac's
    ``world.step(render=False)`` is a no-op while paused, even though direct
    articulation readback already returns the newly written tensor values.
    Resume first and request the physics-to-Fabric update explicitly so Hydra
    sees the same link transforms as PhysX.
    """

    if not world.is_playing():
        world.play()
    step_before = int(world.current_time_step_index)
    root_pose = np.asarray(root_pose_xyzw, dtype=np.float32)
    joints = np.asarray(joint_positions, dtype=np.float32)
    robot.set_world_poses(
        positions=root_pose[None, :3],
        orientations=root_pose[[6, 3, 4, 5]][None],
    )
    robot.set_joint_positions(joints[None])
    robot.set_joint_velocities(zero_joint_velocity)
    robot.set_velocities(zero_root_velocity)
    world.step(render=False, update_fabric=True)
    step_after = int(world.current_time_step_index)
    step_delta = step_after - step_before
    if step_delta <= 0:
        raise RuntimeError(
            "Isaac articulation render synchronization failed: the "
            "physics-only commit did not advance after resuming the timeline"
        )
    return step_delta


def _look_at_usd_wxyz(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a USD-camera quaternion looking from eye toward target."""

    forward = np.asarray(target, dtype=np.float64) - np.asarray(
        eye,
        dtype=np.float64,
    )
    forward_norm = np.linalg.norm(forward)
    if not np.isfinite(forward_norm) or forward_norm <= 1e-8:
        raise ValueError("Scene camera eye and target must differ")
    forward /= forward_norm
    world_up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) <= 1e-6:
        world_up = np.asarray([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack([right, up, -forward])
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]]


def _resolve_scene_camera(
    *,
    scan: SceneScan,
    render_intrinsic: np.ndarray,
    render_size: tuple[int, int],
    requested_eye: tuple[float, float, float] | None,
    requested_target: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    bounds_center = (scan.bounds_min + scan.bounds_max) * 0.5
    target = (
        bounds_center
        if requested_target is None
        else np.asarray(requested_target, dtype=np.float64)
    )
    target_source = (
        "scene-bounds center"
        if requested_target is None
        else "explicit CLI value"
    )

    if requested_eye is None:
        extent = scan.bounds_max - scan.bounds_min
        radius = max(0.5 * float(np.linalg.norm(extent)), 0.05)
        width, height = render_size
        horizontal_fov = 2.0 * np.arctan(
            width / (2.0 * render_intrinsic[0, 0])
        )
        vertical_fov = 2.0 * np.arctan(
            height / (2.0 * render_intrinsic[1, 1])
        )
        limiting_half_fov = 0.5 * min(horizontal_fov, vertical_fov)
        distance = 1.2 * radius / np.sin(limiting_half_fov)
        direction = np.asarray([-1.6, -1.8, 1.0], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        eye = target + max(distance, 0.5) * direction
        eye_source = "automatic fit to all-frame scene bounds"
    else:
        eye = np.asarray(requested_eye, dtype=np.float64)
        eye_source = "explicit CLI value"

    if (
        eye.shape != (3,)
        or target.shape != (3,)
        or not np.isfinite(eye).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Scene camera eye and target must be finite XYZ values")
    _look_at_usd_wxyz(eye, target)
    return eye, target, eye_source, target_source


def _distribution(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or len(finite) == 0 or not np.isfinite(finite).all():
        raise ValueError("Expected a non-empty finite diagnostic vector")
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }


def _rgb8(rgba: np.ndarray, expected_shape: tuple[int, int, int]) -> np.ndarray:
    rgb = np.asarray(rgba)[..., :3]
    if rgb.shape != expected_shape:
        raise RuntimeError(f"Unexpected Isaac camera shape: {rgb.shape}")
    if rgb.dtype == np.uint8:
        return rgb
    if np.issubdtype(rgb.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0
        return np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _resolve_isaac_lighting(
    prepared_root: Path,
    requested_profile: Path | None,
) -> dict[str, object]:
    automatic_path = prepared_root / "lighting_profile.json"
    profile_path = (
        requested_profile.expanduser().resolve()
        if requested_profile is not None
        else (automatic_path if automatic_path.is_file() else None)
    )
    profile = load_lighting_profile(profile_path)
    values = dict(DEFAULT_ISAAC_PROFILE)
    if profile is not None:
        values.update(profile["isaac"])

    def finite_scalar(key: str, minimum: float, maximum: float) -> float:
        value = float(values[key])
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"Lighting {key} must be finite and in "
                f"[{minimum}, {maximum}], got {value}"
            )
        return value

    rotation = np.asarray(values["key_rotation"], dtype=np.float64)
    if rotation.shape != (3,) or not np.isfinite(rotation).all():
        raise ValueError("Lighting key_rotation must be three finite degrees")
    source = (
        "explicit profile"
        if requested_profile is not None
        else (
            "prepared lighting_profile.json"
            if profile_path is not None
            else "stable defaults"
        )
    )
    return {
        "schema_version": 1,
        "applied": True,
        "source": source,
        "profile_path": str(profile_path) if profile_path is not None else None,
        "profile_version": (
            int(profile["profile_version"]) if profile is not None else None
        ),
        "profile_scope": profile.get("scope") if profile is not None else None,
        "dome": {
            "intensity": finite_scalar("dome_intensity", 0.0, 1.0e7),
            "color_temperature_kelvin": finite_scalar(
                "dome_temperature", 1000.0, 20000.0
            ),
        },
        "key": {
            "type": "UsdLux.DistantLight",
            "intensity": finite_scalar("key_intensity", 0.0, 1.0e7),
            "color_temperature_kelvin": finite_scalar(
                "key_temperature", 1000.0, 20000.0
            ),
            "angle_degrees": finite_scalar("key_angle", 0.0, 180.0),
            "rotation_xyz_degrees": rotation.tolist(),
        },
        "confidence": profile.get("confidence") if profile is not None else None,
        "warnings": list(profile.get("warnings", ())) if profile is not None else [],
        "composite": (
            dict(profile["composite"])
            if profile is not None
            else dict(DEFAULT_COMPOSITE_PROFILE)
        ),
        "composite_applied": False,
    }


def _resolve_run_options(
    args: argparse.Namespace,
    *,
    source_fps: float,
) -> RunOptions:
    """Validate mode-dependent CLI options before Isaac is launched."""

    interactive = bool(args.interactive)
    loop = bool(args.loop)
    if interactive and args.headless:
        raise ValueError("--interactive requires --no-headless")
    if loop and not interactive:
        raise ValueError("--loop requires --interactive")
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("Prepared source FPS must be positive and finite")
    playback_fps = float(source_fps)

    if interactive:
        if args.output is not None:
            raise ValueError(
                "--interactive does not record; omit --output"
            )
        return RunOptions(
            interactive=True,
            loop=loop,
            playback_fps=playback_fps,
            output_video=None,
            report_path=None,
        )

    if args.output is None:
        raise ValueError("--output is required unless --interactive is used")
    output_video = args.output.expanduser().resolve()
    if output_video.suffix.lower() != ".mp4":
        raise ValueError("--output must have an .mp4 suffix")
    report_path = output_video.with_suffix(".json")
    mask_path = output_video.with_suffix(".mask.npz")
    for path in (output_video, report_path, mask_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite output: {path}")
    return RunOptions(
        interactive=False,
        loop=False,
        playback_fps=source_fps,
        output_video=output_video,
        report_path=report_path,
    )


def _next_playback_frame(
    current_frame: int,
    *,
    frame_count: int,
    loop: bool,
) -> tuple[int, bool, bool]:
    """Return next frame, whether a pass ended, and whether playback finished."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if current_frame < 0 or current_frame >= frame_count:
        raise ValueError("current_frame is outside the sequence")
    if current_frame + 1 < frame_count:
        return current_frame + 1, False, False
    if loop:
        return 0, True, False
    return current_frame, True, True


def render(args: argparse.Namespace) -> dict[str, object]:
    scene = load_prepared_scene(args.prepared)
    run_options = _resolve_run_options(args, source_fps=scene.fps)
    lighting_report = _resolve_isaac_lighting(
        scene.root,
        args.lighting_profile,
    )
    if (
        not np.isfinite(args.depth_edge_absolute)
        or args.depth_edge_absolute < 0
    ):
        raise ValueError("--depth-edge-absolute must be finite and non-negative")
    if (
        not np.isfinite(args.depth_edge_relative)
        or args.depth_edge_relative < 0
    ):
        raise ValueError("--depth-edge-relative must be finite and non-negative")
    if args.rt_subframes <= 0:
        raise ValueError("--rt-subframes must be positive")
    capture_camera_poses_world_xyzw: np.ndarray | None = None
    capture_camera_poses_cv_xyzw: np.ndarray | None = None
    if args.capture_view == "robot-camera":
        capture_camera_poses_world_xyzw = head_camera_poses_world(
            urdf_path=scene.g1_urdf,
            joint_names=scene.joint_names,
            joint_positions=scene.joint_positions,
            root_pose_xyzw=scene.root_poses_xyzw,
        )
        capture_camera_poses_cv_xyzw = (
            capture_camera_poses_world_xyzw.copy()
        )
        capture_camera_poses_cv_xyzw[:, 3:7] = Rotation.from_matrix(
            Rotation.from_quat(
                capture_camera_poses_world_xyzw[:, 3:7]
            ).as_matrix()
            @ CAMERA_LINK_FROM_CV_OPTICAL
        ).as_quat()
    mesh_camera_poses_cv_xyzw = scene.camera_poses_cv_xyzw
    mesh_pose_source = (
        "prepared source-world rotation with XYZ anchored to G1 torso FK; "
        "G1 capture-camera mount remains independent"
    )
    if np.any(scene.scene_translation_world_xyz_m):
        mesh_pose_source += " plus prepared constant palm XYZ translation"

    (
        render_intrinsic,
        render_size,
        render_intrinsic_source,
    ) = _resolve_capture_calibration(
        capture_view=args.capture_view,
        source_size=scene.source_size,
        source_intrinsic=scene.intrinsic,
    )
    output_size = render_size
    output_size_source = (
        "native D435 4:3 capture"
        if args.capture_view == "robot-camera"
        else "prepared source_size"
    )
    output_intrinsic = render_intrinsic.copy()
    hand_projection_report: dict[str, object] | None = None
    if capture_camera_poses_cv_xyzw is not None:
        hand_points_camera = robot_frame_points_in_camera_cv(
            urdf_path=scene.g1_urdf,
            joint_names=scene.joint_names,
            joint_positions=scene.joint_positions,
            root_pose_xyzw=scene.root_poses_xyzw,
            camera_pose_cv_xyzw=capture_camera_poses_cv_xyzw,
            frame_names=ROBOT_HAND_FRAMES,
        )
        hand_projection_report = _hand_projection_report(
            hand_points_camera_cv=hand_points_camera,
            intrinsic=render_intrinsic,
            capture_size=render_size,
        )
        for warning in hand_projection_report["warnings"]:
            print(f"WARNING: {warning}", flush=True)
    scan = scan_scene(
        scene,
        camera_poses_cv_xyzw=mesh_camera_poses_cv_xyzw,
        absolute_threshold_m=args.depth_edge_absolute,
        relative_threshold=args.depth_edge_relative,
    )
    scene_eye, scene_target, eye_source, target_source = (
        _resolve_scene_camera(
            scan=scan,
            render_intrinsic=render_intrinsic,
            render_size=render_size,
            requested_eye=args.scene_eye,
            requested_target=args.scene_target,
        )
    )
    render_width, render_height = render_size
    output_width, output_height = output_size
    letterbox_report = {
        "output_size": [output_width, output_height],
        "content_size": [render_width, render_height],
        "offsets_px": {
            "left": 0,
            "top": 0,
            "right": 0,
            "bottom": 0,
        },
        "placement": "none",
        "fill_rgb": None,
    }
    capture_camera_report: dict[str, object]
    if args.capture_view == "robot-camera":
        assert hand_projection_report is not None
        capture_camera_report = _capture_camera_report(
            content_intrinsic=render_intrinsic,
            output_intrinsic=output_intrinsic,
            content_size=render_size,
            output_size=output_size,
            letterbox=letterbox_report,
            hand_projection=hand_projection_report,
        )
    else:
        capture_camera_report = {
            "profile": "fixed_third_person_scene_camera_v1",
            "type": "fixed third-person",
            "eye_xyz_m": scene_eye.tolist(),
            "eye_source": eye_source,
            "target_xyz_m": scene_target.tolist(),
            "target_source": target_source,
            "content_size": list(render_size),
            "output_size": list(output_size),
            "letterbox": letterbox_report,
        }
    if run_options.output_video is not None:
        run_options.output_video.parent.mkdir(parents=True, exist_ok=True)

    # SimulationApp must exist before importing omni, pxr, or Isaac APIs.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": "RaytracedLighting",
            "width": render_width,
            "height": render_height,
            "enable_crashreporter": False,
            "extra_args": list(ISAAC_NO_LOG_ARGS),
        }
    )
    writer: cv2.VideoWriter | None = None
    started = time.perf_counter()
    try:
        import omni.kit.app
        import omni.timeline
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate(
            "isaacsim.asset.importer.urdf",
            True,
        )
        extension_manager.set_extension_enabled_immediate(
            "isaacsim.sensors.camera",
            True,
        )
        simulation_app.update()

        context = omni.usd.get_context()
        context.new_stage()
        simulation_app.update()
        stage = context.get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / scene.fps,
            rendering_dt=1.0 / scene.fps,
        )
        world.get_physics_context().set_gravity(0.0)
        timeline = omni.timeline.get_timeline_interface()

        robot_path = import_g1(scene.g1_urdf)
        robot = world.scene.add(
            Articulation(
                prim_paths_expr=robot_path,
                name="g1_vda_replay",
            )
        )
        dome = UsdLux.DomeLight.Define(
            stage,
            Sdf.Path("/World/G1DomeLight"),
        )
        dome_settings = lighting_report["dome"]
        assert isinstance(dome_settings, dict)
        dome.CreateIntensityAttr(float(dome_settings["intensity"]))
        dome.CreateEnableColorTemperatureAttr(True)
        dome.CreateColorTemperatureAttr(
            float(dome_settings["color_temperature_kelvin"])
        )
        key_settings = lighting_report["key"]
        assert isinstance(key_settings, dict)
        key = UsdLux.DistantLight.Define(
            stage,
            Sdf.Path("/World/G1KeyLight"),
        )
        key.CreateIntensityAttr(float(key_settings["intensity"]))
        key.CreateEnableColorTemperatureAttr(True)
        key.CreateColorTemperatureAttr(
            float(key_settings["color_temperature_kelvin"])
        )
        key.CreateAngleAttr(float(key_settings["angle_degrees"]))
        key_rotation = key_settings["rotation_xyz_degrees"]
        assert isinstance(key_rotation, list)
        UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(
            Gf.Vec3f(*(float(value) for value in key_rotation))
        )

        mesh = UsdGeom.Mesh.Define(
            stage,
            Sdf.Path("/World/VdaSceneMesh"),
        )
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        uv_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        )
        uv_primvar.Set(
            Vt.Vec2fArray.FromNumpy(
                grid_uvs(*scene.mesh_size).astype(np.float32)
            )
        )

        material = UsdShade.Material.Define(
            stage,
            Sdf.Path("/World/VdaSceneMaterial"),
        )
        surface = UsdShade.Shader.Define(
            stage,
            Sdf.Path("/World/VdaSceneMaterial/Surface"),
        )
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput(
            "diffuseColor",
            Sdf.ValueTypeNames.Color3f,
        ).Set(Gf.Vec3f(0.0))
        surface.CreateInput(
            "roughness",
            Sdf.ValueTypeNames.Float,
        ).Set(1.0)
        texture = UsdShade.Shader.Define(
            stage,
            Sdf.Path("/World/VdaSceneMaterial/Texture"),
        )
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput(
            "sourceColorSpace",
            Sdf.ValueTypeNames.Token,
        ).Set("sRGB")
        reader = UsdShade.Shader.Define(
            stage,
            Sdf.Path("/World/VdaSceneMaterial/StReader"),
        )
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput(
            "varname",
            Sdf.ValueTypeNames.Token,
        ).Set("st")
        texture.CreateInput(
            "st",
            Sdf.ValueTypeNames.Float2,
        ).ConnectToSource(reader.ConnectableAPI(), "result")
        surface.CreateInput(
            "emissiveColor",
            Sdf.ValueTypeNames.Color3f,
        ).ConnectToSource(texture.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(
            surface.ConnectableAPI(),
            "surface",
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        texture_file_input = texture.CreateInput(
            "file",
            Sdf.ValueTypeNames.Asset,
        )

        (
            horizontal_aperture,
            vertical_aperture,
            focal_length,
        ) = _camera_projection(
            render_intrinsic,
            width=render_width,
            height=render_height,
        )
        camera_path = (
            "/World/G1D435Camera"
            if args.capture_view == "robot-camera"
            else "/World/VdaSceneCamera"
        )
        camera_schema = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
        camera_schema.CreateHorizontalApertureAttr(horizontal_aperture * 10.0)
        camera_schema.CreateVerticalApertureAttr(vertical_aperture * 10.0)
        camera_schema.CreateFocalLengthAttr(focal_length * 10.0)
        camera = Camera(
            prim_path=camera_path,
            name=(
                "g1_d435_camera"
                if args.capture_view == "robot-camera"
                else "vda_scene_camera"
            ),
            resolution=(render_width, render_height),
        )
        camera.set_horizontal_aperture(
            horizontal_aperture,
            maintain_square_pixels=False,
        )
        camera.set_vertical_aperture(
            vertical_aperture,
            maintain_square_pixels=False,
        )
        camera.set_focal_length(focal_length)

        world.reset()
        if run_options.interactive:
            world.pause()
            timeline.set_auto_update(False)
            timeline.commit()
        else:
            world.play()
        camera.initialize()
        if not run_options.interactive:
            camera.add_distance_to_image_plane_to_frame()
        scene_diagonal = float(
            np.linalg.norm(scan.bounds_max - scan.bounds_min)
        )
        camera.set_clipping_range(
            0.01,
            max(
                100.0,
                (
                    np.linalg.norm(scene_eye - scene_target)
                    if args.capture_view == "scene"
                    else 0.0
                )
                + 3.0 * scene_diagonal,
            ),
        )
        camera.set_lens_aperture(0.0)
        camera.set_opencv_pinhole_properties(
            cx=float(render_intrinsic[0, 2]),
            cy=float(render_intrinsic[1, 2]),
            fx=float(render_intrinsic[0, 0]),
            fy=float(render_intrinsic[1, 1]),
            pinhole=D435_DISTORTION.tolist(),
        )
        (
            resolved_cx,
            resolved_cy,
            resolved_fx,
            resolved_fy,
            resolved_distortion,
        ) = camera.get_opencv_pinhole_properties()
        resolved_intrinsic = np.asarray(
            [
                [resolved_fx, 0.0, resolved_cx],
                [0.0, resolved_fy, resolved_cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        if not np.allclose(
            resolved_intrinsic,
            render_intrinsic,
            rtol=1e-6,
            atol=1e-3,
        ):
            raise RuntimeError(
                "Isaac camera intrinsic mismatch: "
                f"expected={render_intrinsic.tolist()}, "
                f"actual={resolved_intrinsic.tolist()}"
            )
        if not np.allclose(
            np.asarray(
                resolved_distortion[
                    : len(D435_DISTORTION)
                ],
                dtype=np.float64,
            ),
            D435_DISTORTION,
            rtol=1e-6,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Isaac camera distortion differs from the required "
                "zero-distortion pinhole model"
            )
        if capture_camera_poses_world_xyzw is not None:
            camera_world_pose = capture_camera_poses_world_xyzw[0]
            camera.set_world_pose(
                position=camera_world_pose[:3].astype(np.float32),
                orientation=camera_world_pose[[6, 3, 4, 5]].astype(
                    np.float32
                ),
                camera_axes="world",
            )
        else:
            camera.set_world_pose(
                position=scene_eye.astype(np.float32),
                orientation=_look_at_usd_wxyz(
                    scene_eye,
                    scene_target,
                ).astype(np.float32),
                camera_axes="usd",
            )

        joint_order = isaac_joint_order(
            list(robot.dof_names),
            scene.joint_names,
        )
        zero_joint_velocity = np.zeros(
            (1, len(joint_order)),
            dtype=np.float32,
        )
        zero_root_velocity = np.zeros((1, 6), dtype=np.float32)
        commit_joint_max_abs_rad: list[float] = []
        commit_joint_rmse_rad: list[float] = []
        commit_root_position_m: list[float] = []
        commit_root_orientation_rad: list[float] = []
        commit_physics_step_delta: list[int] = []
        print(
            f"Isaac G1 articulation: {robot_path} · "
            f"{len(joint_order)} DoF",
            flush=True,
        )

        def apply_frame(frame_id: int) -> None:
            if run_options.interactive:
                timeline.set_auto_update(True)
            rgb_path = scene.rgb_paths[frame_id]
            depth_path = scene.depth_paths[frame_id]
            depth_m = _load_depth_m(depth_path, scene.mesh_size)
            points_camera = backproject_z_depth(
                depth_m,
                scene.mesh_intrinsic,
            )
            points_world = camera_points_to_world(
                points_camera,
                mesh_camera_poses_cv_xyzw[frame_id],
            )
            valid = np.isfinite(points_world).all(axis=-1)
            points_world[valid] += scene.scene_translation_world_xyz_m
            points_world[~valid] = 0.0
            triangles = depth_grid_triangles(
                depth_m,
                absolute_threshold_m=args.depth_edge_absolute,
                relative_threshold=args.depth_edge_relative,
            )

            mesh.GetPointsAttr().Set(
                Vt.Vec3fArray.FromNumpy(
                    points_world.reshape(-1, 3).astype(np.float32)
                )
            )
            mesh.GetFaceVertexIndicesAttr().Set(
                Vt.IntArray.FromNumpy(triangles.reshape(-1))
            )
            mesh.GetFaceVertexCountsAttr().Set(
                Vt.IntArray.FromNumpy(
                    np.full(len(triangles), 3, dtype=np.int32)
                )
            )
            valid_points = points_world[valid]
            mesh.GetExtentAttr().Set(
                Vt.Vec3fArray.FromNumpy(
                    np.stack(
                        [
                            valid_points.min(axis=0),
                            valid_points.max(axis=0),
                        ]
                    ).astype(np.float32)
                )
            )
            texture_file_input.Set(
                Sdf.AssetPath(str(rgb_path.resolve()))
            )

            root_pose = scene.root_poses_xyzw[frame_id]
            expected_joint_positions = scene.joint_positions[
                frame_id,
                joint_order,
            ]
            commit_physics_step_delta.append(
                _commit_articulation_for_render(
                    world=world,
                    robot=robot,
                    root_pose_xyzw=root_pose,
                    joint_positions=expected_joint_positions,
                    zero_joint_velocity=zero_joint_velocity,
                    zero_root_velocity=zero_root_velocity,
                )
            )
            actual_joint_positions = robot.get_joint_positions()
            (
                actual_root_positions,
                actual_root_orientations_wxyz,
            ) = robot.get_world_poses()
            commit_error = _articulation_commit_errors(
                expected_joint_positions=expected_joint_positions,
                actual_joint_positions=actual_joint_positions[0],
                expected_root_pose_xyzw=root_pose,
                actual_root_position=actual_root_positions[0],
                actual_root_orientation_wxyz=(
                    actual_root_orientations_wxyz[0]
                ),
            )
            commit_joint_max_abs_rad.append(
                commit_error["joint_max_abs_rad"]
            )
            commit_joint_rmse_rad.append(
                commit_error["joint_rmse_rad"]
            )
            commit_root_position_m.append(
                commit_error["root_position_m"]
            )
            commit_root_orientation_rad.append(
                commit_error["root_orientation_rad"]
            )
            commit_failures = {
                "joint_max_abs_rad": (
                    commit_error["joint_max_abs_rad"],
                    JOINT_COMMIT_ATOL_RAD,
                ),
                "root_position_m": (
                    commit_error["root_position_m"],
                    ROOT_POSITION_COMMIT_ATOL_M,
                ),
                "root_orientation_rad": (
                    commit_error["root_orientation_rad"],
                    ROOT_ORIENTATION_COMMIT_ATOL_RAD,
                ),
            }
            exceeded = {
                name: {"actual": actual, "limit": limit}
                for name, (actual, limit) in commit_failures.items()
                if actual > limit
            }
            if exceeded:
                raise RuntimeError(
                    "Isaac articulation did not commit the requested "
                    f"trajectory frame {frame_id}: {exceeded}"
                )
            if capture_camera_poses_world_xyzw is not None:
                camera_world_pose = capture_camera_poses_world_xyzw[frame_id]
                camera.set_world_pose(
                    position=camera_world_pose[:3].astype(np.float32),
                    orientation=camera_world_pose[
                        [6, 3, 4, 5]
                    ].astype(np.float32),
                    camera_axes="world",
                )
            if run_options.interactive:
                world.pause()
                timeline.set_auto_update(False)
                timeline.commit()

        def articulation_commit_report() -> dict[str, object]:
            if not commit_joint_max_abs_rad:
                raise RuntimeError(
                    "No articulation trajectory frame was committed"
                )
            return {
                "schema_version": 2,
                "method": (
                    "read back Isaac articulation joint/root state after "
                    "each teleport; resume the Replicator-paused timeline, "
                    "then world.step(render=False, update_fabric=True)"
                ),
                "verified_application_count": len(
                    commit_joint_max_abs_rad
                ),
                "strict_validation_passed": True,
                "render_sync_validation": {
                    "passed": True,
                    "condition": (
                        "physics step index must advance for every frame "
                        "before Hydra/RTX capture"
                    ),
                    "physics_step_index_delta": _distribution(
                        np.asarray(
                            commit_physics_step_delta,
                            dtype=np.float64,
                        )
                    ),
                    "fabric_update_requested": True,
                },
                "thresholds": {
                    "joint_max_abs_rad": JOINT_COMMIT_ATOL_RAD,
                    "root_position_m": ROOT_POSITION_COMMIT_ATOL_M,
                    "root_orientation_rad": (
                        ROOT_ORIENTATION_COMMIT_ATOL_RAD
                    ),
                },
                "joint_max_abs_rad": _distribution(
                    np.asarray(commit_joint_max_abs_rad)
                ),
                "joint_rmse_rad": _distribution(
                    np.asarray(commit_joint_rmse_rad)
                ),
                "root_position_m": _distribution(
                    np.asarray(commit_root_position_m)
                ),
                "root_orientation_rad": _distribution(
                    np.asarray(commit_root_orientation_rad)
                ),
            }

        if run_options.interactive:
            from isaacsim.core.utils.viewports import set_camera_view
            from omni.kit.viewport.utility import get_active_viewport

            viewport = None
            for _ in range(10):
                simulation_app.update()
                viewport = get_active_viewport()
                if viewport is not None:
                    break
            if viewport is None:
                raise RuntimeError("Isaac GUI has no active viewport")

            viewport_camera_path = Sdf.Path("/OmniverseKit_Persp")
            viewport.camera_path = viewport_camera_path
            simulation_app.update()
            set_camera_view(
                eye=scene_eye,
                target=scene_target,
                camera_prim_path=str(viewport_camera_path),
                viewport_api=viewport,
            )
            viewport_camera_prim = stage.GetPrimAtPath(viewport_camera_path)
            if viewport_camera_prim.IsValid():
                viewport_camera_prim.CreateAttribute(
                    "omni:kit:cameraLock",
                    Sdf.ValueTypeNames.Bool,
                    custom=True,
                ).Set(False)

            current_frame = 0
            completed_passes = 0
            finished = False
            apply_frame(current_frame)
            world.render()
            world.render()
            frame_period = 1.0 / run_options.playback_fps
            next_frame_at = time.perf_counter() + frame_period
            print(
                "\nInteractive Isaac VDA + G1 viewer\n"
                f"  playback: {run_options.playback_fps:.3f} FPS, "
                f"{'looping' if run_options.loop else 'one pass'}\n"
                "  Alt+Left drag : orbit\n"
                "  Middle drag   : pan\n"
                "  Right drag    : look\n"
                "  Wheel/Alt+Right drag: zoom\n"
                "Close the Isaac window or press Ctrl-C in this terminal "
                "to quit.\n",
                flush=True,
            )
            try:
                while simulation_app.is_running():
                    now = time.perf_counter()
                    if finished:
                        simulation_app.update()
                        time.sleep(frame_period)
                        continue
                    if now < next_frame_at:
                        time.sleep(
                            min(
                                next_frame_at - now,
                                frame_period / 4.0,
                            )
                        )
                        continue

                    (
                        next_frame,
                        pass_ended,
                        finished,
                    ) = _next_playback_frame(
                        current_frame,
                        frame_count=scene.frame_count,
                        loop=run_options.loop,
                    )
                    if pass_ended:
                        completed_passes += 1
                    if finished:
                        print(
                            "Playback finished; holding the last frame. "
                            "Close the window to quit.",
                            flush=True,
                        )
                        continue

                    current_frame = next_frame
                    apply_frame(current_frame)
                    # One normal viewport render both presents this RGB-D frame
                    # and processes GUI camera input. Avoid an additional Kit
                    # update here: that can make source-rate playback miss its
                    # wall-clock deadline on otherwise capable hardware.
                    world.render()
                    after_render = time.perf_counter()
                    next_frame_at = max(
                        next_frame_at + frame_period,
                        after_render,
                    )
            except KeyboardInterrupt:
                pass

            report: dict[str, object] = {
                "schema_version": 3,
                "method": (
                    "video-depth-anything-dynamic-rgbd-mesh"
                    "-plus-g1-articulation"
                ),
                "mode": "interactive",
                "prepared_root": str(scene.root),
                "source_dataset_kind": scene.source_dataset_kind,
                "robot": {
                    "urdf": str(scene.g1_urdf),
                    "articulation_path": robot_path,
                    "dof_count": len(joint_order),
                    "trajectory_source": (
                        "episode_data.npz joint_positions/root_pose_xyzw"
                    ),
                    "render_visibility": {
                        "mode": "always_enabled",
                        "decision_source": (
                            "capture_camera_frustum_and_depth_test"
                        ),
                        "enabled_frames": scene.frame_count,
                        "disabled_frames": 0,
                    },
                    "articulation_commit": (
                        articulation_commit_report()
                    ),
                },
                "frame_count": scene.frame_count,
                "source_fps": scene.fps,
                "playback_fps": run_options.playback_fps,
                "loop": run_options.loop,
                "completed_passes": completed_passes,
                "last_local_frame": current_frame,
                "source_size": list(scene.source_size),
                "render_size": list(render_size),
                "render_size_source": (
                    "largest native-aspect capture contained by "
                    "prepared source_size"
                    if args.capture_view == "robot-camera"
                    else "prepared source_size"
                ),
                "output_size": list(output_size),
                "output_size_source": output_size_source,
                "letterbox": letterbox_report,
                "capture_view": args.capture_view,
                "capture_intrinsic": render_intrinsic.tolist(),
                "capture_intrinsic_source": render_intrinsic_source,
                "capture_distortion": (
                    D435_DISTORTION.tolist()
                ),
                "interactive_viewport_camera": str(viewport_camera_path),
                "interactive_viewport_type": (
                    "adjustable third-person debug view"
                ),
                "initial_view": {
                    "eye_xyz_m": scene_eye.tolist(),
                    "target_xyz_m": scene_target.tolist(),
                },
                "capture_camera": capture_camera_report,
                "articulation_commit": articulation_commit_report(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            print(json.dumps(report, indent=2), flush=True)
            return report

        import omni.replicator.core as rep

        output_video = run_options.output_video
        report_path = run_options.report_path
        assert output_video is not None
        assert report_path is not None
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            scene.fps,
            output_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open MP4 writer: {output_video}")

        packed_hole_masks: list[np.ndarray] = []
        first_render = True
        print(
            f"Isaac VDA + G1 rendered 0/{scene.frame_count}",
            flush=True,
        )
        for frame_id in range(scene.frame_count):
            apply_frame(frame_id)

            if first_render:
                rep.orchestrator.step(
                    rt_subframes=args.rt_subframes,
                    pause_timeline=True,
                    delta_time=0.0,
                    wait_for_render=True,
                )
                first_render = False
            rep.orchestrator.step(
                rt_subframes=args.rt_subframes,
                pause_timeline=True,
                delta_time=0.0,
                wait_for_render=True,
            )
            rgba = camera.get_rgba()
            if rgba is None:
                raise RuntimeError(
                    f"Isaac camera returned no frame at local frame {frame_id}"
                )
            rgb = _rgb8(
                rgba,
                (render_height, render_width, 3),
            )
            depth = camera.get_depth()
            if depth is None:
                raise RuntimeError(
                    f"Isaac camera returned no depth at local frame {frame_id}"
                )
            depth = np.asarray(depth).squeeze()
            if depth.shape != (render_height, render_width):
                raise RuntimeError(
                    f"Isaac depth shape {depth.shape} does not match "
                    f"{render_height}x{render_width}"
                )
            hole_mask = (~np.isfinite(depth) | (depth <= 0.0))
            packed_hole_masks.append(np.packbits(hole_mask.reshape(-1)))
            writer.write(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )
            print(
                f"Isaac VDA + G1 rendered {frame_id + 1}/"
                f"{scene.frame_count}",
                flush=True,
            )

        writer.release()
        writer = None
        mask_path = output_video.with_suffix(".mask.npz")
        np.savez_compressed(
            mask_path,
            masks=np.stack(packed_hole_masks),
            shape=np.asarray(
                [scene.frame_count, render_height, render_width],
                dtype=np.int64,
            ),
        )
        median_depth_delta = np.abs(np.diff(scan.median_depths_m))
        triangle_delta = np.abs(np.diff(scan.triangle_counts))
        valid_ratio = (
            scan.valid_vertex_counts
            / float(scene.mesh_size[0] * scene.mesh_size[1])
        )
        report: dict[str, object] = {
            "schema_version": BATCH_REPORT_SCHEMA_VERSION,
            "method": (
                "video-depth-anything-dynamic-rgbd-mesh"
                "-plus-g1-articulation"
            ),
            "renderer": "Isaac Sim 5.1 RaytracedLighting",
            "prepared_root": str(scene.root),
            "source_dataset_kind": scene.source_dataset_kind,
            "output_video": str(output_video),
            "hole_masks": {
                "path": str(mask_path),
                "format": "numpy-packbits",
                "semantics": "1=inpaint hole, 0=observed rendered geometry",
            },
            "robot": {
                "urdf": str(scene.g1_urdf),
                "articulation_path": robot_path,
                "dof_count": len(joint_order),
                "trajectory_source": (
                    "episode_data.npz joint_positions/root_pose_xyzw"
                ),
                "render_visibility": {
                    "mode": "always_enabled",
                    "decision_source": (
                        "capture_camera_frustum_and_depth_test"
                    ),
                    "enabled_frames": scene.frame_count,
                    "disabled_frames": 0,
                },
                "articulation_commit": articulation_commit_report(),
            },
            "frame_count": scene.frame_count,
            "fps": scene.fps,
            "duration_seconds": scene.frame_count / scene.fps,
            "source_frame_indices": scene.frame_indices.tolist(),
            "source_size": list(scene.source_size),
            "mesh_size": list(scene.mesh_size),
            "render_size": list(render_size),
            "render_size_source": (
                "largest native-aspect capture contained by prepared "
                "source_size"
                if args.capture_view == "robot-camera"
                else "prepared source_size"
            ),
            "output_size": list(output_size),
            "output_size_source": output_size_source,
            "letterbox": letterbox_report,
            "capture_view": args.capture_view,
            "source_intrinsic": scene.intrinsic.tolist(),
            "mesh_intrinsic": scene.mesh_intrinsic.tolist(),
            "render_intrinsic": render_intrinsic.tolist(),
            "render_intrinsic_source": render_intrinsic_source,
            "capture_distortion": D435_DISTORTION.tolist(),
            "mesh_pose_source": mesh_pose_source,
            "camera_axes": "mesh poses are OpenCV world-from-camera",
            "capture_camera": capture_camera_report,
            "articulation_commit": articulation_commit_report(),
            "scene_bounds_xyz_m": {
                "min": scan.bounds_min.tolist(),
                "max": scan.bounds_max.tolist(),
            },
            "depth_unit": "uint16 millimeters",
            "depth_edge_thresholds": {
                "absolute_m": args.depth_edge_absolute,
                "relative": args.depth_edge_relative,
            },
            "valid_vertex_ratio": _distribution(valid_ratio),
            "triangle_count": _distribution(scan.triangle_counts),
            "adjacent_triangle_count_delta": (
                _distribution(triangle_delta)
                if len(triangle_delta)
                else None
            ),
            "median_depth_m": _distribution(scan.median_depths_m),
            "adjacent_median_depth_delta_m": (
                _distribution(median_depth_delta)
                if len(median_depth_delta)
                else None
            ),
            "background_material": "UsdPreviewSurface emissive RGB texture",
            "lighting_affects": "G1; VDA RGB-D mesh is emissive",
            "lighting": lighting_report,
            "headless": bool(args.headless),
            "rt_subframes": args.rt_subframes,
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        if writer is not None:
            writer.release()
        simulation_app.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a prepared Video Depth Anything RGB-D scene and its "
            "frame-aligned G1 trajectory, either to a native 4:3 G1 "
            "head-camera MP4 or in an adjustable third-person viewport."
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
            "Batch output MP4 and same-stem JSON report; omit in "
            "--interactive mode"
        ),
    )
    parser.add_argument(
        "--capture-view",
        choices=("robot-camera", "scene"),
        default="robot-camera",
        help=(
            "torso/head profile (default), or a fixed third-person "
            "diagnostic view. Robot-camera output uses its native 4:3 "
            "capture size"
        ),
    )
    parser.add_argument(
        "--scene-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Fixed --capture-view scene camera or initial interactive eye; "
            "default fits bounds"
        ),
    )
    parser.add_argument(
        "--scene-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Fixed --capture-view scene or initial interactive target; "
            "default is bounds center"
        ),
    )
    parser.add_argument(
        "--depth-edge-absolute",
        type=float,
        default=0.05,
        help="Absolute depth jump rejected during triangulation (default: 0.05 m)",
    )
    parser.add_argument(
        "--depth-edge-relative",
        type=float,
        default=0.05,
        help="Relative depth jump rejected during triangulation (default: 0.05)",
    )
    parser.add_argument(
        "--rt-subframes",
        type=int,
        default=2,
        help="RTX subframes per batch-captured frame (default: 2)",
    )
    parser.add_argument(
        "--lighting-profile",
        type=Path,
        help=(
            "Lighting profile JSON; default automatically uses "
            "<prepared>/lighting_profile.json, then stable defaults"
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Keep an adjustable Isaac viewport open; requires "
            "--no-headless and does not record"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the sequence in --interactive mode",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run without an Isaac window (default: true)",
    )
    return parser


def main() -> None:
    render(build_parser().parse_args())


if __name__ == "__main__":
    main()
