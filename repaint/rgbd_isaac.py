#!/usr/bin/env python3
"""Render a G1 trajectory against a dynamic RGB-D mesh in Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.isaac_replay import import_g1, isaac_joint_order  # noqa: E402
from repaint.rgbd_geometry import (  # noqa: E402
    backproject_z_depth,
    camera_cv_pose_to_usd_wxyz,
    camera_points_to_world,
    center_patch_median_depth,
    depth_grid_triangles,
    grid_uvs,
    image_plane_corners_camera_cv,
    minimum_image_plane_distance_for_foreground,
    scale_intrinsic,
    temporal_median_depth,
)
from repaint.robot_camera import (  # noqa: E402
    CAMERA_LINK_FROM_CV_OPTICAL,
    camera_pose_delta_report,
    co_centered_source_camera_poses_cv,
    d435_camera_poses_cv,
    fit_co_centered_source_scale,
    load_source_camera_poses_cv,
    load_source_hand_points_camera_cv,
    rigid_source_scene_camera_poses_cv,
    robot_frame_points_in_camera_cv,
)


# Nominal head-camera calibration in BeingBeyond_Isaac_sim_scene:
# envs/tasks/G1_with_hand_head_CylinderEnv.py.  Its declared CameraCfg pose is
# the same d435_joint pose as this project's G1 URDF and it uses an ideal
# (zero-distortion) Isaac pinhole model.  We intentionally keep deterministic
# replay at that declared pose instead of reproducing reset-time perturbations
# (including the source environment's occlusion-rejection nudge).
BEINGBEYOND_D435_IMAGE_SIZE = (640, 480)
BEINGBEYOND_D435_INTRINSIC = np.asarray(
    [
        [607.212158203125, 0.0, 311.9999084472656],
        [0.0, 607.0084838867188, 261.152099609375],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
BEINGBEYOND_D435_DISTORTION = np.zeros(12, dtype=np.float64)
BEINGBEYOND_D435_REFERENCE = (
    "/home/ai01/Downloads/BeingBeyond_Isaac_sim_scene/"
    "envs/tasks/G1_with_hand_head_CylinderEnv.py"
)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _numbered_paths(directory: Path, suffix: str) -> list[Path]:
    paths = sorted(directory.glob(f"*{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No {suffix} files in {directory}")
    return paths


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
    # These aperture values provide the undistorted perspective seed. Exact
    # fx/fy/cx/cy and lens coefficients are applied below through Isaac's
    # OpenCV pinhole camera model.
    focal_length = fx / width * horizontal_aperture
    vertical_aperture = focal_length * height / fy
    return horizontal_aperture, vertical_aperture, focal_length


def _resolve_d435_calibration(
    *,
    render_scale: float,
) -> tuple[np.ndarray, tuple[int, int], str]:
    """Scale the one fixed BeingBeyond K to the render resolution."""

    if not np.isfinite(render_scale) or render_scale <= 0.0:
        raise ValueError("--render-scale must be positive and finite")
    render_size = (
        int(round(BEINGBEYOND_D435_IMAGE_SIZE[0] * render_scale)),
        int(round(BEINGBEYOND_D435_IMAGE_SIZE[1] * render_scale)),
    )
    if min(render_size) <= 0:
        raise ValueError("--render-scale produced an invalid output size")
    render_intrinsic = scale_intrinsic(
        BEINGBEYOND_D435_INTRINSIC,
        source_size=BEINGBEYOND_D435_IMAGE_SIZE,
        target_size=render_size,
    )
    calibration_source = (
        "fixed BeingBeyond Isaac G1 head-camera calibration "
        f"({BEINGBEYOND_D435_REFERENCE})"
    )
    return render_intrinsic, render_size, calibration_source


def _look_at_usd_wxyz(
    eye: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Return a USD-camera quaternion looking from eye toward target."""

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-8:
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


def _resolve_interactive_command(
    command: str,
    *,
    current_frame: int,
    frame_indices: np.ndarray,
) -> tuple[int, bool, str | None]:
    """Resolve one terminal command without touching the Isaac event loop."""

    normalized = command.strip().lower()
    if normalized in {"q", "quit", "exit"}:
        return current_frame, True, None
    if normalized in {"", "n", "next"}:
        return min(current_frame + 1, len(frame_indices) - 1), False, None
    if normalized in {"p", "prev", "previous"}:
        return max(current_frame - 1, 0), False, None
    if normalized.startswith("g "):
        try:
            source_frame = int(normalized.split(maxsplit=1)[1])
        except ValueError:
            return current_frame, False, "usage: g SOURCE_FRAME"
        matches = np.flatnonzero(frame_indices == source_frame)
        if len(matches) == 0:
            return (
                current_frame,
                False,
                f"source frame {source_frame} is not in this prepared "
                "sequence",
            )
        return int(matches[0]), False, None
    return current_frame, False, "unknown command; use s, n, p, g N, or q"


def _validate_archive(
    archive: np.lib.npyio.NpzFile,
) -> tuple[int, tuple[int, int], tuple[int, int]]:
    required = {
        "fps",
        "source_size",
        "mesh_size",
        "frame_indices",
        "joint_names",
        "joint_positions",
        "root_pose_xyzw",
        "camera_pose_cv_xyzw",
        "intrinsic",
        "mesh_intrinsic",
        "scene_scale",
        "source_to_robot_rotation",
        "g1_urdf",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise KeyError(f"Prepared archive is missing: {missing}")
    frame_count = int(len(archive["frame_indices"]))
    if frame_count <= 0:
        raise ValueError("Prepared episode has no frames")
    joint_names = tuple(str(name) for name in archive["joint_names"].tolist())
    if archive["joint_positions"].shape != (frame_count, len(joint_names)):
        raise ValueError("joint_positions has an invalid shape")
    for key in ("root_pose_xyzw", "camera_pose_cv_xyzw"):
        if archive[key].shape != (frame_count, 7):
            raise ValueError(f"{key} has an invalid shape")
    if (
        "source_camera_pose_cv_xyzw" in archive.files
        and archive["source_camera_pose_cv_xyzw"].shape != (frame_count, 7)
    ):
        raise ValueError("source_camera_pose_cv_xyzw has an invalid shape")
    if archive["source_to_robot_rotation"].shape != (3, 3):
        raise ValueError("source_to_robot_rotation has an invalid shape")
    source_size = tuple(
        int(value) for value in archive["source_size"].tolist()
    )
    mesh_size = tuple(
        int(value) for value in archive["mesh_size"].tolist()
    )
    if len(source_size) != 2 or len(mesh_size) != 2:
        raise ValueError("source_size and mesh_size must contain W,H")
    return frame_count, source_size, mesh_size


def render(args: argparse.Namespace) -> dict[str, object]:
    prepared_root = args.prepared.resolve()
    archive_path = prepared_root / "episode_data.npz"
    lighting_path = prepared_root / "lighting_profile.json"
    prepare_report_path = prepared_root / "prepare_report.json"
    if not archive_path.is_file() or not lighting_path.is_file():
        raise FileNotFoundError(
            f"Prepared RGB-D episode is incomplete: {prepared_root}"
        )
    archive = np.load(archive_path, allow_pickle=False)
    prepare_report = (
        _read_json(prepare_report_path)
        if prepare_report_path.is_file()
        else None
    )
    frame_count, source_size, mesh_size = _validate_archive(archive)
    rgb_paths = _numbered_paths(prepared_root / "rgb", ".png")
    depth_paths: list[Path] = []
    needs_depth_sequence = (
        args.background_mode == "rgbd-mesh"
        or (
            args.background_mode == "video-plane"
            and args.camera_pose_source == "d435-centered"
            and args.video_plane_distance is None
        )
    )
    if needs_depth_sequence:
        depth_paths = _numbered_paths(prepared_root / "depth_mm", ".png")
    if len(rgb_paths) != frame_count or (
        needs_depth_sequence and len(depth_paths) != frame_count
    ):
        raise ValueError(
            "RGB/depth sequence length does not match episode archive"
        )

    fps = float(archive["fps"])
    joint_names = tuple(str(name) for name in archive["joint_names"].tolist())
    joint_positions = np.asarray(
        archive["joint_positions"],
        dtype=np.float64,
    )
    root_pose = np.asarray(archive["root_pose_xyzw"], dtype=np.float64)
    mapped_source_camera_pose = np.asarray(
        archive["camera_pose_cv_xyzw"],
        dtype=np.float64,
    ).copy()
    if not np.isfinite(args.scene_z_offset):
        raise ValueError("--scene-z-offset must be finite")
    intrinsic = np.asarray(archive["intrinsic"], dtype=np.float64)
    mesh_intrinsic = np.asarray(
        archive["mesh_intrinsic"],
        dtype=np.float64,
    )
    urdf_path = Path(str(archive["g1_urdf"].item())).resolve()
    optical_center_in_link = np.asarray(
        args.d435_optical_center_in_link,
        dtype=np.float64,
    )
    d435_camera_pose = d435_camera_poses_cv(
        urdf_path=urdf_path,
        joint_names=joint_names,
        joint_positions=joint_positions,
        root_pose_xyzw=root_pose,
        frame_name=args.d435_frame,
        optical_center_in_link_xyz=optical_center_in_link,
    )
    if (
        args.camera_pose_source != "mapped-source"
        and not np.isclose(args.scene_z_offset, 0.0)
    ):
        raise ValueError(
            "--scene-z-offset is only available with "
            "--camera-pose-source mapped-source"
        )
    source_camera_pose: np.ndarray | None = None
    source_to_robot_rotation = np.asarray(
        archive["source_to_robot_rotation"],
        dtype=np.float64,
    )
    if args.camera_pose_source in {
        "source-extrinsic",
        "d435-centered",
    }:
        if "source_camera_pose_cv_xyzw" in archive.files:
            source_camera_pose = np.asarray(
                archive["source_camera_pose_cv_xyzw"],
                dtype=np.float64,
            )
        else:
            if "source_hdf5" not in archive.files:
                raise KeyError(
                    "Prepared archive has neither source camera poses nor "
                    "source_hdf5"
                )
            source_camera_pose = load_source_camera_poses_cv(
                Path(str(archive["source_hdf5"].item())),
                np.asarray(archive["frame_indices"], dtype=np.int64),
            )
    if args.camera_pose_source == "source-extrinsic":
        assert source_camera_pose is not None
        mesh_camera_pose = rigid_source_scene_camera_poses_cv(
            source_camera_pose_cv_xyzw=source_camera_pose,
            source_to_robot_rotation=source_to_robot_rotation,
            robot_camera_anchor_pose_cv_xyzw=d435_camera_pose[0],
            scene_scale=float(archive["scene_scale"]),
        )
        camera_pose = d435_camera_pose
    elif args.camera_pose_source == "d435-centered":
        assert source_camera_pose is not None
        mesh_camera_pose = co_centered_source_camera_poses_cv(
            source_camera_pose_cv_xyzw=source_camera_pose,
            source_to_robot_rotation=source_to_robot_rotation,
            robot_camera_pose_cv_xyzw=d435_camera_pose,
        )
        camera_pose = d435_camera_pose
    elif args.camera_pose_source == "d435-optical":
        # Camera-attached diagnostic: exact source-image reprojection, but the
        # background rotates with the fixed robot camera instead of retaining
        # the source-world extrinsic orientation.
        mesh_camera_pose = d435_camera_pose
        camera_pose = d435_camera_pose
    else:
        # This legacy-free diagnostic mode retains the source-extrinsic
        # mapping but does not claim that the camera is attached to the G1.
        camera_pose = mapped_source_camera_pose.copy()
        camera_pose[:, 2] += args.scene_z_offset
        mesh_camera_pose = camera_pose
    source_to_d435_delta = camera_pose_delta_report(
        mapped_source_camera_pose,
        d435_camera_pose,
    )
    render_camera_pose_source = (
        "mapped-source"
        if args.camera_pose_source == "mapped-source"
        else "d435-optical"
    )
    mesh_pose_source = {
        "source-extrinsic": "rigid-source-extrinsic",
        "d435-centered": "D435-centered source-camera rotation",
        "d435-optical": "d435-camera-attached",
        "mapped-source": "mapped-source",
    }[args.camera_pose_source]
    video_plane_distance: float | None = None
    video_plane_distance_source: str | None = None
    video_plane_scale: float | None = None
    video_plane_source_depth: float | None = None
    video_plane_distances: np.ndarray | None = None
    video_plane_depth_anchor_report: dict[str, object] | None = None
    co_centered_scale_report: dict[str, object] | None = None
    video_plane_points_camera: np.ndarray | None = None
    if args.video_plane_scale is not None and (
        args.background_mode != "video-plane"
        or args.camera_pose_source != "d435-centered"
    ):
        raise ValueError(
            "--video-plane-scale is only valid for a d435-centered "
            "video-plane"
        )
    if args.background_mode == "video-plane":
        if (
            not np.isfinite(args.video_plane_clearance)
            or args.video_plane_clearance < 0.0
        ):
            raise ValueError(
                "--video-plane-clearance must be finite and non-negative"
            )
        robot_hand_points_camera = robot_frame_points_in_camera_cv(
            urdf_path=urdf_path,
            joint_names=joint_names,
            joint_positions=joint_positions,
            root_pose_xyzw=root_pose,
            camera_pose_cv_xyzw=camera_pose,
            frame_names=(
                "left_hand_base_link",
                "right_hand_base_link",
            ),
        )
        if args.camera_pose_source == "d435-centered":
            assert source_camera_pose is not None
            if "source_hdf5" not in archive.files:
                raise KeyError(
                    "D435-centered mode requires source_hdf5 in the "
                    "prepared archive"
                )
            source_hand_points, source_hand_confidence = (
                load_source_hand_points_camera_cv(
                    Path(str(archive["source_hdf5"].item())),
                    np.asarray(
                        archive["frame_indices"],
                        dtype=np.int64,
                    ),
                )
            )
            _, co_centered_scale_report = (
                fit_co_centered_source_scale(
                    source_hand_points_camera_cv=source_hand_points,
                    source_hand_confidence=source_hand_confidence,
                    source_camera_pose_cv_xyzw=source_camera_pose,
                    source_to_robot_rotation=source_to_robot_rotation,
                    robot_camera_pose_cv_xyzw=d435_camera_pose,
                    robot_hand_points_camera_cv=robot_hand_points_camera,
                    confidence_threshold=(
                        args.video_plane_hand_confidence
                    ),
                )
            )
            if args.video_plane_scale is None:
                video_plane_scale = float(archive["scene_scale"])
                if (
                    not np.isfinite(video_plane_scale)
                    or video_plane_scale <= 0.0
                ):
                    raise ValueError(
                        "Prepared scene_scale must be positive and finite"
                    )
                co_centered_scale_report["scale_source"] = (
                    "prepared per-task scene_scale"
                )
            else:
                if (
                    not np.isfinite(args.video_plane_scale)
                    or args.video_plane_scale <= 0.0
                ):
                    raise ValueError(
                        "--video-plane-scale must be positive and finite"
                    )
                video_plane_scale = float(args.video_plane_scale)
                co_centered_scale_report["scale_source"] = (
                    "explicit CLI override"
                )
            co_centered_scale_report["applied_scale"] = (
                video_plane_scale
            )
            if args.video_plane_distance is None:
                if (
                    not np.isfinite(
                        args.video_plane_depth_center_fraction
                    )
                    or args.video_plane_depth_center_fraction <= 0.0
                    or args.video_plane_depth_center_fraction > 0.5
                ):
                    raise ValueError(
                        "--video-plane-depth-center-fraction must lie "
                        "in (0, 0.5]"
                    )
                if (
                    args.video_plane_depth_median_window <= 0
                    or args.video_plane_depth_median_window % 2 == 0
                ):
                    raise ValueError(
                        "--video-plane-depth-median-window must be a "
                        "positive odd integer"
                    )
                if not np.isfinite(args.video_plane_depth_offset):
                    raise ValueError(
                        "--video-plane-depth-offset must be finite"
                    )
                stored_center_depth = np.empty(
                    frame_count,
                    dtype=np.float64,
                )
                for frame_id, depth_path in enumerate(depth_paths):
                    depth_mm = cv2.imread(
                        str(depth_path),
                        cv2.IMREAD_UNCHANGED,
                    )
                    if (
                        depth_mm is None
                        or depth_mm.dtype != np.uint16
                        or depth_mm.shape
                        != (mesh_size[1], mesh_size[0])
                    ):
                        raise ValueError(
                            f"Invalid prepared depth frame: {depth_path}"
                        )
                    stored_center_depth[frame_id] = (
                        center_patch_median_depth(
                            depth_mm.astype(np.float64) / 1000.0,
                            half_fraction=(
                                args.video_plane_depth_center_fraction
                            ),
                        )
                    )
                prepared_depth_scale = float(archive["scene_scale"])
                if (
                    not np.isfinite(prepared_depth_scale)
                    or prepared_depth_scale <= 0.0
                ):
                    raise ValueError(
                        "Prepared scene_scale must be positive and finite"
                    )
                source_center_depth = (
                    stored_center_depth / prepared_depth_scale
                )
                filtered_source_depth = temporal_median_depth(
                    source_center_depth,
                    window=args.video_plane_depth_median_window,
                )
                video_plane_distances = (
                    video_plane_scale * filtered_source_depth
                    + args.video_plane_depth_offset
                )
                if (
                    not np.isfinite(video_plane_distances).all()
                    or np.any(video_plane_distances <= 0.0)
                ):
                    raise ValueError(
                        "Scaled center-depth plane distances must be "
                        "positive and finite"
                    )
                video_plane_distance = float(
                    np.median(video_plane_distances)
                )
                video_plane_source_depth = float(
                    np.median(filtered_source_depth)
                )
                video_plane_distance_source = (
                    "temporally filtered prepared center depth, with prepared "
                    "scene scale removed and co-centered scale applied"
                )
                video_plane_depth_anchor_report = {
                    "method": "prepared metric-depth center-ROI temporal median",
                    "depth_input": "processed removed-arm RGB",
                    "depth_provenance": (
                        prepare_report.get("depth")
                        if prepare_report is not None
                        else None
                    ),
                    "center_half_fraction": (
                        args.video_plane_depth_center_fraction
                    ),
                    "temporal_median_window": (
                        args.video_plane_depth_median_window
                    ),
                    "prepared_depth_scale_removed": (
                        prepared_depth_scale
                    ),
                    "source_to_robot_scale_applied": video_plane_scale,
                    "robot_depth_offset_m": (
                        args.video_plane_depth_offset
                    ),
                    "stored_center_depth_m": {
                        "median": float(
                            np.median(stored_center_depth)
                        ),
                        "min": float(np.min(stored_center_depth)),
                        "max": float(np.max(stored_center_depth)),
                        "std": float(np.std(stored_center_depth)),
                    },
                    "source_center_depth_m": {
                        "median": float(
                            np.median(filtered_source_depth)
                        ),
                        "min": float(np.min(filtered_source_depth)),
                        "max": float(np.max(filtered_source_depth)),
                        "std": float(np.std(filtered_source_depth)),
                    },
                    "robot_plane_distance_m": {
                        "median": video_plane_distance,
                        "min": float(np.min(video_plane_distances)),
                        "max": float(np.max(video_plane_distances)),
                        "std": float(np.std(video_plane_distances)),
                    },
                }
            else:
                if (
                    not np.isfinite(args.video_plane_distance)
                    or args.video_plane_distance <= 0.0
                ):
                    raise ValueError(
                        "--video-plane-distance must be positive and finite"
                    )
                video_plane_distance = float(args.video_plane_distance)
                video_plane_distance_source = "explicit CLI value"
            print(
                json.dumps(
                    {
                        "d435_centered_video_plane": {
                            "source_center_depth_m": (
                                video_plane_source_depth
                            ),
                            "source_to_robot_scale": video_plane_scale,
                            "plane_distance_m": video_plane_distance,
                            "depth_anchor": (
                                video_plane_depth_anchor_report
                            ),
                            "scale_fit": co_centered_scale_report,
                        }
                    },
                    indent=2,
                ),
                flush=True,
            )
        elif args.video_plane_distance is None:
            video_plane_distance = (
                minimum_image_plane_distance_for_foreground(
                    intrinsic=intrinsic,
                    image_size=source_size,
                    plane_camera_pose_cv_xyzw=mesh_camera_pose,
                    render_camera_pose_cv_xyzw=camera_pose,
                    foreground_points_render_camera=(
                        robot_hand_points_camera
                    ),
                    clearance_m=args.video_plane_clearance,
                )
            )
            video_plane_distance_source = (
                "minimum all-frame distance keeping all four video-plane "
                "corners behind both G1 hands plus clearance"
            )
        else:
            if (
                not np.isfinite(args.video_plane_distance)
                or args.video_plane_distance <= 0.0
            ):
                raise ValueError(
                    "--video-plane-distance must be positive and finite"
                )
            video_plane_distance = float(args.video_plane_distance)
            video_plane_distance_source = "explicit CLI value"
        if video_plane_distances is None:
            video_plane_points_camera = image_plane_corners_camera_cv(
                intrinsic,
                image_size=source_size,
                distance_m=video_plane_distance,
            )
        else:
            video_plane_points_camera = np.stack(
                [
                    image_plane_corners_camera_cv(
                        intrinsic,
                        image_size=source_size,
                        distance_m=float(distance),
                    )
                    for distance in video_plane_distances
                ],
                axis=0,
            )
    lighting = _read_json(lighting_path)
    isaac_lighting = lighting.get("isaac")
    if not isinstance(isaac_lighting, dict):
        raise ValueError(f"No Isaac lighting section in {lighting_path}")

    render_intrinsic, render_size, render_intrinsic_source = (
        _resolve_d435_calibration(
            render_scale=args.render_scale,
        )
    )
    render_width, render_height = render_size
    d435_distortion = BEINGBEYOND_D435_DISTORTION.copy()
    render_method = (
        "processed_video_plane_plus_g1_articulation"
        if args.background_mode == "video-plane"
        else "dynamic_rgbd_mesh_plus_g1_articulation"
    )
    output_video: Path | None = None
    if args.robot_camera_window and not args.interactive:
        raise ValueError("--robot-camera-window requires --interactive")
    if args.loop and not args.interactive:
        raise ValueError("--loop requires --interactive")
    if args.playback_fps is not None and not args.interactive:
        raise ValueError("--playback-fps is only valid with --interactive")
    playback_fps = (
        fps if args.playback_fps is None else float(args.playback_fps)
    )
    if not np.isfinite(playback_fps) or playback_fps <= 0.0:
        raise ValueError("--playback-fps must be positive and finite")
    if args.interactive:
        if args.headless:
            raise ValueError("--interactive requires --no-headless")
    else:
        if args.output is None:
            raise ValueError("--output is required unless --interactive is used")
        output_video = args.output.resolve()
        output_video.parent.mkdir(parents=True, exist_ok=True)
        if output_video.exists():
            raise FileExistsError(output_video)

    # SimulationApp must exist before importing omni, pxr, or Isaac APIs.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": "RaytracedLighting",
            "width": render_width,
            "height": render_height,
        }
    )
    writer: cv2.VideoWriter | None = None
    robot_camera_window = None
    started = time.perf_counter()
    try:
        import omni.kit.app
        import omni.usd
        import omni.replicator.core as rep
        import omni.timeline
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera
        from omni.kit.viewport.utility import get_active_viewport_and_window
        from omni.kit.viewport.window import ViewportWindow
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

        extension_manager = (
            omni.kit.app.get_app().get_extension_manager()
        )
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
            physics_dt=1.0 / fps,
            rendering_dt=1.0 / fps,
        )
        timeline = omni.timeline.get_timeline_interface()
        robot_path = import_g1(urdf_path)
        robot = world.scene.add(
            Articulation(
                prim_paths_expr=robot_path,
                name="g1_rgbd_repaint",
            )
        )

        dome = UsdLux.DomeLight.Define(
            stage,
            Sdf.Path("/World/RobotDomeLight"),
        )
        dome.CreateIntensityAttr(float(isaac_lighting["dome_intensity"]))
        dome.CreateEnableColorTemperatureAttr(True)
        dome.CreateColorTemperatureAttr(
            float(isaac_lighting["dome_temperature"])
        )
        key = UsdLux.DistantLight.Define(
            stage,
            Sdf.Path("/World/RobotKeyLight"),
        )
        key.CreateIntensityAttr(float(isaac_lighting["key_intensity"]))
        key.CreateEnableColorTemperatureAttr(True)
        key.CreateColorTemperatureAttr(
            float(isaac_lighting["key_temperature"])
        )
        key.CreateAngleAttr(float(isaac_lighting["key_angle"]))
        key_rotation = [
            float(value) for value in isaac_lighting["key_rotation"]
        ]
        key.AddRotateXYZOp().Set(Gf.Vec3f(*key_rotation))

        mesh = UsdGeom.Mesh.Define(
            stage,
            Sdf.Path("/World/VideoRgbdMesh"),
        )
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        uv_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        )
        if args.background_mode == "video-plane":
            uv_primvar.Set(
                Vt.Vec2fArray.FromNumpy(
                    np.asarray(
                        [
                            [0.0, 1.0],
                            [1.0, 1.0],
                            [0.0, 0.0],
                            [1.0, 0.0],
                        ],
                        dtype=np.float32,
                    )
                )
            )
        else:
            uv_primvar.Set(
                Vt.Vec2fArray.FromNumpy(
                    grid_uvs(*mesh_size).astype(np.float32)
                )
            )

        material = UsdShade.Material.Define(
            stage,
            Sdf.Path("/World/VideoMaterial"),
        )
        surface = UsdShade.Shader.Define(
            stage,
            Sdf.Path("/World/VideoMaterial/Surface"),
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
            Sdf.Path("/World/VideoMaterial/Texture"),
        )
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput(
            "sourceColorSpace",
            Sdf.ValueTypeNames.Token,
        ).Set("sRGB")
        reader = UsdShade.Shader.Define(
            stage,
            Sdf.Path("/World/VideoMaterial/StReader"),
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
        camera_path = "/World/RobotCamera"
        camera_schema = UsdGeom.Camera.Define(
            stage,
            Sdf.Path(camera_path),
        )
        camera_schema.CreateHorizontalApertureAttr(
            horizontal_aperture * 10.0
        )
        camera_schema.CreateVerticalApertureAttr(
            vertical_aperture * 10.0
        )
        camera_schema.CreateFocalLengthAttr(focal_length * 10.0)
        camera = Camera(
            prim_path=camera_path,
            name="robot_camera",
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

        world.get_physics_context().set_gravity(0.0)
        world.reset()
        world.play()
        camera.initialize()
        camera.set_clipping_range(0.01, 100.0)
        camera.set_lens_aperture(0.0)
        camera.set_opencv_pinhole_properties(
            cx=float(render_intrinsic[0, 2]),
            cy=float(render_intrinsic[1, 2]),
            fx=float(render_intrinsic[0, 0]),
            fy=float(render_intrinsic[1, 1]),
            pinhole=d435_distortion.tolist(),
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
        resolved_distortion_array = np.asarray(
            resolved_distortion[: len(d435_distortion)],
            dtype=np.float64,
        )
        if not np.allclose(
            resolved_distortion_array,
            d435_distortion,
            rtol=1e-6,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Isaac camera distortion mismatch: "
                f"expected={d435_distortion.tolist()}, "
                f"actual={resolved_distortion_array.tolist()}"
            )

        capture_camera = camera
        capture_camera_path = camera_path
        scene_camera_eye: list[float] | None = None
        scene_camera_target: list[float] | None = None
        if args.capture_view == "scene":
            scene_camera_path = "/World/SceneCamera"
            scene_camera_schema = UsdGeom.Camera.Define(
                stage,
                Sdf.Path(scene_camera_path),
            )
            scene_camera_schema.CreateHorizontalApertureAttr(
                horizontal_aperture * 10.0
            )
            scene_camera_schema.CreateVerticalApertureAttr(
                vertical_aperture * 10.0
            )
            scene_camera_schema.CreateFocalLengthAttr(focal_length * 10.0)
            scene_camera = Camera(
                prim_path=scene_camera_path,
                name="scene_camera",
                resolution=(render_width, render_height),
            )
            scene_camera.set_horizontal_aperture(
                horizontal_aperture,
                maintain_square_pixels=False,
            )
            scene_camera.set_vertical_aperture(
                vertical_aperture,
                maintain_square_pixels=False,
            )
            scene_camera.set_focal_length(focal_length)
            scene_camera.initialize()
            scene_camera.set_clipping_range(0.01, 100.0)
            scene_camera.set_lens_aperture(0.0)
            default_target = (
                root_pose[0, :3] + mesh_camera_pose[0, :3]
            ) * 0.5
            default_eye = default_target + np.asarray([1.6, -1.8, 1.0])
            scene_eye = np.asarray(
                args.scene_eye
                if args.scene_eye is not None
                else default_eye,
                dtype=np.float64,
            )
            scene_target = np.asarray(
                args.scene_target
                if args.scene_target is not None
                else default_target,
                dtype=np.float64,
            )
            scene_camera.set_world_pose(
                position=scene_eye.astype(np.float32),
                orientation=_look_at_usd_wxyz(
                    scene_eye,
                    scene_target,
                ).astype(np.float32),
                camera_axes="usd",
            )
            capture_camera = scene_camera
            capture_camera_path = scene_camera_path
            scene_camera_eye = scene_eye.tolist()
            scene_camera_target = scene_target.tolist()

        if args.interactive:
            viewport, _ = get_active_viewport_and_window()
            if viewport is None:
                raise RuntimeError("Isaac GUI has no active viewport")
            if args.capture_view == "scene":
                # A regular USD capture camera does not provide the standard
                # free-orbit interaction. Initialize Isaac's free perspective
                # camera from scene-eye/target for the interactive viewport.
                interactive_viewport_camera_path = "/OmniverseKit_Persp"
                viewport.set_active_camera(interactive_viewport_camera_path)
                simulation_app.update()
                assert scene_eye is not None and scene_target is not None
                set_camera_view(
                    eye=scene_eye,
                    target=scene_target,
                    camera_prim_path=interactive_viewport_camera_path,
                    viewport_api=viewport,
                )
            else:
                interactive_viewport_camera_path = capture_camera_path
                viewport.set_active_camera(capture_camera_path)
            if args.robot_camera_window:
                robot_camera_window = ViewportWindow(
                    "G1 First-Person Camera",
                    width=max(640, render_width // 2),
                    height=max(360, render_height // 2),
                )
                robot_camera_window.viewport_api.set_active_camera(camera_path)
                simulation_app.update()

        joint_order = isaac_joint_order(
            list(robot.dof_names),
            joint_names,
        )
        zero_velocity = np.zeros((1, len(joint_order)), dtype=np.float32)
        zero_root_velocity = np.zeros((1, 6), dtype=np.float32)
        topology_counts: list[int] = []
        first_render = True

        def apply_frame(
            frame_id: int,
            *,
            capture: bool,
        ) -> np.ndarray | None:
            nonlocal first_render
            if args.interactive:
                timeline.set_auto_update(True)
                world.play()
            if args.background_mode == "video-plane":
                assert video_plane_points_camera is not None
                frame_plane_points_camera = (
                    video_plane_points_camera[frame_id]
                    if video_plane_points_camera.ndim == 3
                    else video_plane_points_camera
                )
                points_world = camera_points_to_world(
                    frame_plane_points_camera,
                    mesh_camera_pose[frame_id],
                )
                valid_points = np.ones(4, dtype=bool)
                triangles = np.asarray(
                    [[0, 2, 3], [0, 3, 1]],
                    dtype=np.int32,
                )
            else:
                depth_mm = cv2.imread(
                    str(depth_paths[frame_id]),
                    cv2.IMREAD_UNCHANGED,
                )
                if (
                    depth_mm is None
                    or depth_mm.dtype != np.uint16
                    or depth_mm.shape != (mesh_size[1], mesh_size[0])
                ):
                    raise ValueError(
                        f"Invalid prepared depth frame: "
                        f"{depth_paths[frame_id]}"
                    )
                depth_m = depth_mm.astype(np.float64) / 1000.0
                points_camera = backproject_z_depth(
                    depth_m,
                    mesh_intrinsic,
                )
                points_world = camera_points_to_world(
                    points_camera,
                    mesh_camera_pose[frame_id],
                )
                valid_points = np.isfinite(points_world).all(axis=-1)
                points_world[~valid_points] = 0.0
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
            mesh.GetExtentAttr().Set(
                Vt.Vec3fArray.FromNumpy(
                    np.stack(
                        [
                            points_world[valid_points].min(axis=0),
                            points_world[valid_points].max(axis=0),
                        ]
                    ).astype(np.float32)
                )
            )
            topology_counts.append(len(triangles))
            texture_file_input.Set(
                Sdf.AssetPath(str(rgb_paths[frame_id].resolve()))
            )

            root = root_pose[frame_id]
            robot.set_world_poses(
                positions=root[None, :3].astype(np.float32),
                orientations=root[[6, 3, 4, 5]][None].astype(np.float32),
            )
            robot.set_joint_positions(
                joint_positions[frame_id, joint_order][None].astype(
                    np.float32
                )
            )
            robot.set_joint_velocities(zero_velocity)
            robot.set_velocities(zero_root_velocity)
            # Commit articulation tensor writes to Fabric/USD before the
            # Replicator camera captures this pose.
            world.step(render=False)
            camera_position, camera_wxyz = (
                camera_cv_pose_to_usd_wxyz(camera_pose[frame_id])
            )
            camera.set_world_pose(
                position=camera_position.astype(np.float32),
                orientation=camera_wxyz.astype(np.float32),
                camera_axes="usd",
            )

            if args.interactive:
                world.pause()
                timeline.set_auto_update(False)
                timeline.commit()
                # Keep Replicator out of interactive mode: it owns the
                # timeline while running.  A normal viewport render is enough
                # because no sensor pixels are read back here.
                world.render()
                if first_render:
                    world.render()
                    first_render = False
                return None

            # The first texture/material upload can lag the first captured
            # frame even when rt_subframes is greater than one. Render and
            # discard one fully posed frame so frame zero is not a shader
            # compilation/texture upload artifact.
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
            if not capture:
                return None
            rgba = capture_camera.get_rgba()
            if rgba is None:
                raise RuntimeError(
                    f"Isaac camera returned no frame at {frame_id}"
                )
            rgb = np.asarray(rgba)[..., :3]
            if rgb.shape != (render_height, render_width, 3):
                raise RuntimeError(f"Unexpected camera shape: {rgb.shape}")
            return rgb

        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        if args.interactive:
            current_frame = 0
            completed_passes = 0
            playing = bool(args.loop)
            frame_period = 1.0 / playback_fps
            apply_frame(current_frame, capture=False)
            next_frame_at = time.perf_counter() + frame_period
            print(
                "\nInteractive Isaac RGB-D viewer\n"
                f"  playback: {playback_fps:.3f} FPS, "
                f"{'looping' if args.loop else 'manual stepping'}\n"
                "  s          : pause/resume playback\n"
                "  Enter or n : next frame\n"
                "  p          : previous frame\n"
                "  g N        : jump to source frame N\n"
                "  q          : quit\n"
                "The Isaac GUI stays live while the timeline is paused.\n"
                "Use normal viewport mouse controls in the main scene "
                "window.\n",
                flush=True,
            )
            print(
                f"showing local 0/{frame_count - 1}, "
                f"source frame {int(frame_indices[0])}",
                flush=True,
            )
            print("command> ", end="", flush=True)
            try:
                while simulation_app.is_running():
                    # Keep Kit processing mouse/keyboard/window events while
                    # the Isaac timeline remains paused.  Trajectory playback
                    # advances on wall-clock time so GUI camera interaction is
                    # independent from physics.
                    simulation_app.update()
                    readable, _, _ = select.select(
                        [sys.stdin],
                        [],
                        [],
                        0.0,
                    )
                    if readable:
                        line = sys.stdin.readline()
                        if line == "":
                            break
                        normalized = line.strip().lower()
                        if normalized in {"s", "play", "pause"}:
                            playing = not playing
                            next_frame_at = (
                                time.perf_counter() + frame_period
                            )
                            print(
                                "playback resumed"
                                if playing
                                else "playback paused",
                                flush=True,
                            )
                        else:
                            requested_frame, should_quit, message = (
                                _resolve_interactive_command(
                                    line,
                                    current_frame=current_frame,
                                    frame_indices=frame_indices,
                                )
                            )
                            if should_quit:
                                break
                            if message is not None:
                                print(message, flush=True)
                            else:
                                playing = False
                                if requested_frame != current_frame:
                                    current_frame = requested_frame
                                    apply_frame(
                                        current_frame,
                                        capture=False,
                                    )
                            print(
                                f"showing local "
                                f"{current_frame}/{frame_count - 1}, "
                                f"source frame "
                                f"{int(frame_indices[current_frame])}",
                                flush=True,
                            )
                        print("command> ", end="", flush=True)
                        continue

                    now = time.perf_counter()
                    if playing and now >= next_frame_at:
                        current_frame += 1
                        if current_frame >= frame_count:
                            current_frame = 0
                            completed_passes += 1
                        apply_frame(current_frame, capture=False)
                        next_frame_at = max(
                            next_frame_at + frame_period,
                            time.perf_counter(),
                        )
                        continue
                    time.sleep(1.0 / 120.0)
            except KeyboardInterrupt:
                pass
            report: dict[str, object] = {
                "schema_version": 1,
                "method": render_method,
                "mode": "interactive",
                "prepared_root": str(prepared_root),
                "prepared_depth_provenance": (
                    prepare_report.get("depth")
                    if prepare_report is not None
                    else None
                ),
                "frame_count": frame_count,
                "source_fps": fps,
                "playback_fps": playback_fps,
                "loop": bool(args.loop),
                "completed_passes": completed_passes,
                "last_local_frame": current_frame,
                "source_size": list(source_size),
                "render_size": [render_width, render_height],
                "source_intrinsic": intrinsic.tolist(),
                "render_intrinsic": render_intrinsic.tolist(),
                "render_intrinsic_source": render_intrinsic_source,
                "resolved_isaac_intrinsic": resolved_intrinsic.tolist(),
                "d435_distortion": d435_distortion.tolist(),
                "resolved_isaac_distortion": (
                    resolved_distortion_array.tolist()
                ),
                "capture_view": args.capture_view,
                "interactive_viewport_camera": (
                    interactive_viewport_camera_path
                ),
                "render_camera_pose_source": render_camera_pose_source,
                "mesh_pose_source": mesh_pose_source,
                "background_mode": args.background_mode,
                "video_plane_distance_m": video_plane_distance,
                "video_plane_distance_source": video_plane_distance_source,
                "video_plane_source_to_robot_scale": video_plane_scale,
                "video_plane_source_center_depth_m": (
                    video_plane_source_depth
                ),
                "video_plane_depth_anchor": (
                    video_plane_depth_anchor_report
                ),
                "co_centered_scale_alignment": (
                    co_centered_scale_report
                ),
                "robot_camera_frame": args.d435_frame,
                "d435_optical_center_in_link_xyz_m": (
                    optical_center_in_link.tolist()
                ),
                "mapped_source_to_d435_delta": source_to_d435_delta,
                "scene_world_offset_xyz_m": [
                    0.0,
                    0.0,
                    args.scene_z_offset,
                ],
                "scene_camera_eye": scene_camera_eye,
                "scene_camera_target": scene_camera_target,
                "elapsed_seconds": time.perf_counter() - started,
            }
            print(json.dumps(report, indent=2), flush=True)
            return report

        assert output_video is not None
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (render_width, render_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_video}")
        print(f"Isaac RGB-D rendered 0/{frame_count}", flush=True)
        for frame_id in range(frame_count):
            rgb = apply_frame(frame_id, capture=True)
            assert rgb is not None
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(
                f"Isaac RGB-D rendered {frame_id + 1}/{frame_count}",
                flush=True,
            )
        writer.release()
        writer = None
        report: dict[str, object] = {
            "schema_version": 1,
            "method": render_method,
            "prepared_root": str(prepared_root),
            "prepared_depth_provenance": (
                prepare_report.get("depth")
                if prepare_report is not None
                else None
            ),
            "output_video": str(output_video),
            "frame_count": frame_count,
            "fps": fps,
            "source_size": list(source_size),
            "mesh_size": list(mesh_size),
            "render_size": [render_width, render_height],
            "source_intrinsic": intrinsic.tolist(),
            "render_intrinsic": render_intrinsic.tolist(),
            "render_intrinsic_source": render_intrinsic_source,
            "resolved_isaac_intrinsic": resolved_intrinsic.tolist(),
            "d435_distortion": d435_distortion.tolist(),
            "resolved_isaac_distortion": (
                resolved_distortion_array.tolist()
            ),
            "camera_axes": "R_world_from_camera_usd = R_cv @ diag(1,-1,-1)",
            "render_camera_pose_source": render_camera_pose_source,
            "mesh_pose_source": mesh_pose_source,
            "background_mode": args.background_mode,
            "video_plane_distance_m": video_plane_distance,
            "video_plane_distance_source": video_plane_distance_source,
            "video_plane_source_to_robot_scale": video_plane_scale,
            "video_plane_source_center_depth_m": video_plane_source_depth,
            "video_plane_depth_anchor": video_plane_depth_anchor_report,
            "co_centered_scale_alignment": co_centered_scale_report,
            "robot_camera_frame": args.d435_frame,
            "d435_link_from_cv_optical_rotation": (
                CAMERA_LINK_FROM_CV_OPTICAL.tolist()
            ),
            "d435_optical_center_in_link_xyz_m": (
                optical_center_in_link.tolist()
            ),
            "mapped_source_to_d435_delta": source_to_d435_delta,
            "scene_world_offset_xyz_m": [
                0.0,
                0.0,
                args.scene_z_offset,
            ],
            "capture_view": args.capture_view,
            "scene_camera_eye": scene_camera_eye,
            "scene_camera_target": scene_camera_target,
            "background_material": "UsdPreviewSurface emissive texture",
            "lighting_affects": "G1; RGB-D background is emissive",
            "mean_triangle_count": float(np.mean(topology_counts)),
            "min_triangle_count": int(np.min(topology_counts)),
            "max_triangle_count": int(np.max(topology_counts)),
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = output_video.with_suffix(".json")
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        if writer is not None:
            writer.release()
        if robot_camera_window is not None:
            robot_camera_window.destroy()
        archive.close()
        simulation_app.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a prepared processed-video plane or RGB-D mesh and a "
            "LeRobot G1 articulation from its physical D435 optical camera."
        )
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4; omit it in --interactive mode",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Keep the Isaac GUI open for adjustable-view playback or "
            "terminal frame stepping"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "Continuously loop the sequence in --interactive mode; press s "
            "to pause or resume"
        ),
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        help=(
            "Interactive wall-clock playback rate; default is the prepared "
            "episode FPS"
        ),
    )
    parser.add_argument(
        "--robot-camera-window",
        action="store_true",
        help=(
            "Open a second interactive viewport for /World/RobotCamera"
        ),
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=1.0,
        help=(
            "Output resolution scale relative to the fixed 640x480 "
            "BeingBeyond calibration; K is scaled by the same factor, so "
            "the physical field of view is unchanged"
        ),
    )
    parser.add_argument(
        "--background-mode",
        choices=("video-plane", "rgbd-mesh"),
        default="video-plane",
        help=(
            "Reproject RGB on one calibrated plane without depth (default), "
            "or use the experimental per-pixel RGB-D mesh"
        ),
    )
    parser.add_argument(
        "--video-plane-distance",
        type=float,
        help=(
            "Video plane distance in its source camera frame, in robot-world "
            "meters; default is derived from calibrated hand depth"
        ),
    )
    parser.add_argument(
        "--video-plane-clearance",
        type=float,
        default=0.05,
        help=(
            "Extra robot-world distance used by non-depth automatic plane "
            "placement (default: 0.05 m)"
        ),
    )
    parser.add_argument(
        "--video-plane-scale",
        type=float,
        help=(
            "Source-to-robot scale override in d435-centered mode; default "
            "is the prepared per-task scene_scale"
        ),
    )
    parser.add_argument(
        "--video-plane-hand-confidence",
        type=float,
        default=0.5,
        help=(
            "Minimum source HDF5 hand confidence used by d435-centered "
            "scale/depth calibration (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--video-plane-depth-center-fraction",
        type=float,
        default=0.05,
        help=(
            "Half-width/height fraction of the prepared metric-depth center "
            "ROI in "
            "d435-centered mode (default: 0.05)"
        ),
    )
    parser.add_argument(
        "--video-plane-depth-median-window",
        type=int,
        default=9,
        help=(
            "Odd temporal median-filter window for center depth "
            "(default: 9 frames)"
        ),
    )
    parser.add_argument(
        "--video-plane-depth-offset",
        type=float,
        default=0.0,
        help=(
            "Robot-world offset added after scaling center depth "
            "(default: 0 m)"
        ),
    )
    parser.add_argument(
        "--camera-pose-source",
        choices=(
            "source-extrinsic",
            "d435-centered",
            "d435-optical",
            "mapped-source",
        ),
        default="source-extrinsic",
        help=(
            "Place the mesh with one rigid source-extrinsic world transform "
            "while rendering from G1 D435 (default); d435-centered forces "
            "the mapped human/D435 optical centers equal while retaining "
            "human-camera rotation; d435-optical attaches the scene to the "
            "camera, and mapped-source is diagnostic"
        ),
    )
    parser.add_argument(
        "--d435-frame",
        default="d435_link",
        help="URDF frame used as the robot camera mount",
    )
    parser.add_argument(
        "--d435-optical-center-in-link",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help=(
            "D435 RGB optical-center translation expressed in d435_link "
            "coordinates, in meters"
        ),
    )
    parser.add_argument(
        "--scene-z-offset",
        type=float,
        default=0.0,
        help=(
            "Diagnostic mapped-source-only world-Z translation; rejected "
            "in physical D435 modes"
        ),
    )
    parser.add_argument(
        "--depth-edge-absolute",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--depth-edge-relative",
        type=float,
        default=0.05,
    )
    parser.add_argument("--rt-subframes", type=int, default=2)
    parser.add_argument(
        "--capture-view",
        choices=("robot-camera", "scene"),
        default="robot-camera",
    )
    parser.add_argument(
        "--scene-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--scene-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    render(build_parser().parse_args())


if __name__ == "__main__":
    main()
