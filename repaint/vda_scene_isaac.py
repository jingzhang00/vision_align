#!/usr/bin/env python3
"""Replay a prepared Video Depth Anything scene and G1 trajectory in Isaac.

Each prepared RGB-D frame is reconstructed as a textured USD mesh in the
recorded camera pose while the frame-aligned G1 root and joint trajectory are
applied to an Isaac articulation.  The combined replay can either be captured
to one fixed-camera MP4 or viewed in an interactive Isaac viewport.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
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
    camera_points_to_world,
    depth_grid_triangles,
    grid_uvs,
    scale_intrinsic,
)
from repaint.scene_hand_alignment import (  # noqa: E402
    align_scene_camera_poses_to_initial_hands,
)


@dataclass(frozen=True)
class PreparedScene:
    """Validated scene and robot data needed by the combined renderer."""

    root: Path
    rgb_paths: tuple[Path, ...]
    depth_paths: tuple[Path, ...]
    frame_indices: np.ndarray
    fps: float
    source_size: tuple[int, int]
    mesh_size: tuple[int, int]
    intrinsic: np.ndarray
    mesh_intrinsic: np.ndarray
    camera_poses_cv_xyzw: np.ndarray
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    root_poses_xyzw: np.ndarray
    g1_urdf: Path
    source_hdf5: Path | None
    scene_scale: float | None

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
            "source_size",
            "mesh_size",
            "frame_indices",
            "camera_pose_cv_xyzw",
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

        source_hdf5: Path | None = None
        if "source_hdf5" in archive:
            source_hdf5_value = np.asarray(archive["source_hdf5"])
            if source_hdf5_value.shape != ():
                raise ValueError("source_hdf5 must be a scalar path")
            source_hdf5 = Path(
                str(source_hdf5_value.item())
            ).expanduser().resolve()

        scene_scale: float | None = None
        if "scene_scale" in archive:
            scene_scale = float(archive["scene_scale"])
            if not np.isfinite(scene_scale) or scene_scale <= 0.0:
                raise ValueError(
                    "Prepared scene_scale must be positive and finite"
                )

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
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        frame_indices=frame_indices,
        fps=fps,
        source_size=source_size,
        mesh_size=mesh_size,
        intrinsic=intrinsic,
        mesh_intrinsic=mesh_intrinsic,
        camera_poses_cv_xyzw=camera_poses,
        joint_names=joint_names,
        joint_positions=joint_positions,
        root_poses_xyzw=root_poses,
        g1_urdf=g1_urdf,
        source_hdf5=source_hdf5,
        scene_scale=scene_scale,
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
            scene.camera_poses_cv_xyzw[frame_id],
        )
        valid = np.isfinite(points_world).all(axis=-1)
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
    if args.playback_fps is not None and not interactive:
        raise ValueError("--playback-fps is only valid with --interactive")

    playback_fps = (
        source_fps
        if args.playback_fps is None
        else float(args.playback_fps)
    )
    if not np.isfinite(playback_fps) or playback_fps <= 0.0:
        raise ValueError("--playback-fps must be positive and finite")

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
    for path in (output_video, report_path):
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
    if args.scene_alignment == "initial-hands":
        if scene.source_hdf5 is None or scene.scene_scale is None:
            raise ValueError(
                "--scene-alignment initial-hands requires source_hdf5 and "
                "scene_scale in prepared episode_data.npz; use "
                "--scene-alignment none only for legacy prepared data"
            )
        alignment = align_scene_camera_poses_to_initial_hands(
            source_hdf5=scene.source_hdf5,
            frame_indices=scene.frame_indices,
            scene_scale=scene.scene_scale,
            scene_camera_pose_cv_xyzw=scene.camera_poses_cv_xyzw,
            urdf_path=scene.g1_urdf,
            joint_names=scene.joint_names,
            joint_positions=scene.joint_positions,
            root_pose_xyzw=scene.root_poses_xyzw,
            window_frames=args.scene_alignment_window_frames,
            confidence_threshold=args.scene_hand_confidence,
        )
        scene = replace(
            scene,
            camera_poses_cv_xyzw=alignment.camera_pose_cv_xyzw,
        )
        scene_alignment_report = alignment.report
        scene_translation_world = alignment.translation_world_xyz_m
        print(
            "Initial hand-anchor scene placement: "
            f"delta_world={scene_translation_world.tolist()} m, "
            f"frames={scene_alignment_report['paired_frame_count']}",
            flush=True,
        )
    else:
        scene_translation_world = np.zeros(3, dtype=np.float64)
        scene_alignment_report = {
            "schema_version": 1,
            "method": "disabled",
            "requested_mode": "none",
            "world_translation_xyz_m": [0.0, 0.0, 0.0],
            "application_count": 0,
        }
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

    render_size = (
        scene.source_size
        if args.render_size is None
        else tuple(int(value) for value in args.render_size)
    )
    if min(render_size) <= 1:
        raise ValueError("--render-size dimensions must exceed one pixel")
    if any(dimension % 2 for dimension in render_size):
        raise ValueError("--render-size dimensions must be even")
    render_intrinsic = scale_intrinsic(
        scene.intrinsic,
        source_size=scene.source_size,
        target_size=render_size,
    )
    scan = scan_scene(
        scene,
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
        dome.CreateIntensityAttr(500.0)

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
        camera_path = "/World/VdaSceneCamera"
        camera_schema = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
        camera_schema.CreateHorizontalApertureAttr(horizontal_aperture * 10.0)
        camera_schema.CreateVerticalApertureAttr(vertical_aperture * 10.0)
        camera_schema.CreateFocalLengthAttr(focal_length * 10.0)
        camera = Camera(
            prim_path=camera_path,
            name="vda_scene_camera",
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
        scene_diagonal = float(
            np.linalg.norm(scan.bounds_max - scan.bounds_min)
        )
        camera.set_clipping_range(
            0.01,
            max(
                100.0,
                np.linalg.norm(scene_eye - scene_target)
                + 3.0 * scene_diagonal,
            ),
        )
        camera.set_lens_aperture(0.0)
        camera.set_opencv_pinhole_properties(
            cx=float(render_intrinsic[0, 2]),
            cy=float(render_intrinsic[1, 2]),
            fx=float(render_intrinsic[0, 0]),
            fy=float(render_intrinsic[1, 1]),
            pinhole=np.zeros(12, dtype=np.float64).tolist(),
        )
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
        print(
            f"Isaac G1 articulation: {robot_path} · "
            f"{len(joint_order)} DoF",
            flush=True,
        )

        def apply_frame(frame_id: int) -> None:
            if run_options.interactive:
                timeline.set_auto_update(True)
                world.play()
            rgb_path = scene.rgb_paths[frame_id]
            depth_path = scene.depth_paths[frame_id]
            depth_m = _load_depth_m(depth_path, scene.mesh_size)
            points_camera = backproject_z_depth(
                depth_m,
                scene.mesh_intrinsic,
            )
            points_world = camera_points_to_world(
                points_camera,
                scene.camera_poses_cv_xyzw[frame_id],
            )
            valid = np.isfinite(points_world).all(axis=-1)
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
            robot.set_world_poses(
                positions=root_pose[None, :3].astype(np.float32),
                orientations=root_pose[[6, 3, 4, 5]][None].astype(
                    np.float32
                ),
            )
            robot.set_joint_positions(
                scene.joint_positions[
                    frame_id,
                    joint_order,
                ][None].astype(np.float32)
            )
            robot.set_joint_velocities(zero_joint_velocity)
            robot.set_velocities(zero_root_velocity)
            # Commit both the articulation tensor writes and mesh update before
            # this frame is presented or captured.
            world.step(render=False)
            if run_options.interactive:
                world.pause()
                timeline.set_auto_update(False)
                timeline.commit()

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
                        time.sleep(1.0 / 120.0)
                        continue
                    if now < next_frame_at:
                        time.sleep(
                            min(
                                next_frame_at - now,
                                1.0 / 240.0,
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
                    # update here: that would make 30 FPS playback miss its
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
                "schema_version": 1,
                "method": (
                    "video-depth-anything-dynamic-rgbd-mesh"
                    "-plus-g1-articulation"
                ),
                "mode": "interactive",
                "prepared_root": str(scene.root),
                "robot": {
                    "urdf": str(scene.g1_urdf),
                    "articulation_path": robot_path,
                    "dof_count": len(joint_order),
                    "trajectory_source": (
                        "episode_data.npz joint_positions/root_pose_xyzw"
                    ),
                },
                "frame_count": scene.frame_count,
                "source_fps": scene.fps,
                "playback_fps": run_options.playback_fps,
                "loop": run_options.loop,
                "completed_passes": completed_passes,
                "last_local_frame": current_frame,
                "render_size": list(render_size),
                "interactive_viewport_camera": str(viewport_camera_path),
                "initial_view": {
                    "eye_xyz_m": scene_eye.tolist(),
                    "target_xyz_m": scene_target.tolist(),
                },
                "scene_alignment": scene_alignment_report,
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
            render_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open MP4 writer: {output_video}")

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
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(
                f"Isaac VDA + G1 rendered {frame_id + 1}/"
                f"{scene.frame_count}",
                flush=True,
            )

        writer.release()
        writer = None
        median_depth_delta = np.abs(np.diff(scan.median_depths_m))
        triangle_delta = np.abs(np.diff(scan.triangle_counts))
        valid_ratio = (
            scan.valid_vertex_counts
            / float(scene.mesh_size[0] * scene.mesh_size[1])
        )
        report: dict[str, object] = {
            "schema_version": 1,
            "method": (
                "video-depth-anything-dynamic-rgbd-mesh"
                "-plus-g1-articulation"
            ),
            "renderer": "Isaac Sim 5.1 RaytracedLighting",
            "prepared_root": str(scene.root),
            "output_video": str(output_video),
            "robot": {
                "urdf": str(scene.g1_urdf),
                "articulation_path": robot_path,
                "dof_count": len(joint_order),
                "trajectory_source": (
                    "episode_data.npz joint_positions/root_pose_xyzw"
                ),
            },
            "frame_count": scene.frame_count,
            "fps": scene.fps,
            "duration_seconds": scene.frame_count / scene.fps,
            "source_frame_indices": scene.frame_indices.tolist(),
            "source_size": list(scene.source_size),
            "mesh_size": list(scene.mesh_size),
            "render_size": list(render_size),
            "source_intrinsic": scene.intrinsic.tolist(),
            "mesh_intrinsic": scene.mesh_intrinsic.tolist(),
            "render_intrinsic": render_intrinsic.tolist(),
            "mesh_pose_source": (
                "prepared camera_pose_cv_xyzw plus one constant world "
                "translation from initial bilateral hand midpoint"
                if args.scene_alignment == "initial-hands"
                else "prepared camera_pose_cv_xyzw"
            ),
            "camera_axes": "prepared poses are OpenCV world-from-camera",
            "scene_alignment": scene_alignment_report,
            "capture_camera": {
                "type": "fixed third-person",
                "eye_xyz_m": scene_eye.tolist(),
                "eye_source": eye_source,
                "target_xyz_m": scene_target.tolist(),
                "target_source": target_source,
            },
            "scene_bounds_xyz_m": {
                "min": scan.bounds_min.tolist(),
                "max": scan.bounds_max.tolist(),
                "unaligned_min": (
                    scan.bounds_min - scene_translation_world
                ).tolist(),
                "unaligned_max": (
                    scan.bounds_max - scene_translation_world
                ).tolist(),
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
            "frame-aligned G1 trajectory, either to a fixed-camera MP4 or "
            "in an adjustable interactive viewport."
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
        "--render-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Even render dimensions; default is the prepared source_size",
    )
    parser.add_argument(
        "--scene-eye",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Fixed batch camera or initial interactive eye; default fits bounds"
        ),
    )
    parser.add_argument(
        "--scene-target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Fixed batch or initial interactive target; default is bounds center"
        ),
    )
    parser.add_argument(
        "--scene-alignment",
        choices=("initial-hands", "none"),
        default="initial-hands",
        help=(
            "At replay start, place the complete scene with one robust "
            "source/G1 bilateral-hand-midpoint translation, or disable it "
            "for legacy data (default: initial-hands)"
        ),
    )
    parser.add_argument(
        "--scene-alignment-window-frames",
        type=int,
        default=15,
        help=(
            "Initial frames used once for hand-anchor scene placement "
            "(default: 15, i.e. 0.5 s at 30 FPS)"
        ),
    )
    parser.add_argument(
        "--scene-hand-confidence",
        type=float,
        default=0.5,
        help=(
            "Minimum source-HDF5 confidence for both hands during initial "
            "scene placement (default: 0.5; absent confidences mean 1)"
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
        "--playback-fps",
        type=float,
        help=(
            "Interactive wall-clock playback rate; default is the prepared FPS"
        ),
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
