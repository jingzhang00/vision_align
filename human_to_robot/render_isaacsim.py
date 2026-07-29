from __future__ import annotations

import argparse
import json
import os
import traceback
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .lighting_profile import (
    DEFAULT_ISAAC_PROFILE,
    load_lighting_profile,
    resolve_profile_value,
)
from .video_io import video_metadata


ARM_LINKS = {
    f"{side}_{link}"
    for side in ("left", "right")
    for link in (
        "shoulder_pitch_link",
        "shoulder_roll_link",
        "shoulder_yaw_link",
        "elbow_link",
        "wrist_roll_link",
        "wrist_pitch_link",
        "wrist_yaw_link",
    )
}

HAND_LINK_PREFIXES = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in (
        "hand_",
        "thumb_",
        "index_",
        "middle_",
        "ring_",
        "pinky_",
    )
)


def _camera_projection_from_intrinsic(
    intrinsic: np.ndarray,
    *,
    width: int,
    height: int,
    horizontal_aperture: float = 2.0955,
) -> tuple[float, float, float]:
    """Convert the supported OpenCV pinhole K to Isaac/USD camera values."""

    isaac_intrinsic = _isaac_intrinsic_from_source(
        intrinsic,
        width=width,
        height=height,
    )
    fx = float(isaac_intrinsic[0, 0])
    fy = float(isaac_intrinsic[1, 1])
    focal_length = fx / float(width) * horizontal_aperture
    vertical_aperture = (
        focal_length * float(height) / fy
    )
    return horizontal_aperture, vertical_aperture, focal_length


def _isaac_intrinsic_from_source(
    intrinsic: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Normalize a centered source K to Isaac's image-edge coordinates."""

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("Camera intrinsic must be a finite 3x3 matrix")
    if width <= 0 or height <= 0:
        raise ValueError("Camera resolution must be positive")
    canonical = intrinsic.copy()
    canonical[0, 0] = 1.0
    canonical[1, 1] = 1.0
    canonical[0, 2] = 0.0
    canonical[1, 2] = 0.0
    if not np.allclose(
        canonical,
        np.eye(3),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Camera intrinsic must have zero skew and bottom row [0, 0, 1]"
        )
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera focal lengths must be positive")
    if not np.isclose(fx, fy, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "Isaac Camera only supports square pixels, but "
            f"fx={fx} and fy={fy}"
        )
    isaac_center = (width / 2.0, height / 2.0)
    pixel_center = ((width - 1) / 2.0, (height - 1) / 2.0)
    centered = any(
        np.isclose(cx, center_x, rtol=0.0, atol=1e-6)
        and np.isclose(cy, center_y, rtol=0.0, atol=1e-6)
        for center_x, center_y in (isaac_center, pixel_center)
    )
    if not centered:
        raise ValueError(
            "Isaac Camera does not use aperture offsets; expected a "
            "centered principal point in either image-edge or pixel-center "
            f"coordinates, got ({cx}, {cy}) for {width}x{height}"
        )
    normalized = intrinsic.copy()
    normalized[0, 2], normalized[1, 2] = isaac_center
    return normalized


def _is_render_link(name: str) -> bool:
    return name in ARM_LINKS or name.startswith(HAND_LINK_PREFIXES)


def _resolve_urdf_mesh(source: Path, filename: str) -> Path:
    mesh_path = Path(filename)
    if mesh_path.is_absolute():
        resolved = mesh_path.resolve()
    else:
        parts = mesh_path.parts
        if (
            len(parts) >= 4
            and parts[0].lower() == "linker_hand_o6"
            and parts[1] in {"left", "right"}
            and parts[2] == "meshes"
        ):
            resolved = (
                source.parent.parent
                / "hand"
                / f"{parts[1]}_hand"
                / "meshes"
                / Path(*parts[3:])
            ).resolve()
        else:
            resolved = (source.parent / mesh_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"URDF mesh does not exist: {filename} -> {resolved}"
        )
    return resolved


def _build_arm_hand_urdf(
    source: Path,
    destination: Path,
) -> set[str]:
    """Keep the complete G1 tree but render only both arms and new hands."""

    source = source.resolve()
    destination = destination.resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    if root.tag != "robot" or not root.get("name"):
        raise ValueError(f"Invalid robot URDF: {source}")

    retained_links: set[str] = set()
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            raise ValueError(f"URDF contains an unnamed link: {source}")
        if not _is_render_link(link_name):
            for element_name in ("visual", "collision"):
                for element in list(link.findall(element_name)):
                    link.remove(element)
            continue

        if link.find("visual") is not None:
            retained_links.add(link_name)
        for mesh in link.findall("./visual/geometry/mesh") + link.findall(
            "./collision/geometry/mesh"
        ):
            filename = mesh.get("filename")
            if not filename:
                raise ValueError(
                    f"Mesh in {link_name} has no filename: {source}"
                )
            resolved = _resolve_urdf_mesh(source, filename)
            mesh.set(
                "filename",
                os.path.relpath(resolved, start=destination.parent),
            )

    expected_hand_bases = {"left_hand_base_link", "right_hand_base_link"}
    if not expected_hand_bases.issubset(retained_links):
        missing = sorted(expected_hand_bases - retained_links)
        raise ValueError(
            "New G1 hand links are missing from the URDF: "
            f"{', '.join(missing)}"
        )
    if not ARM_LINKS.issubset(retained_links):
        missing = sorted(ARM_LINKS - retained_links)
        raise ValueError(
            f"G1 arm links are missing from the URDF: {', '.join(missing)}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return retained_links


def _validate_archive(archive: np.lib.npyio.NpzFile) -> int:
    required = {
        "kinematics",
        "qpos",
        "joint_names",
        "base_position",
        "base_rotation",
        "link_names",
        "link_positions",
        "link_quaternions",
        "camera_positions",
        "camera_gl_quaternions",
        "scale",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        raise ValueError(f"Trajectory archive is missing: {', '.join(missing)}")
    if str(np.asarray(archive["kinematics"]).item()) != "pinocchio":
        raise ValueError("Isaac Sim requires a Pinocchio trajectory archive")
    frame_count = int(archive["qpos"].shape[0])
    if archive["camera_positions"].shape != (frame_count, 3):
        raise ValueError("camera_positions must have shape [frame_count, 3]")
    if archive["camera_gl_quaternions"].shape != (frame_count, 4):
        raise ValueError("camera_gl_quaternions must have shape [frame_count, 4]")
    if archive["base_position"].shape != (3,):
        raise ValueError("base_position must have shape [3]")
    if archive["base_rotation"].shape != (3, 3):
        raise ValueError("base_rotation must have shape [3, 3]")
    link_count = int(archive["link_names"].shape[0])
    if archive["link_positions"].shape != (
        frame_count,
        link_count,
        3,
    ):
        raise ValueError(
            "link_positions must have shape [frame_count, link_count, 3]"
        )
    if archive["link_quaternions"].shape != (
        frame_count,
        link_count,
        4,
    ):
        raise ValueError(
            "link_quaternions must have shape [frame_count, link_count, 4]"
        )
    return frame_count


def _resolve_render_settings(
    args: argparse.Namespace,
) -> tuple[dict[str, object] | None, Path | None]:
    profile_path = (
        args.lighting_profile.resolve()
        if args.lighting_profile is not None
        else None
    )
    profile = load_lighting_profile(profile_path)
    for key in (
        "dome_intensity",
        "dome_temperature",
        "key_intensity",
        "key_temperature",
        "key_angle",
        "key_rotation",
    ):
        value = resolve_profile_value(
            explicit=getattr(args, key),
            profile=profile,
            section="isaac",
            key=key,
            default=DEFAULT_ISAAC_PROFILE[key],
        )
        if key == "key_rotation":
            rotation = tuple(float(component) for component in value)  # type: ignore[arg-type]
            if len(rotation) != 3:
                raise ValueError("key_rotation must contain exactly three values")
            setattr(args, key, rotation)
        else:
            setattr(args, key, float(value))  # type: ignore[arg-type]
    return profile, profile_path


def render(args: argparse.Namespace) -> dict[str, object]:
    if (args.inpaint_video is None) != (args.composite_output is None):
        raise ValueError(
            "--inpaint-video and --composite-output must be supplied together"
        )
    if args.composite_output is not None and args.scene_depths is None:
        raise ValueError(
            "--scene-depths is required for in-process compositing"
        )
    lighting_profile, lighting_profile_path = _resolve_render_settings(args)
    source_video = args.video.resolve()
    render_width, render_height, source_frame_count, source_fps = (
        video_metadata(source_video)
    )
    if (
        min(render_width, render_height, source_frame_count) <= 0
        or source_fps <= 0
    ):
        raise ValueError(f"Invalid source video metadata: {source_video}")
    # SimulationApp must be constructed before importing any omni/pxr modules.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": args.renderer,
            "width": render_width,
            "height": render_height,
            "anti_aliasing": 0,
        }
    )

    try:
        import cv2
        import h5py
        import omni.kit.app
        import omni.kit.commands
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

        from .composite import InMemoryVideoCompositor
        from .video_io import write_video

        trajectory_path = args.trajectory.resolve()
        source_urdf = args.urdf.resolve()
        source_hdf5 = args.hdf5.resolve()
        output_root = args.output.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        stream_composite = args.composite_output is not None
        inpaint_video = (
            args.inpaint_video.resolve()
            if args.inpaint_video is not None
            else None
        )
        composite_output = (
            args.composite_output.resolve()
            if args.composite_output is not None
            else None
        )
        scene_depth_root = (
            args.scene_depths.resolve()
            if args.scene_depths is not None
            else None
        )
        render_urdf = output_root / "g1_arms_hands_isaacsim.urdf"
        visible_link_names = _build_arm_hand_urdf(
            source_urdf,
            render_urdf,
        )

        archive = np.load(trajectory_path)
        total_frames = _validate_archive(archive)
        archive_joint_names = [
            str(name) for name in archive["joint_names"].tolist()
        ]
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        if qpos.shape != (total_frames, len(archive_joint_names)):
            raise ValueError(
                "qpos must contain one value per archived joint: "
                f"expected {(total_frames, len(archive_joint_names))}, "
                f"got {qpos.shape}"
            )
        archived_link_names = [
            str(name) for name in archive["link_names"].tolist()
        ]
        link_indices = {
            name: index for index, name in enumerate(archived_link_names)
        }
        if len(link_indices) != len(archived_link_names):
            raise ValueError("Archived Pinocchio link names must be unique")
        missing_visible_poses = sorted(visible_link_names - set(link_indices))
        if missing_visible_poses:
            raise ValueError(
                "Rendered URDF links have no archived Pinocchio pose: "
                f"{missing_visible_poses}"
            )
        with h5py.File(source_hdf5, "r") as source_data:
            intrinsic = np.asarray(
                source_data["camera/intrinsic"], dtype=np.float64
            )
            hdf5_frame_count = int(
                source_data["transforms/camera"].shape[0]
            )
        if (
            source_frame_count != total_frames
            or hdf5_frame_count != total_frames
        ):
            raise ValueError(
                "Video, HDF5, and qpos trajectory frame counts must agree: "
                f"video={source_frame_count}, hdf5={hdf5_frame_count}, "
                f"qpos={total_frames}"
            )
        frame_count = (
            min(total_frames, args.max_frames)
            if args.max_frames is not None
            else total_frames
        )
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate(
            "isaacsim.asset.importer.urdf", True
        )
        extension_manager.set_extension_enabled_immediate(
            "isaacsim.sensors.camera", True
        )
        simulation_app.update()

        context = omni.usd.get_context()
        context.new_stage()
        simulation_app.update()
        stage = context.get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not np.isclose(meters_per_unit, 1.0):
            raise RuntimeError(
                "Isaac stage must use meters for metric depth, got "
                f"metersPerUnit={meters_per_unit}"
            )

        status, import_config = omni.kit.commands.execute(
            "URDFCreateImportConfig"
        )
        if not status:
            raise RuntimeError("URDFCreateImportConfig failed")
        import_config.set_fix_base(True)
        import_config.set_import_inertia_tensor(True)
        import_config.set_merge_fixed_joints(False)
        import_config.set_parse_mimic(True)
        import_config.set_self_collision(False)
        import_config.set_collision_from_visuals(False)
        import_config.set_create_physics_scene(False)
        parsed, urdf_robot = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path=str(render_urdf),
            import_config=import_config,
        )
        if not parsed or urdf_robot is None:
            raise RuntimeError(f"URDFParseFile failed: {render_urdf}")
        imported, robot_prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_path=str(render_urdf),
            urdf_robot=urdf_robot,
            import_config=import_config,
            dest_path="",
            get_articulation_root=False,
        )
        if not imported or not robot_prim_path:
            raise RuntimeError(f"URDFImportRobot failed: {render_urdf}")
        simulation_app.update()
        robot_prim_path = str(robot_prim_path)
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        if not robot_prim.IsValid():
            raise RuntimeError(
                "Isaac Sim returned an invalid URDF root path: "
                f"{robot_prim_path}"
            )

        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(args.dome_intensity)
        dome.CreateColorAttr((1.0, 1.0, 1.0))
        dome.CreateEnableColorTemperatureAttr(True)
        dome.CreateColorTemperatureAttr(args.dome_temperature)

        key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
        key.CreateIntensityAttr(args.key_intensity)
        key.CreateColorAttr((1.0, 1.0, 1.0))
        key.CreateEnableColorTemperatureAttr(True)
        key.CreateColorTemperatureAttr(args.key_temperature)
        key.CreateAngleAttr(args.key_angle)
        key.AddRotateXYZOp().Set(Gf.Vec3f(*args.key_rotation))

        (
            horizontal_aperture,
            vertical_aperture,
            focal_length,
        ) = _camera_projection_from_intrinsic(
            intrinsic,
            width=render_width,
            height=render_height,
        )
        isaac_intrinsic = _isaac_intrinsic_from_source(
            intrinsic,
            width=render_width,
            height=render_height,
        )
        camera_path = "/World/SynthesisCamera"
        camera_schema = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
        camera_schema.CreateHorizontalApertureAttr(
            horizontal_aperture * 10.0
        )
        camera_schema.CreateVerticalApertureAttr(
            vertical_aperture * 10.0
        )
        camera_schema.CreateFocalLengthAttr(focal_length * 10.0)
        camera = Camera(
            prim_path=camera_path,
            name="synthesis_camera",
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
        camera.initialize()
        resolved_intrinsic = np.asarray(
            camera.get_intrinsics_matrix(),
            dtype=np.float64,
        )
        if not np.allclose(
            resolved_intrinsic,
            isaac_intrinsic,
            rtol=1e-6,
            atol=1e-4,
        ):
            raise RuntimeError(
                "Isaac camera calibration does not match source K: "
                f"source={intrinsic.tolist()}, "
                f"expected={isaac_intrinsic.tolist()}, "
                f"actual={resolved_intrinsic.tolist()}"
            )
        camera.add_distance_to_image_plane_to_frame()
        simulation_app.update()
        simulation_app.update()

        prim_candidates: dict[str, list[object]] = {}
        archive_link_set = set(archived_link_names)
        archive_joint_set = set(archive_joint_names)
        isaac_joint_names: set[str] = set()
        for prim in Usd.PrimRange(
            robot_prim, Usd.TraverseInstanceProxies()
        ):
            prim_name = prim.GetName()
            if prim_name in archive_joint_set:
                isaac_joint_names.add(prim_name)
            if prim_name not in archive_link_set or prim.IsInstanceProxy():
                continue
            xformable = UsdGeom.Xformable(prim)
            if not xformable.GetPrim().IsValid():
                continue
            prim_candidates.setdefault(prim_name, []).append(prim)

        def candidate_score(prim: object) -> tuple[int, int, int]:
            path = str(prim.GetPath())  # type: ignore[attr-defined]
            is_render_child = int(
                "/visuals/" in path
                or "/collisions/" in path
                or path.endswith("/visuals")
                or path.endswith("/collisions")
            )
            is_rigid_body = int(
                prim.HasAPI(UsdPhysics.RigidBodyAPI)  # type: ignore[attr-defined]
            )
            return (
                is_rigid_body,
                -is_render_child,
                -path.count("/"),
            )

        selected_link_prims: dict[str, object] = {}
        for link_name, candidates in prim_candidates.items():
            selected_link_prims[link_name] = max(
                candidates,
                key=candidate_score,
            )

        missing_visible_prims = sorted(
            visible_link_names - selected_link_prims.keys()
        )
        if missing_visible_prims:
            raise RuntimeError(
                "Isaac URDF import omitted rendered links: "
                f"{missing_visible_prims}"
            )

        body_pose_ops: list[tuple[int, object, object]] = []
        for link_name, body_prim in selected_link_prims.items():
            link_index = link_indices[link_name]
            xformable = UsdGeom.Xformable(body_prim)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble,
                opSuffix="pinocchio_playback",
            )
            orient_op = xformable.AddOrientOp(
                UsdGeom.XformOp.PrecisionDouble,
                opSuffix="pinocchio_playback",
            )
            xformable.SetXformOpOrder([translate_op, orient_op])
            body_pose_ops.append((link_index, translate_op, orient_op))

        matched_link_names = sorted(selected_link_prims)
        missing_link_names = sorted(
            archive_link_set - selected_link_prims.keys()
        )
        isaac_joint_name_list = sorted(isaac_joint_names)
        missing_joint_names = sorted(
            archive_joint_set - isaac_joint_names
        )
        print(
            "Isaac Sim URDF links matched by archive name "
            f"({len(matched_link_names)}/{len(archived_link_names)}); "
            f"missing={missing_link_names}",
            flush=True,
        )
        print(
            "Isaac Sim URDF joints matched by archive name "
            f"({len(isaac_joint_name_list)}/{len(archive_joint_names)}); "
            f"missing={missing_joint_names}",
            flush=True,
        )
        camera.set_clipping_range(0.01, 100.0)
        camera.set_lens_aperture(0.0)

        rgb_dir = output_root / "robot_rgb"
        mask_dir = output_root / "robot_masks"
        depth_dir = output_root / "robot_depth_mm"
        if not stream_composite:
            for directory in (rgb_dir, mask_dir, depth_dir):
                directory.mkdir(parents=True, exist_ok=True)

        base_position = np.asarray(
            archive["base_position"],
            dtype=np.float64,
        )
        base_rotation = np.asarray(
            archive["base_rotation"],
            dtype=np.float64,
        )
        camera_positions = np.einsum(
            "ij,nj->ni",
            base_rotation.T,
            np.asarray(archive["camera_positions"], dtype=np.float64)
            - base_position,
        )
        camera_world_quaternions = np.asarray(
            archive["camera_gl_quaternions"], dtype=np.float64
        )
        camera_world_rotations = Rotation.from_quat(
            np.roll(camera_world_quaternions, -1, axis=1)
        ).as_matrix()
        camera_base_rotations = np.einsum(
            "ij,njk->nik",
            base_rotation.T,
            camera_world_rotations,
        )
        camera_quaternions = np.roll(
            Rotation.from_matrix(camera_base_rotations).as_quat(),
            1,
            axis=1,
        )
        video_frames: list[np.ndarray] = []
        mask_fractions: list[float] = []
        first_frame_diagnostics: dict[str, object] = {}
        link_positions = np.asarray(
            archive["link_positions"], dtype=np.float64
        )
        link_quaternions = np.asarray(
            archive["link_quaternions"], dtype=np.float64
        )
        compositor: InMemoryVideoCompositor | None = None
        with ExitStack() as render_stack:
            if stream_composite:
                if inpaint_video is None or composite_output is None:
                    raise RuntimeError(
                        "Streaming composite paths were not resolved"
                    )
                if scene_depth_root is None:
                    raise RuntimeError(
                        "Streaming scene depth path was not resolved"
                    )
                compositor = render_stack.enter_context(
                    InMemoryVideoCompositor(
                        source_video=source_video,
                        inpaint_video=inpaint_video,
                        trajectory_path=trajectory_path,
                        scene_depth_root=scene_depth_root,
                        output_root=composite_output,
                        g1_root=output_root,
                        lighting_profile_path=lighting_profile_path,
                        hardware_encoding=args.hardware_encoding,
                    )
                )
                if compositor.frame_count != frame_count:
                    raise ValueError(
                        "Rendered and composite frame counts disagree: "
                        f"{frame_count} != {compositor.frame_count}"
                    )
                if (compositor.width, compositor.height) != (
                    render_width,
                    render_height,
                ):
                    raise ValueError(
                        "Inpaint and render resolutions disagree: "
                        f"{compositor.width}x{compositor.height} != "
                        f"{render_width}x{render_height}"
                    )

            print(
                f"Isaac Sim rendered 0/{frame_count}",
                flush=True,
            )
            for frame_id in range(frame_count):
                for link_index, translate_op, orient_op in body_pose_ops:
                    relative_position = link_positions[frame_id, link_index]
                    relative_quaternion = link_quaternions[
                        frame_id, link_index
                    ]
                    translate_op.Set(Gf.Vec3d(*relative_position))
                    orient_op.Set(
                        Gf.Quatd(
                            float(relative_quaternion[0]),
                            Gf.Vec3d(*relative_quaternion[1:]),
                        )
                    )
                camera.set_world_pose(
                    position=np.asarray(
                        camera_positions[frame_id], dtype=np.float32
                    ),
                    orientation=np.asarray(
                        camera_quaternions[frame_id], dtype=np.float32
                    ),
                    camera_axes="usd",
                )
                rep.orchestrator.step(
                    rt_subframes=2,
                    pause_timeline=True,
                    delta_time=0.0,
                    wait_for_render=True,
                )
                rgba = camera.get_rgba()
                depth = camera.get_depth()
                if rgba is None or depth is None:
                    raise RuntimeError(
                        f"Camera returned no data at frame {frame_id}"
                    )
                rgba = np.asarray(rgba)
                depth = np.asarray(depth, dtype=np.float32)
                if rgba.shape[:2] != (render_height, render_width):
                    raise RuntimeError(f"Unexpected RGBA shape: {rgba.shape}")
                if depth.shape != (render_height, render_width):
                    raise RuntimeError(f"Unexpected depth shape: {depth.shape}")

                mask = (
                    np.isfinite(depth)
                    & (depth > 0.01)
                    & (depth < 100.0)
                )
                rgb = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                mask_u8 = mask.astype(np.uint8) * 255

                if compositor is not None:
                    compositor.write(rgb, mask_u8, depth)
                else:
                    depth_mm = np.where(
                        mask,
                        np.clip(
                            np.rint(depth * 1000.0),
                            0.0,
                            65535.0,
                        ),
                        0.0,
                    ).astype(np.uint16)
                    if not cv2.imwrite(
                        str(rgb_dir / f"{frame_id:05d}.png"), rgb
                    ):
                        raise RuntimeError(
                            f"Failed to write rendered RGB frame {frame_id}"
                        )
                    if not cv2.imwrite(
                        str(mask_dir / f"{frame_id:05d}.png"), mask_u8
                    ):
                        raise RuntimeError(
                            f"Failed to write rendered mask frame {frame_id}"
                        )
                    if not cv2.imwrite(
                        str(depth_dir / f"{frame_id:05d}.png"), depth_mm
                    ):
                        raise RuntimeError(
                            f"Failed to write rendered depth frame {frame_id}"
                        )
                if args.debug_artifacts:
                    video_frames.append(rgb)
                mask_fractions.append(float(mask.mean()))
                if (
                    frame_id == 0
                    or (frame_id + 1) % 10 == 0
                    or frame_id + 1 == frame_count
                ):
                    print(
                        f"Isaac Sim rendered {frame_id + 1}/{frame_count}",
                        flush=True,
                    )
                if frame_id == 0:
                    bbox_cache = UsdGeom.BBoxCache(
                        Usd.TimeCode.Default(),
                        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                        useExtentsHint=False,
                    )
                    bbox_range = (
                        bbox_cache.ComputeWorldBound(robot_prim)
                        .ComputeAlignedRange()
                    )
                    actual_camera_pose = camera.get_world_pose(
                        camera_axes="usd"
                    )
                    first_frame_diagnostics = {
                        "robot_bbox_min": list(bbox_range.GetMin()),
                        "robot_bbox_max": list(bbox_range.GetMax()),
                        "camera_position": actual_camera_pose[0].tolist(),
                        "camera_quaternion": actual_camera_pose[1].tolist(),
                        "finite_depth_pixels": int(mask.sum()),
                    }
                    print(
                        "First-frame diagnostics: "
                        f"{first_frame_diagnostics}",
                        flush=True,
                    )

        if args.debug_artifacts:
            write_video(
                output_root / "robot_only.mp4",
                video_frames,
                source_fps,
                source_video=source_video,
                hardware_encoding=args.hardware_encoding,
            )
        report: dict[str, object] = {
            "renderer": f"Isaac Sim 5.1 ({args.renderer})",
            "ik": "Pinocchio (precomputed link poses)",
            "frame_count": frame_count,
            "source_frame_count": total_frames,
            "resolution": [render_width, render_height],
            "source_resolution": [render_width, render_height],
            "source_fps": source_fps,
            "camera_intrinsic": intrinsic.tolist(),
            "isaac_camera_intrinsic": isaac_intrinsic.tolist(),
            "resolved_camera_intrinsic": resolved_intrinsic.tolist(),
            "camera_projection": {
                "horizontal_aperture": horizontal_aperture,
                "vertical_aperture": vertical_aperture,
                "focal_length": focal_length,
                "stage_meters_per_unit": meters_per_unit,
            },
            "lighting": {
                "profile": (
                    str(lighting_profile_path)
                    if lighting_profile_path is not None
                    else None
                ),
                "dome_intensity": args.dome_intensity,
                "dome_temperature_kelvin": args.dome_temperature,
                "key_type": "UsdLux.DistantLight",
                "key_intensity": args.key_intensity,
                "key_temperature_kelvin": args.key_temperature,
                "key_angle_degrees": args.key_angle,
                "key_rotation_xyz_degrees": list(args.key_rotation),
                "profile_confidence": (
                    lighting_profile.get("confidence")
                    if lighting_profile is not None
                    else None
                ),
            },
            "joint_count": len(isaac_joint_name_list),
            "joint_names": isaac_joint_name_list,
            "missing_archive_joint_names": missing_joint_names,
            "matched_link_count": len(matched_link_names),
            "matched_link_names": matched_link_names,
            "missing_archive_link_names": missing_link_names,
            "visible_link_names": sorted(visible_link_names),
            "first_frame_diagnostics": first_frame_diagnostics,
            "mean_robot_area_fraction": float(np.mean(mask_fractions)),
            "max_robot_area_fraction": float(np.max(mask_fractions)),
            "output_mode": (
                "in_memory_composite"
                if compositor is not None
                else "png_sequences"
            ),
            "composite": (
                compositor.report if compositor is not None else None
            ),
            "trajectory": str(trajectory_path),
            "urdf": str(source_urdf),
            "render_urdf": str(render_urdf),
            "robot_prim_path": robot_prim_path,
            "asset_import": (
                "Isaac Sim URDF importer; motion playback uses "
                "precomputed Pinocchio link poses"
            ),
        }
        (output_root / "isaacsim_render_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
        return report
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a Pinocchio G1 IK trajectory with Isaac Sim 5.1"
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inpaint-video",
        type=Path,
        help=(
            "ProPainter video for in-process compositing; requires "
            "--composite-output"
        ),
    )
    parser.add_argument(
        "--composite-output",
        type=Path,
        help=(
            "Write synthetic_g1.mp4 directly while rendering instead of "
            "writing RGB/mask/depth PNG sequences"
        ),
    )
    parser.add_argument(
        "--scene-depths",
        type=Path,
        help=(
            "Numbered native-metric DA3 depth PNGs in uint16 millimeters; "
            "required with --composite-output"
        ),
    )
    parser.add_argument(
        "--renderer",
        choices=("RaytracedLighting", "PathTracing"),
        default="RaytracedLighting",
    )
    parser.add_argument(
        "--lighting-profile",
        type=Path,
        help="Task-level JSON generated by human_to_robot.pipeline",
    )
    parser.add_argument("--dome-intensity", type=float)
    parser.add_argument("--dome-temperature", type=float)
    parser.add_argument("--key-intensity", type=float)
    parser.add_argument("--key-temperature", type=float)
    parser.add_argument("--key-angle", type=float)
    parser.add_argument(
        "--key-rotation",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug-artifacts", action="store_true")
    parser.add_argument(
        "--hardware-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVIDIA NVENC for output videos when available (default: true)",
    )
    return parser


def main() -> None:
    render(build_parser().parse_args())


if __name__ == "__main__":
    main()
