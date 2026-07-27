from __future__ import annotations

import argparse
import json
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .lighting_profile import (
    DEFAULT_ISAAC_PROFILE,
    load_lighting_profile,
    resolve_profile_value,
)


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
    lighting_profile, lighting_profile_path = _resolve_render_settings(args)
    # SimulationApp must be constructed before importing any omni/pxr modules.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": args.renderer,
            "width": args.width,
            "height": args.height,
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

        from .video_io import video_metadata, write_video

        trajectory_path = args.trajectory.resolve()
        source_urdf = args.urdf.resolve()
        source_video = args.video.resolve()
        source_hdf5 = args.hdf5.resolve()
        output_root = args.output.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
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
        source_width, source_height, source_frame_count, source_fps = (
            video_metadata(source_video)
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

        camera = Camera(
            prim_path="/World/SynthesisCamera",
            name="synthesis_camera",
            resolution=(args.width, args.height),
        )
        camera.initialize()
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
        # Match the OpenCV pinhole calibration. USD stores focal length and
        # aperture in the same arbitrary unit, so their ratio is what matters.
        horizontal_aperture = 20.955
        focal_length = (
            float(intrinsic[0, 0])
            / float(source_width)
            * horizontal_aperture
        )
        camera.set_horizontal_aperture(horizontal_aperture)
        camera.set_focal_length(focal_length)
        camera.set_clipping_range(0.01, 100.0)
        camera.set_lens_aperture(0.0)

        rgb_dir = output_root / "robot_rgb"
        mask_dir = output_root / "robot_masks"
        depth_dir = output_root / "robot_depth_mm"
        for directory in (rgb_dir, mask_dir, depth_dir):
            directory.mkdir(parents=True, exist_ok=True)

        base_position = np.asarray(archive["base_position"], dtype=np.float64)
        camera_positions = (
            np.asarray(archive["camera_positions"], dtype=np.float64)
            - base_position
        )
        camera_quaternions = np.asarray(
            archive["camera_gl_quaternions"], dtype=np.float64
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
                position=np.asarray(camera_positions[frame_id], dtype=np.float32),
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
                raise RuntimeError(f"Camera returned no data at frame {frame_id}")
            rgba = np.asarray(rgba)
            depth = np.asarray(depth, dtype=np.float32)
            if rgba.shape[:2] != (args.height, args.width):
                raise RuntimeError(f"Unexpected RGBA shape: {rgba.shape}")
            if depth.shape != (args.height, args.width):
                raise RuntimeError(f"Unexpected depth shape: {depth.shape}")

            mask = (
                np.isfinite(depth)
                & (depth > 0.01)
                & (depth < 100.0)
            )
            depth_mm = np.where(
                mask, np.clip(depth * 1000.0, 0.0, 65535.0), 0.0
            ).astype(np.uint16)
            rgb = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            mask_u8 = mask.astype(np.uint8) * 255

            cv2.imwrite(str(rgb_dir / f"{frame_id:05d}.png"), rgb)
            cv2.imwrite(str(mask_dir / f"{frame_id:05d}.png"), mask_u8)
            cv2.imwrite(str(depth_dir / f"{frame_id:05d}.png"), depth_mm)
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
                actual_camera_pose = camera.get_world_pose(camera_axes="usd")
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
            "resolution": [args.width, args.height],
            "source_resolution": [source_width, source_height],
            "source_fps": source_fps,
            "focal_length_px": float(intrinsic[0, 0]),
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
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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
        help="Use NVIDIA NVENC for debug videos when available (default: true)",
    )
    return parser


def main() -> None:
    render(build_parser().parse_args())


if __name__ == "__main__":
    main()
