from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ARM_MESHES = {
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
        "rubber_hand",
    )
}


def _build_arm_only_mjcf(source: Path, destination: Path) -> None:
    """Keep G1 kinematics intact while removing every non-arm render geom."""

    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"No <compiler> element in {source}")
    mesh_directory = (source.parent / "../meshes/g1").resolve()
    compiler.set(
        "meshdir",
        os.path.relpath(mesh_directory, start=destination.parent.resolve()),
    )

    worldbodies = root.findall("worldbody")
    if not worldbodies:
        raise ValueError(f"No <worldbody> element in {source}")
    for worldbody in worldbodies:
        for parent in worldbody.iter():
            for geom in list(parent.findall("geom")):
                mesh = geom.get("mesh")
                if geom.get("class") != "visual" or mesh not in ARM_MESHES:
                    parent.remove(geom)
            for light in list(parent.findall("light")):
                parent.remove(light)

    # Isaac's MJCF importer converts every declared mesh, including unused
    # assets. Remove non-arm assets so it does not create *_tmp USD caches for
    # the torso and legs next to the source STL files.
    for asset in root.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") not in ARM_MESHES:
                asset.remove(mesh)
    for contact in root.findall("contact"):
        root.remove(contact)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")


def _mujoco_joint_lookup(
    mjcf: Path,
) -> tuple[object, list[str]]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    names: list[str] = []
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name
        if name == "floating_base_joint":
            continue
        names.append(name)
    return model, names


def _validate_archive(archive: np.lib.npyio.NpzFile) -> int:
    required = {
        "qpos",
        "camera_positions",
        "camera_gl_quaternions",
        "scale",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        raise ValueError(f"Trajectory archive is missing: {', '.join(missing)}")
    frame_count = int(archive["qpos"].shape[0])
    if archive["camera_positions"].shape != (frame_count, 3):
        raise ValueError("camera_positions must have shape [frame_count, 3]")
    if archive["camera_gl_quaternions"].shape != (frame_count, 4):
        raise ValueError("camera_gl_quaternions must have shape [frame_count, 4]")
    return frame_count


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import cv2

    if not frames:
        raise ValueError("No frames were rendered")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def render(args: argparse.Namespace) -> dict[str, object]:
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
        import omni.kit.app
        import omni.kit.commands
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
        from scipy.spatial.transform import Rotation

        from .trajectory import load_single_point_trajectory
        from .video_io import video_metadata

        trajectory_path = args.trajectory.resolve()
        source_mjcf = args.mjcf.resolve()
        source_video = args.video.resolve()
        source_hdf5 = args.hdf5.resolve()
        output_root = args.output.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        arm_mjcf = output_root / "g1_arms_isaacsim.xml"
        _build_arm_only_mjcf(source_mjcf, arm_mjcf)

        archive = np.load(trajectory_path)
        total_frames = _validate_archive(archive)
        source_width, source_height, source_frame_count, source_fps = (
            video_metadata(source_video)
        )
        hand_trajectory = load_single_point_trajectory(source_hdf5)
        if (
            source_frame_count != total_frames
            or hand_trajectory.frame_count != total_frames
        ):
            raise ValueError(
                "Video, HDF5, and qpos trajectory frame counts must agree: "
                f"video={source_frame_count}, hdf5={hand_trajectory.frame_count}, "
                f"qpos={total_frames}"
            )
        frame_count = (
            min(total_frames, args.max_frames)
            if args.max_frames is not None
            else total_frames
        )
        mujoco_model, mujoco_joint_names = _mujoco_joint_lookup(source_mjcf)

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate(
            "isaacsim.asset.importer.mjcf", True
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
            "MJCFCreateImportConfig"
        )
        if not status:
            raise RuntimeError("MJCFCreateImportConfig failed")
        import_config.set_fix_base(True)
        import_config.set_import_inertia_tensor(True)
        imported = omni.kit.commands.execute(
            "MJCFCreateAsset",
            mjcf_path=str(arm_mjcf),
            import_config=import_config,
            prim_path="/World/G1",
        )
        if not imported[0]:
            raise RuntimeError("MJCFCreateAsset failed")
        simulation_app.update()

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

        body_pose_ops: list[tuple[int, object, object]] = []
        for body_id in range(1, mujoco_model.nbody):
            body_name = mujoco_model.body(body_id).name
            body_path = f"/World/G1/pelvis/{body_name}"
            body_prim = stage.GetPrimAtPath(body_path)
            if not body_prim.IsValid():
                raise RuntimeError(f"Isaac Sim body prim not found: {body_path}")
            xformable = UsdGeom.Xformable(body_prim)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble,
                opSuffix="mujoco_playback",
            )
            orient_op = xformable.AddOrientOp(
                UsdGeom.XformOp.PrecisionDouble,
                opSuffix="mujoco_playback",
            )
            xformable.SetXformOpOrder([translate_op, orient_op])
            body_pose_ops.append((body_id, translate_op, orient_op))

        isaac_joint_names = [
            name
            for name in mujoco_joint_names
            if stage.GetPrimAtPath(f"/World/G1/joints/{name}").IsValid()
        ]
        print(
            f"Isaac Sim joints ({len(isaac_joint_names)}): "
            f"{isaac_joint_names}",
            flush=True,
        )
        missing_in_isaac = sorted(set(mujoco_joint_names) - set(isaac_joint_names))
        extra_in_isaac = sorted(set(isaac_joint_names) - set(mujoco_joint_names))
        if missing_in_isaac or extra_in_isaac:
            raise RuntimeError(
                "MuJoCo/Isaac joint mismatch: "
                f"missing_in_isaac={missing_in_isaac}, "
                f"extra_in_isaac={extra_in_isaac}"
            )
        # Match the OpenCV pinhole calibration. USD stores focal length and
        # aperture in the same arbitrary unit, so their ratio is what matters.
        horizontal_aperture = 20.955
        focal_length = (
            float(hand_trajectory.intrinsic[0, 0])
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

        qpos = archive["qpos"]
        world_camera_positions = archive["camera_positions"]
        world_camera_quaternions = archive["camera_gl_quaternions"]
        base_rotations = Rotation.from_quat(
            np.roll(qpos[:, 3:7], -1, axis=1)
        )
        world_camera_rotations = Rotation.from_quat(
            np.roll(world_camera_quaternions, -1, axis=1)
        )
        camera_positions = base_rotations.inv().apply(
            world_camera_positions - qpos[:, :3]
        )
        camera_quaternions = np.roll(
            (base_rotations.inv() * world_camera_rotations).as_quat(),
            1,
            axis=1,
        )
        video_frames: list[np.ndarray] = []
        mask_fractions: list[float] = []
        first_frame_diagnostics: dict[str, object] = {}
        import mujoco

        mujoco_data = mujoco.MjData(mujoco_model)

        for frame_id in range(frame_count):
            mujoco_data.qpos[:] = qpos[frame_id]
            mujoco.mj_forward(mujoco_model, mujoco_data)
            base_rotation = Rotation.from_quat(
                np.roll(qpos[frame_id, 3:7], -1)
            )
            for body_id, translate_op, orient_op in body_pose_ops:
                relative_position = base_rotation.inv().apply(
                    mujoco_data.xpos[body_id] - qpos[frame_id, :3]
                )
                world_body_rotation = Rotation.from_quat(
                    np.roll(mujoco_data.xquat[body_id], -1)
                )
                relative_quaternion = np.roll(
                    (base_rotation.inv() * world_body_rotation).as_quat(),
                    1,
                )
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
            video_frames.append(rgb)
            mask_fractions.append(float(mask.mean()))
            if frame_id == 0 or (frame_id + 1) % 10 == 0:
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
                    bbox_cache.ComputeWorldBound(
                        stage.GetPrimAtPath("/World/G1")
                    )
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

        _write_video(output_root / "robot_only.mp4", video_frames, source_fps)
        shutil.copy2(trajectory_path, output_root / "g1_trajectory.npz")
        report: dict[str, object] = {
            "renderer": f"Isaac Sim 5.1 ({args.renderer})",
            "ik": "MuJoCo + Mink (reused qpos trajectory)",
            "frame_count": frame_count,
            "source_frame_count": total_frames,
            "resolution": [args.width, args.height],
            "source_resolution": [source_width, source_height],
            "source_fps": source_fps,
            "focal_length_px": float(hand_trajectory.intrinsic[0, 0]),
            "lighting": {
                "dome_intensity": args.dome_intensity,
                "dome_temperature_kelvin": args.dome_temperature,
                "key_type": "UsdLux.DistantLight",
                "key_intensity": args.key_intensity,
                "key_temperature_kelvin": args.key_temperature,
                "key_angle_degrees": args.key_angle,
                "key_rotation_xyz_degrees": list(args.key_rotation),
            },
            "joint_count": len(isaac_joint_names),
            "joint_names": isaac_joint_names,
            "first_frame_diagnostics": first_frame_diagnostics,
            "mean_robot_area_fraction": float(np.mean(mask_fractions)),
            "max_robot_area_fraction": float(np.max(mask_fractions)),
            "trajectory": str(trajectory_path),
            "mjcf": str(source_mjcf),
            "arm_only_mjcf": str(arm_mjcf),
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
        description="Render a MuJoCo/Mink G1 IK trajectory with Isaac Sim 5.1"
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
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
    parser.add_argument("--dome-intensity", type=float, default=900.0)
    parser.add_argument("--dome-temperature", type=float, default=4300.0)
    parser.add_argument("--key-intensity", type=float, default=180.0)
    parser.add_argument("--key-temperature", type=float, default=4500.0)
    parser.add_argument("--key-angle", type=float, default=8.0)
    parser.add_argument(
        "--key-rotation",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(-20.0, 15.0, 0.0),
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    render(build_parser().parse_args())


if __name__ == "__main__":
    main()
