#!/usr/bin/env python3
"""Replay a LeRobot v3 G1 trajectory in Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.traj_replay import (  # noqa: E402
    DEFAULT_G1_URDF,
    SUPPORTED_FIELDS,
    Trajectory,
    load_trajectory,
)


DEFAULT_DATASET = WORKSPACE_ROOT / "output_new/basic_pick_place"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a LeRobot v3 G1 trajectory as an Isaac articulation."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--field",
        choices=SUPPORTED_FIELDS,
        default="observation.state",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_G1_URDF)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run one pass without opening the Isaac Sim window.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after one pass instead of looping until the window closes.",
    )
    return parser.parse_args()


def validate_trajectory(
    trajectory: Trajectory,
    urdf_path: Path,
) -> None:
    urdf_path = urdf_path.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"G1 URDF does not exist: {urdf_path}")
    model = pin.buildModelFromUrdf(str(urdf_path))
    if tuple(model.names[1:]) != trajectory.joint_names:
        raise ValueError(
            "LeRobot joint names/order do not match main_with_hand_new.urdf"
        )


def import_g1(urdf_path: Path) -> str:
    import omni.kit.commands

    status, import_config = omni.kit.commands.execute(
        "URDFCreateImportConfig"
    )
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    import_config.set_fix_base(False)
    import_config.set_import_inertia_tensor(True)
    import_config.set_merge_fixed_joints(False)
    import_config.set_parse_mimic(False)
    import_config.set_self_collision(False)
    import_config.set_collision_from_visuals(False)
    import_config.set_create_physics_scene(False)
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path.resolve()),
        import_config=import_config,
        dest_path="",
        get_articulation_root=True,
    )
    if not status or not prim_path:
        raise RuntimeError(f"Isaac failed to import {urdf_path}")
    return str(prim_path)


def isaac_joint_order(
    isaac_names: list[str],
    trajectory_names: tuple[str, ...],
) -> np.ndarray:
    if len(set(isaac_names)) != len(isaac_names):
        raise ValueError("Isaac articulation contains duplicate joint names")
    missing = sorted(set(trajectory_names) - set(isaac_names))
    extra = sorted(set(isaac_names) - set(trajectory_names))
    if missing or extra:
        raise ValueError(
            "Isaac and LeRobot joint sets differ: "
            f"missing={missing}, extra={extra}"
        )
    source_index = {
        name: index for index, name in enumerate(trajectory_names)
    }
    return np.asarray(
        [source_index[name] for name in isaac_names],
        dtype=np.int64,
    )


def replay(args: argparse.Namespace, trajectory: Trajectory) -> None:
    # SimulationApp must exist before importing omni, pxr, or Isaac APIs.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": "RaytracedLighting",
        }
    )
    try:
        import omni.kit.app
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Sdf, UsdLux

        extension_manager = (
            omni.kit.app.get_app().get_extension_manager()
        )
        extension_manager.set_extension_enabled_immediate(
            "isaacsim.asset.importer.urdf",
            True,
        )
        simulation_app.update()

        context = omni.usd.get_context()
        context.new_stage()
        simulation_app.update()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / trajectory.fps,
            rendering_dt=1.0 / trajectory.fps,
        )
        world.scene.add_default_ground_plane()

        robot_path = import_g1(args.urdf)
        robot = world.scene.add(
            Articulation(
                prim_paths_expr=robot_path,
                name="g1_replay",
            )
        )

        stage = context.get_stage()
        dome = UsdLux.DomeLight.Define(
            stage,
            Sdf.Path("/World/DomeLight"),
        )
        dome.CreateIntensityAttr(500.0)
        center = trajectory.root_pose_xyzw[:, :3].mean(axis=0)
        set_camera_view(
            eye=center + np.array([2.2, 2.2, 1.2]),
            target=center + np.array([0.0, 0.0, 0.25]),
            camera_prim_path="/OmniverseKit_Persp",
        )

        world.reset()
        world.pause()
        joint_order = isaac_joint_order(
            list(robot.dof_names),
            trajectory.joint_names,
        )
        print(
            f"Isaac articulation: {robot_path} · {len(joint_order)} DoF",
            flush=True,
        )
        print(
            f"Episode {trajectory.episode_index}: "
            f"{trajectory.frame_count} frames at {trajectory.fps:g} FPS",
            flush=True,
        )

        zero_velocity = np.zeros((1, len(joint_order)), dtype=np.float32)
        repeat = not (args.headless or args.once)
        while simulation_app.is_running():
            deadline = time.perf_counter()
            for index in range(trajectory.frame_count):
                if not simulation_app.is_running():
                    break
                root = trajectory.root_pose_xyzw[index]
                root_wxyz = root[[6, 3, 4, 5]]
                robot.set_world_poses(
                    positions=root[None, :3].astype(np.float32),
                    orientations=root_wxyz[None].astype(np.float32),
                )
                robot.set_joint_positions(
                    trajectory.joint_positions[
                        index, joint_order
                    ][None].astype(np.float32)
                )
                robot.set_joint_velocities(zero_velocity)
                world.render()

                deadline += 1.0 / trajectory.fps
                time.sleep(max(0.0, deadline - time.perf_counter()))
            if not repeat:
                break
    finally:
        simulation_app.close()


def main() -> None:
    args = parse_args()
    trajectory = load_trajectory(
        args.dataset,
        args.episode,
        args.field,
        args.start_frame,
        args.end_frame,
    )
    validate_trajectory(trajectory, args.urdf)
    replay(args, trajectory)


if __name__ == "__main__":
    main()
