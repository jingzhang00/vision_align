#!/usr/bin/env python3
"""Replay a retargeted LeRobot v3 trajectory with MuJoCo, without Isaac.

The replay is kinematic: for every frame, the saved floating-base pose and
29-DoF G1 joint state are copied into MuJoCo qpos and followed by mj_forward().
No controller, physics rollout, IK, or Isaac renderer is involved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import mujoco as mj
import numpy as np
import pyarrow.parquet as pq


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    WORKSPACE_ROOT / "dataset/basic_pick_place_g1_lerobot_v3"
)
DEFAULT_G1_XML = (
    WORKSPACE_ROOT
    / "stream/assets/unitree_g1/g1_mocap_29dof.xml"
)
SUPPORTED_FIELDS = ("observation.state", "action")
LEFT_ARROW_KEY = 263
RIGHT_ARROW_KEY = 262


@dataclass(frozen=True)
class Trajectory:
    dataset_root: Path
    episode_index: int
    fps: float
    field: str
    joint_names: tuple[str, ...]
    frame_indices: np.ndarray
    timestamps: np.ndarray
    root_pose_xyzw: np.ndarray
    joint_positions: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_indices))


@dataclass
class ReplayControls:
    paused: bool = False
    reset_requested: bool = False
    step_delta: int = 0
    exit_requested: bool = False

    def on_key(self, keycode: int) -> None:
        if keycode == ord(" "):
            self.paused = not self.paused
            print("paused" if self.paused else "playing")
        elif keycode in (ord("R"), ord("r")):
            self.reset_requested = True
        elif keycode in (RIGHT_ARROW_KEY, ord("D"), ord("d")):
            self.paused = True
            self.step_delta += 1
        elif keycode in (LEFT_ARROW_KEY, ord("A"), ord("a")):
            self.paused = True
            self.step_delta -= 1
        elif keycode in (ord("Q"), ord("q")):
            self.exit_requested = True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _episode_metadata_rows(dataset_root: Path) -> list[dict[str, Any]]:
    metadata_paths = sorted(
        (dataset_root / "meta/episodes").glob("chunk-*/*.parquet")
    )
    if not metadata_paths:
        raise FileNotFoundError(
            f"No LeRobot episode metadata under {dataset_root / 'meta/episodes'}"
        )
    rows: list[dict[str, Any]] = []
    for path in metadata_paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def load_trajectory(
    dataset_root: Path,
    episode_index: int,
    field: str,
    start_frame: int,
    end_frame: int | None,
) -> Trajectory:
    dataset_root = dataset_root.resolve()
    info = _read_json(dataset_root / "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"{dataset_root} is not a LeRobot v3.0 dataset"
        )
    if info.get("robot_type") != "unitree_g1":
        raise ValueError(
            "This simple replay currently supports robot_type=unitree_g1, "
            f"got {info.get('robot_type')!r}"
        )
    if field not in SUPPORTED_FIELDS:
        raise ValueError(f"Unsupported trajectory field: {field}")

    features = info.get("features", {})
    feature = features.get(field)
    if not isinstance(feature, dict):
        raise KeyError(f"Dataset does not contain {field}")
    joint_names_raw = feature.get("names")
    if (
        not isinstance(joint_names_raw, list)
        or len(joint_names_raw) != 29
        or not all(isinstance(name, str) for name in joint_names_raw)
    ):
        raise ValueError(f"{field} must declare exactly 29 G1 joint names")
    joint_names = tuple(joint_names_raw)

    episode_rows = _episode_metadata_rows(dataset_root)
    available = sorted(int(row["episode_index"]) for row in episode_rows)
    matching_rows = [
        row
        for row in episode_rows
        if int(row["episode_index"]) == episode_index
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            f"Episode {episode_index} was not found exactly once; "
            f"available episodes: {available}"
        )
    episode_row = matching_rows[0]

    data_pattern = info.get("data_path")
    if not isinstance(data_pattern, str):
        raise ValueError("meta/info.json does not define data_path")
    data_relative = data_pattern.format(
        chunk_index=int(episode_row["data/chunk_index"]),
        file_index=int(episode_row["data/file_index"]),
    )
    data_path = dataset_root / data_relative
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing episode data: {data_path}")

    required_columns = (
        field,
        "observation.root_pose",
        "episode_index",
        "frame_index",
        "timestamp",
    )
    table = pq.read_table(data_path, columns=list(required_columns))
    episode_ids = np.asarray(
        table["episode_index"].to_pylist(),
        dtype=np.int64,
    )
    mask = episode_ids == episode_index
    if not np.any(mask):
        raise ValueError(
            f"{data_path} contains no rows for episode {episode_index}"
        )

    frame_indices = np.asarray(
        table["frame_index"].to_pylist(),
        dtype=np.int64,
    )[mask]
    timestamps = np.asarray(
        table["timestamp"].to_pylist(),
        dtype=np.float64,
    )[mask]
    root_pose = np.asarray(
        table["observation.root_pose"].to_pylist(),
        dtype=np.float64,
    )[mask]
    joint_positions = np.asarray(
        table[field].to_pylist(),
        dtype=np.float64,
    )[mask]

    order = np.argsort(frame_indices)
    frame_indices = frame_indices[order]
    timestamps = timestamps[order]
    root_pose = root_pose[order]
    joint_positions = joint_positions[order]

    declared_length = int(episode_row["length"])
    if len(frame_indices) != declared_length:
        raise ValueError(
            f"Episode {episode_index} has {len(frame_indices)} rows, "
            f"metadata declares {declared_length}"
        )
    if not np.array_equal(
        frame_indices,
        np.arange(declared_length, dtype=np.int64),
    ):
        raise ValueError(
            f"Episode {episode_index} frame indices are not contiguous"
        )
    if root_pose.shape != (declared_length, 7):
        raise ValueError(
            f"observation.root_pose has invalid shape {root_pose.shape}"
        )
    if joint_positions.shape != (declared_length, 29):
        raise ValueError(
            f"{field} has invalid shape {joint_positions.shape}"
        )
    if not (
        np.isfinite(root_pose).all()
        and np.isfinite(joint_positions).all()
        and np.isfinite(timestamps).all()
    ):
        raise ValueError(f"Episode {episode_index} contains non-finite values")

    quaternion_norm = np.linalg.norm(root_pose[:, 3:7], axis=1)
    if not np.allclose(quaternion_norm, 1.0, atol=2e-4):
        raise ValueError(
            f"Episode {episode_index} contains non-unit root quaternions"
        )

    if start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    stop = declared_length if end_frame is None else end_frame
    if not start_frame < stop <= declared_length:
        raise ValueError(
            f"Expected 0 <= start-frame < end-frame <= {declared_length}, "
            f"got {start_frame} and {stop}"
        )

    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Dataset FPS must be positive, got {fps}")

    selection = slice(start_frame, stop)
    return Trajectory(
        dataset_root=dataset_root,
        episode_index=episode_index,
        fps=fps,
        field=field,
        joint_names=joint_names,
        frame_indices=frame_indices[selection],
        timestamps=timestamps[selection],
        root_pose_xyzw=root_pose[selection],
        joint_positions=joint_positions[selection],
    )


def load_g1_model(
    xml_path: Path,
    joint_names: tuple[str, ...],
) -> tuple[mj.MjModel, mj.MjData, np.ndarray]:
    xml_path = xml_path.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"Unitree G1 MuJoCo XML not found: {xml_path}")
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)

    free_joint_ids = np.flatnonzero(
        model.jnt_type == mj.mjtJoint.mjJNT_FREE
    )
    if (
        len(free_joint_ids) != 1
        or int(model.jnt_qposadr[free_joint_ids[0]]) != 0
    ):
        raise ValueError(
            "Expected one floating-base joint at qpos address zero"
        )

    qpos_addresses: list[int] = []
    for joint_name in joint_names:
        joint_id = mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise KeyError(
                f"Dataset joint {joint_name!r} is absent from {xml_path}"
            )
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (
            mj.mjtJoint.mjJNT_HINGE,
            mj.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(
                f"Dataset joint {joint_name!r} is not a scalar MuJoCo joint"
            )
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    if len(set(qpos_addresses)) != len(qpos_addresses):
        raise ValueError("Dataset joint mapping contains duplicate qpos addresses")
    return model, data, np.asarray(qpos_addresses, dtype=np.int64)


def apply_frame(
    model: mj.MjModel,
    data: mj.MjData,
    qpos_addresses: np.ndarray,
    trajectory: Trajectory,
    local_frame: int,
) -> None:
    data.qpos[:] = model.qpos0
    root_pose = trajectory.root_pose_xyzw[local_frame]
    data.qpos[:3] = root_pose[:3]
    # LeRobot stores quaternion XYZW; MuJoCo qpos uses WXYZ.
    data.qpos[3:7] = root_pose[[6, 3, 4, 5]]
    data.qpos[qpos_addresses] = trajectory.joint_positions[local_frame]
    data.qvel[:] = 0.0
    mj.mj_forward(model, data)
    if not (
        np.isfinite(data.qpos).all()
        and np.isfinite(data.xpos).all()
        and np.isfinite(data.xquat).all()
    ):
        absolute_frame = int(trajectory.frame_indices[local_frame])
        raise ValueError(
            f"MuJoCo produced non-finite kinematics at frame {absolute_frame}"
        )


def replay_headless(
    model: mj.MjModel,
    data: mj.MjData,
    qpos_addresses: np.ndarray,
    trajectory: Trajectory,
) -> None:
    for local_frame in range(trajectory.frame_count):
        apply_frame(
            model,
            data,
            qpos_addresses,
            trajectory,
            local_frame,
        )
    print(
        "Headless replay OK: "
        f"episode={trajectory.episode_index}, "
        f"frames={trajectory.frame_count}, "
        f"field={trajectory.field}, "
        f"fps={trajectory.fps:g}"
    )


def replay_interactive(
    model: mj.MjModel,
    data: mj.MjData,
    qpos_addresses: np.ndarray,
    trajectory: Trajectory,
    speed: float,
    loop: bool,
) -> None:
    import mujoco.viewer as mj_viewer

    controls = ReplayControls()
    frame_period = 1.0 / (trajectory.fps * speed)
    cursor = 0
    finished = False

    apply_frame(model, data, qpos_addresses, trajectory, cursor)
    with mj_viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=controls.on_key,
    ) as viewer:
        viewer.cam.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, 0.55])
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -12.0
        viewer.cam.azimuth = 145.0
        viewer.sync()

        print(
            "MuJoCo replay started. Controls: "
            "Space=pause, A/Left=previous, D/Right=next, "
            "R=restart, Q=quit."
        )
        while viewer.is_running() and not controls.exit_requested:
            if controls.reset_requested:
                controls.reset_requested = False
                cursor = 0
                finished = False
                controls.paused = False

            if controls.step_delta:
                cursor = int(
                    np.clip(
                        cursor + controls.step_delta,
                        0,
                        trajectory.frame_count - 1,
                    )
                )
                controls.step_delta = 0
                finished = False
                apply_frame(
                    model,
                    data,
                    qpos_addresses,
                    trajectory,
                    cursor,
                )
                viewer.cam.lookat[:] = (
                    data.qpos[:3] + np.array([0.0, 0.0, 0.55])
                )
                viewer.sync()

            if controls.paused or finished:
                viewer.sync()
                time.sleep(0.01)
                continue

            frame_start = time.perf_counter()
            apply_frame(
                model,
                data,
                qpos_addresses,
                trajectory,
                cursor,
            )
            viewer.cam.lookat[:] = (
                data.qpos[:3] + np.array([0.0, 0.0, 0.55])
            )
            viewer.sync()

            cursor += 1
            if cursor >= trajectory.frame_count:
                if loop:
                    cursor = 0
                else:
                    cursor = trajectory.frame_count - 1
                    finished = True
                    controls.paused = True
                    print(
                        "Replay finished; holding the last frame. "
                        "Press R to restart or Q to quit."
                    )

            remaining = frame_period - (time.perf_counter() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kinematically replay a Unitree G1 LeRobot v3 trajectory in "
            "MuJoCo, without Isaac."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="LeRobot v3 dataset root.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="LeRobot episode_index to replay.",
    )
    parser.add_argument(
        "--field",
        choices=SUPPORTED_FIELDS,
        default="observation.state",
        help="Joint trajectory field to replay.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_G1_XML,
        help="Unitree G1 MuJoCo XML.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First episode-relative frame to replay.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive episode-relative end frame; defaults to episode end.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for the interactive viewer.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop continuously instead of holding the last frame.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run every frame through MuJoCo kinematics without opening a "
            "viewer; useful on servers and for validation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0.0:
        raise ValueError("--speed must be finite and positive")
    if args.headless and args.loop:
        raise ValueError("--loop cannot be combined with --headless")

    trajectory = load_trajectory(
        dataset_root=args.dataset,
        episode_index=args.episode,
        field=args.field,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    model, data, qpos_addresses = load_g1_model(
        args.xml,
        trajectory.joint_names,
    )
    print(
        f"Loaded episode {trajectory.episode_index}: "
        f"{trajectory.frame_count} frames at {trajectory.fps:g} FPS "
        f"from {trajectory.dataset_root}"
    )

    if args.headless:
        replay_headless(
            model,
            data,
            qpos_addresses,
            trajectory,
        )
    else:
        replay_interactive(
            model,
            data,
            qpos_addresses,
            trajectory,
            speed=args.speed,
            loop=args.loop,
        )


if __name__ == "__main__":
    main()
