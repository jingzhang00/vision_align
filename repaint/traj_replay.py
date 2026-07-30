#!/usr/bin/env python3
"""Visualize a LeRobot v3 G1 trajectory with Pinocchio kinematics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from utils.visualize_g1_trajectory import (  # noqa: E402
    BACKGROUND,
    FOREGROUND,
    LEFT,
    RIGHT,
    TARGET,
    draw_pose,
    set_equal_3d_limits,
    style_3d_axis,
    style_axis,
    urdf_edges,
)


DEFAULT_DATASET = WORKSPACE_ROOT / "output_new/basic_pick_place"
DEFAULT_G1_URDF = (
    WORKSPACE_ROOT
    / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
)
SUPPORTED_FIELDS = ("observation.state", "action")


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _episode_metadata_rows(dataset_root: Path) -> list[dict[str, Any]]:
    paths = sorted(
        (dataset_root / "meta/episodes").glob("chunk-*/*.parquet")
    )
    if not paths:
        raise FileNotFoundError(
            f"No LeRobot episode metadata under {dataset_root / 'meta/episodes'}"
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
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
        raise ValueError(f"{dataset_root} is not a LeRobot v3.0 dataset")
    if info.get("robot_type") not in {
        "unitree_g1",
        "unitree_g1_with_hands",
    }:
        raise ValueError(
            f"Unsupported robot_type: {info.get('robot_type')!r}"
        )
    feature = info.get("features", {}).get(field)
    if field not in SUPPORTED_FIELDS or not isinstance(feature, dict):
        raise KeyError(f"Dataset does not contain supported field {field!r}")
    joint_names = tuple(feature.get("names", ()))
    if not joint_names:
        raise ValueError(f"{field} does not declare any G1 joint names")

    episode_rows = _episode_metadata_rows(dataset_root)
    matching = [
        row
        for row in episode_rows
        if int(row["episode_index"]) == episode_index
    ]
    if len(matching) != 1:
        available = sorted(
            int(row["episode_index"])
            for row in episode_rows
        )
        raise ValueError(
            f"Episode {episode_index} was not found exactly once; "
            f"available episodes: {available}"
        )
    episode_row = matching[0]
    data_path = dataset_root / str(info["data_path"]).format(
        chunk_index=int(episode_row["data/chunk_index"]),
        file_index=int(episode_row["data/file_index"]),
    )
    table = pq.read_table(
        data_path,
        columns=[
            field,
            "observation.root_pose",
            "episode_index",
            "frame_index",
            "timestamp",
        ],
    )
    mask = (
        np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        == episode_index
    )
    frame_indices = np.asarray(
        table["frame_index"].to_pylist(),
        dtype=np.int64,
    )[mask]
    order = np.argsort(frame_indices)
    frame_indices = frame_indices[order]
    timestamps = np.asarray(
        table["timestamp"].to_pylist(),
        dtype=np.float64,
    )[mask][order]
    root_pose = np.asarray(
        table["observation.root_pose"].to_pylist(),
        dtype=np.float64,
    )[mask][order]
    joint_positions = np.asarray(
        table[field].to_pylist(),
        dtype=np.float64,
    )[mask][order]

    length = int(episode_row["length"])
    if not np.array_equal(frame_indices, np.arange(length)):
        raise ValueError(
            f"Episode {episode_index} frame indices are not contiguous"
        )
    if root_pose.shape != (length, 7):
        raise ValueError(f"Invalid root pose shape: {root_pose.shape}")
    if joint_positions.shape != (length, len(joint_names)):
        raise ValueError(f"Invalid joint trajectory shape: {joint_positions.shape}")
    if not (
        np.isfinite(root_pose).all()
        and np.isfinite(joint_positions).all()
        and np.isfinite(timestamps).all()
    ):
        raise ValueError(f"Episode {episode_index} contains non-finite values")
    if not np.allclose(
        np.linalg.norm(root_pose[:, 3:7], axis=1),
        1.0,
        atol=2e-4,
    ):
        raise ValueError(f"Episode {episode_index} has non-unit root quaternions")

    stop = length if end_frame is None else end_frame
    if not 0 <= start_frame < stop <= length:
        raise ValueError(
            f"Expected 0 <= start-frame < end-frame <= {length}, "
            f"got {start_frame} and {stop}"
        )
    selection = slice(start_frame, stop)
    return Trajectory(
        dataset_root=dataset_root,
        episode_index=episode_index,
        fps=float(info["fps"]),
        field=field,
        joint_names=joint_names,
        frame_indices=frame_indices[selection],
        timestamps=timestamps[selection],
        root_pose_xyzw=root_pose[selection],
        joint_positions=joint_positions[selection],
    )


def pinocchio_link_positions(
    trajectory: Trajectory,
    urdf_path: Path,
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    model = pin.buildModelFromUrdf(str(urdf_path.resolve()))
    if tuple(model.names[1:]) != trajectory.joint_names:
        raise ValueError(
            "LeRobot joint order does not match the Pinocchio G1 URDF"
        )
    frame_ids = [
        frame_id
        for frame_id, frame in enumerate(model.frames)
        if frame.type == pin.FrameType.BODY
    ]
    names = [model.frames[frame_id].name for frame_id in frame_ids]
    positions = np.empty(
        (trajectory.frame_count, len(frame_ids), 3),
        dtype=np.float64,
    )
    data = model.createData()
    for index, qpos in enumerate(trajectory.joint_positions):
        pin.forwardKinematics(model, data, qpos)
        pin.updateFramePlacements(model, data)
        local = np.stack(
            [data.oMf[frame_id].translation for frame_id in frame_ids]
        )
        rotation = Rotation.from_quat(
            trajectory.root_pose_xyzw[index, 3:7]
        ).as_matrix()
        positions[index] = (
            local @ rotation.T
            + trajectory.root_pose_xyzw[index, :3]
        )
    return positions, names, urdf_edges(urdf_path, names)


def plot_overview(
    trajectory: Trajectory,
    positions: np.ndarray,
    names: list[str],
    edges: list[tuple[int, int]],
    output_path: Path,
) -> None:
    frames = trajectory.frame_count
    x = trajectory.frame_indices
    fig = plt.figure(
        figsize=(16, 9),
        facecolor=BACKGROUND,
        constrained_layout=True,
    )
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1.0))
    pose_axis = fig.add_subplot(grid[:, 0], projection="3d")
    joint_axis = fig.add_subplot(grid[0, 1])
    step_axis = fig.add_subplot(grid[1, 1])
    style_3d_axis(pose_axis)
    style_axis(joint_axis)
    style_axis(step_axis)

    samples = np.unique(np.linspace(0, frames - 1, 5, dtype=int))
    for frame in samples[:-1]:
        draw_pose(
            pose_axis,
            positions[frame],
            edges,
            names,
            alpha=0.16,
            linewidth=1.2,
        )
    draw_pose(
        pose_axis,
        positions[samples[-1]],
        edges,
        names,
        linewidth=2.4,
    )
    set_equal_3d_limits(pose_axis, positions.reshape(-1, 3))
    pose_axis.set(
        title="Pinocchio FK trajectory (five sampled poses)",
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z (m)",
    )

    arm_indices = [
        index
        for index, name in enumerate(trajectory.joint_names)
        if "shoulder" in name or "elbow" in name or "wrist" in name
    ]
    for index in arm_indices:
        name = trajectory.joint_names[index]
        joint_axis.plot(
            x,
            trajectory.joint_positions[:, index],
            color=LEFT if name.startswith("left_") else RIGHT,
            alpha=0.62,
            linewidth=1.0,
        )
    joint_axis.set(
        title="Arm joint trajectory",
        xlabel="Frame",
        ylabel="Position (rad)",
    )

    step = np.max(
        np.abs(np.diff(trajectory.joint_positions, axis=0)),
        axis=1,
    )
    step_axis.plot(x[1:], step, color=TARGET, linewidth=1.5)
    step_axis.set(
        title="Maximum joint step per frame",
        xlabel="Frame",
        ylabel="|Δq| max (rad)",
    )
    fig.suptitle(
        f"G1 trajectory · episode {trajectory.episode_index} · "
        f"{frames} frames · {trajectory.fps:g} FPS\n"
        "Kinematic inspection only; full replay belongs in Isaac",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def plot_animation(
    trajectory: Trajectory,
    positions: np.ndarray,
    names: list[str],
    edges: list[tuple[int, int]],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BACKGROUND)
    axis = fig.add_subplot(111, projection="3d")
    style_3d_axis(axis)
    set_equal_3d_limits(axis, positions.reshape(-1, 3))
    axis.set(
        title="G1 Pinocchio trajectory",
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z (m)",
    )
    lines = []
    for parent, child in edges:
        line, = axis.plot(
            [],
            [],
            [],
            color=(
                LEFT
                if "left_" in f"{names[parent]} {names[child]}"
                else (
                    RIGHT
                    if "right_" in f"{names[parent]} {names[child]}"
                    else "#8b949e"
                )
            ),
            linewidth=2.4,
            solid_capstyle="round",
        )
        lines.append((line, parent, child))
    frame_text = axis.text2D(
        0.02,
        0.96,
        "",
        transform=axis.transAxes,
        color=FOREGROUND,
        fontsize=11,
    )

    def update(frame: int) -> list[Any]:
        current = positions[frame]
        for line, parent, child in lines:
            segment = current[[parent, child]]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])
        frame_text.set_text(
            f"Episode {trajectory.episode_index} · "
            f"frame {int(trajectory.frame_indices[frame])}"
        )
        return [*(line for line, _, _ in lines), frame_text]

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=trajectory.frame_count,
        interval=1000.0 / trajectory.fps,
        blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=trajectory.fps,
        codec="libx264",
        bitrate=4000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    movie.save(output_path, writer=writer)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Pinocchio-FK overview and 3D MP4 for a G1 "
            "LeRobot v3 trajectory."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--field",
        choices=SUPPORTED_FIELDS,
        default="observation.state",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_G1_URDF)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--overview-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory = load_trajectory(
        args.dataset,
        args.episode,
        args.field,
        args.start_frame,
        args.end_frame,
    )
    positions, names, edges = pinocchio_link_positions(
        trajectory,
        args.urdf,
    )
    output_dir = (
        args.output_dir
        or trajectory.dataset_root
        / "visualization"
        / f"episode_{trajectory.episode_index:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = output_dir / "g1_trajectory_overview.png"
    plot_overview(trajectory, positions, names, edges, overview)
    print(f"Wrote {overview}")
    if not args.overview_only:
        video = output_dir / "g1_trajectory_3d.mp4"
        plot_animation(trajectory, positions, names, edges, video)
        print(f"Wrote {video}")


if __name__ == "__main__":
    main()
