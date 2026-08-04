#!/usr/bin/env python3
"""Fast trajectory-only G1 replay using Pinocchio forward kinematics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
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
from repaint.trajectory_io import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_G1_URDF,
    SUPPORTED_FIELDS,
    Trajectory,
    find_dataset_roots,
    list_episode_indices,
    load_trajectory,
)


def pinocchio_link_positions(
    trajectory: Trajectory,
    urdf_path: Path,
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    model = pin.buildModelFromUrdf(str(urdf_path.resolve()))
    if tuple(model.names[1:]) != trajectory.joint_names:
        raise ValueError(
            "Trajectory joint order does not match the Pinocchio G1 URDF"
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
        "Kinematic inspection only; scene replay belongs in Isaac",
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
    width, height = trajectory.source_size
    dpi = 100
    fig = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor=BACKGROUND,
    )
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
            "Create a fast Pinocchio-FK overview and 3D MP4 for a G1 "
            "trajectory. FPS and MP4 size come from the episode source video."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Native Nymeria episode/dataset, LeRobot root, or folder "
            "recursively containing either format (default: canonical "
            "egodex/basic_pick_place/lerobot)."
        ),
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to render; omit to render every episode.",
    )
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


def render_episode(args: argparse.Namespace, episode_index: int) -> None:
    trajectory = load_trajectory(
        args.dataset,
        episode_index,
        args.field,
        args.start_frame,
        args.end_frame,
    )
    positions, names, edges = pinocchio_link_positions(
        trajectory,
        args.urdf,
    )
    episode_dir = f"episode_{trajectory.episode_index:06d}"
    if args.output_dir is None:
        output_dir = trajectory.dataset_root / "trajectory_check" / episode_dir
    elif args.episode is None:
        output_dir = args.output_dir / episode_dir
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = output_dir / "g1_trajectory_overview.png"
    plot_overview(trajectory, positions, names, edges, overview)
    print(f"Wrote {overview}")
    if not args.overview_only:
        video = output_dir / "g1_trajectory_3d.mp4"
        plot_animation(trajectory, positions, names, edges, video)
        print(f"Wrote {video}")


def main() -> None:
    args = parse_args()
    input_root = args.dataset.expanduser().resolve()
    output_root = args.output_dir
    dataset_roots = find_dataset_roots(input_root)
    if (
        args.episode is None
        and (input_root / "trajectory.npz").is_file()
        and input_root.name.startswith("episode_")
    ):
        args.episode = int(input_root.name.removeprefix("episode_"))
    for dataset_root in dataset_roots:
        args.dataset = dataset_root
        if output_root is not None and len(dataset_roots) > 1:
            args.output_dir = (
                output_root / dataset_root.relative_to(input_root)
            )
        episode_indices = (
            (args.episode,)
            if args.episode is not None
            else list_episode_indices(dataset_root)
        )
        print(
            f"{dataset_root}: processing {len(episode_indices)} episode(s): "
            + ", ".join(str(index) for index in episode_indices)
        )
        for episode_index in episode_indices:
            render_episode(args, episode_index)


if __name__ == "__main__":
    main()
