#!/usr/bin/env python3
"""Visualize a saved G1 trajectory without requiring Isaac Sim.

The trajectory stores link positions in the pelvis frame.  Targets are stored
in world coordinates, so the base pose is applied before plotting the links.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BACKGROUND = "#0d1117"
FOREGROUND = "#e6edf3"
GRID = "#30363d"
LEFT = "#56d4f5"
RIGHT = "#ff9f43"
NEUTRAL = "#8b949e"
TARGET = "#7ee787"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an overview PNG and 3D MP4 from g1_trajectory.npz."
    )
    parser.add_argument("trajectory", type=Path, help="Path to g1_trajectory.npz")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("assets/unitree_description/urdf/g1/main_with_hand_new.urdf"),
        help="G1 URDF used to recover the skeleton topology",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (default: TRAJECTORY_DIR/visualization)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--overview-only",
        action="store_true",
        help="Only create the static overview PNG",
    )
    return parser.parse_args()


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def world_link_positions(data: dict[str, np.ndarray]) -> np.ndarray:
    positions = np.asarray(data["link_positions"], dtype=float)
    rotation = np.asarray(data["base_rotation"], dtype=float)
    translation = np.asarray(data["base_position"], dtype=float)
    return positions @ rotation.T + translation


def urdf_edges(urdf_path: Path, link_names: list[str]) -> list[tuple[int, int]]:
    name_to_index = {name: index for index, name in enumerate(link_names)}
    root = ET.parse(urdf_path).getroot()
    edges: list[tuple[int, int]] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib["link"]
        child_name = child.attrib["link"]
        if parent_name in name_to_index and child_name in name_to_index:
            edges.append((name_to_index[parent_name], name_to_index[child_name]))
    if not edges:
        raise ValueError(f"No matching trajectory links found in {urdf_path}")
    return edges


def edge_color(parent_name: str, child_name: str) -> str:
    joined = f"{parent_name} {child_name}"
    if "left_" in joined:
        return LEFT
    if "right_" in joined:
        return RIGHT
    return NEUTRAL


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors=FOREGROUND, labelsize=8)
    axis.xaxis.label.set_color(FOREGROUND)
    axis.yaxis.label.set_color(FOREGROUND)
    axis.title.set_color(FOREGROUND)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(True, color=GRID, linewidth=0.55, alpha=0.65)


def style_3d_axis(axis: plt.Axes) -> None:
    style_axis(axis)
    axis.zaxis.label.set_color(FOREGROUND)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor(BACKGROUND)
        pane.set_edgecolor(GRID)
    axis.view_init(elev=18, azim=-55)


def set_equal_3d_limits(axis: plt.Axes, points: np.ndarray) -> None:
    finite = points[np.isfinite(points).all(axis=1)]
    low = finite.min(axis=0)
    high = finite.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 0.1) * 1.08
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def draw_pose(
    axis: plt.Axes,
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    names: list[str],
    *,
    alpha: float = 1.0,
    linewidth: float = 2.0,
) -> None:
    for parent, child in edges:
        segment = positions[[parent, child]]
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=edge_color(names[parent], names[child]),
            alpha=alpha,
            linewidth=linewidth,
            solid_capstyle="round",
        )


def plot_overview(
    data: dict[str, np.ndarray],
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    output_path: Path,
) -> None:
    names = data["link_names"].tolist()
    frames = positions.shape[0]
    x = np.arange(frames)
    position_errors = np.column_stack((data["left_errors"], data["right_errors"]))
    orientation_errors = np.rad2deg(
        np.column_stack(
            (data["left_orientation_errors"], data["right_orientation_errors"])
        )
    )

    fig = plt.figure(figsize=(16, 9), facecolor=BACKGROUND, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0))
    pose_axis = fig.add_subplot(grid[:, 0], projection="3d")
    error_axis = fig.add_subplot(grid[0, 1])
    action_axis = fig.add_subplot(grid[1, 1])

    style_3d_axis(pose_axis)
    sample_frames = np.unique(np.linspace(0, frames - 1, 5, dtype=int))
    for sample in sample_frames[:-1]:
        draw_pose(
            pose_axis,
            positions[sample],
            edges,
            names,
            alpha=0.14,
            linewidth=1.2,
        )
    draw_pose(
        pose_axis,
        positions[sample_frames[-1]],
        edges,
        names,
        alpha=0.95,
        linewidth=2.3,
    )
    pose_axis.plot(
        data["left_targets"][:, 0],
        data["left_targets"][:, 1],
        data["left_targets"][:, 2],
        color=LEFT,
        linewidth=1.5,
        alpha=0.9,
        label="Left target path",
    )
    pose_axis.plot(
        data["right_targets"][:, 0],
        data["right_targets"][:, 1],
        data["right_targets"][:, 2],
        color=RIGHT,
        linewidth=1.5,
        alpha=0.9,
        label="Right target path",
    )
    all_points = np.concatenate(
        (
            positions.reshape(-1, 3),
            data["left_targets"],
            data["right_targets"],
        ),
        axis=0,
    )
    set_equal_3d_limits(pose_axis, all_points)
    pose_axis.set(
        title="3D link trajectory (five sampled poses)",
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z (m)",
    )
    pose_axis.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=FOREGROUND)

    style_axis(error_axis)
    safe_errors = np.maximum(position_errors * 1000.0, 1e-3)
    error_axis.semilogy(x, safe_errors[:, 0], color=LEFT, label="Left position")
    error_axis.semilogy(x, safe_errors[:, 1], color=RIGHT, label="Right position")
    error_axis.set(
        title="IK tracking error",
        xlabel="Frame",
        ylabel="Position error (mm, log scale)",
    )
    orientation_axis = error_axis.twinx()
    orientation_axis.plot(
        x,
        orientation_errors[:, 0],
        color=LEFT,
        linestyle=":",
        alpha=0.55,
        label="Left orientation",
    )
    orientation_axis.plot(
        x,
        orientation_errors[:, 1],
        color=RIGHT,
        linestyle=":",
        alpha=0.55,
        label="Right orientation",
    )
    orientation_axis.set_ylabel("Orientation error (deg)", color=FOREGROUND)
    orientation_axis.tick_params(colors=FOREGROUND, labelsize=8)
    orientation_axis.spines["right"].set_color(GRID)
    handles = error_axis.lines + orientation_axis.lines
    error_axis.legend(
        handles,
        [line.get_label() for line in handles],
        loc="upper left",
        ncols=2,
        fontsize=8,
        frameon=False,
        labelcolor=FOREGROUND,
    )

    style_axis(action_axis)
    commands = np.asarray(data["hand_motor_commands"], dtype=float)
    motor_names = data["hand_motor_order"].tolist()
    linestyles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)))
    for side, color in enumerate((LEFT, RIGHT)):
        for motor, linestyle in enumerate(linestyles):
            action_axis.plot(
                x,
                commands[:, side, motor],
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
                alpha=0.82,
            )
    action_axis.set(
        title="Independent hand motor commands",
        xlabel="Frame",
        ylabel="Command (rad)",
    )
    side_handles = [
        Line2D([0], [0], color=LEFT, label="Left"),
        Line2D([0], [0], color=RIGHT, label="Right"),
    ]
    motor_handles = [
        Line2D([0], [0], color=FOREGROUND, linestyle=linestyles[index], label=name)
        for index, name in enumerate(motor_names)
    ]
    action_axis.legend(
        handles=side_handles + motor_handles,
        loc="upper right",
        ncols=2,
        fontsize=7,
        frameon=False,
        labelcolor=FOREGROUND,
    )

    report = (
        f"{frames} frames · {len(data['joint_names'])} joints · "
        f"{len(names)} links · schema v{int(data['trajectory_schema_version'])}\n"
        f"Mean position error: L {position_errors[:, 0].mean() * 1000:.2f} mm · "
        f"R {position_errors[:, 1].mean() * 1000:.2f} mm"
    )
    fig.suptitle(
        "G1 trajectory overview\n" + report,
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def plot_animation(
    data: dict[str, np.ndarray],
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    output_path: Path,
    fps: float,
) -> None:
    names = data["link_names"].tolist()
    frames = positions.shape[0]
    left_targets = np.asarray(data["left_targets"])
    right_targets = np.asarray(data["right_targets"])
    position_errors = (
        np.column_stack((data["left_errors"], data["right_errors"])) * 1000.0
    )
    closures = np.asarray(data["hand_closures"])
    x = np.arange(frames)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BACKGROUND)
    layout = fig.add_gridspec(
        2, 2, width_ratios=(2.15, 1.0), height_ratios=(1, 1), wspace=0.22, hspace=0.3
    )
    pose_axis = fig.add_subplot(layout[:, 0], projection="3d")
    error_axis = fig.add_subplot(layout[0, 1])
    closure_axis = fig.add_subplot(layout[1, 1])
    style_3d_axis(pose_axis)
    style_axis(error_axis)
    style_axis(closure_axis)

    all_points = np.concatenate(
        (positions.reshape(-1, 3), left_targets, right_targets), axis=0
    )
    set_equal_3d_limits(pose_axis, all_points)
    pose_axis.set(
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z (m)",
    )
    pose_axis.set_title("G1 3D trajectory", fontsize=13, fontweight="bold")

    edge_lines = []
    for parent, child in edges:
        line, = pose_axis.plot(
            [],
            [],
            [],
            color=edge_color(names[parent], names[child]),
            linewidth=2.4,
            solid_capstyle="round",
        )
        edge_lines.append((line, parent, child))
    left_target_dot, = pose_axis.plot(
        [], [], [], marker="o", color=TARGET, markersize=6, linestyle=""
    )
    right_target_dot, = pose_axis.plot(
        [], [], [], marker="o", color=TARGET, markersize=6, linestyle=""
    )
    left_trail, = pose_axis.plot([], [], [], color=LEFT, linewidth=1.25, alpha=0.8)
    right_trail, = pose_axis.plot([], [], [], color=RIGHT, linewidth=1.25, alpha=0.8)
    frame_text = pose_axis.text2D(
        0.02, 0.96, "", transform=pose_axis.transAxes, color=FOREGROUND, fontsize=10
    )

    safe_errors = np.maximum(position_errors, 1e-3)
    error_axis.semilogy(x, safe_errors[:, 0], color=LEFT, label="Left")
    error_axis.semilogy(x, safe_errors[:, 1], color=RIGHT, label="Right")
    error_axis.set(title="Position error", ylabel="mm (log)", xlim=(0, frames - 1))
    error_axis.legend(frameon=False, labelcolor=FOREGROUND, fontsize=8)
    error_cursor = error_axis.axvline(0, color=TARGET, linewidth=1.1)

    closure_axis.plot(x, closures[:, 0], color=LEFT, label="Left")
    closure_axis.plot(x, closures[:, 1], color=RIGHT, label="Right")
    closure_axis.set(
        title="Hand closure",
        xlabel="Frame",
        ylabel="Normalized command",
        xlim=(0, frames - 1),
        ylim=(0, max(1.0, float(np.nanmax(closures)) * 1.05)),
    )
    closure_axis.legend(frameon=False, labelcolor=FOREGROUND, fontsize=8)
    closure_cursor = closure_axis.axvline(0, color=TARGET, linewidth=1.1)

    fig.subplots_adjust(left=0.02, right=0.97, bottom=0.08, top=0.94)

    def update(frame: int):
        current = positions[frame]
        for line, parent, child in edge_lines:
            segment = current[[parent, child]]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])
        left_target_dot.set_data(
            [left_targets[frame, 0]], [left_targets[frame, 1]]
        )
        left_target_dot.set_3d_properties([left_targets[frame, 2]])
        right_target_dot.set_data(
            [right_targets[frame, 0]], [right_targets[frame, 1]]
        )
        right_target_dot.set_3d_properties([right_targets[frame, 2]])
        start = max(0, frame - int(round(fps)))
        left_trail.set_data(
            left_targets[start : frame + 1, 0], left_targets[start : frame + 1, 1]
        )
        left_trail.set_3d_properties(left_targets[start : frame + 1, 2])
        right_trail.set_data(
            right_targets[start : frame + 1, 0],
            right_targets[start : frame + 1, 1],
        )
        right_trail.set_3d_properties(right_targets[start : frame + 1, 2])
        frame_text.set_text(
            f"Frame {frame + 1:03d}/{frames:03d}\n"
            f"IK error  L {position_errors[frame, 0]:.2f} mm  ·  "
            f"R {position_errors[frame, 1]:.2f} mm"
        )
        error_cursor.set_xdata([frame, frame])
        closure_cursor.set_xdata([frame, frame])
        return [
            *(line for line, _, _ in edge_lines),
            left_target_dot,
            right_target_dot,
            left_trail,
            right_trail,
            frame_text,
            error_cursor,
            closure_cursor,
        ]

    movie = animation.FuncAnimation(
        fig, update, frames=frames, interval=1000.0 / fps, blit=False
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=4000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    movie.save(output_path, writer=writer)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.trajectory.parent / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_trajectory(args.trajectory)
    positions = world_link_positions(data)
    names = data["link_names"].tolist()
    edges = urdf_edges(args.urdf, names)

    overview_path = output_dir / "g1_trajectory_overview.png"
    plot_overview(data, positions, edges, overview_path)
    print(f"Wrote {overview_path}")
    if not args.overview_only:
        animation_path = output_dir / "g1_trajectory_3d.mp4"
        plot_animation(data, positions, edges, animation_path, args.fps)
        print(f"Wrote {animation_path}")


if __name__ == "__main__":
    main()
