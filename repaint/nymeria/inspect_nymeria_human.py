#!/usr/bin/env python3
"""Replay source Nymeria 208-D human motion as a 22-joint skeleton."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.nymeria.nymeria_adapter import (  # noqa: E402
    NymeriaClip,
    discover_nymeria_clips,
    load_motion_metadata,
    load_participant_height_m,
    probe_display_video,
    resample_motion_to_video,
)
from utils.visualize_g1_trajectory import (  # noqa: E402
    BACKGROUND,
    FOREGROUND,
    LEFT,
    NEUTRAL,
    RIGHT,
    TARGET,
    set_equal_3d_limits,
    style_3d_axis,
    style_axis,
)


DEFAULT_DATASET = WORKSPACE_ROOT / "dataset/nymeria"
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
SMPL_PARENT_NAMES = {
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "spine2": "spine1",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine2",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "neck": "spine3",
    "left_collar": "spine3",
    "right_collar": "spine3",
    "head": "neck",
    "left_shoulder": "left_collar",
    "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
}
NAME_TO_INDEX = {
    name: index for index, name in enumerate(SMPL_JOINT_NAMES)
}
SMPL_EDGES = tuple(
    (NAME_TO_INDEX[parent], NAME_TO_INDEX[child])
    for child, parent in SMPL_PARENT_NAMES.items()
)


@dataclass(frozen=True)
class NymeriaHumanTrajectory:
    dataset_root: Path
    clip: NymeriaClip
    task: str
    participant_height_m: float
    video_fps: float
    video_size: tuple[int, int]
    full_frame_count: int
    motion_fps: float
    motion_sample_count: int
    frame_indices: np.ndarray
    timestamps: np.ndarray
    source_sample_positions: np.ndarray
    positions_z_up: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_indices))


def _y_up_to_z_up(values: np.ndarray) -> np.ndarray:
    """Apply the Nymeria pipeline's [0.5, 0.5, 0.5, 0.5] rotation."""

    return np.asarray(values, dtype=np.float64)[..., (2, 0, 1)].copy()


def _world_joint_positions(
    root_position_y_up: np.ndarray,
    root_rotation_matrix: np.ndarray,
    relative_joint_positions: np.ndarray,
) -> np.ndarray:
    positions_y_up = (
        root_position_y_up[:, None, :]
        + np.einsum(
            "nij,nkj->nki",
            root_rotation_matrix,
            relative_joint_positions,
        )
    )
    return _y_up_to_z_up(positions_y_up)


def load_nymeria_human_trajectory(
    *,
    dataset_root: Path,
    clip: NymeriaClip,
    start_frame: int,
    end_frame: int | None,
) -> NymeriaHumanTrajectory:
    video = probe_display_video(clip.video_path)
    metadata = load_motion_metadata(clip)
    motion = np.load(clip.motion_208_path)
    resampled = resample_motion_to_video(
        motion,
        motion_fps=metadata.fps,
        display_timestamps=video.display_timestamps,
    )
    if resampled.relative_joint_positions.shape != (
        video.frame_count,
        len(SMPL_JOINT_NAMES),
        3,
    ):
        raise ValueError(
            "Unexpected Nymeria joint position shape: "
            f"{resampled.relative_joint_positions.shape}"
        )

    stop = video.frame_count if end_frame is None else end_frame
    if not 0 <= start_frame < stop <= video.frame_count:
        raise ValueError(
            f"Expected 0 <= start < end <= {video.frame_count}, got "
            f"{start_frame} and {stop}"
        )
    positions = _world_joint_positions(
        resampled.root_position_y_up,
        resampled.root_rotation.as_matrix(),
        resampled.relative_joint_positions,
    )
    if not np.isfinite(positions).all():
        raise ValueError(f"Non-finite human position in {clip.motion_208_path}")

    selection = slice(start_frame, stop)
    return NymeriaHumanTrajectory(
        dataset_root=dataset_root.expanduser().resolve(),
        clip=clip,
        task=metadata.action_text,
        participant_height_m=load_participant_height_m(clip),
        video_fps=video.fps,
        video_size=(video.width, video.height),
        full_frame_count=video.frame_count,
        motion_fps=metadata.fps,
        motion_sample_count=metadata.sample_count,
        frame_indices=np.arange(start_frame, stop, dtype=np.int64),
        timestamps=resampled.timestamps[selection],
        source_sample_positions=resampled.source_sample_positions[selection],
        positions_z_up=positions[selection],
    )


def _edge_color(parent: int, child: int) -> str:
    names = f"{SMPL_JOINT_NAMES[parent]} {SMPL_JOINT_NAMES[child]}"
    if "left_" in names:
        return LEFT
    if "right_" in names:
        return RIGHT
    return NEUTRAL


def draw_pose(
    axis: plt.Axes,
    positions: np.ndarray,
    *,
    alpha: float = 1.0,
    linewidth: float = 2.3,
) -> None:
    for parent, child in SMPL_EDGES:
        segment = positions[[parent, child]]
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=_edge_color(parent, child),
            alpha=alpha,
            linewidth=linewidth,
            solid_capstyle="round",
        )


def _trajectory_limits(
    positions: np.ndarray,
) -> np.ndarray:
    """Include all joints while keeping the ground plane visible."""

    ground = positions.reshape(-1, 3).mean(axis=0)
    ground[2] = 0.0
    return np.concatenate(
        (positions.reshape(-1, 3), ground[None]),
        axis=0,
    )


def plot_overview(
    trajectory: NymeriaHumanTrajectory,
    output_path: Path,
) -> None:
    positions = trajectory.positions_z_up
    frames = trajectory.frame_count
    x = trajectory.frame_indices
    pelvis = positions[:, NAME_TO_INDEX["pelvis"]]
    left_wrist = positions[:, NAME_TO_INDEX["left_wrist"]]
    right_wrist = positions[:, NAME_TO_INDEX["right_wrist"]]

    fig = plt.figure(
        figsize=(16, 9),
        facecolor=BACKGROUND,
        constrained_layout=True,
    )
    grid = fig.add_gridspec(2, 2, width_ratios=(1.18, 1.0))
    pose_axis = fig.add_subplot(grid[:, 0], projection="3d")
    translation_axis = fig.add_subplot(grid[0, 1])
    step_axis = fig.add_subplot(grid[1, 1])
    style_3d_axis(pose_axis)
    style_axis(translation_axis)
    style_axis(step_axis)

    samples = np.unique(np.linspace(0, frames - 1, 5, dtype=int))
    for frame in samples[:-1]:
        draw_pose(
            pose_axis,
            positions[int(frame)],
            alpha=0.15,
            linewidth=1.3,
        )
    draw_pose(
        pose_axis,
        positions[int(samples[-1])],
        linewidth=2.5,
    )
    pose_axis.plot(
        pelvis[:, 0],
        pelvis[:, 1],
        pelvis[:, 2],
        color=TARGET,
        linewidth=1.2,
        linestyle=":",
        alpha=0.9,
        label="pelvis path",
    )
    pose_axis.plot(
        left_wrist[:, 0],
        left_wrist[:, 1],
        left_wrist[:, 2],
        color=LEFT,
        linewidth=1.2,
        alpha=0.8,
        label="left wrist path",
    )
    pose_axis.plot(
        right_wrist[:, 0],
        right_wrist[:, 1],
        right_wrist[:, 2],
        color=RIGHT,
        linewidth=1.2,
        alpha=0.8,
        label="right wrist path",
    )
    set_equal_3d_limits(
        pose_axis,
        _trajectory_limits(positions),
    )
    pose_axis.set(
        title="Nymeria human trajectory (five sampled poses)",
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z / up (m)",
    )
    pose_axis.legend(
        loc="upper left",
        fontsize=8,
        frameon=False,
        labelcolor=FOREGROUND,
    )

    displacement = pelvis - pelvis[:1]
    for dimension, color in enumerate((LEFT, TARGET, RIGHT)):
        translation_axis.plot(
            x,
            displacement[:, dimension],
            color=color,
            linewidth=1.35,
            label=("X", "Y", "Z")[dimension],
        )
    translation_axis.set(
        title="Pelvis translation from first frame",
        xlabel="Displayed video frame",
        ylabel="Displacement (m)",
    )
    translation_axis.legend(
        loc="upper right",
        fontsize=8,
        ncols=3,
        frameon=False,
        labelcolor=FOREGROUND,
    )

    delta_time = np.diff(trajectory.timestamps)
    displacement_per_frame = np.linalg.norm(
        np.diff(positions, axis=0),
        axis=2,
    )
    speed = displacement_per_frame / delta_time[:, None]
    step_axis.plot(
        x[1:],
        np.max(speed, axis=1),
        color=TARGET,
        linewidth=1.4,
        label="maximum landmark",
    )
    step_axis.plot(
        x[1:],
        speed[:, NAME_TO_INDEX["left_wrist"]],
        color=LEFT,
        linewidth=1.0,
        alpha=0.8,
        label="left wrist",
    )
    step_axis.plot(
        x[1:],
        speed[:, NAME_TO_INDEX["right_wrist"]],
        color=RIGHT,
        linewidth=1.0,
        alpha=0.8,
        label="right wrist",
    )
    step_axis.set(
        title="Human landmark speed",
        xlabel="Displayed video frame",
        ylabel="Speed (m/s)",
    )
    step_axis.legend(
        loc="upper right",
        fontsize=8,
        frameon=False,
        labelcolor=FOREGROUND,
    )

    task = textwrap.shorten(
        trajectory.task,
        width=110,
        placeholder="…",
    )
    title = (
        f"Nymeria human · {trajectory.clip.stem} · {frames} frames · "
        f"video {trajectory.video_fps:g} FPS / motion "
        f"{trajectory.motion_fps:g} FPS"
    )
    if task:
        title += f"\n{task}"
    fig.suptitle(
        title,
        color=FOREGROUND,
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def plot_animation(
    trajectory: NymeriaHumanTrajectory,
    output_path: Path,
) -> None:
    positions = trajectory.positions_z_up
    dpi = 100
    fig = plt.figure(
        figsize=(OUTPUT_WIDTH / dpi, OUTPUT_HEIGHT / dpi),
        dpi=dpi,
        facecolor=BACKGROUND,
    )
    axis = fig.add_subplot(111, projection="3d")
    style_3d_axis(axis)
    set_equal_3d_limits(axis, _trajectory_limits(positions))
    axis.set(
        title="Nymeria source human trajectory",
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z / up (m)",
    )

    lines: list[tuple[Any, int, int]] = []
    for parent, child in SMPL_EDGES:
        line, = axis.plot(
            [],
            [],
            [],
            color=_edge_color(parent, child),
            linewidth=2.8,
            solid_capstyle="round",
        )
        lines.append((line, parent, child))

    pelvis_index = NAME_TO_INDEX["pelvis"]
    axis.plot(
        positions[:, pelvis_index, 0],
        positions[:, pelvis_index, 1],
        positions[:, pelvis_index, 2],
        color=TARGET,
        linewidth=0.9,
        linestyle=":",
        alpha=0.35,
    )
    pelvis_marker, = axis.plot(
        [],
        [],
        [],
        marker="o",
        color=TARGET,
        markersize=5,
        linestyle="",
    )
    trails: list[tuple[Any, int]] = []
    for name, color in (("left_wrist", LEFT), ("right_wrist", RIGHT)):
        trail, = axis.plot(
            [],
            [],
            [],
            color=color,
            linewidth=1.5,
            alpha=0.82,
        )
        trails.append((trail, NAME_TO_INDEX[name]))

    frame_text = axis.text2D(
        0.02,
        0.88,
        "",
        transform=axis.transAxes,
        color=FOREGROUND,
        fontsize=10,
    )
    trail_frames = max(1, int(round(2.0 * trajectory.video_fps)))

    def update(frame: int) -> list[Any]:
        current = positions[frame]
        for line, parent, child in lines:
            segment = current[[parent, child]]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])

        pelvis = current[pelvis_index]
        pelvis_marker.set_data([pelvis[0]], [pelvis[1]])
        pelvis_marker.set_3d_properties([pelvis[2]])

        start = max(0, frame - trail_frames + 1)
        for trail, index in trails:
            points = positions[start : frame + 1, index]
            trail.set_data(points[:, 0], points[:, 1])
            trail.set_3d_properties(points[:, 2])

        frame_text.set_text(
            f"{trajectory.clip.stem}\n"
            f"display frame {int(trajectory.frame_indices[frame])} · "
            f"{frame + 1}/{trajectory.frame_count} · "
            f"t={trajectory.timestamps[frame]:.3f} s\n"
            f"motion sample {trajectory.source_sample_positions[frame]:.2f} "
            f"/ {trajectory.motion_sample_count - 1} · "
            f"height {trajectory.participant_height_m:.2f} m"
        )
        return [
            *(line for line, _, _ in lines),
            *(trail for trail, _ in trails),
            pelvis_marker,
            frame_text,
        ]

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=trajectory.frame_count,
        interval=1000.0 / trajectory.video_fps,
        blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=trajectory.video_fps,
        codec="libx264",
        bitrate=4000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": f"Nymeria human {trajectory.clip.stem}"},
    )
    movie.save(output_path, writer=writer)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 3D overview and MP4 replay of source Nymeria human "
            "motion. The 208-D track is timestamp-resampled onto ffprobe's "
            "displayed MP4 frame timeline."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Nymeria source root (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Session name to render; may be supplied more than once.",
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=None,
        help="Full clip stem to render; may be supplied more than once.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--overview-only", action="store_true")
    return parser.parse_args()


def select_clips(
    dataset: Path,
    sessions: list[str] | None,
    clip_stems: list[str] | None,
) -> tuple[NymeriaClip, ...]:
    dataset = dataset.expanduser().resolve()
    available = discover_nymeria_clips(dataset)
    session_filter = None if sessions is None else set(sessions)
    clip_filter = None if clip_stems is None else set(clip_stems)
    selected = tuple(
        clip
        for clip in available
        if (session_filter is None or clip.session in session_filter)
        and (clip_filter is None or clip.stem in clip_filter)
    )
    if not selected:
        filters: list[str] = []
        if session_filter is not None:
            filters.append("sessions=" + ",".join(sorted(session_filter)))
        if clip_filter is not None:
            filters.append("clips=" + ",".join(sorted(clip_filter)))
        detail = " for " + "; ".join(filters) if filters else ""
        raise FileNotFoundError(f"No Nymeria clips selected{detail}")

    selected_sessions = {clip.session for clip in selected}
    if session_filter is not None:
        missing_sessions = session_filter - selected_sessions
        if missing_sessions:
            raise FileNotFoundError(
                "Unknown or empty Nymeria session(s): "
                + ", ".join(sorted(missing_sessions))
            )
    selected_stems = {clip.stem for clip in selected}
    if clip_filter is not None:
        missing_clips = clip_filter - selected_stems
        if missing_clips:
            raise FileNotFoundError(
                "Unknown Nymeria clip stem(s): "
                + ", ".join(sorted(missing_clips))
            )
    return selected


def render_clip(
    args: argparse.Namespace,
    clip: NymeriaClip,
    *,
    multiple: bool,
) -> None:
    trajectory = load_nymeria_human_trajectory(
        dataset_root=args.dataset,
        clip=clip,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    if args.output_dir is None:
        output_dir = (
            trajectory.dataset_root
            / "human_trajectory_check"
            / clip.session
            / clip.stem
        )
    elif multiple:
        output_dir = (
            args.output_dir.expanduser().resolve()
            / clip.session
            / clip.stem
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overview = output_dir / "nymeria_human_trajectory_overview.png"
    plot_overview(trajectory, overview)
    print(f"Wrote {overview}", flush=True)
    if not args.overview_only:
        video = output_dir / "nymeria_human_trajectory_3d.mp4"
        plot_animation(trajectory, video)
        print(f"Wrote {video}", flush=True)


def main() -> None:
    args = parse_args()
    clips = select_clips(args.dataset, args.session, args.clip)
    print(
        f"Processing {len(clips)} Nymeria human clip(s): "
        + ", ".join(clip.stem for clip in clips),
        flush=True,
    )
    for clip in clips:
        render_clip(args, clip, multiple=len(clips) > 1)


if __name__ == "__main__":
    main()
