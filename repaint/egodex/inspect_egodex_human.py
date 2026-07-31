#!/usr/bin/env python3
"""Replay EgoDex human trajectories stored as HDF5 transform tracks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import textwrap
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

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
from repaint.vda_pipeline_common import probe_video  # noqa: E402


DEFAULT_DATASET = WORKSPACE_ROOT / "dataset/flip_coin"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

BODY_CHAINS = (
    (
        "hip",
        "spine1",
        "spine2",
        "spine3",
        "spine4",
        "spine5",
        "spine6",
        "spine7",
        "neck1",
        "neck2",
        "neck3",
        "neck4",
    ),
    ("spine7", "leftShoulder", "leftArm", "leftForearm", "leftHand"),
    ("spine7", "rightShoulder", "rightArm", "rightForearm", "rightHand"),
)
FINGER_CHAINS = (
    (
        "Hand",
        "ThumbKnuckle",
        "ThumbIntermediateBase",
        "ThumbIntermediateTip",
        "ThumbTip",
    ),
    (
        "Hand",
        "IndexFingerMetacarpal",
        "IndexFingerKnuckle",
        "IndexFingerIntermediateBase",
        "IndexFingerIntermediateTip",
        "IndexFingerTip",
    ),
    (
        "Hand",
        "MiddleFingerMetacarpal",
        "MiddleFingerKnuckle",
        "MiddleFingerIntermediateBase",
        "MiddleFingerIntermediateTip",
        "MiddleFingerTip",
    ),
    (
        "Hand",
        "RingFingerMetacarpal",
        "RingFingerKnuckle",
        "RingFingerIntermediateBase",
        "RingFingerIntermediateTip",
        "RingFingerTip",
    ),
    (
        "Hand",
        "LittleFingerMetacarpal",
        "LittleFingerKnuckle",
        "LittleFingerIntermediateBase",
        "LittleFingerIntermediateTip",
        "LittleFingerTip",
    ),
)


def _all_chains() -> tuple[tuple[str, ...], ...]:
    hand_chains = tuple(
        tuple(f"{side}{name}" for name in chain)
        for side in ("left", "right")
        for chain in FINGER_CHAINS
    )
    return BODY_CHAINS + hand_chains


SKELETON_CHAINS = _all_chains()
CANONICAL_NAMES = tuple(
    dict.fromkeys(name for chain in SKELETON_CHAINS for name in chain)
)


@dataclass(frozen=True)
class HumanTrajectory:
    dataset_root: Path
    stem: str
    hdf5_path: Path
    video_path: Path
    description: str
    fps: float
    source_size: tuple[int, int]
    full_frame_count: int
    frame_indices: np.ndarray
    names: tuple[str, ...]
    positions_world: np.ndarray
    confidence: np.ndarray
    camera_positions_world: np.ndarray
    intrinsic: np.ndarray
    edges: tuple[tuple[int, int], ...]

    @property
    def frame_count(self) -> int:
        return int(self.positions_world.shape[0])


def _video_metadata(path: Path) -> tuple[int, int, int, float]:
    metadata = probe_video(path)
    return (
        metadata.width,
        metadata.height,
        metadata.frame_count,
        metadata.fps,
    )


def _attribute_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return ""
    return str(value)


def _skeleton_edges(
    names: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    name_to_index = {name: index for index, name in enumerate(names)}
    edges: list[tuple[int, int]] = []
    for chain in SKELETON_CHAINS:
        for parent, child in zip(chain, chain[1:]):
            if parent in name_to_index and child in name_to_index:
                edge = (name_to_index[parent], name_to_index[child])
                if edge not in edges:
                    edges.append(edge)
    if not edges:
        raise ValueError("No recognized human skeleton edges were found")
    return tuple(edges)


def load_human_trajectory(
    hdf5_path: Path,
    start_frame: int,
    end_frame: int | None,
) -> HumanTrajectory:
    hdf5_path = hdf5_path.expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    video_path = hdf5_path.with_suffix(".mp4")
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Expected a same-name source video next to the HDF5: {video_path}"
        )

    width, height, video_frame_count, fps = _video_metadata(video_path)
    with h5py.File(hdf5_path, "r") as handle:
        if "transforms/camera" not in handle:
            raise KeyError(f"{hdf5_path} has no transforms/camera dataset")
        camera_dataset = handle["transforms/camera"]
        if camera_dataset.ndim != 3 or camera_dataset.shape[1:] != (4, 4):
            raise ValueError(
                "transforms/camera must have shape (frames, 4, 4), got "
                f"{camera_dataset.shape}"
            )
        full_frame_count = int(camera_dataset.shape[0])
        if video_frame_count != full_frame_count:
            raise ValueError(
                "Source video and HDF5 lengths disagree: "
                f"{video_frame_count} != {full_frame_count}"
            )
        stop = full_frame_count if end_frame is None else end_frame
        if not 0 <= start_frame < stop <= full_frame_count:
            raise ValueError(
                f"Expected 0 <= start < end <= {full_frame_count}, got "
                f"{start_frame} and {stop}"
            )
        selection = slice(start_frame, stop)

        if "camera/intrinsic" not in handle:
            raise KeyError(f"{hdf5_path} has no camera/intrinsic dataset")
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("camera/intrinsic must be a finite 3x3 matrix")

        transform_group = handle.get("transforms")
        if transform_group is None:
            raise KeyError(f"{hdf5_path} has no transforms group")
        names = tuple(name for name in CANONICAL_NAMES if name in transform_group)
        if not names:
            raise ValueError(f"{hdf5_path} has no recognized human joints")

        position_values: list[np.ndarray] = []
        confidence_values: list[np.ndarray] = []
        for name in names:
            transform = handle[f"transforms/{name}"]
            if transform.shape != (full_frame_count, 4, 4):
                raise ValueError(
                    f"transforms/{name} has invalid shape {transform.shape}"
                )
            position_values.append(
                np.asarray(transform[selection, :3, 3], dtype=np.float64)
            )
            confidence_key = f"confidences/{name}"
            if confidence_key in handle:
                confidence_dataset = handle[confidence_key]
                if confidence_dataset.shape != (full_frame_count,):
                    raise ValueError(
                        f"{confidence_key} has invalid shape "
                        f"{confidence_dataset.shape}"
                    )
                confidence_values.append(
                    np.asarray(confidence_dataset[selection], dtype=np.float64)
                )
            else:
                confidence_values.append(
                    np.ones(stop - start_frame, dtype=np.float64)
                )

        positions_world = np.stack(position_values, axis=1)
        confidence = np.clip(
            np.stack(confidence_values, axis=1),
            0.0,
            1.0,
        )
        camera_positions_world = np.asarray(
            camera_dataset[selection, :3, 3],
            dtype=np.float64,
        )
        description = _attribute_text(
            handle.attrs.get(
                "description",
                handle.attrs.get("llm_description", ""),
            )
        )

    if not np.isfinite(positions_world).all():
        raise ValueError(f"{hdf5_path} contains non-finite joint positions")
    if not np.isfinite(confidence).all():
        raise ValueError(f"{hdf5_path} contains non-finite confidence values")
    if not np.isfinite(camera_positions_world).all():
        raise ValueError(f"{hdf5_path} contains non-finite camera positions")

    return HumanTrajectory(
        dataset_root=hdf5_path.parent,
        stem=hdf5_path.stem,
        hdf5_path=hdf5_path,
        video_path=video_path,
        description=description,
        fps=fps,
        source_size=(width, height),
        full_frame_count=full_frame_count,
        frame_indices=np.arange(start_frame, stop, dtype=np.int64),
        names=names,
        positions_world=positions_world,
        confidence=confidence,
        camera_positions_world=camera_positions_world,
        intrinsic=intrinsic,
        edges=_skeleton_edges(names),
    )


def display_coordinates(points_world: np.ndarray) -> np.ndarray:
    """Map EgoDex's Y-up world coordinates to a Z-up plot."""

    result = np.asarray(points_world, dtype=np.float64)[..., (0, 2, 1)].copy()
    result[..., 1] *= -1.0
    return result


def _edge_color(parent_name: str, child_name: str) -> str:
    joined = f"{parent_name} {child_name}"
    if "left" in joined:
        return LEFT
    if "right" in joined:
        return RIGHT
    return NEUTRAL


def _is_finger_edge(parent_name: str, child_name: str) -> bool:
    joined = f"{parent_name} {child_name}"
    return "Finger" in joined or "Thumb" in joined


def _confidence_alpha(
    confidence: float,
    threshold: float,
    maximum: float,
) -> float:
    if confidence < threshold:
        return 0.18 * maximum
    denominator = max(1.0 - threshold, np.finfo(float).eps)
    strength = min(1.0, (confidence - threshold) / denominator)
    return maximum * (0.45 + 0.55 * strength)


def draw_human_pose(
    axis: plt.Axes,
    trajectory: HumanTrajectory,
    positions: np.ndarray,
    frame: int,
    *,
    confidence_threshold: float,
    alpha: float = 1.0,
    linewidth_scale: float = 1.0,
) -> None:
    for parent, child in trajectory.edges:
        segment = positions[frame, [parent, child]]
        edge_confidence = float(
            min(
                trajectory.confidence[frame, parent],
                trajectory.confidence[frame, child],
            )
        )
        finger = _is_finger_edge(
            trajectory.names[parent],
            trajectory.names[child],
        )
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=_edge_color(
                trajectory.names[parent],
                trajectory.names[child],
            ),
            alpha=_confidence_alpha(
                edge_confidence,
                confidence_threshold,
                alpha,
            ),
            linewidth=(1.15 if finger else 2.3) * linewidth_scale,
            solid_capstyle="round",
        )


def _joint_index(
    trajectory: HumanTrajectory,
    name: str,
) -> int | None:
    try:
        return trajectory.names.index(name)
    except ValueError:
        return None


def _group_confidence(
    trajectory: HumanTrajectory,
    predicate: Any,
) -> np.ndarray | None:
    indices = [
        index
        for index, name in enumerate(trajectory.names)
        if predicate(name)
    ]
    if not indices:
        return None
    return np.median(trajectory.confidence[:, indices], axis=1)


def plot_overview(
    trajectory: HumanTrajectory,
    output_path: Path,
    confidence_threshold: float,
) -> None:
    positions = display_coordinates(trajectory.positions_world)
    camera_positions = display_coordinates(
        trajectory.camera_positions_world
    )
    frames = trajectory.frame_count
    x = trajectory.frame_indices

    fig = plt.figure(
        figsize=(16, 9),
        facecolor=BACKGROUND,
        constrained_layout=True,
    )
    grid = fig.add_gridspec(2, 2, width_ratios=(1.18, 1.0))
    pose_axis = fig.add_subplot(grid[:, 0], projection="3d")
    confidence_axis = fig.add_subplot(grid[0, 1])
    step_axis = fig.add_subplot(grid[1, 1])
    style_3d_axis(pose_axis)
    style_axis(confidence_axis)
    style_axis(step_axis)

    samples = np.unique(np.linspace(0, frames - 1, 5, dtype=int))
    for frame in samples[:-1]:
        draw_human_pose(
            pose_axis,
            trajectory,
            positions,
            int(frame),
            confidence_threshold=confidence_threshold,
            alpha=0.16,
            linewidth_scale=0.72,
        )
    draw_human_pose(
        pose_axis,
        trajectory,
        positions,
        int(samples[-1]),
        confidence_threshold=confidence_threshold,
        alpha=1.0,
    )

    for name, color in (("leftHand", LEFT), ("rightHand", RIGHT)):
        index = _joint_index(trajectory, name)
        if index is not None:
            pose_axis.plot(
                positions[:, index, 0],
                positions[:, index, 1],
                positions[:, index, 2],
                color=color,
                linewidth=1.35,
                alpha=0.82,
                label=f"{name} path",
            )
    pose_axis.plot(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        color=TARGET,
        linewidth=1.0,
        linestyle=":",
        alpha=0.8,
        label="camera path",
    )
    all_points = np.concatenate(
        (positions.reshape(-1, 3), camera_positions),
        axis=0,
    )
    set_equal_3d_limits(pose_axis, all_points)
    pose_axis.set(
        title="Human trajectory (five sampled poses)",
        xlabel="World X (m)",
        ylabel="-World Z (m)",
        zlabel="World Y / up (m)",
    )
    pose_axis.legend(
        loc="upper left",
        fontsize=8,
        frameon=False,
        labelcolor=FOREGROUND,
    )

    confidence_groups = (
        (
            "Torso",
            NEUTRAL,
            lambda name: not name.startswith(("left", "right")),
        ),
        ("Left", LEFT, lambda name: name.startswith("left")),
        ("Right", RIGHT, lambda name: name.startswith("right")),
    )
    for label, color, predicate in confidence_groups:
        values = _group_confidence(trajectory, predicate)
        if values is not None:
            confidence_axis.plot(
                x,
                values,
                color=color,
                linewidth=1.4,
                label=label,
            )
    confidence_axis.axhline(
        confidence_threshold,
        color=TARGET,
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
        label=f"threshold {confidence_threshold:g}",
    )
    confidence_axis.set(
        title="Median tracking confidence",
        xlabel="Source frame",
        ylabel="Confidence",
        ylim=(-0.02, 1.02),
    )
    confidence_axis.legend(
        loc="lower right",
        fontsize=8,
        frameon=False,
        labelcolor=FOREGROUND,
    )

    displacement = np.linalg.norm(
        np.diff(trajectory.positions_world, axis=0),
        axis=2,
    )
    valid_pair = np.minimum(
        trajectory.confidence[:-1],
        trajectory.confidence[1:],
    ) >= confidence_threshold
    valid_displacement = np.where(valid_pair, displacement, -np.inf)
    maximum_step = np.max(valid_displacement, axis=1)
    maximum_step[~np.isfinite(maximum_step)] = np.nan
    step_axis.plot(
        x[1:],
        maximum_step * 100.0,
        color=TARGET,
        linewidth=1.5,
    )
    step_axis.set(
        title="Maximum confident landmark step",
        xlabel="Source frame",
        ylabel="Displacement (cm/frame)",
    )

    description = textwrap.shorten(
        trajectory.description,
        width=110,
        placeholder="…",
    )
    subtitle = (
        f"Human trajectory · {trajectory.stem} · {frames} frames · "
        f"{trajectory.fps:g} FPS"
    )
    if description:
        subtitle += f"\n{description}"
    fig.suptitle(
        subtitle,
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def plot_animation(
    trajectory: HumanTrajectory,
    output_path: Path,
    confidence_threshold: float,
) -> None:
    positions = display_coordinates(trajectory.positions_world)
    camera_positions = display_coordinates(
        trajectory.camera_positions_world
    )
    width, height = trajectory.source_size
    dpi = 100
    fig = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor=BACKGROUND,
    )
    axis = fig.add_subplot(111, projection="3d")
    style_3d_axis(axis)
    set_equal_3d_limits(
        axis,
        np.concatenate((positions.reshape(-1, 3), camera_positions), axis=0),
    )
    axis.set(
        title="EgoDex human trajectory",
        xlabel="World X (m)",
        ylabel="-World Z (m)",
        zlabel="World Y / up (m)",
    )

    edge_lines: list[tuple[Any, int, int]] = []
    for parent, child in trajectory.edges:
        finger = _is_finger_edge(
            trajectory.names[parent],
            trajectory.names[child],
        )
        line, = axis.plot(
            [],
            [],
            [],
            color=_edge_color(
                trajectory.names[parent],
                trajectory.names[child],
            ),
            linewidth=1.35 if finger else 2.7,
            solid_capstyle="round",
        )
        edge_lines.append((line, parent, child))

    hand_trails: list[tuple[Any, int]] = []
    for name, color in (("leftHand", LEFT), ("rightHand", RIGHT)):
        index = _joint_index(trajectory, name)
        if index is not None:
            trail, = axis.plot(
                [],
                [],
                [],
                color=color,
                linewidth=1.4,
                alpha=0.82,
            )
            hand_trails.append((trail, index))

    axis.plot(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        color=TARGET,
        linewidth=0.8,
        linestyle=":",
        alpha=0.35,
    )
    camera_marker, = axis.plot(
        [],
        [],
        [],
        marker="^",
        color=TARGET,
        markersize=6,
        linestyle="",
    )
    frame_text = axis.text2D(
        0.02,
        0.96,
        "",
        transform=axis.transAxes,
        color=FOREGROUND,
        fontsize=11,
    )
    left_hand = _joint_index(trajectory, "leftHand")
    right_hand = _joint_index(trajectory, "rightHand")
    trail_frames = max(1, int(round(trajectory.fps)))

    def hand_confidence(frame: int, index: int | None) -> str:
        if index is None:
            return "n/a"
        return f"{trajectory.confidence[frame, index]:.2f}"

    def update(frame: int) -> list[Any]:
        current = positions[frame]
        for line, parent, child in edge_lines:
            segment = current[[parent, child]]
            line.set_data(segment[:, 0], segment[:, 1])
            line.set_3d_properties(segment[:, 2])
            edge_confidence = float(
                min(
                    trajectory.confidence[frame, parent],
                    trajectory.confidence[frame, child],
                )
            )
            line.set_alpha(
                _confidence_alpha(
                    edge_confidence,
                    confidence_threshold,
                    1.0,
                )
            )

        start = max(0, frame - trail_frames + 1)
        for trail, index in hand_trails:
            points = positions[start : frame + 1, index]
            trail.set_data(points[:, 0], points[:, 1])
            trail.set_3d_properties(points[:, 2])

        camera = camera_positions[frame]
        camera_marker.set_data([camera[0]], [camera[1]])
        camera_marker.set_3d_properties([camera[2]])
        frame_text.set_text(
            f"Sequence {trajectory.stem} · source frame "
            f"{int(trajectory.frame_indices[frame])} · "
            f"{frame + 1}/{trajectory.frame_count}\n"
            f"Hand confidence  L {hand_confidence(frame, left_hand)} · "
            f"R {hand_confidence(frame, right_hand)}"
        )
        return [
            *(line for line, _, _ in edge_lines),
            *(trail for trail, _ in hand_trails),
            camera_marker,
            frame_text,
        ]

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
        bitrate=5000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": f"Human trajectory {trajectory.stem}"},
    )
    movie.save(output_path, writer=writer)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a confidence-aware 3D overview and MP4 replay from "
            "EgoDex human transform tracks. Each HDF5 must have a same-name "
            "source MP4; replay FPS and size are read from that video."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Directory containing paired .hdf5/.mp4 files, or one HDF5 "
            f"(default: {DEFAULT_DATASET})."
        ),
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="HDF5 filename stem to render; omit to render every sequence.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Fade edges below this tracking confidence (default: 0.5).",
    )
    parser.add_argument("--overview-only", action="store_true")
    return parser.parse_args()


def list_hdf5_paths(
    dataset: Path,
    episode: str | None,
) -> tuple[Path, ...]:
    dataset = dataset.expanduser().resolve()
    if dataset.is_file():
        if dataset.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Expected an HDF5 file, got {dataset}")
        paths = (dataset,)
    elif dataset.is_dir():
        paths = tuple(
            sorted(
                (
                    path
                    for path in dataset.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in {".h5", ".hdf5"}
                ),
                key=lambda path: (
                    0,
                    int(path.stem),
                )
                if path.stem.isdigit()
                else (1, path.stem),
            )
        )
    else:
        raise FileNotFoundError(dataset)

    if episode is not None:
        paths = tuple(path for path in paths if path.stem == episode)
    if not paths:
        detail = f" matching episode {episode!r}" if episode is not None else ""
        raise FileNotFoundError(f"No HDF5 trajectories{detail} in {dataset}")
    return paths


def render_trajectory(
    args: argparse.Namespace,
    hdf5_path: Path,
    multiple: bool,
) -> None:
    trajectory = load_human_trajectory(
        hdf5_path,
        args.start_frame,
        args.end_frame,
    )
    if args.output_dir is None:
        output_dir = (
            trajectory.dataset_root
            / "human_trajectory_check"
            / trajectory.stem
        )
    elif multiple:
        output_dir = args.output_dir.expanduser().resolve() / trajectory.stem
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overview = output_dir / "human_trajectory_overview.png"
    plot_overview(
        trajectory,
        overview,
        args.confidence_threshold,
    )
    print(f"Wrote {overview}")
    if not args.overview_only:
        video = output_dir / "human_trajectory_3d.mp4"
        plot_animation(
            trajectory,
            video,
            args.confidence_threshold,
        )
        print(f"Wrote {video}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")
    paths = list_hdf5_paths(args.dataset, args.episode)
    print(
        f"Processing {len(paths)} human trajectory sequence(s): "
        + ", ".join(path.stem for path in paths)
    )
    for path in paths:
        render_trajectory(args, path, multiple=len(paths) > 1)


if __name__ == "__main__":
    main()
