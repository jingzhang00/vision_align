#!/usr/bin/env python3
"""Insert actual G1 mesh silhouettes onto the already handless videos.

Human-arm masks are used only to estimate a 2-D root, direction, and reach.
They never clip the inserted robot. The rendered G1 forearm/hand therefore
keeps its own alpha silhouette instead of inheriting the human arm shape.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from mesh_texture_g1_arms import (
    Atlas,
    DEFAULT_MASK_ROOT,
    DEFAULT_URDF,
    VideoInfo,
    make_comparison,
    read_mask,
    render_mesh_atlas,
    significant_components,
    start_video_writer,
    video_info,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HANDLESS_ROOT = (
    WORKSPACE_ROOT
    / "output_new"
    / "basic_pick_place"
    / "videos_handless"
    / "observation.images.ego"
    / "chunk-000"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "insert_output"
SOURCE_EPISODE_NAMES = ("9", "10")


@dataclass(frozen=True)
class Episode:
    name: str
    handless_video: Path
    mask_dir: Path


@dataclass
class Pose2D:
    root: np.ndarray
    tip: np.ndarray
    width: float

    def smoothed(self, previous: "Pose2D", weight: float) -> "Pose2D":
        return Pose2D(
            root=previous.root * weight + self.root * (1.0 - weight),
            tip=previous.tip * weight + self.tip * (1.0 - weight),
            width=previous.width * weight + self.width * (1.0 - weight),
        )


def discover_episodes(
    handless_root: Path,
    mask_root: Path,
    requested: set[str] | None,
) -> list[Episode]:
    videos = sorted(handless_root.glob("file-*.mp4"))
    mask_dirs = sorted(
        path for path in mask_root.glob("file-*") if path.is_dir()
    )
    if len(videos) != len(mask_dirs) or len(videos) != len(
        SOURCE_EPISODE_NAMES
    ):
        raise ValueError(
            f"Expected {len(SOURCE_EPISODE_NAMES)} handless videos/mask "
            f"directories; found {len(videos)} and {len(mask_dirs)}"
        )
    episodes = [
        Episode(name, video, mask_dir)
        for name, video, mask_dir in zip(
            SOURCE_EPISODE_NAMES, videos, mask_dirs, strict=True
        )
        if requested is None or name in requested
    ]
    if not episodes:
        raise ValueError("No requested episode found")
    return episodes


def component_pose(component: np.ndarray) -> Pose2D:
    height, width = component.shape
    y, x = np.nonzero(component)
    points = np.stack([x, y], axis=1).astype(np.float32)
    border_distance = np.minimum.reduce(
        [x, width - 1 - x, y, height - 1 - y]
    )
    border_limit = max(5, int(min(height, width) * 0.008))
    border = points[border_distance <= border_limit]
    if len(border) >= 8:
        root = np.median(border, axis=0)
    else:
        root = np.median(
            points[points[:, 1] >= np.percentile(points[:, 1], 95)],
            axis=0,
        )

    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered))
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    if float(np.dot(axis, np.mean(points, axis=0) - root)) < 0:
        axis = -axis
    perpendicular = np.array([-axis[1], axis[0]], dtype=np.float32)
    relative = points - root
    longitudinal = relative @ axis
    lateral = relative @ perpendicular
    far = points[longitudinal >= np.percentile(longitudinal, 98)]
    tip = np.median(far, axis=0)
    component_width = float(
        np.percentile(lateral, 97) - np.percentile(lateral, 3)
    )
    return Pose2D(root=root, tip=tip, width=max(8.0, component_width))


def side_for_pose(pose: Pose2D, frame_width: int) -> str:
    return "left" if pose.root[0] < frame_width / 2 else "right"


def crop_atlas(atlas: Atlas) -> np.ndarray:
    rgba = atlas.rendered_rgba
    y, x = np.nonzero(rgba[..., 3] > 0)
    if len(x) == 0:
        raise ValueError("Atlas has no non-transparent pixels")
    padding = 2
    x0 = max(0, int(x.min()) - padding)
    x1 = min(rgba.shape[1], int(x.max()) + padding + 1)
    y0 = max(0, int(y.min()) - padding)
    y1 = min(rgba.shape[0], int(y.max()) + padding + 1)
    return rgba[y0:y1, x0:x1].copy()


def warp_robot(
    sprite_rgba: np.ndarray,
    pose: Pose2D,
    frame_shape: tuple[int, int],
    *,
    width_scale: float,
    reach_scale: float,
) -> np.ndarray:
    frame_height, frame_width = frame_shape
    sprite_height, sprite_width = sprite_rgba.shape[:2]
    vector = pose.tip - pose.root
    length = float(np.linalg.norm(vector))
    if length < 8:
        return np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
    direction = vector / length
    perpendicular = np.array(
        [-direction[1], direction[0]], dtype=np.float32
    )
    target_tip = pose.root + vector * reach_scale
    # Preserve the actual G1 sprite proportions, with one global width tuning
    # factor. No human-mask boundary enters the transform.
    proportional_width = (
        length * reach_scale * sprite_width / max(1, sprite_height)
    )
    target_width = max(
        pose.width * 0.75,
        proportional_width * width_scale,
    )

    source = np.float32(
        [
            [sprite_width / 2.0, sprite_height - 1.0],
            [sprite_width / 2.0, 0.0],
            [sprite_width - 1.0, sprite_height - 1.0],
        ]
    )
    target = np.float32(
        [
            pose.root,
            target_tip,
            pose.root + perpendicular * (target_width / 2.0),
        ]
    )
    transform = cv2.getAffineTransform(source, target)
    return cv2.warpAffine(
        sprite_rgba,
        transform,
        (frame_width, frame_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def tint_to_scene(
    robot_rgba: np.ndarray,
    handless_rgb: np.ndarray,
    original_arm_mask: np.ndarray,
) -> np.ndarray:
    output = robot_rgba.copy()
    gray = cv2.cvtColor(handless_rgb, cv2.COLOR_RGB2GRAY)
    nearby = cv2.dilate(
        original_arm_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
    )
    values = gray[nearby > 0]
    median = float(np.median(values)) if len(values) else 150.0
    gain = float(np.clip(median / 155.0, 0.78, 1.12))
    output[..., :3] = np.clip(
        output[..., :3].astype(np.float32) * gain, 0, 255
    ).astype(np.uint8)
    return output


def alpha_composite(
    background_rgb: np.ndarray,
    layers: list[np.ndarray],
    edge_feather: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = background_rgb.astype(np.float32)
    combined_alpha = np.zeros(background_rgb.shape[:2], dtype=np.float32)
    for layer in layers:
        alpha = layer[..., 3].astype(np.float32) / 255.0
        if edge_feather > 0:
            alpha = cv2.GaussianBlur(
                alpha, (0, 0), sigmaX=edge_feather
            )
            alpha = np.clip(alpha, 0.0, 1.0)
        result = (
            layer[..., :3].astype(np.float32) * alpha[..., None]
            + result * (1.0 - alpha[..., None])
        )
        combined_alpha = np.maximum(combined_alpha, alpha)
    return (
        np.clip(np.rint(result), 0, 255).astype(np.uint8),
        np.clip(np.rint(combined_alpha * 255.0), 0, 255).astype(np.uint8),
    )


def selected_frames(value: str | None, frame_count: int) -> set[int] | None:
    if value is None:
        return None
    result: set[int] = set()
    for token in value.split(","):
        index = int(token.strip())
        if index < 0:
            index += frame_count
        if not 0 <= index < frame_count:
            raise ValueError(f"Frame {token!r} is outside this episode")
        result.add(index)
    return result


def process_episode(
    episode: Episode,
    sprites: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, object]:
    info: VideoInfo = video_info(episode.handless_video)
    mask_paths = sorted(episode.mask_dir.glob("*.png"))
    if len(mask_paths) != info.frame_count:
        raise ValueError(
            f"{episode.name}: {info.frame_count} frames, "
            f"{len(mask_paths)} masks"
        )
    selected = selected_frames(args.frames, info.frame_count)
    complete_video = selected is None
    selected = selected or set(range(info.frame_count))

    output_root = args.output_dir / episode.name
    frames_dir = output_root / "frames"
    alpha_dir = output_root / "robot_alpha"
    comparisons_dir = output_root / "comparisons"
    for directory in (frames_dir, alpha_dir, comparisons_dir):
        directory.mkdir(parents=True, exist_ok=True)
    temporary_video = output_root / f".{episode.name}_g1_insert.tmp.mp4"
    final_video = output_root / f"{episode.name}_g1_insert.mp4"
    writer = (
        start_video_writer(temporary_video, info) if complete_video else None
    )

    capture = cv2.VideoCapture(str(episode.handless_video))
    previous: dict[str, Pose2D] = {}
    try:
        for index in tqdm(
            range(info.frame_count),
            desc=f"insert episode {episode.name}",
            unit="frame",
            dynamic_ncols=True,
        ):
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index}")
            handless = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mask = read_mask(
                mask_paths[index], (info.height, info.width)
            )
            poses: dict[str, Pose2D] = {}
            for component in significant_components(mask):
                pose = component_pose(component)
                side = side_for_pose(pose, info.width)
                if side in poses:
                    # Keep the larger reach if SAM split one arm.
                    old_length = np.linalg.norm(
                        poses[side].tip - poses[side].root
                    )
                    new_length = np.linalg.norm(pose.tip - pose.root)
                    if new_length <= old_length:
                        continue
                poses[side] = pose

            layers: list[np.ndarray] = []
            for side in ("left", "right"):
                pose = poses.get(side)
                if pose is None:
                    continue
                if side in previous and args.pose_smoothing > 0:
                    pose = pose.smoothed(
                        previous[side], args.pose_smoothing
                    )
                previous[side] = pose
                layer = warp_robot(
                    sprites[side],
                    pose,
                    (info.height, info.width),
                    width_scale=args.width_scale,
                    reach_scale=args.reach_scale,
                )
                layer = tint_to_scene(layer, handless, mask)
                layers.append(layer)

            result, robot_alpha = alpha_composite(
                handless, layers, args.edge_feather
            )
            if index in selected:
                Image.fromarray(result).save(
                    frames_dir / f"{index:06d}.png"
                )
                Image.fromarray(robot_alpha).save(
                    alpha_dir / f"{index:06d}.png"
                )
                comparison = make_comparison(handless, robot_alpha, result)
                Image.fromarray(comparison).save(
                    comparisons_dir / f"{index:06d}.jpg",
                    quality=92,
                    subsampling=0,
                )
            if writer is not None:
                assert writer.stdin is not None
                writer.stdin.write(np.ascontiguousarray(result).tobytes())
    finally:
        capture.release()
        if writer is not None:
            assert writer.stdin is not None
            writer.stdin.close()
            status = writer.wait()
            if status:
                raise RuntimeError(f"ffmpeg exited with status {status}")
    if writer is not None:
        temporary_video.replace(final_video)
    return {
        "episode": episode.name,
        "handless_video": str(episode.handless_video),
        "human_mask_used_for_clipping": False,
        "frame_count": info.frame_count,
        "video": str(final_video) if complete_video else None,
        "frames_dir": str(frames_dir),
        "robot_alpha_dir": str(alpha_dir),
        "comparisons_dir": str(comparisons_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Insert G1 mesh sprites onto handless videos; human masks are "
            "placement signals only."
        )
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--handless-root", type=Path, default=DEFAULT_HANDLESS_ROOT
    )
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", nargs="+")
    parser.add_argument(
        "--frames",
        help="Comma-separated preview frames; omit for complete videos",
    )
    parser.add_argument("--width-scale", type=float, default=0.72)
    parser.add_argument("--reach-scale", type=float, default=1.0)
    parser.add_argument(
        "--pose-smoothing",
        type=float,
        default=0.55,
        help="Previous 2-D placement weight; no robot kinematics are used",
    )
    parser.add_argument("--edge-feather", type=float, default=0.7)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("width_scale", "reach_scale"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.pose_smoothing < 1:
        raise ValueError("--pose-smoothing must be in [0, 1)")
    if args.edge_feather < 0:
        raise ValueError("--edge-feather must be non-negative")
    args.urdf = args.urdf.resolve()
    args.handless_root = args.handless_root.resolve()
    args.mask_root = args.mask_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    left = render_mesh_atlas(args.urdf, "left")
    right = render_mesh_atlas(args.urdf, "right")
    sprites = {"left": crop_atlas(left), "right": crop_atlas(right)}
    atlas_dir = args.output_dir / "sprites"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sprites["left"]).save(atlas_dir / "left_g1.png")
    Image.fromarray(sprites["right"]).save(atlas_dir / "right_g1.png")

    requested = set(args.episodes) if args.episodes else None
    episodes = discover_episodes(
        args.handless_root, args.mask_root, requested
    )
    results = [
        process_episode(episode, sprites, args) for episode in episodes
    ]
    manifest = {
        "method": "G1 mesh silhouette insertion onto handless video",
        "urdf": str(args.urdf),
        "human_mask_role": "2-D placement only; never clips robot alpha",
        "per_frame_robot_kinematics": False,
        "width_scale": args.width_scale,
        "reach_scale": args.reach_scale,
        "pose_smoothing": args.pose_smoothing,
        "edge_feather": args.edge_feather,
        "episodes": results,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

