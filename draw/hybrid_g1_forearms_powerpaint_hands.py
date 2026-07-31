#!/usr/bin/env python3
"""Combine stable G1 mesh forearms with the previous PowerPaint hands.

The already-handless video is the background. The human-arm mask is split
along its root-to-tip direction: a rendered G1 mesh supplies the proximal
forearm, while the distal hand region comes from the previous PowerPaint
result so that the original grasp pose is retained. The two sources overlap
through a soft wrist transition.
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

from mesh_insert_g1_arms import (
    DEFAULT_HANDLESS_ROOT,
    Episode,
    Pose2D,
    alpha_composite,
    component_pose,
    crop_atlas,
    discover_episodes,
    selected_frames,
    side_for_pose,
    tint_to_scene,
    warp_robot,
)
from mesh_texture_g1_arms import (
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


DEFAULT_POWERPAINT_OUTPUT = Path(__file__).resolve().parent / "output"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "hybrid_output"


@dataclass(frozen=True)
class HybridEpisode:
    episode: Episode
    powerpaint_frames: Path


def discover_hybrid_episodes(
    handless_root: Path,
    mask_root: Path,
    powerpaint_root: Path,
    requested: set[str] | None,
) -> list[HybridEpisode]:
    episodes = discover_episodes(handless_root, mask_root, requested)
    result: list[HybridEpisode] = []
    for episode in episodes:
        frames = powerpaint_root / episode.name / "frames"
        if not frames.is_dir():
            raise FileNotFoundError(
                f"Missing previous PowerPaint frames for episode "
                f"{episode.name}: {frames}"
            )
        result.append(HybridEpisode(episode, frames))
    return result


def smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def longitudinal_fraction(
    pose: Pose2D, shape: tuple[int, int]
) -> np.ndarray:
    height, width = shape
    vector = pose.tip - pose.root
    length = max(1.0, float(np.linalg.norm(vector)))
    direction = vector / length
    y, x = np.mgrid[:height, :width]
    return (
        (x.astype(np.float32) - pose.root[0]) * direction[0]
        + (y.astype(np.float32) - pose.root[1]) * direction[1]
    ) / length


def keep_forearm_only(
    layer: np.ndarray,
    pose: Pose2D,
    full_until: float,
    zero_after: float,
) -> np.ndarray:
    fraction = longitudinal_fraction(pose, layer.shape[:2])
    fade = 1.0 - smoothstep(
        (fraction - full_until) / (zero_after - full_until)
    )
    result = layer.copy()
    result[..., 3] = np.clip(
        np.rint(result[..., 3].astype(np.float32) * fade),
        0,
        255,
    ).astype(np.uint8)
    return result


def hand_alpha_for_component(
    component: np.ndarray,
    pose: Pose2D,
    start_at: float,
    full_at: float,
    edge_feather: float,
) -> np.ndarray:
    fraction = longitudinal_fraction(pose, component.shape)
    wrist_ramp = smoothstep(
        (fraction - start_at) / (full_at - start_at)
    )
    binary = component.astype(np.uint8)
    if edge_feather > 0:
        distance = cv2.distanceTransform(
            binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        boundary = np.clip(distance / edge_feather, 0.0, 1.0)
    else:
        boundary = binary.astype(np.float32)
    return wrist_ramp * boundary * binary


def composite_powerpaint_hands(
    base_rgb: np.ndarray,
    powerpaint_rgb: np.ndarray,
    hand_alpha: np.ndarray,
) -> np.ndarray:
    alpha = np.clip(hand_alpha, 0.0, 1.0)[..., None]
    result = (
        powerpaint_rgb.astype(np.float32) * alpha
        + base_rgb.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def process_episode(
    hybrid: HybridEpisode,
    sprites: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, object]:
    episode = hybrid.episode
    info: VideoInfo = video_info(episode.handless_video)
    mask_paths = sorted(episode.mask_dir.glob("*.png"))
    powerpaint_paths = sorted(hybrid.powerpaint_frames.glob("*.png"))
    if len(mask_paths) != info.frame_count:
        raise ValueError(
            f"{episode.name}: {info.frame_count} frames, "
            f"{len(mask_paths)} masks"
        )
    if len(powerpaint_paths) != info.frame_count:
        raise ValueError(
            f"{episode.name}: {info.frame_count} frames, "
            f"{len(powerpaint_paths)} PowerPaint frames"
        )

    selected = selected_frames(args.frames, info.frame_count)
    complete_video = selected is None
    selected = selected or set(range(info.frame_count))
    output_root = args.output_dir / episode.name
    frames_dir = output_root / "frames"
    arm_alpha_dir = output_root / "forearm_alpha"
    hand_alpha_dir = output_root / "hand_alpha"
    comparisons_dir = output_root / "comparisons"
    for directory in (
        frames_dir,
        arm_alpha_dir,
        hand_alpha_dir,
        comparisons_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    temporary_video = output_root / f".{episode.name}_g1_hybrid.tmp.mp4"
    final_video = output_root / f"{episode.name}_g1_hybrid.mp4"
    writer = (
        start_video_writer(temporary_video, info) if complete_video else None
    )
    capture = cv2.VideoCapture(str(episode.handless_video))
    previous: dict[str, Pose2D] = {}
    try:
        for index in tqdm(
            range(info.frame_count),
            desc=f"hybrid episode {episode.name}",
            unit="frame",
            dynamic_ncols=True,
        ):
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index}")
            handless = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            powerpaint = np.asarray(
                Image.open(powerpaint_paths[index]).convert("RGB")
            )
            if powerpaint.shape != handless.shape:
                raise ValueError(
                    f"{powerpaint_paths[index]} has shape "
                    f"{powerpaint.shape}; expected {handless.shape}"
                )
            mask = read_mask(
                mask_paths[index], (info.height, info.width)
            )

            poses: dict[str, Pose2D] = {}
            components: dict[str, np.ndarray] = {}
            for component in significant_components(mask):
                pose = component_pose(component)
                side = side_for_pose(pose, info.width)
                if side in poses:
                    old_length = np.linalg.norm(
                        poses[side].tip - poses[side].root
                    )
                    new_length = np.linalg.norm(pose.tip - pose.root)
                    if new_length <= old_length:
                        continue
                poses[side] = pose
                components[side] = component

            forearm_layers: list[np.ndarray] = []
            hand_alpha = np.zeros(
                (info.height, info.width), dtype=np.float32
            )
            for side in ("left", "right"):
                pose = poses.get(side)
                component = components.get(side)
                if pose is None or component is None:
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
                layer = keep_forearm_only(
                    layer,
                    pose,
                    args.forearm_full_until,
                    args.forearm_zero_after,
                )
                forearm_layers.append(layer)
                hand_alpha = np.maximum(
                    hand_alpha,
                    hand_alpha_for_component(
                        component,
                        pose,
                        args.hand_start_at,
                        args.hand_full_at,
                        args.hand_edge_feather,
                    ),
                )

            with_forearms, forearm_alpha = alpha_composite(
                handless, forearm_layers, args.forearm_edge_feather
            )
            result = composite_powerpaint_hands(
                with_forearms, powerpaint, hand_alpha
            )
            hand_alpha_u8 = np.clip(
                np.rint(hand_alpha * 255.0), 0, 255
            ).astype(np.uint8)
            combined_alpha = np.maximum(
                forearm_alpha, hand_alpha_u8
            )

            if index in selected:
                Image.fromarray(result).save(
                    frames_dir / f"{index:06d}.png"
                )
                Image.fromarray(forearm_alpha).save(
                    arm_alpha_dir / f"{index:06d}.png"
                )
                Image.fromarray(hand_alpha_u8).save(
                    hand_alpha_dir / f"{index:06d}.png"
                )
                comparison = make_comparison(
                    handless, combined_alpha, result
                )
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
        "powerpaint_frames": str(hybrid.powerpaint_frames),
        "frame_count": info.frame_count,
        "video": str(final_video) if complete_video else None,
        "frames_dir": str(frames_dir),
        "forearm_alpha_dir": str(arm_alpha_dir),
        "hand_alpha_dir": str(hand_alpha_dir),
        "comparisons_dir": str(comparisons_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use stable G1 mesh forearms and retain previous PowerPaint "
            "hands on a handless background."
        )
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--handless-root", type=Path, default=DEFAULT_HANDLESS_ROOT
    )
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument(
        "--powerpaint-output",
        type=Path,
        default=DEFAULT_POWERPAINT_OUTPUT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", nargs="+")
    parser.add_argument(
        "--frames",
        help="Comma-separated preview frames; omit for complete videos",
    )
    parser.add_argument("--width-scale", type=float, default=0.72)
    parser.add_argument("--reach-scale", type=float, default=0.82)
    parser.add_argument("--pose-smoothing", type=float, default=0.55)
    parser.add_argument("--forearm-full-until", type=float, default=0.70)
    parser.add_argument("--forearm-zero-after", type=float, default=0.84)
    parser.add_argument("--hand-start-at", type=float, default=0.68)
    parser.add_argument("--hand-full-at", type=float, default=0.80)
    parser.add_argument(
        "--forearm-edge-feather", type=float, default=0.7
    )
    parser.add_argument("--hand-edge-feather", type=float, default=1.4)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("width_scale", "reach_scale"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.pose_smoothing < 1:
        raise ValueError("--pose-smoothing must be in [0, 1)")
    if not (
        args.forearm_full_until < args.forearm_zero_after
        and args.hand_start_at < args.hand_full_at
    ):
        raise ValueError("Each transition start must precede its end")
    if min(
        args.forearm_edge_feather, args.hand_edge_feather
    ) < 0:
        raise ValueError("Edge feather values must be non-negative")

    args.urdf = args.urdf.resolve()
    args.handless_root = args.handless_root.resolve()
    args.mask_root = args.mask_root.resolve()
    args.powerpaint_output = args.powerpaint_output.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sprites = {
        side: crop_atlas(
            render_mesh_atlas(args.urdf, side, include_hand=False)
        )
        for side in ("left", "right")
    }
    sprite_dir = args.output_dir / "sprites"
    sprite_dir.mkdir(parents=True, exist_ok=True)
    for side, sprite in sprites.items():
        Image.fromarray(sprite).save(
            sprite_dir / f"{side}_g1_source.png"
        )

    requested = set(args.episodes) if args.episodes else None
    episodes = discover_hybrid_episodes(
        args.handless_root,
        args.mask_root,
        args.powerpaint_output,
        requested,
    )
    results = [
        process_episode(episode, sprites, args) for episode in episodes
    ]
    manifest = {
        "method": (
            "stable G1 mesh forearms plus previous PowerPaint hands on "
            "handless video"
        ),
        "urdf": str(args.urdf),
        "g1_mesh_scope": (
            "elbow and wrist links only; hand/finger meshes excluded"
        ),
        "powerpaint_output": str(args.powerpaint_output),
        "human_mask_role": (
            "2-D placement and distal hand selection; never clips the "
            "G1 forearm silhouette"
        ),
        "per_frame_robot_kinematics": False,
        "parameters": {
            "width_scale": args.width_scale,
            "reach_scale": args.reach_scale,
            "pose_smoothing": args.pose_smoothing,
            "forearm_full_until": args.forearm_full_until,
            "forearm_zero_after": args.forearm_zero_after,
            "hand_start_at": args.hand_start_at,
            "hand_full_at": args.hand_full_at,
            "forearm_edge_feather": args.forearm_edge_feather,
            "hand_edge_feather": args.hand_edge_feather,
        },
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
