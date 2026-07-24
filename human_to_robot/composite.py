from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .trajectory import load_single_point_trajectory
from .video_io import read_video, write_video


def _read_numbered_images(
    directory: Path, flags: int = cv2.IMREAD_COLOR
) -> list[np.ndarray]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {directory}")
    images = [cv2.imread(str(path), flags) for path in paths]
    if any(image is None for image in images):
        raise RuntimeError(f"Failed to read one or more images from {directory}")
    return images  # type: ignore[return-value]


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (260, 42), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def compose_video(
    *,
    source_video: Path,
    inpaint_video: Path,
    hdf5_path: Path,
    g1_root: Path,
    object_mask_root: Path | None,
    output_root: Path,
) -> dict[str, float | int | str]:
    source_frames, source_fps = read_video(source_video)
    backgrounds, background_fps = read_video(inpaint_video)
    robot_frames = _read_numbered_images(g1_root / "robot_rgb")
    robot_masks = _read_numbered_images(
        g1_root / "robot_masks", cv2.IMREAD_GRAYSCALE
    )
    robot_depth_mm = _read_numbered_images(
        g1_root / "robot_depth_mm", cv2.IMREAD_UNCHANGED
    )
    object_masks = (
        _read_numbered_images(object_mask_root, cv2.IMREAD_GRAYSCALE)
        if object_mask_root is not None and object_mask_root.is_dir()
        else None
    )
    counts = {
        len(source_frames),
        len(backgrounds),
        len(robot_frames),
        len(robot_masks),
        len(robot_depth_mm),
    }
    if object_masks is not None:
        counts.add(len(object_masks))
    if len(counts) != 1:
        raise ValueError(f"Input frame counts disagree: {sorted(counts)}")
    if abs(source_fps - background_fps) > 0.01:
        raise ValueError(f"FPS mismatch: source={source_fps}, inpaint={background_fps}")

    trajectory = load_single_point_trajectory(hdf5_path)
    archive = np.load(g1_root / "g1_trajectory.npz")
    scale = float(archive["scale"])
    if trajectory.frame_count != len(source_frames):
        raise ValueError("HDF5 trajectory and videos have different lengths")

    height, width = backgrounds[0].shape[:2]
    synthetic_frames: list[np.ndarray] = []
    comparison_frames: list[np.ndarray] = []
    robot_fractions: list[float] = []
    object_occlusion_fractions: list[float] = []
    object_depths = np.maximum(
        0.05, scale * trajectory.right_observation_camera[:, 2] - 0.02
    )

    for frame_id in range(len(source_frames)):
        source = cv2.resize(
            source_frames[frame_id], (width, height), interpolation=cv2.INTER_AREA
        )
        background = backgrounds[frame_id]
        robot = robot_frames[frame_id]
        binary_robot = robot_masks[frame_id] > 0
        depth = robot_depth_mm[frame_id].astype(np.float32) / 1000.0

        occluder = np.zeros((height, width), dtype=bool)
        if object_masks is not None:
            object_mask = cv2.resize(
                object_masks[frame_id],
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            source_object_depth = object_depths[frame_id]
            occluder = (
                (object_mask > 0)
                & binary_robot
                & (source_object_depth <= depth + 0.015)
            )

        visible_robot = binary_robot & ~occluder
        alpha = cv2.GaussianBlur(
            visible_robot.astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=0.8
        )
        alpha[visible_robot] = np.maximum(alpha[visible_robot], 0.94)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        synthetic = (
            robot.astype(np.float32) * alpha
            + background.astype(np.float32) * (1.0 - alpha)
        )
        synthetic = np.clip(synthetic, 0, 255).astype(np.uint8)
        synthetic[occluder] = source[occluder]

        synthetic_frames.append(synthetic)
        comparison_frames.append(
            np.hstack(
                [
                    _label(source, "original"),
                    _label(background, "ProPainter"),
                    _label(synthetic, "G1 synthesis"),
                ]
            )
        )
        robot_fractions.append(float(visible_robot.mean()))
        object_occlusion_fractions.append(float(occluder.mean()))

    output_root.mkdir(parents=True, exist_ok=True)
    write_video(output_root / "synthetic_g1.mp4", synthetic_frames, source_fps)
    write_video(
        output_root / "comparison_original_inpaint_g1.mp4",
        comparison_frames,
        source_fps,
    )
    selected = [0, 25, 50, 75, 100, 125]
    contact = np.vstack(
        [
            np.hstack([synthetic_frames[index] for index in selected[:3]]),
            np.hstack([synthetic_frames[index] for index in selected[3:]]),
        ]
    )
    cv2.imwrite(str(output_root / "synthetic_contact.jpg"), contact)

    report = {
        "frame_count": len(synthetic_frames),
        "fps": source_fps,
        "resolution": [width, height],
        "mean_robot_area_fraction": float(np.mean(robot_fractions)),
        "max_robot_area_fraction": float(np.max(robot_fractions)),
        "mean_object_depth_occlusion_fraction": float(
            np.mean(object_occlusion_fractions)
        ),
        "source_video": str(source_video),
        "inpaint_video": str(inpaint_video),
        "g1_root": str(g1_root),
    }
    (output_root / "composite_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Depth-aware compositing of rendered G1 arms"
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--inpaint-video", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--object-masks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = compose_video(
        source_video=args.source_video.resolve(),
        inpaint_video=args.inpaint_video.resolve(),
        hdf5_path=args.hdf5.resolve(),
        g1_root=args.g1_root.resolve(),
        object_mask_root=(
            args.object_masks.resolve() if args.object_masks is not None else None
        ),
        output_root=args.output.resolve(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
