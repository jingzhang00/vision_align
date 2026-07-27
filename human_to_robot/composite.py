from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .lighting_profile import (
    DEFAULT_COMPOSITE_PROFILE,
    load_lighting_profile,
    resolve_profile_value,
)
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


def _grade_robot(
    frame: np.ndarray,
    *,
    exposure: float,
    rgb_gain: tuple[float, float, float],
    contrast: float,
    saturation: float,
) -> np.ndarray:
    """Apply one fixed, video-editor-style grade to the robot layer."""

    rgb = frame[..., ::-1].astype(np.float32) / 255.0
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    linear *= np.asarray(rgb_gain, dtype=np.float32) * (2.0**exposure)
    rgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )
    luma = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    rgb = luma[..., None] + saturation * (rgb - luma[..., None])
    rgb = (rgb - 0.5) * contrast + 0.5
    return np.clip(rgb[..., ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)


def _soften_robot(
    frame: np.ndarray, mask: np.ndarray, sigma: float
) -> np.ndarray:
    """Soften the robot without bleeding the render background into its edge."""

    if sigma <= 0.0:
        return frame
    weight = mask.astype(np.float32)
    blurred_weight = cv2.GaussianBlur(weight, (0, 0), sigma)
    premultiplied = frame.astype(np.float32) * weight[..., None]
    blurred = cv2.GaussianBlur(premultiplied, (0, 0), sigma)
    valid = blurred_weight > 1e-5
    result = frame.astype(np.float32)
    result[valid] = blurred[valid] / blurred_weight[valid, None]
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def compose_video(
    *,
    source_video: Path,
    inpaint_video: Path,
    hdf5_path: Path,
    g1_root: Path,
    object_mask_root: Path | None,
    output_root: Path,
    lighting_profile_path: Path | None = None,
    robot_exposure: float | None = None,
    robot_rgb_gain: tuple[float, float, float] | None = None,
    robot_contrast: float | None = None,
    robot_saturation: float | None = None,
    robot_soften_sigma: float | None = None,
) -> dict[str, object]:
    lighting_profile = load_lighting_profile(lighting_profile_path)
    resolved_grade: dict[str, object] = {}
    explicit_values: dict[str, object | None] = {
        "robot_exposure": robot_exposure,
        "robot_rgb_gain": robot_rgb_gain,
        "robot_contrast": robot_contrast,
        "robot_saturation": robot_saturation,
        "robot_soften_sigma": robot_soften_sigma,
    }
    for key, explicit in explicit_values.items():
        resolved_grade[key] = resolve_profile_value(
            explicit=explicit,
            profile=lighting_profile,
            section="composite",
            key=key,
            default=DEFAULT_COMPOSITE_PROFILE[key],
        )
    robot_exposure = float(resolved_grade["robot_exposure"])
    robot_rgb_gain = tuple(
        float(value) for value in resolved_grade["robot_rgb_gain"]  # type: ignore[union-attr]
    )
    if len(robot_rgb_gain) != 3:
        raise ValueError("robot_rgb_gain must contain exactly three values")
    robot_contrast = float(resolved_grade["robot_contrast"])
    robot_saturation = float(resolved_grade["robot_saturation"])
    robot_soften_sigma = float(resolved_grade["robot_soften_sigma"])

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
        robot = _grade_robot(
            robot_frames[frame_id],
            exposure=robot_exposure,
            rgb_gain=robot_rgb_gain,
            contrast=robot_contrast,
            saturation=robot_saturation,
        )
        binary_robot = robot_masks[frame_id] > 0
        robot = _soften_robot(robot, binary_robot, robot_soften_sigma)
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
        "lighting_profile": (
            str(lighting_profile_path) if lighting_profile_path is not None else None
        ),
        "robot_grade": {
            "exposure_stops": robot_exposure,
            "rgb_gain": list(robot_rgb_gain),
            "contrast": robot_contrast,
            "saturation": robot_saturation,
            "soften_sigma_px": robot_soften_sigma,
        },
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
    parser.add_argument(
        "--lighting-profile",
        type=Path,
        help="Task-level JSON generated by human_to_robot.pipeline",
    )
    parser.add_argument("--robot-exposure", type=float)
    parser.add_argument(
        "--robot-rgb-gain",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
    )
    parser.add_argument("--robot-contrast", type=float)
    parser.add_argument("--robot-saturation", type=float)
    parser.add_argument("--robot-soften-sigma", type=float)
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
        lighting_profile_path=(
            args.lighting_profile.resolve()
            if args.lighting_profile is not None
            else None
        ),
        robot_exposure=args.robot_exposure,
        robot_rgb_gain=(
            tuple(args.robot_rgb_gain) if args.robot_rgb_gain is not None else None
        ),
        robot_contrast=args.robot_contrast,
        robot_saturation=args.robot_saturation,
        robot_soften_sigma=args.robot_soften_sigma,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
