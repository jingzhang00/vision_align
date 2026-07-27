from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .video_io import video_metadata


DEFAULT_ISAAC_PROFILE: dict[str, object] = {
    "dome_intensity": 900.0,
    "dome_temperature": 4300.0,
    "key_intensity": 180.0,
    "key_temperature": 4500.0,
    "key_angle": 8.0,
    "key_rotation": [-20.0, 15.0, 0.0],
}

DEFAULT_COMPOSITE_PROFILE: dict[str, object] = {
    "robot_exposure": -0.10,
    "robot_rgb_gain": [1.0, 1.0, 1.0],
    "robot_contrast": 0.92,
    "robot_saturation": 0.82,
    "robot_soften_sigma": 0.55,
}


def _write_profile(
    path: Path,
    profile: dict[str, Any],
    *,
    skip_if_unchanged: bool = False,
) -> None:
    serialized = json.dumps(profile, indent=2)
    if (
        skip_if_unchanged
        and path.is_file()
        and path.read_text(encoding="utf-8") == serialized
    ):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def load_lighting_profile(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if profile.get("profile_version") != 1:
        raise ValueError(
            f"Unsupported lighting profile version in {path}: "
            f"{profile.get('profile_version')!r}"
        )
    for section in ("isaac", "composite"):
        if not isinstance(profile.get(section), dict):
            raise ValueError(f"Lighting profile has no '{section}' section: {path}")
    return profile


def resolve_profile_value(
    *,
    explicit: object | None,
    profile: dict[str, Any] | None,
    section: str,
    key: str,
    default: object,
) -> object:
    """Resolve CLI override > lighting profile > stable fallback."""

    if explicit is not None:
        return explicit
    if profile is not None and key in profile[section]:
        return profile[section][key]
    return default


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def _linear_rgb_to_cct(rgb: np.ndarray) -> tuple[float, list[float]]:
    """Approximate CCT from an sRGB-like linear RGB observation."""

    xyz = np.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    ) @ np.asarray(rgb, dtype=np.float64)
    total = float(xyz.sum())
    if not np.isfinite(total) or total <= 1e-8:
        return 6500.0, [0.3127, 0.3290]
    x = float(xyz[0] / total)
    y = float(xyz[1] / total)
    denominator = y - 0.1858
    if abs(denominator) < 1e-6:
        return 6500.0, [x, y]
    n = (x - 0.3320) / denominator
    cct = -449.0 * n**3 + 3525.0 * n**2 - 6823.3 * n + 5520.33
    if not np.isfinite(cct):
        cct = 6500.0
    return float(np.clip(cct, 2500.0, 10000.0)), [x, y]


def _ffprobe_color_metadata(path: Path) -> dict[str, str | None]:
    keys = (
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
        "pix_fmt",
        "codec_name",
    )
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                f"stream={','.join(keys)}",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(completed.stdout).get("streams", [])
        stream = streams[0] if streams else {}
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        stream = {}
    return {
        key: (
            None
            if stream.get(key) in (None, "", "unknown", "unspecified")
            else str(stream[key])
        )
        for key in keys
    }


def _sample_frames(
    video_path: Path,
    *,
    sample_count: int,
    max_dimension: int,
) -> tuple[list[int], list[np.ndarray], float]:
    width, height, frame_count, fps = video_metadata(video_path)
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    indices = np.unique(
        np.linspace(0, frame_count - 1, min(sample_count, frame_count))
        .round()
        .astype(np.int64)
    ).tolist()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    scale = min(1.0, float(max_dimension) / float(max(width, height)))
    frames: list[np.ndarray] = []
    try:
        for frame_id in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not decode sampled frame {frame_id}: {video_path}"
                )
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (
                        max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(frame)
    finally:
        capture.release()
    return indices, frames, fps


def _read_reliability_masks(
    directory: Path | None,
    indices: list[int],
    frames: list[np.ndarray],
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for frame_id, frame in zip(indices, frames, strict=True):
        height, width = frame.shape[:2]
        if directory is None:
            masks.append(np.zeros((height, width), dtype=bool))
            continue
        path = directory / f"{frame_id:05d}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Reliability mask not found: {path}")
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        dilation_radius = max(1, int(round(min(width, height) * 0.0125)))
        kernel_size = 2 * dilation_radius + 1
        mask = cv2.dilate(
            (mask > 0).astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        )
        masks.append(mask > 0)
    return masks


def _frame_measurements(
    frame_bgr: np.ndarray,
    excluded: np.ndarray,
    *,
    tile_columns: int = 10,
    tile_rows: int = 6,
) -> tuple[float, list[np.ndarray], np.ndarray, dict[str, float]]:
    rgb = frame_bgr[..., ::-1].astype(np.float32) / 255.0
    linear = _srgb_to_linear(rgb)
    linear_luma = linear @ np.asarray(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    reliable = ~excluded
    if int(reliable.sum()) < 512:
        raise ValueError("Sampled frame has too few pixels outside the SAM mask")
    luminance_p75 = float(np.percentile(linear_luma[reliable], 75.0))

    height, width = frame_bgr.shape[:2]
    x_edges = np.linspace(0, width, tile_columns + 1).astype(np.int64)
    y_edges = np.linspace(0, height, tile_rows + 1).astype(np.int64)
    tile_colors: list[np.ndarray] = []
    selected = np.zeros((height, width), dtype=bool)
    valid_tile_count = 0
    for top, bottom in zip(y_edges[:-1], y_edges[1:], strict=True):
        for left, right in zip(x_edges[:-1], x_edges[1:], strict=True):
            tile_reliable = reliable[top:bottom, left:right]
            if float(tile_reliable.mean()) < 0.60:
                continue
            valid_tile_count += 1
            tile_rgb_pixels = rgb[top:bottom, left:right]
            tile_rgb = np.median(tile_rgb_pixels[tile_reliable], axis=0)
            tile_luma = float(
                tile_rgb
                @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
            )
            tile_maximum = float(tile_rgb.max())
            tile_saturation = float(
                (tile_maximum - tile_rgb.min()) / max(tile_maximum, 1e-6)
            )
            if not (0.10 < tile_luma < 0.85 and tile_saturation < 0.40):
                continue
            tile_colors.append(tile_rgb)
            selected[top:bottom, left:right] |= tile_reliable

    diagnostics = {
        "valid_pixel_fraction": float(reliable.mean()),
        "neutral_tile_fraction": (
            float(len(tile_colors) / valid_tile_count)
            if valid_tile_count
            else 0.0
        ),
        "neutral_tile_count": float(len(tile_colors)),
        "valid_tile_count": float(valid_tile_count),
    }
    return luminance_p75, tile_colors, selected, diagnostics


def _render_temperature(
    apparent_temperature: float, color_confidence: float
) -> tuple[float, float]:
    # Treat the observation as an effective scene cast, not a recovered
    # physical illuminant. Pull only a bounded fraction away from D65 so beige
    # walls and tabletops cannot make the neutral G1 strongly yellow.
    neutral_temperature = 5500.0
    dome_temperature = float(
        np.clip(
            neutral_temperature
            + 0.70
            * color_confidence
            * (apparent_temperature - neutral_temperature),
            3800.0,
            6500.0,
        )
    )
    key_temperature = float(np.clip(dome_temperature + 200.0, 4000.0, 6700.0))
    return dome_temperature, key_temperature


def _write_diagnostic(
    path: Path,
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    indices: list[int],
) -> None:
    thumbnails: list[np.ndarray] = []
    for frame, mask, frame_id in zip(frames, masks, indices, strict=True):
        preview = frame.copy()
        tint = np.zeros_like(preview)
        tint[..., 1] = 255
        alpha = (mask.astype(np.float32) * 0.22)[..., None]
        preview = np.clip(
            preview.astype(np.float32) * (1.0 - alpha)
            + tint.astype(np.float32) * alpha,
            0.0,
            255.0,
        ).astype(np.uint8)
        cv2.rectangle(preview, (0, 0), (150, 27), (0, 0, 0), -1)
        cv2.putText(
            preview,
            f"frame {frame_id}",
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbnails.append(preview)

    columns = min(4, len(thumbnails))
    rows = int(math.ceil(len(thumbnails) / columns))
    height, width = thumbnails[0].shape[:2]
    while len(thumbnails) < rows * columns:
        thumbnails.append(np.zeros((height, width, 3), dtype=np.uint8))
    montage = np.vstack(
        [
            np.hstack(thumbnails[row * columns : (row + 1) * columns])
            for row in range(rows)
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), montage):
        raise RuntimeError(f"Could not write lighting diagnostic: {path}")


def estimate_lighting_profile(
    *,
    inpaint_video: Path,
    output_path: Path | None = None,
    reliability_mask_root: Path | None = None,
    sample_count: int = 16,
    max_dimension: int = 384,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    if write_artifacts and output_path is None:
        raise ValueError("output_path is required when write_artifacts is enabled")
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3")
    if max_dimension < 64:
        raise ValueError("max_dimension must be at least 64")

    width, height, frame_count, fps = video_metadata(inpaint_video)
    indices, frames, sampled_fps = _sample_frames(
        inpaint_video,
        sample_count=sample_count,
        max_dimension=max_dimension,
    )
    if abs(fps - sampled_fps) > 0.01:
        raise RuntimeError("Video FPS changed while sampling")
    excluded_masks = _read_reliability_masks(
        reliability_mask_root, indices, frames
    )

    frame_srgb: list[np.ndarray] = []
    frame_luminance: list[float] = []
    frame_temperatures: list[float] = []
    selected_masks: list[np.ndarray] = []
    valid_fractions: list[float] = []
    neutral_tile_fractions: list[float] = []
    selection_diagnostics: list[dict[str, float]] = []
    for frame, excluded in zip(frames, excluded_masks, strict=True):
        (
            luminance_p75,
            tile_colors,
            selected,
            diagnostics,
        ) = _frame_measurements(frame, excluded)
        if tile_colors:
            median_srgb = np.median(np.stack(tile_colors), axis=0)
            temperature, _ = _linear_rgb_to_cct(
                _srgb_to_linear(median_srgb)
            )
            frame_srgb.append(median_srgb)
            frame_temperatures.append(temperature)
        frame_luminance.append(luminance_p75)
        selected_masks.append(selected)
        valid_fractions.append(diagnostics["valid_pixel_fraction"])
        neutral_tile_fractions.append(diagnostics["neutral_tile_fraction"])
        selection_diagnostics.append(diagnostics)

    eligible_frame_ratio = float(len(frame_srgb) / len(frames))
    if not frame_srgb:
        robust_srgb = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        robust_rgb = _srgb_to_linear(robust_srgb)
        apparent_temperature = 5500.0
        chromaticity_xy = [0.3127, 0.3290]
        log_chroma_temporal_mad = float("inf")
    else:
        robust_srgb = np.median(np.stack(frame_srgb), axis=0)
        robust_rgb = _srgb_to_linear(robust_srgb)
        apparent_temperature, chromaticity_xy = _linear_rgb_to_cct(robust_rgb)
        log_chroma = np.log(
            np.maximum(np.stack(frame_srgb), 1e-6)
            / np.maximum(np.stack(frame_srgb)[:, 1:2], 1e-6)
        )
        log_chroma_temporal_mad = float(
            np.median(
                np.linalg.norm(
                    log_chroma - np.median(log_chroma, axis=0),
                    axis=1,
                )
            )
        )
    robust_luminance = float(np.median(frame_luminance))
    luminance_stops = np.log2(
        np.maximum(np.asarray(frame_luminance), 1e-6)
    )
    luminance_stops_mad = float(
        np.median(np.abs(luminance_stops - np.median(luminance_stops)))
    )
    luminance_p90_p10_stops = float(
        np.percentile(luminance_stops, 90.0)
        - np.percentile(luminance_stops, 10.0)
    )
    valid_fraction = float(np.median(valid_fractions))
    neutral_tile_coverage = float(np.median(neutral_tile_fractions))
    support_confidence = min(1.0, neutral_tile_coverage / 0.15)
    temporal_confidence = max(
        0.0, 1.0 - log_chroma_temporal_mad / 0.08
    )
    color_confidence = float(
        np.clip(
            support_confidence * temporal_confidence * eligible_frame_ratio,
            0.0,
            1.0,
        )
    )

    dome_temperature, key_temperature = _render_temperature(
        apparent_temperature, color_confidence
    )
    light_ev = float(
        np.clip(
            0.65 * math.log2(max(robust_luminance, 1e-4) / 0.26),
            -0.65,
            0.53,
        )
    )
    dome_intensity = float(np.clip(900.0 * 2.0**light_ev, 550.0, 1300.0))
    key_intensity = 0.20 * dome_intensity

    warnings: list[str] = []
    if valid_fraction < 0.50:
        warnings.append("unmasked_valid_area_below_50_percent")
    if neutral_tile_coverage < 0.03:
        warnings.append("neutral_tile_coverage_below_3_percent")
    if eligible_frame_ratio < 0.50:
        warnings.append("eligible_frame_ratio_below_50_percent")
    if luminance_p90_p10_stops > 0.50:
        warnings.append("luminance_p90_p10_above_0.5_stops")
    if log_chroma_temporal_mad > 0.08:
        warnings.append("log_chroma_temporal_mad_above_0.08")

    color_reliable = not any(
        warning
        in {
            "unmasked_valid_area_below_50_percent",
            "neutral_tile_coverage_below_3_percent",
            "eligible_frame_ratio_below_50_percent",
            "log_chroma_temporal_mad_above_0.08",
        }
        for warning in warnings
    )
    brightness_reliable = not any(
        warning
        in {
            "unmasked_valid_area_below_50_percent",
            "luminance_p90_p10_above_0.5_stops",
        }
        for warning in warnings
    )
    if not color_reliable:
        dome_temperature, key_temperature = 5500.0, 5700.0
    if not brightness_reliable:
        dome_intensity, key_intensity = 900.0, 180.0

    if write_artifacts:
        if output_path is None:
            raise ValueError(
                "output_path is required when write_artifacts is enabled"
            )
        diagnostic_path = output_path.with_name(
            f"{output_path.stem}_samples.jpg"
        )
        _write_diagnostic(diagnostic_path, frames, selected_masks, indices)
    else:
        diagnostic_path = None
    observed_srgb = np.clip(robust_srgb, 0.0, 1.0)
    profile: dict[str, Any] = {
        "profile_version": 1,
        "analysis_video": str(inpaint_video.resolve()),
        "analysis_video_metadata": {
            "resolution": [width, height],
            "frame_count": frame_count,
            "fps": fps,
            "encoded_color": _ffprobe_color_metadata(inpaint_video),
        },
        "decode_policy": {
            "working_assumption": "decoded RGB is sRGB-like display-referred",
            "linearization": "sRGB EOTF working approximation",
            "note": (
                "No physical camera response or illuminant is inferred from "
                "untagged SDR pixels."
            ),
        },
        "sampling": {
            "sample_count": len(indices),
            "sample_indices": indices,
            "analysis_max_dimension": max_dimension,
            "diagnostic_image": (
                str(diagnostic_path.resolve())
                if diagnostic_path is not None
                else None
            ),
            "reliability_mask_directory": (
                str(reliability_mask_root.resolve())
                if reliability_mask_root is not None
                else None
            ),
            "selection": (
                "10x6 tiles with >=60% unmasked, middle-luminance, "
                "low-saturation background pixels"
            ),
        },
        "observations": {
            "selected_linear_rgb": robust_rgb.tolist(),
            "selected_srgb_8bit": (observed_srgb * 255.0).tolist(),
            "selected_linear_luminance": robust_luminance,
            "apparent_cct_kelvin": apparent_temperature,
            "chromaticity_xy": chromaticity_xy,
            "median_valid_pixel_fraction": valid_fraction,
            "neutral_tile_coverage": neutral_tile_coverage,
            "eligible_frame_ratio": eligible_frame_ratio,
            "log_chroma_temporal_mad": log_chroma_temporal_mad,
            "luminance_temporal_mad_stops": luminance_stops_mad,
            "luminance_p90_p10_stops": luminance_p90_p10_stops,
            "per_sample_cct_kelvin": frame_temperatures,
            "per_sample_linear_luminance": frame_luminance,
            "per_sample_selection": selection_diagnostics,
        },
        "confidence": {
            "color": color_confidence,
            "support": support_confidence,
            "temporal_color_stability": temporal_confidence,
            "fixed_profile_reliable": color_reliable and brightness_reliable,
        },
        "warnings": warnings,
        "isaac": {
            "dome_intensity": dome_intensity,
            "dome_temperature": dome_temperature,
            "key_intensity": key_intensity,
            "key_temperature": key_temperature,
            "key_angle": DEFAULT_ISAAC_PROFILE["key_angle"],
            "key_rotation": DEFAULT_ISAAC_PROFILE["key_rotation"],
            "interpretation": (
                "bounded effective render cast; not a recovered physical illuminant"
            ),
        },
        "composite": dict(DEFAULT_COMPOSITE_PROFILE),
    }
    if write_artifacts:
        if output_path is None:
            raise RuntimeError("validated output_path unexpectedly missing")
        _write_profile(output_path, profile)
    return profile


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    midpoint = 0.5 * float(ordered_weights.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weights), midpoint, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def aggregate_task_lighting_profiles(
    *,
    task_name: str,
    episode_profiles: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Aggregate episode observations into one fixed profile for a task."""

    if not episode_profiles:
        raise ValueError("Cannot build a task profile without episode observations")

    episode_srgb = (
        np.asarray(
            [
                profile["observations"]["selected_srgb_8bit"]
                for profile in episode_profiles
            ],
            dtype=np.float64,
        )
        / 255.0
    )
    episode_color_confidence = np.asarray(
        [profile["confidence"]["color"] for profile in episode_profiles],
        dtype=np.float64,
    )
    color_weights = np.maximum(episode_color_confidence, 0.02)
    robust_srgb = np.asarray(
        [
            _weighted_median(episode_srgb[:, channel], color_weights)
            for channel in range(3)
        ],
        dtype=np.float64,
    )
    robust_linear_rgb = _srgb_to_linear(robust_srgb)
    apparent_temperature, chromaticity_xy = _linear_rgb_to_cct(
        robust_linear_rgb
    )

    log_chroma = np.log(
        np.maximum(episode_srgb, 1e-6)
        / np.maximum(episode_srgb[:, 1:2], 1e-6)
    )
    log_chroma_center = np.asarray(
        [
            _weighted_median(log_chroma[:, channel], color_weights)
            for channel in range(log_chroma.shape[1])
        ]
    )
    between_episode_color_mad = _weighted_median(
        np.linalg.norm(log_chroma - log_chroma_center, axis=1),
        color_weights,
    )
    within_episode_confidence = _weighted_median(
        episode_color_confidence, color_weights
    )
    between_episode_stability = max(
        0.0, 1.0 - between_episode_color_mad / 0.12
    )
    task_color_confidence = float(
        np.clip(
            within_episode_confidence * between_episode_stability,
            0.0,
            1.0,
        )
    )

    episode_luminance = np.asarray(
        [
            profile["observations"]["selected_linear_luminance"]
            for profile in episode_profiles
        ],
        dtype=np.float64,
    )
    robust_luminance = float(np.median(episode_luminance))
    episode_luminance_stops = np.log2(np.maximum(episode_luminance, 1e-6))
    between_episode_luminance_p90_p10 = float(
        np.percentile(episode_luminance_stops, 90.0)
        - np.percentile(episode_luminance_stops, 10.0)
    )

    dome_temperature, key_temperature = _render_temperature(
        apparent_temperature, task_color_confidence
    )
    light_ev = float(
        np.clip(
            0.65 * math.log2(max(robust_luminance, 1e-4) / 0.26),
            -0.65,
            0.53,
        )
    )
    dome_intensity = float(np.clip(900.0 * 2.0**light_ev, 550.0, 1300.0))
    key_intensity = 0.20 * dome_intensity

    warnings: list[str] = []
    if task_color_confidence < 0.05:
        warnings.append("task_color_confidence_below_0.05")
        dome_temperature, key_temperature = 5500.0, 5700.0
    if between_episode_color_mad > 0.08:
        warnings.append("task_color_varies_between_episodes")
    if between_episode_luminance_p90_p10 > 0.75:
        warnings.append("task_exposure_varies_between_episodes")

    per_episode = []
    for profile in episode_profiles:
        per_episode.append(
            {
                "analysis_video": profile["analysis_video"],
                "frame_count": profile["analysis_video_metadata"]["frame_count"],
                "selected_srgb_8bit": profile["observations"][
                    "selected_srgb_8bit"
                ],
                "selected_linear_luminance": profile["observations"][
                    "selected_linear_luminance"
                ],
                "apparent_cct_kelvin": profile["observations"][
                    "apparent_cct_kelvin"
                ],
                "color_confidence": profile["confidence"]["color"],
                "warnings": profile["warnings"],
            }
        )

    task_profile: dict[str, Any] = {
        "profile_version": 1,
        "scope": "task",
        "task_name": task_name,
        "episode_count": len(episode_profiles),
        "analysis_episodes": per_episode,
        "decode_policy": {
            "working_assumption": "decoded RGB is sRGB-like display-referred",
            "linearization": "sRGB EOTF working approximation",
            "note": (
                "One robust fixed profile is shared by all episodes in this "
                "task; it is an effective display match, not inverse rendering."
            ),
        },
        "aggregation": {
            "color": "confidence-weighted median of episode observations",
            "luminance": "median of episode linear-luminance P75",
        },
        "observations": {
            "selected_srgb_8bit": (robust_srgb * 255.0).tolist(),
            "selected_linear_rgb": robust_linear_rgb.tolist(),
            "selected_linear_luminance": robust_luminance,
            "apparent_cct_kelvin": apparent_temperature,
            "chromaticity_xy": chromaticity_xy,
            "between_episode_log_chroma_mad": between_episode_color_mad,
            "between_episode_luminance_p90_p10_stops": (
                between_episode_luminance_p90_p10
            ),
        },
        "confidence": {
            "color": task_color_confidence,
            "within_episode_color": within_episode_confidence,
            "between_episode_color_stability": between_episode_stability,
            "fixed_profile_reliable": not warnings,
        },
        "warnings": warnings,
        "isaac": {
            "dome_intensity": dome_intensity,
            "dome_temperature": dome_temperature,
            "key_intensity": key_intensity,
            "key_temperature": key_temperature,
            "key_angle": DEFAULT_ISAAC_PROFILE["key_angle"],
            "key_rotation": DEFAULT_ISAAC_PROFILE["key_rotation"],
            "interpretation": (
                "one bounded effective render cast shared by the task"
            ),
        },
        "composite": dict(DEFAULT_COMPOSITE_PROFILE),
    }
    _write_profile(output_path, task_profile, skip_if_unchanged=True)
    return task_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate one fixed render-lighting profile from ProPainter video"
    )
    parser.add_argument("--inpaint-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reliability-mask-root",
        type=Path,
        help="SAM arm masks to exclude ProPainter-generated pixels",
    )
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--max-dimension", type=int, default=384)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = estimate_lighting_profile(
        inpaint_video=args.inpaint_video.resolve(),
        output_path=args.output.resolve(),
        reliability_mask_root=(
            args.reliability_mask_root.resolve()
            if args.reliability_mask_root is not None
            else None
        ),
        sample_count=args.sample_count,
        max_dimension=args.max_dimension,
    )
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
