#!/usr/bin/env python3
"""Prepare one handless video as a metric VDA RGB-D sequence.

This is deliberately independent from the existing DA3 preparation pipeline.
It reuses only the already-validated episode camera metadata and RGB frames,
then replaces the depth sequence with one offline Video-Depth-Anything pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np

from repaint.vda_depth import (
    DEFAULT_CHECKPOINT,
    DEFAULT_VDA_METRIC_SCALE,
    DecodedVideoFrames,
    INFERENCE_OVERLAP_FRAMES,
    INFERENCE_WINDOW_FRAMES,
    MODEL_CONFIGS,
    MODEL_SCALE_NAMES,
    decode_contiguous_video_frames_rgb,
    depth_m_to_uint16_mm,
    file_sha256,
    infer_metric_vda_sequence,
    load_metric_vda_model,
    scale_and_resize_metric_depths,
    validate_metric_checkpoint,
)
from repaint.vda_scale_validation import (
    infer_and_estimate_handless_metric_scale,
)


def _sha256(path: Path) -> str:
    """Compatibility wrapper for the original preparation helper."""

    return file_sha256(path)


def _video_frames_rgb(
    path: Path,
    *,
    max_resolution: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Compatibility wrapper that decodes the complete input video."""

    decoded = decode_contiguous_video_frames_rgb(
        path,
        max_resolution=max_resolution,
    )
    return decoded.frames_rgb, decoded.fps, decoded.source_size


def _load_model(checkpoint: Path, encoder: str) -> tuple[object, object]:
    """Compatibility wrapper for strict metric VDA model loading."""

    return load_metric_vda_model(checkpoint, encoder)


def _temporary_output(
    output_root: Path,
) -> tuple[Path, Path, Path]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-",
            dir=output_root.parent,
        )
    )
    rgb_root = temporary_root / "rgb"
    depth_root = temporary_root / "depth_mm"
    rgb_root.mkdir(parents=True, exist_ok=True)
    depth_root.mkdir(parents=True, exist_ok=True)
    return temporary_root, rgb_root, depth_root


def _load_template(
    template_root: Path,
) -> dict[str, np.ndarray]:
    archive_path = template_root / "episode_data.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "fps",
        "source_size",
        "mesh_size",
        "frame_indices",
        "camera_pose_cv_xyzw",
        "intrinsic",
        "mesh_intrinsic",
        "scene_scale",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"Template episode archive is missing: {missing}")
    frame_indices = payload["frame_indices"]
    if (
        frame_indices.ndim != 1
        or len(frame_indices) < 2
        or np.any(np.diff(frame_indices.astype(np.int64)) != 1)
    ):
        raise ValueError(
            "Template frame_indices must contain at least two contiguous frames"
        )
    for key in ("source_size", "mesh_size"):
        if payload[key].shape != (2,) or np.any(payload[key] <= 1):
            raise ValueError(f"Template {key} must contain valid WIDTH,HEIGHT")
    for key in ("intrinsic", "mesh_intrinsic"):
        matrix = np.asarray(payload[key], dtype=np.float64)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0.0
            or matrix[1, 1] <= 0.0
        ):
            raise ValueError(f"Template {key} is not a valid pinhole matrix")
    source_width, source_height = (
        int(value) for value in payload["source_size"]
    )
    mesh_width, mesh_height = (
        int(value) for value in payload["mesh_size"]
    )
    expected_mesh_intrinsic = np.asarray(
        payload["intrinsic"],
        dtype=np.float64,
    ).copy()
    expected_mesh_intrinsic[0] *= mesh_width / source_width
    expected_mesh_intrinsic[1] *= mesh_height / source_height
    if not np.allclose(
        payload["mesh_intrinsic"],
        expected_mesh_intrinsic,
        rtol=1e-7,
        atol=1e-7,
    ):
        raise ValueError(
            "Template mesh_intrinsic is inconsistent with source/mesh sizes"
        )
    camera_poses = np.asarray(
        payload["camera_pose_cv_xyzw"],
        dtype=np.float64,
    )
    if (
        camera_poses.shape != (len(frame_indices), 7)
        or not np.isfinite(camera_poses).all()
        or np.any(np.linalg.norm(camera_poses[:, 3:7], axis=1) <= 1e-8)
    ):
        raise ValueError("Template camera poses are invalid")
    fps = float(payload["fps"])
    scene_scale = float(payload["scene_scale"])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("Template FPS must be positive and finite")
    if not np.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError("Template scene_scale must be positive and finite")
    return payload


def _scalar_template_path(
    template: dict[str, np.ndarray],
    key: str,
) -> Path:
    if key not in template:
        raise KeyError(
            f"Automatic VDA scale requires template field {key!r}"
        )
    value = np.asarray(template[key])
    if value.shape != ():
        raise ValueError(f"Template {key} must be a scalar path")
    path = Path(str(value.item())).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _auto_scale_sources(
    *,
    args: argparse.Namespace,
    template: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    source_video = (
        args.vda_scale_source_video.expanduser().resolve()
        if args.vda_scale_source_video is not None
        else _scalar_template_path(template, "source_video")
    )
    source_hdf5 = (
        args.vda_scale_source_hdf5.expanduser().resolve()
        if args.vda_scale_source_hdf5 is not None
        else _scalar_template_path(template, "source_hdf5")
    )
    for path in (source_video, source_hdf5):
        if not path.is_file():
            raise FileNotFoundError(path)
    if source_video == args.input_video.expanduser().resolve():
        raise ValueError(
            "Automatic VDA scale needs the original hand-containing video; "
            "it cannot be the same path as --input-video"
        )
    return source_video, source_hdf5


def _resolve_metric_scale(
    *,
    args: argparse.Namespace,
    template: dict[str, np.ndarray],
    handless_decoded: DecodedVideoFrames,
    handless_depth_raw_m: np.ndarray,
    checkpoint: Path,
) -> tuple[float, dict[str, object]]:
    fixed_scale = float(args.vda_metric_scale)
    if not np.isfinite(fixed_scale) or fixed_scale <= 0.0:
        raise ValueError("--vda-metric-scale must be positive and finite")
    if args.vda_scale_mode == "fixed":
        return fixed_scale, {
            "schema_version": 1,
            "requested_mode": "fixed",
            "resolved_mode": "fixed",
            "resolved_vda_metric_scale": fixed_scale,
            "scale_excludes_scene_scale": True,
        }

    source_video, source_hdf5 = _auto_scale_sources(
        args=args,
        template=template,
    )

    calibration = infer_and_estimate_handless_metric_scale(
        original_video=source_video,
        handless_video=args.input_video,
        source_hdf5=source_hdf5,
        frame_indices=np.asarray(template["frame_indices"], dtype=np.int64),
        handless_decoded=handless_decoded,
        handless_depth_raw_m=handless_depth_raw_m,
        checkpoint=checkpoint,
        encoder=args.encoder,
        input_size=args.input_size,
        max_resolution=args.max_resolution,
        fp32=args.fp32,
        mask_dir=args.vda_scale_mask_dir,
    )
    status = str(calibration["decision"]["status"])
    if status == "validated":
        calibration["requested_mode"] = "auto"
        calibration["resolved_mode"] = "auto"
        return float(calibration["resolved_vda_metric_scale"]), calibration
    reasons = calibration["decision"]["reasons"]
    if args.vda_scale_failure_policy == "fixed":
        calibration["requested_mode"] = "auto"
        calibration["resolved_mode"] = "fixed_fallback"
        calibration["automatic_candidate_vda_metric_scale"] = calibration[
            "resolved_vda_metric_scale"
        ]
        calibration["resolved_vda_metric_scale"] = fixed_scale
        calibration["fallback_reason"] = reasons
        print(
            "WARNING: automatic VDA scale rejected; using explicitly allowed "
            f"fixed fallback {fixed_scale}: {reasons}",
            flush=True,
        )
        return fixed_scale, calibration
    raise RuntimeError(
        "Automatic VDA metric scale failed quality gates: "
        f"{reasons}. Use --vda-scale-mode fixed only after choosing an "
        "episode-specific scale."
    )


def _write_rgb_frames(
    video_path: Path,
    output_root: Path,
    *,
    expected_count: int,
    expected_size: tuple[int, int],
    frame_indices: np.ndarray | None = None,
) -> None:
    """Decode texture PNGs from the same video used for VDA inference."""

    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    selected = (
        np.arange(expected_count, dtype=np.int64)
        if frame_indices is None
        else np.asarray(frame_indices, dtype=np.int64)
    )
    if (
        selected.shape != (expected_count,)
        or int(selected[0]) < 0
        or np.any(np.diff(selected) != 1)
    ):
        raise ValueError(
            "RGB frame_indices must match the expected count and be contiguous"
        )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to reopen video: {video_path}")
    start_frame = int(selected[0])
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if actual != start_frame:
            capture.release()
            raise RuntimeError(
                f"Could not seek {video_path} to frame {start_frame}; got {actual}"
            )
    completed = 0
    try:
        while completed < expected_count:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_bgr.shape[:2] != (
                expected_size[1],
                expected_size[0],
            ):
                raise ValueError(
                    f"Unexpected decoded RGB size: {frame_bgr.shape[:2]}"
                )
            frame_path = output_root / f"{completed:05d}.png"
            if not cv2.imwrite(str(frame_path), frame_bgr):
                raise RuntimeError(f"Failed to write RGB frame: {frame_path}")
            completed += 1
    finally:
        capture.release()
    if completed != expected_count:
        raise ValueError(
            f"Decoded RGB selection changed: {completed} != {expected_count}"
        )


def _write_depth_preview(
    path: Path,
    depths_m: np.ndarray,
    *,
    fps: float,
) -> tuple[float, float]:
    positive = depths_m[np.isfinite(depths_m) & (depths_m > 0.0)]
    if positive.size == 0:
        raise ValueError("VDA returned no positive depths")
    near, far = np.percentile(positive, [1.0, 99.0]).tolist()
    if far <= near:
        far = near + 1e-6
    height, width = depths_m.shape[1:]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open depth preview writer: {path}")
    try:
        for depth in depths_m:
            valid = np.isfinite(depth) & (depth > 0.0)
            normalized = np.clip((depth - near) / (far - near), 0.0, 1.0)
            gray = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
            color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
            color[~valid] = 0
            writer.write(color)
    finally:
        writer.release()
    return float(near), float(far)


def prepare(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    input_video = args.input_video.resolve()
    checkpoint = args.checkpoint.resolve()
    template_root = args.template_prepared.resolve()
    output_root = args.output.resolve()
    if not input_video.is_file():
        raise FileNotFoundError(input_video)
    if args.input_size <= 0:
        raise ValueError("--input-size must be positive")
    checkpoint = validate_metric_checkpoint(checkpoint, args.encoder)

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_root}")
    template = _load_template(template_root)
    if args.vda_scale_mode == "auto":
        # Fail before either expensive CUDA inference when an old template
        # lacks the original-video/HDF5 calibration inputs.
        _auto_scale_sources(args=args, template=template)
    selected_indices = np.asarray(template["frame_indices"], dtype=np.int64)
    decoded = decode_contiguous_video_frames_rgb(
        input_video,
        frame_indices=selected_indices,
        max_resolution=args.max_resolution,
    )
    frame_count = decoded.frame_count
    expected_count = len(template["frame_indices"])
    if frame_count != expected_count:
        raise ValueError(
            f"Video/template frame mismatch: {frame_count} != {expected_count}"
        )
    template_fps = float(template["fps"])
    template_size = tuple(
        int(value) for value in template["source_size"].tolist()
    )
    if not np.isclose(decoded.fps, template_fps, atol=1e-3):
        raise ValueError(
            f"Video/template FPS mismatch: {decoded.fps} != {template_fps}"
        )
    if decoded.source_size != template_size:
        raise ValueError(
            f"Video/template size mismatch: "
            f"{decoded.source_size} != {template_size}"
        )

    depths_raw = infer_metric_vda_sequence(
        decoded,
        checkpoint=checkpoint,
        encoder=args.encoder,
        input_size=args.input_size,
        fp32=args.fp32,
    )
    # The calibration pass needs only handless-video metadata. Release the
    # full handless RGB tensor before decoding the original video.
    decoded = decoded.without_rgb_frames()

    resolved_vda_metric_scale, scale_calibration = _resolve_metric_scale(
        args=args,
        template=template,
        handless_decoded=decoded,
        handless_depth_raw_m=depths_raw,
        checkpoint=checkpoint,
    )
    scene_scale = float(template["scene_scale"])
    mesh_width, mesh_height = (
        int(value) for value in template["mesh_size"].tolist()
    )
    depths_mesh_m = scale_and_resize_metric_depths(
        depths_raw,
        scene_scale=scene_scale,
        vda_metric_scale=resolved_vda_metric_scale,
        mesh_size=(mesh_width, mesh_height),
    )
    positive = depths_mesh_m[depths_mesh_m > 0.0]
    checkpoint_sha256 = file_sha256(checkpoint)
    video_sha256 = file_sha256(input_video)
    temporary_root, rgb_root, depth_root = _temporary_output(output_root)
    try:
        _write_rgb_frames(
            input_video,
            rgb_root,
            expected_count=frame_count,
            expected_size=decoded.source_size,
            frame_indices=decoded.frame_indices,
        )
        lighting_source = template_root / "lighting_profile.json"
        if lighting_source.is_file():
            shutil.copy2(
                lighting_source,
                temporary_root / lighting_source.name,
            )
        else:
            (temporary_root / "lighting_profile.json").write_text(
                json.dumps({"isaac": {}}, indent=2),
                encoding="utf-8",
            )

        for frame_id, resized in enumerate(depths_mesh_m):
            depth_mm = depth_m_to_uint16_mm(resized)
            depth_path = depth_root / f"{frame_id:05d}.png"
            if not cv2.imwrite(str(depth_path), depth_mm):
                raise RuntimeError(f"Failed to write depth frame: {depth_path}")

        np.savez_compressed(
            temporary_root / "depth_m.npz",
            depths=depths_mesh_m,
        )
        preview_near, preview_far = _write_depth_preview(
            temporary_root / "depth_preview.mp4",
            depths_mesh_m,
            fps=decoded.fps,
        )
        template["removed_video"] = np.asarray(str(input_video))
        if "source_video" in scale_calibration:
            template["source_video"] = np.asarray(
                str(scale_calibration["source_video"])
            )
        if "source_hdf5" in scale_calibration:
            template["source_hdf5"] = np.asarray(
                str(scale_calibration["source_hdf5"])
            )
        template["vda_metric_scale"] = np.asarray(
            resolved_vda_metric_scale,
            dtype=np.float64,
        )
        template["vda_scale_mode"] = np.asarray(
            str(scale_calibration["resolved_mode"])
        )
        np.savez_compressed(temporary_root / "episode_data.npz", **template)

        percentiles = np.percentile(
            positive,
            [0.0, 1.0, 50.0, 99.0, 100.0],
        )
        frame_medians = np.asarray(
            [
                np.median(depth[depth > 0.0])
                for depth in depths_mesh_m
            ],
            dtype=np.float64,
        )
        median_steps = np.abs(np.diff(frame_medians))
        model_scale_name = MODEL_SCALE_NAMES[args.encoder]
        report: dict[str, object] = {
            "schema_version": 1,
            "method": "offline_metric_video_depth_anything",
            "model": {
                "name": (
                    f"Metric-Video-Depth-Anything-{model_scale_name}"
                ),
                "encoder": args.encoder,
                "metric": True,
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": checkpoint_sha256,
                "input_size": args.input_size,
                "max_video_resolution": args.max_resolution,
                "precision": "fp32" if args.fp32 else "autocast-fp16",
                "window_frames": INFERENCE_WINDOW_FRAMES,
                "window_overlap_frames": INFERENCE_OVERLAP_FRAMES,
            },
            "input": {
                "video": str(input_video),
                "video_bytes": input_video.stat().st_size,
                "video_sha256": video_sha256,
                "human_removed": True,
                "frame_count": frame_count,
                "video_frame_count": decoded.video_frame_count,
                "selected_frame_indices": decoded.frame_indices.tolist(),
                "selected_frame_start": int(decoded.frame_indices[0]),
                "selected_frame_end_exclusive": int(
                    decoded.frame_indices[-1] + 1
                ),
                "selection_is_contiguous": True,
                "fps": decoded.fps,
                "source_size": list(decoded.source_size),
                "inference_size": list(decoded.inference_size),
                "rgb_texture_source": (
                    "fresh full-resolution decode of the inference video"
                ),
            },
            "geometry": {
                "depth_interpretation": (
                    "metric camera-z prediction calibrated to an automatic "
                    "episode source-depth scale, then scaled into robot/world"
                ),
                "vda_metric_scale": resolved_vda_metric_scale,
                "vda_metric_scale_source": (
                    "automatic per-episode HDF5 hand anchor plus unchanged "
                    "background transfer"
                    if scale_calibration["resolved_mode"] == "auto"
                    else "explicit fixed value or explicitly allowed fallback"
                ),
                "vda_scale_calibration": scale_calibration,
                "vda_metric_scale_application_count": 1,
                "scene_scale": scene_scale,
                "scene_scale_application_count": 1,
                "effective_raw_depth_to_robot_scale": (
                    resolved_vda_metric_scale * scene_scale
                ),
                "mesh_size": [mesh_width, mesh_height],
                "mesh_intrinsic": template["mesh_intrinsic"].tolist(),
                "depth_resize_interpolation": "nearest",
                "depth_unit": (
                    "meters (depth_m.npz) and uint16 millimeters"
                ),
                "depth_percentiles_m": {
                    "min": float(percentiles[0]),
                    "p01": float(percentiles[1]),
                    "p50": float(percentiles[2]),
                    "p99": float(percentiles[3]),
                    "max": float(percentiles[4]),
                },
                "positive_pixel_fraction": float(
                    np.mean(depths_mesh_m > 0.0)
                ),
            },
            "quick_temporal_check": {
                "frame_median_depth_min_m": float(frame_medians.min()),
                "frame_median_depth_max_m": float(frame_medians.max()),
                "adjacent_median_step_p50_m": float(
                    np.percentile(median_steps, 50.0)
                ),
                "adjacent_median_step_p95_m": float(
                    np.percentile(median_steps, 95.0)
                ),
                "adjacent_median_step_max_m": float(median_steps.max()),
            },
            "artifacts": {
                "prepared_root": str(output_root),
                "depth_npz": str(output_root / "depth_m.npz"),
                "depth_preview": str(output_root / "depth_preview.mp4"),
                "episode_data": str(output_root / "episode_data.npz"),
            },
            "preview_range_m": {
                "near_p01": preview_near,
                "far_p99": preview_far,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = temporary_root / "prepare_report.json"
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        temporary_root.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one offline metric Video-Depth-Anything pass over a "
            "handless episode and prepare RGB-D frames for Isaac."
        )
    )
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--template-prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--encoder",
        choices=tuple(MODEL_CONFIGS),
        default="vitl",
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-resolution", type=int, default=1280)
    parser.add_argument(
        "--vda-scale-mode",
        choices=("auto", "fixed"),
        default="auto",
        help=(
            "Estimate one per-episode VDA-to-source metric scale from the "
            "original video and HDF5 hand anchors, or use a fixed value "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--vda-metric-scale",
        type=float,
        default=DEFAULT_VDA_METRIC_SCALE,
        help=(
            "Fixed scale used by --vda-scale-mode fixed, or by an explicitly "
            "allowed fixed fallback (default: legacy 0.525)"
        ),
    )
    parser.add_argument(
        "--vda-scale-failure-policy",
        choices=("error", "fixed"),
        default="error",
        help=(
            "Fail when automatic calibration is unreliable, or explicitly "
            "fall back to --vda-metric-scale (default: error)"
        ),
    )
    parser.add_argument(
        "--vda-scale-source-video",
        type=Path,
        help=(
            "Original hand-containing video for automatic scale; default is "
            "template episode_data.npz source_video"
        ),
    )
    parser.add_argument(
        "--vda-scale-source-hdf5",
        type=Path,
        help=(
            "Source HDF5 hand anchors for automatic scale; default is "
            "template episode_data.npz source_hdf5"
        ),
    )
    parser.add_argument(
        "--vda-scale-mask-dir",
        type=Path,
        help=(
            "Optional hand-mask PNG directory indexed by source frame; "
            "default derives the changed region from original/handless RGB"
        ),
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable CUDA autocast (uses more memory and is slower)",
    )
    return parser


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()
