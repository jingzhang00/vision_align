"""Estimate and validate an episode-level VDA metric scale.

This is intentionally a validation tool rather than part of the reconstruction
pipeline.  It keeps the two scale conversions separate:

* raw VDA depth -> source-camera meters, estimated from the original video;
* source-camera meters -> robot/world scale, owned by the caller.

The original-video estimate is transferred to the handless prediction using
unchanged background pixels.  An optional DA3 depth directory is used only as
an independent compatibility check, never as the primary metric anchor.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from human_to_robot.trajectory import HAND_LANDMARK_NAMES
from repaint.vda_depth import (
    DEFAULT_CHECKPOINT,
    DecodedVideoFrames,
    decode_contiguous_video_frames_rgb,
    file_sha256,
    infer_metric_vda_sequence,
    validate_frame_indices,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _video_metadata(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(width, height, frame_count) <= 0 or fps <= 0.0:
        raise ValueError(f"Invalid video metadata: {path}")
    return width, height, frame_count, fps


def _declared_hdf5_source_video(path: Path) -> Path | None:
    candidates: list[str] = []

    def append_attribute(value: Any) -> None:
        if isinstance(value, (bytes, np.bytes_)):
            candidates.append(bytes(value).decode("utf-8"))
        else:
            candidates.append(str(value))

    with h5py.File(path, "r") as handle:
        if "source_video" in handle.attrs:
            append_attribute(handle.attrs["source_video"])

        def collect(_name: str, item: h5py.Group | h5py.Dataset) -> None:
            if "source_video" in item.attrs:
                append_attribute(item.attrs["source_video"])

        handle.visititems(collect)
    unique = {value for value in candidates if value}
    if len(unique) > 1:
        raise ValueError(
            f"{path} declares multiple source_video values: {sorted(unique)}"
        )
    if not unique:
        return None
    return Path(unique.pop()).expanduser().resolve()


def _validate_source_frame_provenance(
    *,
    original_video: Path,
    handless_video_frame_count: int,
    source_hdf5: Path,
    hdf5_frame_count: int,
    selected_frame_indices: np.ndarray,
) -> None:
    """Reject ambiguous temporal mappings while allowing a declared prefix."""

    selected = validate_frame_indices(selected_frame_indices)
    original_frame_count = _video_metadata(original_video)[2]
    if handless_video_frame_count != hdf5_frame_count:
        raise ValueError(
            "Handless video and source HDF5 total frame counts disagree: "
            f"{handless_video_frame_count} != {hdf5_frame_count}"
        )
    if int(selected[-1]) >= min(original_frame_count, hdf5_frame_count):
        raise ValueError(
            "Selected frames are not covered by both the original video and "
            "source HDF5"
        )
    if original_frame_count == hdf5_frame_count:
        return

    declared_source = _declared_hdf5_source_video(source_hdf5)
    if original_frame_count < hdf5_frame_count or declared_source is None:
        raise ValueError(
            "Original video and source HDF5 total frame counts disagree "
            f"({original_frame_count} != {hdf5_frame_count}); this is only "
            "allowed when the HDF5/handless sequence is a covered frame-zero "
            "prefix and the HDF5 declares source_video provenance"
        )
    if declared_source != original_video.resolve():
        raise ValueError(
            "Source HDF5 declares a different source_video: "
            f"{declared_source} != {original_video.resolve()}"
        )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _distribution(
    values: np.ndarray,
    *,
    positive_only: bool = True,
) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if positive_only:
        finite = finite[finite > 0.0]
    if finite.size == 0:
        return {
            "count": 0,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "log_mad": None,
        }
    quantiles = np.percentile(finite, [5, 25, 50, 75, 95])
    positive = finite[finite > 0.0]
    if positive.size:
        log_values = np.log(positive)
        log_median = float(np.median(log_values))
        log_mad: float | None = float(
            np.median(np.abs(log_values - log_median))
        )
    else:
        log_mad = None
    return {
        "count": int(finite.size),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "log_mad": log_mad,
    }


def _frame_equal_summary(
    frame_values: list[np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    frame_medians = np.full(len(frame_values), np.nan, dtype=np.float64)
    for frame_id, values in enumerate(frame_values):
        array = np.asarray(values)
        if array.size > 0:
            frame_medians[frame_id] = np.median(array)
    valid_frame_medians = frame_medians[
        np.isfinite(frame_medians) & (frame_medians > 0.0)
    ]
    summary = _distribution(valid_frame_medians)
    summary["aggregation"] = "median within frame, then median across frames"
    summary["supporting_frames"] = int(valid_frame_medians.size)
    summary["raw_sample_count"] = int(
        sum(np.asarray(values).size for values in frame_values)
    )
    return summary, frame_medians


def _minimum_anchor_frame_count(selected_frame_count: int) -> int:
    if selected_frame_count <= 0:
        raise ValueError("selected_frame_count must be positive")
    return max(10, int(math.ceil(0.25 * selected_frame_count)))


def _anchor_summary_is_supported(
    summary: dict[str, Any],
    *,
    selected_frame_count: int,
) -> bool:
    """Use the same support gate for root selection and final validation."""

    return (
        int(summary["supporting_frames"])
        >= _minimum_anchor_frame_count(selected_frame_count)
        and int(summary["raw_sample_count"]) >= 30
        and summary["log_mad"] is not None
        and float(summary["log_mad"]) <= 0.15
    )


def _bootstrap_product(
    first: np.ndarray,
    second: np.ndarray,
    *,
    samples: int = 2000,
) -> dict[str, float | int | None]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    values_first = first[np.isfinite(first) & (first > 0.0)]
    values_second = second[np.isfinite(second) & (second > 0.0)]
    if min(values_first.size, values_second.size) < 10:
        return {
            "first_frame_count": int(values_first.size),
            "second_frame_count": int(values_second.size),
            "sample_count": 0,
            "p025": None,
            "p50": None,
            "p975": None,
            "relative_95pct_half_width": None,
        }
    generator = np.random.default_rng(20260731)
    # Bound the temporary index storage for long episodes. The previous
    # samples-by-frames allocation could consume hundreds of megabytes.
    maximum_indices_per_batch = 2_000_000
    batch_size = min(
        samples,
        max(
            1,
            maximum_indices_per_batch
            // max(values_first.size, values_second.size),
        ),
    )
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        current_batch = stop - start
        first_indices = generator.integers(
            0,
            values_first.size,
            size=(current_batch, values_first.size),
        )
        second_indices = generator.integers(
            0,
            values_second.size,
            size=(current_batch, values_second.size),
        )
        estimates[start:stop] = (
            np.median(values_first[first_indices], axis=1)
            * np.median(values_second[second_indices], axis=1)
        )
    p025, p50, p975 = np.percentile(estimates, [2.5, 50.0, 97.5])
    return {
        "first_frame_count": int(values_first.size),
        "second_frame_count": int(values_second.size),
        "sample_count": int(samples),
        "p025": float(p025),
        "p50": float(p50),
        "p975": float(p975),
        "relative_95pct_half_width": float((p975 - p025) / (2.0 * p50)),
    }


def _scaled_intrinsic(
    intrinsic: np.ndarray,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    target_width, target_height = target_size
    result = np.asarray(intrinsic, dtype=np.float64).copy()
    result[0] *= target_width / source_width
    result[1] *= target_height / source_height
    return result


def _load_hdf5_geometry(
    path: Path,
    *,
    frame_indices: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    float,
    int,
]:
    selected = (
        None
        if frame_indices is None
        else validate_frame_indices(frame_indices)
    )
    with h5py.File(path, "r") as handle:
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        camera_dataset = handle["transforms/camera"]
        if camera_dataset.ndim != 3 or camera_dataset.shape[1:] != (4, 4):
            raise ValueError(
                "transforms/camera has invalid shape "
                f"{camera_dataset.shape}"
            )
        frame_count = int(camera_dataset.shape[0])
        if selected is not None and int(selected[-1]) >= frame_count:
            raise IndexError(
                f"Selected HDF5 frame {int(selected[-1])} exceeds "
                f"{frame_count} frames: {path}"
            )
        selection = (
            slice(None)
            if selected is None
            else slice(int(selected[0]), int(selected[-1]) + 1)
        )
        camera_to_world = np.asarray(
            camera_dataset[selection],
            dtype=np.float64,
        )
        transform_values: list[np.ndarray] = []
        confidence_values: list[np.ndarray] = []
        labels: list[str] = []
        for side in ("left", "right"):
            for landmark in HAND_LANDMARK_NAMES:
                transform_key = f"transforms/{side}{landmark}"
                if transform_key not in handle:
                    continue
                transform_dataset = handle[transform_key]
                if transform_dataset.shape != (frame_count, 4, 4):
                    raise ValueError(
                        f"{transform_key} has invalid shape "
                        f"{transform_dataset.shape}"
                    )
                transforms = np.asarray(
                    transform_dataset[selection],
                    dtype=np.float64,
                )
                confidence_key = f"confidences/{side}{landmark}"
                if confidence_key in handle:
                    confidence_dataset = handle[confidence_key]
                    if confidence_dataset.shape != (frame_count,):
                        raise ValueError(
                            f"{confidence_key} has invalid shape "
                            f"{confidence_dataset.shape}"
                        )
                    confidence = np.asarray(
                        confidence_dataset[selection],
                        dtype=np.float64,
                    )
                else:
                    confidence = np.ones(
                        len(camera_to_world),
                        dtype=np.float64,
                    )
                transform_values.append(transforms)
                confidence_values.append(confidence)
                labels.append(f"{side}{landmark}")
    if not transform_values:
        raise ValueError(f"{path} has no hand transforms")
    transforms_world = np.stack(transform_values, axis=1)
    confidence = np.stack(confidence_values, axis=1)
    points_world = transforms_world[..., :3, 3]
    world_to_camera = np.linalg.inv(camera_to_world)
    homogeneous = np.concatenate(
        [
            points_world,
            np.ones((*points_world.shape[:2], 1), dtype=np.float64),
        ],
        axis=2,
    )
    points_camera = np.einsum(
        "nij,nkj->nki",
        world_to_camera,
        homogeneous,
    )[..., :3]
    translations = camera_to_world[:, :3, 3]
    # Exact maximum pairwise baseline with bounded memory. The Gram-matrix
    # form avoids materializing an N x N x 3 tensor.
    squared_norm = np.einsum("ni,ni->n", translations, translations)
    maximum_squared = 0.0
    for start in range(0, len(translations), 256):
        stop = min(start + 256, len(translations))
        squared_distance = (
            squared_norm[start:stop, None]
            + squared_norm[None, :]
            - 2.0 * translations[start:stop] @ translations.T
        )
        maximum_squared = max(
            maximum_squared,
            float(np.max(squared_distance)),
        )
    max_baseline = float(math.sqrt(max(0.0, maximum_squared)))
    return (
        intrinsic,
        camera_to_world,
        points_camera,
        confidence,
        tuple(labels),
        max_baseline,
        frame_count,
    )


def _depth_cache_manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.cache.json")


def _depth_cache_key(
    *,
    video: Path,
    video_metadata: tuple[int, int, int, float],
    video_sha256: str,
    frame_count: int,
    max_resolution: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    encoder: str,
    input_size: int,
    fp32: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "video": str(video.resolve()),
        "video_bytes": int(video.stat().st_size),
        "video_sha256": video_sha256,
        "video_metadata": {
            "width": int(video_metadata[0]),
            "height": int(video_metadata[1]),
            "frame_count": int(video_metadata[2]),
            "fps": float(video_metadata[3]),
        },
        "selected_frame_start": 0,
        "selected_frame_end_exclusive": int(frame_count),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": int(checkpoint.stat().st_size),
        "checkpoint_sha256": checkpoint_sha256,
        "encoder": encoder,
        "input_size": int(input_size),
        "max_resolution": int(max_resolution),
        "fp32": bool(fp32),
    }


def _load_valid_depth_cache(
    *,
    output: Path,
    expected_key: dict[str, Any],
    frame_count: int,
) -> np.ndarray | None:
    manifest_path = _depth_cache_manifest_path(output)
    if not output.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != expected_key:
            return None
        depths = np.load(output, mmap_mode="r")
        expected_shape = tuple(int(value) for value in manifest["depth_shape"])
        if (
            depths.ndim != 3
            or depths.shape != expected_shape
            or depths.shape[0] != frame_count
            or depths.dtype != np.float32
            or int(output.stat().st_size) != int(manifest["depth_bytes"])
            or file_sha256(output) != manifest["depth_sha256"]
        ):
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return depths


def _write_depth_cache(
    *,
    output: Path,
    depths: np.ndarray,
    cache_key: dict[str, Any],
) -> np.ndarray:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(depths, dtype=np.float32))
    cached = np.load(output, mmap_mode="r")
    manifest = {
        "schema_version": 1,
        "cache_key": cache_key,
        "depth_shape": list(cached.shape),
        "depth_dtype": str(cached.dtype),
        "depth_bytes": int(output.stat().st_size),
        "depth_sha256": file_sha256(output),
    }
    manifest_path = _depth_cache_manifest_path(output)
    temporary_manifest = manifest_path.with_name(
        f"{manifest_path.name}.temporary"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return cached


def _infer_or_load(
    *,
    video: Path,
    output: Path,
    frame_count: int,
    max_resolution: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    encoder: str,
    input_size: int,
    fp32: bool,
    reuse: bool,
) -> tuple[
    np.ndarray,
    tuple[int, int],
    tuple[int, int],
    float,
    dict[str, Any],
]:
    metadata = _video_metadata(video)
    if metadata[2] < frame_count:
        raise ValueError(
            f"{video} has {metadata[2]} frames, fewer than the required "
            f"{frame_count}"
        )
    video_sha256 = file_sha256(video)
    cache_key = _depth_cache_key(
        video=video,
        video_metadata=metadata,
        video_sha256=video_sha256,
        frame_count=frame_count,
        max_resolution=max_resolution,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        encoder=encoder,
        input_size=input_size,
        fp32=fp32,
    )
    if reuse:
        depths = _load_valid_depth_cache(
            output=output,
            expected_key=cache_key,
            frame_count=frame_count,
        )
        if depths is not None:
            cache_info = {
                "requested": True,
                "reused": True,
                "manifest": str(_depth_cache_manifest_path(output)),
                "cache_key": cache_key,
            }
            return (
                depths,
                (metadata[0], metadata[1]),
                (int(depths.shape[2]), int(depths.shape[1])),
                metadata[3],
                cache_info,
            )

    selection = np.arange(frame_count, dtype=np.int64)
    decoded = decode_contiguous_video_frames_rgb(
        video,
        frame_indices=selection,
        max_resolution=max_resolution,
    )
    depths = infer_metric_vda_sequence(
        decoded,
        checkpoint=checkpoint,
        encoder=encoder,
        input_size=input_size,
        fp32=fp32,
    )
    cached = _write_depth_cache(
        output=output,
        depths=depths,
        cache_key=cache_key,
    )
    cache_info = {
        "requested": bool(reuse),
        "reused": False,
        "manifest": str(_depth_cache_manifest_path(output)),
        "cache_key": cache_key,
    }
    return (
        cached,
        decoded.source_size,
        decoded.inference_size,
        decoded.fps,
        cache_info,
    )


def _open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return capture


def _read_resized_rgb(
    capture: cv2.VideoCapture,
    *,
    size: tuple[int, int],
    frame_id: int,
) -> np.ndarray:
    ok, frame_bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Video decode stopped at frame {frame_id}")
    if (frame_bgr.shape[1], frame_bgr.shape[0]) != size:
        frame_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _indexed_png_paths(
    directory: Path | None,
    frame_indices: np.ndarray,
    *,
    label: str,
) -> list[Path]:
    """Resolve PNGs by numeric source-frame stem, including zero padding."""

    if directory is None:
        return []
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    by_index: dict[int, Path] = {}
    for path in directory.glob("*.png"):
        if not path.stem.isdigit():
            continue
        source_index = int(path.stem)
        if source_index in by_index:
            raise ValueError(
                f"{directory} has duplicate {label} PNGs for source frame "
                f"{source_index}"
            )
        by_index[source_index] = path
    missing = [
        int(source_index)
        for source_index in frame_indices
        if int(source_index) not in by_index
    ]
    if missing:
        preview = missing[:8]
        raise ValueError(
            f"{directory} has no {label} PNG for selected source frames "
            f"{preview}{'...' if len(missing) > len(preview) else ''}"
        )
    return [by_index[int(source_index)] for source_index in frame_indices]


def _mask_paths(
    mask_dir: Path | None,
    frame_indices: np.ndarray,
) -> list[Path]:
    if mask_dir is None:
        return []
    return _indexed_png_paths(
        mask_dir,
        frame_indices,
        label="mask",
    )


def _foreground_mask(
    *,
    original_rgb: np.ndarray,
    handless_rgb: np.ndarray,
    explicit_path: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = original_rgb.shape[:2]
    difference = np.max(
        np.abs(
            original_rgb.astype(np.int16)
            - handless_rgb.astype(np.int16)
        ),
        axis=2,
    ).astype(np.float32)
    if explicit_path is None:
        threshold = max(10.0, float(np.percentile(difference, 50)) + 7.0)
        foreground = difference >= threshold
        kernel = np.ones((3, 3), dtype=np.uint8)
        foreground = cv2.morphologyEx(
            foreground.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        ).astype(bool)
    else:
        raw = cv2.imread(str(explicit_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise RuntimeError(f"Failed to read mask: {explicit_path}")
        foreground = cv2.resize(
            raw,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    return foreground, difference


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _sample_depth_patch(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    u: float,
    v: float,
    radius: int = 2,
) -> float | None:
    height, width = depth.shape
    center_x = int(round(u))
    center_y = int(round(v))
    if not (0 <= center_x < width and 0 <= center_y < height):
        return None
    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)
    local_depth = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64)
    local_mask = mask[y0:y1, x0:x1]
    values = local_depth[
        local_mask & np.isfinite(local_depth) & (local_depth > 0.0)
    ]
    if values.size < 3:
        return None
    median = float(np.median(values))
    relative_mad = float(np.median(np.abs(values - median))) / median
    if not np.isfinite(relative_mad) or relative_mad > 0.15:
        return None
    return median


def _project_landmarks(
    points_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    z = points_camera[..., 2]
    pixels = np.full((*points_camera.shape[:2], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points_camera).all(axis=2) & (z > 1e-6)
    pixels[..., 0][valid] = (
        intrinsic[0, 0] * points_camera[..., 0][valid] / z[valid]
        + intrinsic[0, 2]
    )
    pixels[..., 1][valid] = (
        intrinsic[1, 1] * points_camera[..., 1][valid] / z[valid]
        + intrinsic[1, 2]
    )
    return pixels


def _da3_depth_paths(
    path: Path | None,
    frame_indices: np.ndarray,
) -> list[Path]:
    if path is None:
        return []
    return _indexed_png_paths(
        path,
        frame_indices,
        label="DA3 depth",
    )


def _seek_capture(
    capture: cv2.VideoCapture,
    *,
    start_frame: int,
    path: Path,
) -> None:
    if start_frame == 0:
        return
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if actual != start_frame:
        raise RuntimeError(
            f"Could not seek {path} to source frame {start_frame}; got {actual}"
        )


def _depth_bin_name(depth_m: np.ndarray) -> np.ndarray:
    edges = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0, np.inf])
    return np.digitize(depth_m, edges[1:-1], right=False)


def _analyze(
    *,
    original_video: Path,
    handless_video: Path,
    mask_dir: Path | None,
    original_depth: np.ndarray,
    handless_depth: np.ndarray,
    source_size: tuple[int, int],
    inference_size: tuple[int, int],
    intrinsic: np.ndarray,
    points_camera: np.ndarray,
    confidence: np.ndarray,
    labels: tuple[str, ...],
    frame_indices: np.ndarray | None = None,
    da3_depth_dir: Path | None = None,
    da3_scene_scale: float | None = None,
) -> dict[str, Any]:
    frame_count = int(original_depth.shape[0])
    selected = (
        np.arange(frame_count, dtype=np.int64)
        if frame_indices is None
        else validate_frame_indices(frame_indices)
    )
    if len(selected) != frame_count:
        raise ValueError("frame_indices and VDA frame counts disagree")
    if original_depth.shape != handless_depth.shape:
        raise ValueError("Original and handless VDA depth shapes disagree")
    if points_camera.shape[:2] != confidence.shape:
        raise ValueError("Landmark points/confidence shapes disagree")
    if points_camera.shape[0] != frame_count:
        raise ValueError("HDF5 and VDA frame counts disagree")

    scaled_k = _scaled_intrinsic(
        intrinsic,
        source_size=source_size,
        target_size=inference_size,
    )
    pixels = _project_landmarks(points_camera, scaled_k)
    explicit_masks = _mask_paths(mask_dir, selected)
    da3_paths = _da3_depth_paths(da3_depth_dir, selected)
    if bool(da3_paths) != (da3_scene_scale is not None):
        raise ValueError(
            "DA3 depth directory and --da3-scene-scale must be supplied together"
        )
    if da3_scene_scale is not None and (
        not np.isfinite(da3_scene_scale) or da3_scene_scale <= 0.0
    ):
        raise ValueError("--da3-scene-scale must be positive")

    original_capture = _open_video(original_video)
    handless_capture = _open_video(handless_video)
    _seek_capture(
        original_capture,
        start_frame=int(selected[0]),
        path=original_video,
    )
    _seek_capture(
        handless_capture,
        start_frame=int(selected[0]),
        path=handless_video,
    )
    hand_scale_frames: list[np.ndarray] = []
    hand_root_scale_frames: list[np.ndarray] = []
    hand_depth_frames: list[np.ndarray] = []
    root_labels = tuple(label for label in labels if label.endswith("Hand"))
    root_scale_frames_by_label: dict[str, list[np.ndarray]] = {
        label: [] for label in root_labels
    }
    transfer_frames: list[np.ndarray] = []
    handless_background_depth_frames: list[np.ndarray] = []
    da3_scale_frames: list[np.ndarray] = []
    mask_coverages: list[float] = []
    unchanged_coverages: list[float] = []
    da3_bins: dict[str, list[np.ndarray]] = {
        "0.0-0.5m": [],
        "0.5-1.0m": [],
        "1.0-2.0m": [],
        "2.0-4.0m": [],
        "4.0m+": [],
    }
    bin_names = tuple(da3_bins)
    try:
        for frame_id in range(frame_count):
            original_rgb = _read_resized_rgb(
                original_capture,
                size=inference_size,
                frame_id=int(selected[frame_id]),
            )
            handless_rgb = _read_resized_rgb(
                handless_capture,
                size=inference_size,
                frame_id=int(selected[frame_id]),
            )
            foreground, rgb_difference = _foreground_mask(
                original_rgb=original_rgb,
                handless_rgb=handless_rgb,
                explicit_path=(
                    explicit_masks[frame_id] if explicit_masks else None
                ),
            )
            mask_coverages.append(float(np.mean(foreground)))
            distance_inside = cv2.distanceTransform(
                foreground.astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            anchor_mask = distance_inside >= 1.0
            dilation_radius = max(
                3,
                int(round(30.0 * inference_size[0] / source_size[0])),
            )
            excluded = _dilate(foreground, dilation_radius)
            background = (
                ~excluded
                & (rgb_difference <= 12.0)
                & np.isfinite(original_depth[frame_id])
                & np.isfinite(handless_depth[frame_id])
                & (original_depth[frame_id] > 0.0)
                & (handless_depth[frame_id] > 0.0)
            )
            unchanged_coverages.append(float(np.mean(background)))
            sampled_background = background[::4, ::4]
            transfer_values = (
                np.asarray(original_depth[frame_id][::4, ::4], dtype=np.float64)[
                    sampled_background
                ]
                / np.asarray(
                    handless_depth[frame_id][::4, ::4],
                    dtype=np.float64,
                )[sampled_background]
            )
            transfer_frames.append(
                transfer_values[
                    np.isfinite(transfer_values) & (transfer_values > 0.0)
                ]
            )
            handless_background_values = np.asarray(
                handless_depth[frame_id][::4, ::4],
                dtype=np.float64,
            )[sampled_background]
            handless_background_depth_frames.append(
                handless_background_values[
                    np.isfinite(handless_background_values)
                    & (handless_background_values > 0.0)
                ]
            )

            frame_scales: list[float] = []
            frame_root_scales: list[float] = []
            frame_hand_depths: list[float] = []
            frame_root_scales_by_label: dict[str, list[float]] = {
                label: [] for label in root_labels
            }
            for landmark_id, label in enumerate(labels):
                if (
                    not np.isfinite(confidence[frame_id, landmark_id])
                    or confidence[frame_id, landmark_id] < 0.5
                ):
                    continue
                point = points_camera[frame_id, landmark_id]
                pixel = pixels[frame_id, landmark_id]
                if (
                    not np.isfinite(point).all()
                    or point[2] <= 0.0
                    or not np.isfinite(pixel).all()
                ):
                    continue
                depth_value = _sample_depth_patch(
                    np.asarray(original_depth[frame_id]),
                    anchor_mask,
                    u=float(pixel[0]),
                    v=float(pixel[1]),
                )
                if depth_value is None:
                    continue
                scale = float(point[2] / depth_value)
                if not np.isfinite(scale) or scale <= 0.0:
                    continue
                frame_scales.append(scale)
                frame_hand_depths.append(float(point[2]))
                if label.endswith("Hand"):
                    frame_root_scales.append(scale)
                    frame_root_scales_by_label[label].append(scale)
            hand_scale_frames.append(np.asarray(frame_scales, dtype=np.float64))
            hand_root_scale_frames.append(
                np.asarray(frame_root_scales, dtype=np.float64)
            )
            hand_depth_frames.append(
                np.asarray(frame_hand_depths, dtype=np.float64)
            )
            for label in root_labels:
                root_scale_frames_by_label[label].append(
                    np.asarray(
                        frame_root_scales_by_label[label],
                        dtype=np.float64,
                    )
                )

            if da3_paths:
                da3_mm = cv2.imread(
                    str(da3_paths[frame_id]),
                    cv2.IMREAD_UNCHANGED,
                )
                if da3_mm is None or da3_mm.ndim != 2:
                    raise RuntimeError(
                        f"Failed to read DA3 depth: {da3_paths[frame_id]}"
                    )
                da3_size = (int(da3_mm.shape[1]), int(da3_mm.shape[0]))
                vda_da3 = cv2.resize(
                    np.asarray(handless_depth[frame_id], dtype=np.float32),
                    da3_size,
                    interpolation=cv2.INTER_NEAREST,
                )
                background_da3 = cv2.resize(
                    background.astype(np.uint8),
                    da3_size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                da3_source_m = (
                    da3_mm.astype(np.float64)
                    / 1000.0
                    / float(da3_scene_scale)
                )
                valid = (
                    background_da3
                    & (da3_mm > 0)
                    & np.isfinite(vda_da3)
                    & (vda_da3 > 0.0)
                )
                valid[1:, :] &= np.abs(
                    da3_source_m[1:, :] - da3_source_m[:-1, :]
                ) < 0.15
                valid[:, 1:] &= np.abs(
                    da3_source_m[:, 1:] - da3_source_m[:, :-1]
                ) < 0.15
                sampled = valid[::2, ::2]
                ratios = (
                    da3_source_m[::2, ::2][sampled]
                    / vda_da3[::2, ::2][sampled]
                )
                ratios = ratios[
                    np.isfinite(ratios)
                    & (ratios > 0.0)
                    & (ratios < 5.0)
                ]
                da3_scale_frames.append(ratios)

                sampled_depth = da3_source_m[::2, ::2][sampled]
                sampled_vda = vda_da3[::2, ::2][sampled]
                bin_ids = _depth_bin_name(sampled_depth)
                bin_ratio = sampled_depth / sampled_vda
                for bin_id, name in enumerate(bin_names):
                    bin_selected = (
                        (bin_ids == bin_id)
                        & np.isfinite(bin_ratio)
                        & (bin_ratio > 0.0)
                        & (bin_ratio < 5.0)
                    )
                    if np.any(bin_selected):
                        da3_bins[name].append(bin_ratio[bin_selected])
    finally:
        original_capture.release()
        handless_capture.release()

    hand_summary, hand_frame_medians = _frame_equal_summary(hand_scale_frames)
    root_summary, root_frame_medians = _frame_equal_summary(
        hand_root_scale_frames
    )
    root_summaries_by_label = {
        label: _frame_equal_summary(frame_values)[0]
        for label, frame_values in root_scale_frames_by_label.items()
    }
    hand_depth_summary, _ = _frame_equal_summary(hand_depth_frames)
    transfer_summary, transfer_frame_medians = _frame_equal_summary(
        transfer_frames
    )
    raw_background_summary, _ = _frame_equal_summary(
        handless_background_depth_frames
    )
    minimum_anchor_frames = _minimum_anchor_frame_count(len(selected))
    root_is_supported = _anchor_summary_is_supported(
        root_summary,
        selected_frame_count=len(selected),
    )
    primary_hand = root_summary if root_is_supported else hand_summary
    primary_hand_frame_medians = (
        root_frame_medians if root_is_supported else hand_frame_medians
    )
    primary_hand_source = (
        "left/right Hand roots" if root_is_supported else "all hand landmarks"
    )
    original_scale = primary_hand["p50"]
    transfer = transfer_summary["p50"]
    handless_scale = (
        None
        if original_scale is None or transfer is None
        else float(original_scale) * float(transfer)
    )
    raw_background_median = raw_background_summary["p50"]
    estimated_background_depth = (
        None
        if handless_scale is None or raw_background_median is None
        else float(handless_scale) * float(raw_background_median)
    )

    result: dict[str, Any] = {
        "selected_frame_indices": selected,
        "source_intrinsic": intrinsic,
        "inference_intrinsic": scaled_k,
        "landmark_labels": labels,
        "mask_source": "explicit PNG masks" if explicit_masks else "RGB difference",
        "mask_coverage": _distribution(
            np.asarray(mask_coverages),
            positive_only=False,
        ),
        "unchanged_background_coverage": _distribution(
            np.asarray(unchanged_coverages),
            positive_only=False,
        ),
        "original_hand_anchor": hand_summary,
        "original_hand_root_anchor": root_summary,
        "original_hand_root_anchor_by_side": root_summaries_by_label,
        "selected_original_anchor": primary_hand,
        "selected_original_anchor_source": primary_hand_source,
        "anchor_selection_requirements": {
            "minimum_supporting_frames": minimum_anchor_frames,
            "minimum_raw_sample_count": 30,
            "maximum_log_mad": 0.15,
        },
        "sampled_hdf5_hand_camera_z_m": hand_depth_summary,
        "original_to_handless_transfer": transfer_summary,
        "handless_background_raw_vda_depth_m": raw_background_summary,
        "estimated_metric_background_depth_m": estimated_background_depth,
        "recommended_handless_metric_scale": handless_scale,
        "recommended_scale_frame_bootstrap": _bootstrap_product(
            primary_hand_frame_medians,
            transfer_frame_medians,
        ),
        "recommended_scale_formula": (
            "median_frame(z_hdf5 / d_vda_original) * "
            "median_frame(d_vda_original / d_vda_handless)"
        ),
        "temporal_diagnostics": {
            "hand_anchor_frame_medians": _distribution(hand_frame_medians),
            "hand_root_frame_medians": _distribution(root_frame_medians),
            "transfer_frame_medians": _distribution(transfer_frame_medians),
        },
    }
    if da3_paths:
        da3_summary, da3_frame_medians = _frame_equal_summary(
            da3_scale_frames
        )
        da3_scale = da3_summary["p50"]
        agreement = (
            None
            if handless_scale is None or da3_scale is None
            else float(handless_scale) / float(da3_scale)
        )
        result["da3_compatibility"] = {
            "scene_scale_removed": float(da3_scene_scale),
            "vda_to_da3_source_metric_scale": da3_summary,
            "hand_anchor_scale_over_da3_scale": agreement,
            "absolute_log_disagreement": (
                None if agreement is None else abs(math.log(agreement))
            ),
            "frame_medians": _distribution(da3_frame_medians),
            "scale_by_da3_source_depth": {
                name: _distribution(
                    np.concatenate(values)
                    if values
                    else np.empty(0, dtype=np.float64)
                )
                for name, values in da3_bins.items()
            },
        }
    return result


def estimate_handless_metric_scale(
    *,
    original_video: Path,
    handless_video: Path,
    source_hdf5: Path,
    frame_indices: np.ndarray,
    original_depth_raw_m: np.ndarray,
    handless_depth_raw_m: np.ndarray,
    source_size: tuple[int, int],
    inference_size: tuple[int, int],
    mask_dir: Path | None = None,
) -> dict[str, Any]:
    """Estimate raw handless-VDA depth to source-camera meters.

    This public CPU stage is shared by the preparation entry points. Both raw
    depth tensors must come from the same VDA configuration and the same
    contiguous source-frame selection. The returned scale deliberately does
    not include ``scene_scale``.
    """

    selected = validate_frame_indices(frame_indices)
    original_video = original_video.expanduser().resolve()
    handless_video = handless_video.expanduser().resolve()
    source_hdf5 = source_hdf5.expanduser().resolve()
    for path in (original_video, handless_video, source_hdf5):
        if not path.is_file():
            raise FileNotFoundError(path)
    if mask_dir is not None:
        mask_dir = mask_dir.expanduser().resolve()

    (
        intrinsic,
        _camera_to_world,
        points_camera,
        confidence,
        labels,
        max_baseline,
        hdf5_frame_count,
    ) = _load_hdf5_geometry(
        source_hdf5,
        frame_indices=selected,
    )
    _validate_source_frame_provenance(
        original_video=original_video,
        handless_video_frame_count=_video_metadata(handless_video)[2],
        source_hdf5=source_hdf5,
        hdf5_frame_count=hdf5_frame_count,
        selected_frame_indices=selected,
    )
    analysis = _analyze(
        original_video=original_video,
        handless_video=handless_video,
        mask_dir=mask_dir,
        original_depth=np.asarray(original_depth_raw_m),
        handless_depth=np.asarray(handless_depth_raw_m),
        source_size=source_size,
        inference_size=inference_size,
        intrinsic=intrinsic,
        points_camera=points_camera,
        confidence=confidence,
        labels=labels,
        frame_indices=selected,
        da3_depth_dir=None,
        da3_scene_scale=None,
    )
    decision = _quality_decision(
        analysis=analysis,
        max_camera_baseline_m=max_baseline,
    )
    scale = analysis["recommended_handless_metric_scale"]
    resolved_scale = (
        None
        if scale is None or not np.isfinite(scale) or float(scale) <= 0.0
        else float(scale)
    )
    return _json_ready({
        "schema_version": 1,
        "method": (
            "source-HDF5 hand depth on original VDA times unchanged-"
            "background original-to-handless transfer"
        ),
        "resolved_vda_metric_scale": resolved_scale,
        "scale_excludes_scene_scale": True,
        "source_video": str(original_video),
        "source_video_sha256": file_sha256(original_video),
        "handless_video": str(handless_video),
        "handless_video_sha256": file_sha256(handless_video),
        "source_hdf5": str(source_hdf5),
        "source_hdf5_sha256": file_sha256(source_hdf5),
        "analysis": analysis,
        "decision": decision,
    })


def infer_and_estimate_handless_metric_scale(
    *,
    original_video: Path,
    handless_video: Path,
    source_hdf5: Path,
    frame_indices: np.ndarray,
    handless_decoded: DecodedVideoFrames,
    handless_depth_raw_m: np.ndarray,
    checkpoint: Path,
    encoder: str,
    input_size: int,
    max_resolution: int,
    fp32: bool,
    mask_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the matching original-video VDA pass, then estimate one scale."""

    selected = validate_frame_indices(frame_indices)
    if not np.array_equal(selected, handless_decoded.frame_indices):
        raise ValueError(
            "Handless VDA selection does not match calibration frame_indices"
        )
    with h5py.File(source_hdf5, "r") as handle:
        hdf5_frame_count = int(handle["transforms/camera"].shape[0])
    _validate_source_frame_provenance(
        original_video=original_video,
        handless_video_frame_count=handless_decoded.video_frame_count,
        source_hdf5=source_hdf5,
        hdf5_frame_count=hdf5_frame_count,
        selected_frame_indices=selected,
    )
    original_decoded = decode_contiguous_video_frames_rgb(
        original_video,
        frame_indices=selected,
        max_resolution=max_resolution,
    )
    if (
        original_decoded.source_size != handless_decoded.source_size
        or original_decoded.inference_size != handless_decoded.inference_size
        or original_decoded.frame_count != handless_decoded.frame_count
        or not np.isclose(
            original_decoded.fps,
            handless_decoded.fps,
            atol=1e-3,
        )
    ):
        raise ValueError(
            "Original and handless videos must have matching frame selection, "
            "selected frame count, FPS, source size, and VDA inference size"
        )
    original_depth_raw_m = infer_metric_vda_sequence(
        original_decoded,
        checkpoint=checkpoint,
        encoder=encoder,
        input_size=input_size,
        fp32=fp32,
    )
    original_source_size = original_decoded.source_size
    original_inference_size = original_decoded.inference_size
    del original_decoded
    try:
        return estimate_handless_metric_scale(
            original_video=original_video,
            handless_video=handless_video,
            source_hdf5=source_hdf5,
            frame_indices=selected,
            original_depth_raw_m=original_depth_raw_m,
            handless_depth_raw_m=handless_depth_raw_m,
            source_size=original_source_size,
            inference_size=original_inference_size,
            mask_dir=mask_dir,
        )
    finally:
        del original_depth_raw_m


def _quality_decision(
    *,
    analysis: dict[str, Any],
    max_camera_baseline_m: float,
) -> dict[str, Any]:
    hand = analysis["selected_original_anchor"]
    transfer = analysis["original_to_handless_transfer"]
    scale = analysis["recommended_handless_metric_scale"]
    selected_frame_count = len(analysis["selected_frame_indices"])
    minimum_anchor_frames = _minimum_anchor_frame_count(selected_frame_count)
    minimum_transfer_frames = max(
        10,
        int(math.ceil(0.5 * selected_frame_count)),
    )
    reasons: list[str] = []
    if scale is None:
        reasons.append("no valid hand-anchor scale")
    elif not 0.1 <= float(scale) <= 2.0:
        reasons.append("estimated VDA metric scale lies outside [0.1, 2.0]")
    if int(hand["supporting_frames"]) < minimum_anchor_frames:
        reasons.append(
            "too few hand-anchor frames: "
            f"{hand['supporting_frames']} < {minimum_anchor_frames}"
        )
    if int(hand["raw_sample_count"]) < 30:
        reasons.append("fewer than 30 hand-anchor samples")
    if hand["log_mad"] is None or float(hand["log_mad"]) > 0.15:
        reasons.append("hand-anchor log-MAD exceeds 0.15")
    side_scales = [
        float(summary["p50"])
        for summary in analysis["original_hand_root_anchor_by_side"].values()
        if summary["p50"] is not None
        and int(summary["supporting_frames"]) >= 10
    ]
    if len(side_scales) >= 2 and max(side_scales) / min(side_scales) > 1.25:
        reasons.append("left/right hand-root scales disagree by over 25%")
    if int(transfer["supporting_frames"]) < minimum_transfer_frames:
        reasons.append(
            "too few unchanged-background transfer frames: "
            f"{transfer['supporting_frames']} < {minimum_transfer_frames}"
        )
    if transfer["log_mad"] is None or float(transfer["log_mad"]) > 0.08:
        reasons.append("content-transfer log-MAD exceeds 0.08")
    unchanged_coverage = analysis["unchanged_background_coverage"]["p50"]
    if unchanged_coverage is None or float(unchanged_coverage) < 0.05:
        reasons.append("median unchanged-background coverage is below 5%")
    bootstrap = analysis["recommended_scale_frame_bootstrap"]
    bootstrap_half_width = bootstrap["relative_95pct_half_width"]
    if (
        bootstrap_half_width is not None
        and float(bootstrap_half_width) > 0.15
    ):
        reasons.append("scale bootstrap relative 95% half-width exceeds 15%")

    da3 = analysis.get("da3_compatibility")
    agreement_status = "not_checked"
    if da3 is not None:
        ratio = da3["hand_anchor_scale_over_da3_scale"]
        if ratio is None:
            agreement_status = "unavailable"
            reasons.append("DA3 compatibility scale unavailable")
        elif abs(math.log(float(ratio))) <= math.log(1.2):
            agreement_status = "within_20_percent"
        else:
            agreement_status = "conflict_over_20_percent"
            reasons.append(
                "hand-anchor and background DA3 scales disagree by over 20%"
            )

    source_depth = analysis["estimated_metric_background_depth_m"]
    approximate_angle = (
        None
        if source_depth is None or source_depth <= 0.0
        else math.degrees(math.atan(max_camera_baseline_m / source_depth))
    )
    if approximate_angle is None or approximate_angle < 1.0:
        multiview = "rejected_low_parallax"
    elif approximate_angle < 2.0:
        multiview = "marginal"
    else:
        multiview = "potentially_observable"

    if not reasons:
        status = "validated"
    elif (
        len(reasons) == 1
        and reasons[0]
        == "hand-anchor and background DA3 scales disagree by over 20%"
    ):
        status = "rejected_single_global_scale"
    else:
        status = "needs_fallback_or_review"
    return {
        "status": status,
        "reasons": reasons,
        "support_requirements": {
            "selected_frame_count": selected_frame_count,
            "minimum_hand_anchor_frames": minimum_anchor_frames,
            "minimum_background_transfer_frames": minimum_transfer_frames,
        },
        "da3_agreement": agreement_status,
        "multiview_observability": {
            "status": multiview,
            "max_camera_baseline_m": max_camera_baseline_m,
            "rough_angle_from_anchor_depth_deg": approximate_angle,
            "note": (
                "This is only an observability gate; it is not a "
                "triangulation scale estimate."
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run raw metric VDA on original/handless video and validate one "
            "episode-level metric scale using HDF5 hand anchors."
        )
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--original-video", type=Path, required=True)
    parser.add_argument("--handless-video", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--da3-depth-dir", type=Path)
    parser.add_argument("--da3-scene-scale", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--encoder", default="vitl", choices=("vits", "vitb", "vitl"))
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-resolution", type=int, default=1280)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--reuse-depth-cache", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    original_video = args.original_video.expanduser().resolve()
    handless_video = args.handless_video.expanduser().resolve()
    hdf5_path = args.hdf5.expanduser().resolve()
    mask_dir = (
        None if args.mask_dir is None else args.mask_dir.expanduser().resolve()
    )
    da3_depth_dir = (
        None
        if args.da3_depth_dir is None
        else args.da3_depth_dir.expanduser().resolve()
    )
    output = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    for path in (original_video, handless_video, hdf5_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if mask_dir is not None and not mask_dir.is_dir():
        raise NotADirectoryError(mask_dir)
    if da3_depth_dir is not None and not da3_depth_dir.is_dir():
        raise NotADirectoryError(da3_depth_dir)
    output.mkdir(parents=True, exist_ok=True)

    (
        intrinsic,
        _camera_to_world,
        points_camera,
        confidence,
        labels,
        max_baseline,
        hdf5_frame_count,
    ) = _load_hdf5_geometry(hdf5_path)
    frame_count = int(points_camera.shape[0])
    if frame_count != hdf5_frame_count:
        raise RuntimeError("Standalone validation must load the full HDF5")
    selected_frame_indices = np.arange(frame_count, dtype=np.int64)
    _validate_source_frame_provenance(
        original_video=original_video,
        handless_video_frame_count=_video_metadata(handless_video)[2],
        source_hdf5=hdf5_path,
        hdf5_frame_count=hdf5_frame_count,
        selected_frame_indices=selected_frame_indices,
    )
    checkpoint_sha256 = file_sha256(checkpoint)
    (
        original_depth,
        source_size,
        inference_size,
        original_fps,
        original_cache,
    ) = _infer_or_load(
        video=original_video,
        output=output / "original_vda_raw_m.npy",
        frame_count=frame_count,
        max_resolution=args.max_resolution,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        encoder=args.encoder,
        input_size=args.input_size,
        fp32=args.fp32,
        reuse=args.reuse_depth_cache,
    )
    (
        handless_depth,
        handless_source_size,
        handless_inference_size,
        handless_fps,
        handless_cache,
    ) = _infer_or_load(
        video=handless_video,
        output=output / "handless_vda_raw_m.npy",
        frame_count=frame_count,
        max_resolution=args.max_resolution,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        encoder=args.encoder,
        input_size=args.input_size,
        fp32=args.fp32,
        reuse=args.reuse_depth_cache,
    )
    if source_size != handless_source_size:
        raise ValueError("Original and handless source sizes disagree")
    if inference_size != handless_inference_size:
        raise ValueError("Original and handless inference sizes disagree")
    if abs(original_fps - handless_fps) > 0.01:
        raise ValueError("Original and handless FPS values disagree")

    analysis = _analyze(
        original_video=original_video,
        handless_video=handless_video,
        mask_dir=mask_dir,
        original_depth=original_depth,
        handless_depth=handless_depth,
        source_size=source_size,
        inference_size=inference_size,
        intrinsic=intrinsic,
        points_camera=points_camera,
        confidence=confidence,
        labels=labels,
        da3_depth_dir=da3_depth_dir,
        da3_scene_scale=args.da3_scene_scale,
    )
    decision = _quality_decision(
        analysis=analysis,
        max_camera_baseline_m=max_baseline,
    )
    report = {
        "schema_version": 1,
        "name": args.name,
        "inputs": {
            "original_video": original_video,
            "original_video_sha256": original_cache[
                "cache_key"
            ]["video_sha256"],
            "handless_video": handless_video,
            "handless_video_sha256": handless_cache[
                "cache_key"
            ]["video_sha256"],
            "hdf5": hdf5_path,
            "hdf5_sha256": file_sha256(hdf5_path),
            "mask_dir": mask_dir,
            "da3_depth_dir": da3_depth_dir,
            "da3_scene_scale": args.da3_scene_scale,
            "frame_count": frame_count,
            "fps": original_fps,
            "source_size": source_size,
            "inference_size": inference_size,
        },
        "vda": {
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "encoder": args.encoder,
            "input_size": args.input_size,
            "max_resolution": args.max_resolution,
            "fp32": bool(args.fp32),
            "depth_cache": {
                "requested": bool(args.reuse_depth_cache),
                "original": original_cache,
                "handless": handless_cache,
                "all_reused": bool(
                    original_cache["reused"]
                    and handless_cache["reused"]
                ),
            },
        },
        "analysis": analysis,
        "decision": decision,
    }
    report_path = output / "scale_validation_report.json"
    report_path.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_ready({
        "name": args.name,
        "recommended_handless_metric_scale": (
            analysis["recommended_handless_metric_scale"]
        ),
        "da3_scale": (
            analysis.get("da3_compatibility", {})
            .get("vda_to_da3_source_metric_scale", {})
            .get("p50")
        ),
        "status": decision["status"],
        "reasons": decision["reasons"],
        "report": report_path,
    }), indent=2))


if __name__ == "__main__":
    main()
