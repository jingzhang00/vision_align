"""Estimate and validate an episode-level VDA metric scale.

This is intentionally a validation tool rather than part of the reconstruction
pipeline.  It keeps the two scale conversions separate:

* raw VDA depth -> source-camera meters, estimated from the original video;
* source-camera meters -> robot/world scale, owned by the caller.

Hand anchors are transferred through unchanged pixels. Datasets with a metric
background map instead calibrate the handless prediction directly.
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
import pandas as pd

from repaint.hand_trajectory import HAND_LANDMARK_NAMES
from repaint.vda_depth import (
    DEFAULT_CHECKPOINT,
    DecodedVideoFrames,
    decode_contiguous_video_frames_rgb,
    file_sha256,
    infer_metric_vda_sequence,
    validate_frame_indices,
)
from repaint.vda_pipeline_common import probe_video


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SEMIDENSE_DISTANCE_STD_MAX_M = 0.05
SEMIDENSE_INVERSE_DISTANCE_STD_MAX = 0.01
SEMIDENSE_SAMPLE_STRIDE = 32
SEMIDENSE_MAX_POINTS = 50_000


def _video_metadata(path: Path) -> tuple[int, int, int, float]:
    metadata = probe_video(path)
    return metadata.width, metadata.height, metadata.frame_count, metadata.fps


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


def _declared_hdf5_source_frame_start(path: Path) -> int | None:
    values: list[int] = []
    attribute_names = (
        "source_frame_start",
        "source_video_frame_start",
        "source_clip_frame_start",
    )

    def collect_attributes(attributes: h5py.AttributeManager) -> None:
        for name in attribute_names:
            if name in attributes:
                values.append(int(attributes[name]))

    with h5py.File(path, "r") as handle:
        collect_attributes(handle.attrs)

        def collect(_name: str, item: h5py.Group | h5py.Dataset) -> None:
            collect_attributes(item.attrs)

        handle.visititems(collect)
    unique = set(values)
    if len(unique) > 1:
        raise ValueError(
            f"{path} declares multiple source-frame starts: {sorted(unique)}"
        )
    return None if not unique else unique.pop()


def _video_frame_is_decodable(path: Path, frame_index: int) -> bool:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if actual != frame_index:
            return False
        ok, _frame = capture.read()
        return bool(ok)
    finally:
        capture.release()


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
    declared_source = _declared_hdf5_source_video(source_hdf5)
    if declared_source is not None and declared_source != original_video.resolve():
        raise ValueError(
            "Source HDF5 declares a different source_video: "
            f"{declared_source} != {original_video.resolve()}"
        )
    if (
        declared_source is None
        and source_hdf5.stem != original_video.stem
    ):
        raise ValueError(
            "Source HDF5 does not identify the original video: add a "
            "source_video attribute or use matching HDF5/video stems "
            f"({source_hdf5.stem!r} != {original_video.stem!r})"
        )
    if original_frame_count == hdf5_frame_count:
        return

    if original_frame_count < hdf5_frame_count or declared_source is None:
        raise ValueError(
            "Original video and source HDF5 total frame counts disagree "
            f"({original_frame_count} != {hdf5_frame_count}); this is only "
            "allowed when the HDF5/handless sequence is a covered frame-zero "
            "prefix and the HDF5 declares source_video provenance"
        )
    # Some containers overstate their frame count (Nymeria c024 declares 460
    # but decodes exactly 450). If a real extra frame is decodable, require an
    # explicit frame-zero mapping rather than inferring a prefix from lengths.
    if (
        _video_frame_is_decodable(original_video, hdf5_frame_count)
        and _declared_hdf5_source_frame_start(source_hdf5) != 0
    ):
        raise ValueError(
            "Original video has decodable frames beyond the source HDF5; "
            "the HDF5 must explicitly declare source_frame_start=0"
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
    return _frame_equal_summary_from_medians(
        frame_medians,
        raw_sample_count=int(
            sum(np.asarray(values).size for values in frame_values)
        ),
    )


def _frame_equal_summary_from_medians(
    frame_medians: np.ndarray,
    *,
    raw_sample_count: int,
) -> tuple[dict[str, Any], np.ndarray]:
    medians = np.asarray(frame_medians, dtype=np.float64)
    valid_frame_medians = medians[
        np.isfinite(medians) & (medians > 0.0)
    ]
    summary = _distribution(valid_frame_medians)
    summary["aggregation"] = "median within frame, then median across frames"
    summary["supporting_frames"] = int(valid_frame_medians.size)
    summary["raw_sample_count"] = int(raw_sample_count)
    return summary, medians


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


def _maximum_camera_baseline(camera_to_world: np.ndarray) -> float:
    translations = camera_to_world[:, :3, 3]
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
    return float(math.sqrt(max(0.0, maximum_squared)))


def _load_hdf5_camera_geometry(
    path: Path,
    *,
    frame_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
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
    return (
        intrinsic,
        camera_to_world,
        _maximum_camera_baseline(camera_to_world),
        frame_count,
    )


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
    selected = None if frame_indices is None else validate_frame_indices(
        frame_indices
    )
    intrinsic, camera_to_world, max_baseline, frame_count = (
        _load_hdf5_camera_geometry(path, frame_indices=selected)
    )
    with h5py.File(path, "r") as handle:
        selection = (
            slice(None)
            if selected is None
            else slice(int(selected[0]), int(selected[-1]) + 1)
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
    transfer_frame_medians = np.full(frame_count, np.nan, dtype=np.float64)
    transfer_raw_sample_count = 0
    handless_background_frame_medians = np.full(
        frame_count,
        np.nan,
        dtype=np.float64,
    )
    handless_background_raw_sample_count = 0
    mask_coverages: list[float] = []
    unchanged_coverages: list[float] = []
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
            transfer_values = transfer_values[
                np.isfinite(transfer_values) & (transfer_values > 0.0)
            ]
            transfer_raw_sample_count += int(transfer_values.size)
            if transfer_values.size:
                transfer_frame_medians[frame_id] = np.median(transfer_values)
            handless_background_values = np.asarray(
                handless_depth[frame_id][::4, ::4],
                dtype=np.float64,
            )[sampled_background]
            handless_background_values = handless_background_values[
                np.isfinite(handless_background_values)
                & (handless_background_values > 0.0)
            ]
            handless_background_raw_sample_count += int(
                handless_background_values.size
            )
            if handless_background_values.size:
                handless_background_frame_medians[frame_id] = np.median(
                    handless_background_values
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
    transfer_summary, transfer_frame_medians = (
        _frame_equal_summary_from_medians(
            transfer_frame_medians,
            raw_sample_count=transfer_raw_sample_count,
        )
    )
    raw_background_summary, _ = _frame_equal_summary_from_medians(
        handless_background_frame_medians,
        raw_sample_count=handless_background_raw_sample_count,
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
    return result


def _load_semidense_points(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    columns = (
        "px_world",
        "py_world",
        "pz_world",
        "inv_dist_std",
        "dist_std",
    )
    sampled: list[np.ndarray] = []
    row_count = 0
    quality_count = 0
    for chunk in pd.read_csv(
        path,
        usecols=list(columns),
        chunksize=250_000,
    ):
        row_count += len(chunk)
        values = chunk.to_numpy(dtype=np.float64)
        quality = (
            np.isfinite(values).all(axis=1)
            & (values[:, 3] <= SEMIDENSE_INVERSE_DISTANCE_STD_MAX)
            & (values[:, 4] <= SEMIDENSE_DISTANCE_STD_MAX_M)
        )
        quality_count += int(np.count_nonzero(quality))
        keep = quality & (
            chunk.index.to_numpy(dtype=np.int64) % SEMIDENSE_SAMPLE_STRIDE == 0
        )
        if np.any(keep):
            sampled.append(values[keep, :3])
    if not sampled:
        raise ValueError(f"No reliable semidense points in {path}")
    points = np.concatenate(sampled)
    if len(points) > SEMIDENSE_MAX_POINTS:
        points = points[
            np.linspace(
                0,
                len(points) - 1,
                SEMIDENSE_MAX_POINTS,
                dtype=np.int64,
            )
        ]
    return points, {
        "source_row_count": row_count,
        "quality_row_count": quality_count,
        "sampled_point_count": len(points),
        "sampling_stride": SEMIDENSE_SAMPLE_STRIDE,
        "maximum_sampled_points": SEMIDENSE_MAX_POINTS,
        "maximum_distance_std_m": SEMIDENSE_DISTANCE_STD_MAX_M,
        "maximum_inverse_distance_std": (
            SEMIDENSE_INVERSE_DISTANCE_STD_MAX
        ),
    }


def _semidense_frame_ratios(
    *,
    points_world: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    depth: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    points_camera = (
        points_world @ world_to_camera[:3, :3].T
        + world_to_camera[:3, 3]
    )
    z = points_camera[:, 2]
    height, width = depth.shape
    projectable = (
        np.isfinite(points_camera).all(axis=1)
        & (z > 0.1)
        & (z < 20.0)
    )
    points_camera = points_camera[projectable]
    z = z[projectable]
    if not len(z):
        return np.empty(0), np.empty(0), 0
    u = intrinsic[0, 0] * points_camera[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * points_camera[:, 1] / z + intrinsic[1, 2]
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    visible = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    x = x[visible]
    y = y[visible]
    z = z[visible]
    if not len(z):
        return np.empty(0), np.empty(0), 0

    # Keep the closest map point at each pixel so hidden surfaces cannot vote.
    pixel = y * width + x
    order = np.lexsort((z, pixel))
    pixel = pixel[order]
    first = np.r_[True, pixel[1:] != pixel[:-1]]
    x = x[order][first]
    y = y[order][first]
    z = z[order][first]
    projected_count = len(z)
    predicted = np.asarray(depth[y, x], dtype=np.float64)
    usable = (
        background[y, x]
        & np.isfinite(predicted)
        & (predicted > 0.0)
    )
    ratios = z[usable] / predicted[usable]
    valid = np.isfinite(ratios) & (ratios > 0.0)
    return ratios[valid], z[usable][valid], projected_count


def _analyze_semidense(
    *,
    original_video: Path,
    handless_video: Path,
    mask_dir: Path | None,
    handless_depth: np.ndarray,
    source_size: tuple[int, int],
    inference_size: tuple[int, int],
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    points_world: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, Any]:
    selected = validate_frame_indices(frame_indices)
    frame_count = len(selected)
    if handless_depth.shape[0] != frame_count:
        raise ValueError("frame_indices and VDA frame counts disagree")
    if camera_to_world.shape != (frame_count, 4, 4):
        raise ValueError("Camera and VDA frame counts disagree")
    scaled_k = _scaled_intrinsic(
        intrinsic,
        source_size=source_size,
        target_size=inference_size,
    )
    explicit_masks = _mask_paths(mask_dir, selected)
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
    world_to_camera = np.linalg.inv(camera_to_world)
    scale_frame_medians = np.full(frame_count, np.nan, dtype=np.float64)
    scale_frame_log_mads = np.full(frame_count, np.nan, dtype=np.float64)
    depth_frame_medians = np.full(frame_count, np.nan, dtype=np.float64)
    background_depth_frame_medians = np.full(
        frame_count,
        np.nan,
        dtype=np.float64,
    )
    raw_sample_count = 0
    background_raw_sample_count = 0
    projected_counts = np.zeros(frame_count, dtype=np.int64)
    usable_counts = np.zeros(frame_count, dtype=np.int64)
    unchanged_coverages = np.zeros(frame_count, dtype=np.float64)
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
            dilation_radius = max(
                3,
                int(round(30.0 * inference_size[0] / source_size[0])),
            )
            depth = np.asarray(handless_depth[frame_id], dtype=np.float64)
            background = (
                ~_dilate(foreground, dilation_radius)
                & (rgb_difference <= 12.0)
                & np.isfinite(depth)
                & (depth > 0.0)
            )
            unchanged_coverages[frame_id] = float(np.mean(background))
            sampled_background = depth[::4, ::4][background[::4, ::4]]
            background_raw_sample_count += len(sampled_background)
            if len(sampled_background):
                background_depth_frame_medians[frame_id] = np.median(
                    sampled_background
                )
            ratios, metric_depths, projected_count = (
                _semidense_frame_ratios(
                    points_world=points_world,
                    world_to_camera=world_to_camera[frame_id],
                    intrinsic=scaled_k,
                    depth=depth,
                    background=background,
                )
            )
            projected_counts[frame_id] = projected_count
            usable_counts[frame_id] = len(ratios)
            raw_sample_count += len(ratios)
            if len(ratios):
                scale_frame_medians[frame_id] = np.median(ratios)
                logs = np.log(ratios)
                scale_frame_log_mads[frame_id] = np.median(
                    np.abs(logs - np.median(logs))
                )
                depth_frame_medians[frame_id] = np.median(metric_depths)
    finally:
        original_capture.release()
        handless_capture.release()

    anchor, scale_frame_medians = _frame_equal_summary_from_medians(
        scale_frame_medians,
        raw_sample_count=raw_sample_count,
    )
    map_depth, _ = _frame_equal_summary_from_medians(
        depth_frame_medians,
        raw_sample_count=raw_sample_count,
    )
    raw_background, _ = _frame_equal_summary_from_medians(
        background_depth_frame_medians,
        raw_sample_count=background_raw_sample_count,
    )
    scale = anchor["p50"]
    raw_background_median = raw_background["p50"]
    return {
        "selected_frame_indices": selected,
        "source_intrinsic": intrinsic,
        "inference_intrinsic": scaled_k,
        "mask_source": "explicit PNG masks" if explicit_masks else "RGB difference",
        "unchanged_background_coverage": _distribution(
            unchanged_coverages,
            positive_only=False,
        ),
        "projected_semidense_points_per_frame": _distribution(
            projected_counts,
            positive_only=False,
        ),
        "usable_semidense_points_per_frame": _distribution(
            usable_counts,
            positive_only=False,
        ),
        "semidense_anchor": anchor,
        "semidense_anchor_frame_log_mad": _distribution(
            scale_frame_log_mads,
            positive_only=False,
        ),
        "sampled_semidense_camera_z_m": map_depth,
        "handless_background_raw_vda_depth_m": raw_background,
        "estimated_metric_background_depth_m": (
            None
            if scale is None or raw_background_median is None
            else float(scale) * float(raw_background_median)
        ),
        "recommended_handless_metric_scale": scale,
        "recommended_scale_formula": (
            "median_frame(z_aria_semidense / d_vda_handless_background)"
        ),
        "temporal_diagnostics": {
            "semidense_anchor_frame_medians": _distribution(
                scale_frame_medians
            ),
        },
    }


def _semidense_quality_decision(analysis: dict[str, Any]) -> dict[str, Any]:
    anchor = analysis["semidense_anchor"]
    scale = analysis["recommended_handless_metric_scale"]
    selected_frame_count = len(analysis["selected_frame_indices"])
    minimum_frames = _minimum_anchor_frame_count(selected_frame_count)
    minimum_samples = max(300, minimum_frames * 10)
    reasons: list[str] = []
    if scale is None:
        reasons.append("no valid Aria semidense scale")
    elif not 0.1 <= float(scale) <= 2.0:
        reasons.append("estimated VDA metric scale lies outside [0.1, 2.0]")
    if int(anchor["supporting_frames"]) < minimum_frames:
        reasons.append(
            "too few semidense-anchor frames: "
            f"{anchor['supporting_frames']} < {minimum_frames}"
        )
    if int(anchor["raw_sample_count"]) < minimum_samples:
        reasons.append(
            "too few semidense-anchor samples: "
            f"{anchor['raw_sample_count']} < {minimum_samples}"
        )
    if anchor["log_mad"] is None or float(anchor["log_mad"]) > 0.15:
        reasons.append("semidense-anchor temporal log-MAD exceeds 0.15")
    unchanged = analysis["unchanged_background_coverage"]["p50"]
    if unchanged is None or float(unchanged) < 0.05:
        reasons.append("median unchanged-background coverage is below 5%")
    return {
        "status": "validated" if not reasons else "needs_fallback_or_review",
        "reasons": reasons,
        "support_requirements": {
            "selected_frame_count": selected_frame_count,
            "minimum_semidense_anchor_frames": minimum_frames,
            "minimum_semidense_anchor_samples": minimum_samples,
        },
        "metric_reference": (
            "Project Aria MPS semidense world points projected by the raw "
            "closed-loop camera trajectory"
        ),
    }


def estimate_semidense_handless_metric_scale(
    *,
    original_video: Path,
    handless_video: Path,
    source_hdf5: Path,
    semidense_points: Path,
    frame_indices: np.ndarray,
    handless_video_frame_count: int,
    handless_depth_raw_m: np.ndarray,
    source_size: tuple[int, int],
    inference_size: tuple[int, int],
    mask_dir: Path | None = None,
) -> dict[str, Any]:
    """Calibrate handless VDA directly from static Aria background points."""

    selected = validate_frame_indices(frame_indices)
    paths = [original_video, handless_video, source_hdf5, semidense_points]
    paths = [path.expanduser().resolve() for path in paths]
    original_video, handless_video, source_hdf5, semidense_points = paths
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if mask_dir is not None:
        mask_dir = mask_dir.expanduser().resolve()
    intrinsic, camera_to_world, _baseline, hdf5_frame_count = (
        _load_hdf5_camera_geometry(
            source_hdf5,
            frame_indices=selected,
        )
    )
    _validate_source_frame_provenance(
        original_video=original_video,
        handless_video_frame_count=handless_video_frame_count,
        source_hdf5=source_hdf5,
        hdf5_frame_count=hdf5_frame_count,
        selected_frame_indices=selected,
    )
    points_world, point_filter = _load_semidense_points(semidense_points)
    analysis = _analyze_semidense(
        original_video=original_video,
        handless_video=handless_video,
        mask_dir=mask_dir,
        handless_depth=np.asarray(handless_depth_raw_m),
        source_size=source_size,
        inference_size=inference_size,
        intrinsic=intrinsic,
        camera_to_world=camera_to_world,
        points_world=points_world,
        frame_indices=selected,
    )
    analysis["semidense_point_filter"] = point_filter
    decision = _semidense_quality_decision(analysis)
    scale = analysis["recommended_handless_metric_scale"]
    resolved_scale = (
        None
        if scale is None or not np.isfinite(scale) or float(scale) <= 0.0
        else float(scale)
    )
    return _json_ready({
        "schema_version": 1,
        "method": "aria-mps-semidense-background-depth",
        "resolved_vda_metric_scale": resolved_scale,
        "scale_excludes_scene_scale": True,
        "source_video": str(original_video),
        "source_video_sha256": file_sha256(original_video),
        "handless_video": str(handless_video),
        "handless_video_sha256": file_sha256(handless_video),
        "source_hdf5": str(source_hdf5),
        "source_hdf5_sha256": file_sha256(source_hdf5),
        "semidense_points": str(semidense_points),
        "semidense_points_sha256": file_sha256(semidense_points),
        "analysis": analysis,
        "decision": decision,
    })


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

    status = "validated" if not reasons else "needs_fallback_or_review"
    return {
        "status": status,
        "reasons": reasons,
        "support_requirements": {
            "selected_frame_count": selected_frame_count,
            "minimum_hand_anchor_frames": minimum_anchor_frames,
            "minimum_background_transfer_frames": minimum_transfer_frames,
        },
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
    output = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    for path in (original_video, handless_video, hdf5_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if mask_dir is not None and not mask_dir.is_dir():
        raise NotADirectoryError(mask_dir)
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
        "status": decision["status"],
        "reasons": decision["reasons"],
        "report": report_path,
    }), indent=2))


if __name__ == "__main__":
    main()
