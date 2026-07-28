from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .video_io import video_metadata


PROCESS_RESOLUTION = 504
INFERENCE_CHUNK_SIZE = 8
METRIC_FOCAL_NORMALIZER = 300.0
DEPTH_CACHE_SCHEMA_VERSION = 2


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _model_root() -> Path:
    return _repository_root() / "models/DA3METRIC-LARGE"


def _submodule_source() -> Path:
    return _repository_root() / "Depth-Anything-3/src"


def _validate_local_model(model_root: Path) -> dict[str, object]:
    config_path = model_root / "config.json"
    weights_path = model_root / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "Depth Anything 3 metric checkpoint is incomplete: "
            f"{model_root}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_name") != "da3metric-large":
        raise ValueError(
            "Scene compositing requires the native metric checkpoint "
            f"da3metric-large, got {payload.get('model_name')!r}"
        )
    if not isinstance(payload.get("config"), dict):
        raise ValueError(f"Invalid DA3 config: {config_path}")
    return payload


def _load_metric_model() -> tuple[object, object, object]:
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    source_root = _submodule_source()
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Depth Anything 3 submodule is missing: {source_root.parent}"
        )
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Depth Anything 3 metric inference requires the configured "
            "NVIDIA CUDA runtime"
        )

    from depth_anything_3.cfg import create_object
    from depth_anything_3.utils.io.input_processor import InputProcessor

    model_root = _model_root()
    payload = _validate_local_model(model_root)
    network = create_object(OmegaConf.create(payload["config"]))
    state = load_file(str(model_root / "model.safetensors"), device="cpu")
    state = {
        (
            name.removeprefix("model.")
            if name.startswith("model.")
            else name
        ): value
        for name, value in state.items()
    }
    missing, unexpected = network.load_state_dict(state, strict=False)
    del state
    if missing or unexpected:
        raise RuntimeError(
            "DA3 metric checkpoint does not match the local submodule: "
            f"missing={missing}, unexpected={unexpected}"
        )
    device = torch.device("cuda")
    network.eval().to(device=device)
    return network, InputProcessor(), device


def _metric_depth(
    raw_depth: np.ndarray,
    processed_intrinsics: np.ndarray,
    *,
    trajectory_scale: float,
) -> np.ndarray:
    if raw_depth.ndim != 3:
        raise ValueError("raw_depth must have shape [N, H, W]")
    if processed_intrinsics.shape != (raw_depth.shape[0], 3, 3):
        raise ValueError(
            "processed_intrinsics must have shape [N, 3, 3]"
        )
    focal_px = 0.5 * (
        processed_intrinsics[:, 0, 0]
        + processed_intrinsics[:, 1, 1]
    )
    if not np.isfinite(focal_px).all() or np.any(focal_px <= 0.0):
        raise ValueError("Processed focal lengths must be positive and finite")
    if not np.isfinite(trajectory_scale) or trajectory_scale <= 0.0:
        raise ValueError("trajectory_scale must be positive and finite")
    return (
        raw_depth.astype(np.float32)
        * focal_px[:, None, None].astype(np.float32)
        / METRIC_FOCAL_NORMALIZER
        * np.float32(trajectory_scale)
    )


def _depth_to_millimeters(
    depth_m: np.ndarray,
    *,
    output_size: tuple[int, int],
) -> np.ndarray:
    width, height = output_size
    resized = cv2.resize(
        depth_m.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    valid = np.isfinite(resized) & (resized > 0.0)
    millimeters = np.zeros((height, width), dtype=np.uint16)
    millimeters[valid] = np.clip(
        np.rint(resized[valid] * 1000.0),
        1,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    return millimeters


def _model_forward(
    network: object,
    processor: object,
    device: object,
    images_rgb: list[np.ndarray],
    intrinsics: np.ndarray,
    *,
    trajectory_scale: float,
) -> np.ndarray:
    import torch

    image_tensor, _, processed_intrinsics = processor(
        images_rgb,
        intrinsics=intrinsics,
        process_res=PROCESS_RESOLUTION,
        process_res_method="upper_bound_resize",
        num_workers=min(8, len(images_rgb)),
        print_progress=False,
    )
    if processed_intrinsics is None:
        raise RuntimeError("DA3 preprocessing discarded camera intrinsics")
    batch = image_tensor[None].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=dtype,
    ):
        output = network(batch)
    raw_depth = (
        output.depth.squeeze(0)
        .squeeze(-1)
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    return _metric_depth(
        raw_depth,
        processed_intrinsics.numpy(),
        trajectory_scale=trajectory_scale,
    )


def _model_forward_with_oom_backoff(
    network: object,
    processor: object,
    device: object,
    images_rgb: list[np.ndarray],
    intrinsics: np.ndarray,
    *,
    trajectory_scale: float,
) -> np.ndarray:
    """Retry an inference chunk in smaller pieces after a CUDA OOM."""

    import torch

    try:
        return _model_forward(
            network,
            processor,
            device,
            images_rgb,
            intrinsics,
            trajectory_scale=trajectory_scale,
        )
    except torch.cuda.OutOfMemoryError:
        if len(images_rgb) <= 1:
            raise
        torch.cuda.empty_cache()
        split = len(images_rgb) // 2
        first = _model_forward_with_oom_backoff(
            network,
            processor,
            device,
            images_rgb[:split],
            intrinsics[:split],
            trajectory_scale=trajectory_scale,
        )
        second = _model_forward_with_oom_backoff(
            network,
            processor,
            device,
            images_rgb[split:],
            intrinsics[split:],
            trajectory_scale=trajectory_scale,
        )
        return np.concatenate([first, second], axis=0)


def estimate_scene_depth(
    *,
    scene_video: Path,
    trajectory_path: Path,
    output_root: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    scene_video = scene_video.resolve()
    trajectory_path = trajectory_path.resolve()
    output_root = output_root.resolve()
    width, height, frame_count, fps = video_metadata(scene_video)
    with np.load(trajectory_path) as archive:
        trajectory_scale = float(archive["scale"])
        intrinsic = np.asarray(archive["intrinsic"], dtype=np.float32)
        trajectory_frame_count = int(archive["qpos"].shape[0])
    if trajectory_frame_count != frame_count:
        raise ValueError(
            "Scene video and trajectory frame counts disagree: "
            f"{frame_count} != {trajectory_frame_count}"
        )
    if intrinsic.shape != (3, 3):
        raise ValueError("Trajectory intrinsic must have shape [3, 3]")
    if not np.isfinite(intrinsic).all():
        raise ValueError("Trajectory intrinsic must be finite")
    if not np.isfinite(trajectory_scale) or trajectory_scale <= 0.0:
        raise ValueError("Trajectory scale must be positive and finite")

    network, processor, device = _load_metric_model()
    depth_root = output_root / "scene_depth_mm"
    report_path = output_root / "scene_depth_report.json"
    report_path.unlink(missing_ok=True)
    if depth_root.exists():
        shutil.rmtree(depth_root)
    depth_root.mkdir(parents=True)

    capture = cv2.VideoCapture(str(scene_video))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open scene video: {scene_video}")
    completed = 0
    valid_pixels = 0
    total_pixels = 0
    depth_png_hasher = hashlib.sha256()
    positive_depth_samples: list[np.ndarray] = []
    print(
        f"Depth Anything 3 processed 0/{frame_count}",
        flush=True,
    )
    try:
        while completed < frame_count:
            images_rgb: list[np.ndarray] = []
            for _ in range(
                min(INFERENCE_CHUNK_SIZE, frame_count - completed)
            ):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Video decode stopped at frame "
                        f"{completed + len(images_rgb)}"
                    )
                images_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            repeated_intrinsics = np.repeat(
                intrinsic[None],
                len(images_rgb),
                axis=0,
            )
            metric_depths = _model_forward_with_oom_backoff(
                network,
                processor,
                device,
                images_rgb,
                repeated_intrinsics,
                trajectory_scale=trajectory_scale,
            )
            for local_index, depth_m in enumerate(metric_depths):
                frame_id = completed + local_index
                depth_mm = _depth_to_millimeters(
                    depth_m,
                    output_size=(width, height),
                )
                depth_path = depth_root / f"{frame_id:05d}.png"
                if not cv2.imwrite(str(depth_path), depth_mm):
                    raise RuntimeError(
                        f"Failed to write scene depth frame {frame_id}"
                    )
                depth_png_hasher.update(depth_path.name.encode("ascii"))
                depth_png_hasher.update(b"\0")
                depth_png_hasher.update(depth_path.read_bytes())
                positive = depth_mm[depth_mm > 0]
                valid_pixels += int(positive.size)
                total_pixels += int(depth_mm.size)
                if positive.size:
                    stride = max(1, positive.size // 4096)
                    positive_depth_samples.append(positive[::stride])
            completed += len(images_rgb)
            print(
                f"Depth Anything 3 processed {completed}/{frame_count}",
                flush=True,
            )
    finally:
        capture.release()

    samples = (
        np.concatenate(positive_depth_samples)
        if positive_depth_samples
        else np.empty(0, dtype=np.uint16)
    )
    percentiles = (
        np.percentile(samples, [1, 50, 99]).tolist()
        if samples.size
        else [None, None, None]
    )
    model_root = _model_root()
    weights = model_root / "model.safetensors"
    scene_stat = scene_video.stat()
    report: dict[str, object] = {
        "schema_version": DEPTH_CACHE_SCHEMA_VERSION,
        "model": "DA3METRIC-LARGE",
        "model_root": str(model_root),
        "model_weights_bytes": weights.stat().st_size,
        "native_metric": True,
        "metric_conversion": (
            "depth_m = ((fx_processed + fy_processed) / 2) "
            "* network_depth / 300 * trajectory_scale"
        ),
        "trajectory_scale": trajectory_scale,
        "unit": "uint16 millimeters; 0 means invalid",
        "scene_video": str(scene_video),
        "input_role": "ProPainter clean background",
        "scene_video_fingerprint": {
            "size": scene_stat.st_size,
            "mtime_ns": scene_stat.st_mtime_ns,
        },
        "trajectory": str(trajectory_path),
        "frame_count": frame_count,
        "fps": fps,
        "source_resolution": [width, height],
        "process_resolution": PROCESS_RESOLUTION,
        "chunk_size": INFERENCE_CHUNK_SIZE,
        "depth_png_sha256": depth_png_hasher.hexdigest(),
        "valid_fraction": valid_pixels / max(total_pixels, 1),
        "depth_percentiles_mm": {
            "p01": percentiles[0],
            "p50": percentiles[1],
            "p99": percentiles[2],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cache native metric scene depth with the repository's "
            "read-only Depth Anything 3 submodule"
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="ProPainter clean-background video at source resolution",
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = estimate_scene_depth(
        scene_video=args.video,
        trajectory_path=args.trajectory,
        output_root=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
