"""Reusable offline metric Video Depth Anything inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
VDA_ROOT = WORKSPACE_ROOT / "Video-Depth-Anything"
DEFAULT_CHECKPOINT = (
    VDA_ROOT / "checkpoints/metric_video_depth_anything_vitl.pth"
)
# Legacy explicit fixed value retained only for reproducibility and an
# opt-in fallback. Preparation defaults to per-episode automatic calibration;
# this value is never silently treated as a cross-task metric constant. Keep
# either choice separate from source-to-robot ``scene_scale``.
DEFAULT_VDA_METRIC_SCALE = 0.525
MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}
MODEL_SCALE_NAMES = {
    "vits": "Small",
    "vitb": "Base",
    "vitl": "Large",
}
INFERENCE_WINDOW_FRAMES = 32
INFERENCE_OVERLAP_FRAMES = 10


@dataclass(frozen=True)
class DecodedVideoFrames:
    """One contiguous video selection prepared for offline VDA inference."""

    frames_rgb: np.ndarray
    fps: float
    source_size: tuple[int, int]
    inference_size: tuple[int, int]
    frame_indices: np.ndarray
    video_frame_count: int

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_indices))

    def without_rgb_frames(self) -> "DecodedVideoFrames":
        """Return lightweight sequence metadata after inference is complete."""

        return DecodedVideoFrames(
            frames_rgb=np.empty((0, 0, 0, 3), dtype=np.uint8),
            fps=self.fps,
            source_size=self.source_size,
            inference_size=self.inference_size,
            frame_indices=self.frame_indices,
            video_frame_count=self.video_frame_count,
        )


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frame_indices(frame_indices: np.ndarray) -> np.ndarray:
    """Validate and copy a non-empty, non-negative contiguous selection."""

    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("frame_indices must be a non-empty vector")
    if int(indices[0]) < 0 or np.any(np.diff(indices) != 1):
        raise ValueError(
            "frame_indices must be non-negative and contiguous in source order"
        )
    return indices.copy()


def _inference_size(
    source_size: tuple[int, int],
    *,
    max_resolution: int,
) -> tuple[int, int]:
    if max_resolution < 0:
        raise ValueError("max_resolution must be non-negative")
    source_width, source_height = source_size
    target_width, target_height = source_width, source_height
    if max_resolution > 0 and max(source_size) > max_resolution:
        scale = max_resolution / max(source_size)
        target_width = max(2, int(round(source_width * scale)))
        target_height = max(2, int(round(source_height * scale)))
        target_width += target_width % 2
        target_height += target_height % 2
    return target_width, target_height


def decode_contiguous_video_frames_rgb(
    path: Path,
    *,
    frame_indices: np.ndarray | None = None,
    max_resolution: int = 1280,
) -> DecodedVideoFrames:
    """Decode one contiguous selection and retain it as an RGB VDA tensor.

    The returned frames are resized only for model inference. Callers should
    decode texture frames independently at source resolution when writing the
    prepared RGB sequence.
    """

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if (
        not np.isfinite(fps)
        or fps <= 0.0
        or min(source_width, source_height) <= 0
    ):
        capture.release()
        raise ValueError(f"Invalid video metadata: {path}")

    if frame_indices is None:
        selected: np.ndarray | None = None
        start_frame = 0
    else:
        selected = validate_frame_indices(frame_indices)
        start_frame = int(selected[0])
        if declared_frame_count > 0 and int(selected[-1]) >= declared_frame_count:
            capture.release()
            raise IndexError(
                f"Selected frame {int(selected[-1])} exceeds video length "
                f"{declared_frame_count}: {path}"
            )
        if start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if actual != start_frame:
                capture.release()
                raise RuntimeError(
                    f"Could not seek {path} to frame {start_frame}; got {actual}"
                )

    source_size = (source_width, source_height)
    target_size = _inference_size(
        source_size,
        max_resolution=max_resolution,
    )
    frames: list[np.ndarray] = []
    expected_count = None if selected is None else len(selected)
    try:
        while expected_count is None or len(frames) < expected_count:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_bgr.shape[:2] != (source_height, source_width):
                raise ValueError(
                    f"Video frame size changed at source frame "
                    f"{start_frame + len(frames)}: {frame_bgr.shape[:2]}"
                )
            if target_size != source_size:
                frame_bgr = cv2.resize(
                    frame_bgr,
                    target_size,
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"Video selection has no decodable frames: {path}")
    if expected_count is not None and len(frames) != expected_count:
        raise RuntimeError(
            f"Video decode stopped at source frame {start_frame + len(frames)}; "
            f"expected {expected_count} selected frames from {path}"
        )
    if selected is None:
        selected = np.arange(len(frames), dtype=np.int64)
    video_frame_count = (
        declared_frame_count if declared_frame_count > 0 else len(frames)
    )
    return DecodedVideoFrames(
        frames_rgb=np.stack(frames, axis=0),
        fps=fps,
        source_size=source_size,
        inference_size=target_size,
        frame_indices=selected,
        video_frame_count=video_frame_count,
    )


def validate_metric_checkpoint(checkpoint: Path, encoder: str) -> Path:
    """Validate the metric checkpoint path and encoder/name pairing."""

    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported VDA encoder: {encoder}")
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_name = f"metric_video_depth_anything_{encoder}.pth"
    if checkpoint.name != expected_name:
        raise ValueError(
            f"Encoder {encoder!r} expects checkpoint {expected_name!r}, "
            f"got {checkpoint.name!r}"
        )
    return checkpoint


def load_metric_vda_model(
    checkpoint: Path,
    encoder: str,
) -> tuple[object, object]:
    """Strictly load one metric VDA checkpoint onto CUDA in eval mode."""

    import torch

    checkpoint = validate_metric_checkpoint(checkpoint, encoder)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Metric Video Depth Anything inference requires CUDA"
        )
    source_text = str(VDA_ROOT)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    try:
        from video_depth_anything.video_depth import VideoDepthAnything
    except ModuleNotFoundError as error:
        if error.name == "easydict":
            raise ModuleNotFoundError(
                "Video-Depth-Anything requires easydict; install it in the "
                "active environment with `python -m pip install easydict`"
            ) from error
        raise

    model = VideoDepthAnything(**MODEL_CONFIGS[encoder], metric=True)
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    try:
        model.load_state_dict(state, strict=True)
    finally:
        del state
    device = torch.device("cuda")
    model.to(device=device).eval()
    return model, device


def infer_metric_vda_sequence(
    frames: DecodedVideoFrames,
    *,
    checkpoint: Path,
    encoder: str,
    input_size: int,
    fp32: bool,
) -> np.ndarray:
    """Infer raw metric camera-Z depth for the complete selected sequence."""

    if input_size <= 0:
        raise ValueError("input_size must be positive")
    if (
        frames.frames_rgb.ndim != 4
        or frames.frames_rgb.shape[0] != frames.frame_count
        or frames.frames_rgb.shape[-1] != 3
    ):
        raise ValueError(
            "DecodedVideoFrames no longer contains the RGB tensor required "
            "for VDA inference"
        )
    checkpoint = validate_metric_checkpoint(checkpoint, encoder)
    model, device = load_metric_vda_model(checkpoint, encoder)
    try:
        depths_raw, inferred_fps = model.infer_video_depth(
            frames.frames_rgb,
            frames.fps,
            input_size=input_size,
            device=str(device),
            fp32=fp32,
        )
    finally:
        del model
        import torch

        torch.cuda.empty_cache()

    expected_shape = (
        frames.frame_count,
        frames.frames_rgb.shape[1],
        frames.frames_rgb.shape[2],
    )
    depths = np.asarray(depths_raw, dtype=np.float32)
    if depths.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected VDA output shape: {depths.shape} != {expected_shape}"
        )
    if not np.isclose(inferred_fps, frames.fps):
        raise RuntimeError(
            f"VDA changed FPS: {inferred_fps} != {frames.fps}"
        )
    finite = np.isfinite(depths)
    positive_rate = np.mean(finite & (depths > 0.0), axis=(1, 2))
    if not finite.all() or float(positive_rate.min()) < 0.99:
        raise RuntimeError(
            "Metric VDA must return finite dense depth; minimum per-frame "
            f"positive rate was {float(positive_rate.min()):.6f}"
        )
    return depths


def scale_and_resize_metric_depths(
    depths_raw_m: np.ndarray,
    *,
    scene_scale: float,
    vda_metric_scale: float = 1.0,
    mesh_size: tuple[int, int],
    maximum_depth_m: float | None = None,
) -> np.ndarray:
    """Calibrate VDA depth, apply scene scale, and nearest-resize it."""

    depths = np.asarray(depths_raw_m, dtype=np.float32)
    if depths.ndim != 3 or len(depths) == 0:
        raise ValueError("depths_raw_m must have shape [frames, height, width]")
    if not np.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError("scene_scale must be positive and finite")
    if not np.isfinite(vda_metric_scale) or vda_metric_scale <= 0.0:
        raise ValueError("vda_metric_scale must be positive and finite")
    if maximum_depth_m is not None and (
        not np.isfinite(maximum_depth_m) or maximum_depth_m <= 0.0
    ):
        raise ValueError("maximum_depth_m must be positive and finite")
    mesh_width, mesh_height = (int(value) for value in mesh_size)
    if min(mesh_width, mesh_height) <= 1:
        raise ValueError("mesh_size must contain valid WIDTH,HEIGHT")

    output = np.empty(
        (len(depths), mesh_height, mesh_width),
        dtype=np.float32,
    )
    for frame_id, raw_depth in enumerate(depths):
        # These are two distinct conversions, each applied exactly once:
        # learned VDA metric scale -> calibrated source depth -> robot world.
        scaled = (
            raw_depth
            * np.float32(vda_metric_scale)
            * np.float32(scene_scale)
        )
        resized = cv2.resize(
            scaled,
            (mesh_width, mesh_height),
            interpolation=cv2.INTER_NEAREST,
        )
        valid = np.isfinite(resized) & (resized > 0.0)
        positive_rate = float(np.mean(valid))
        if positive_rate < 0.99:
            raise RuntimeError(
                f"VDA depth is not dense at frame {frame_id}: "
                f"positive_rate={positive_rate:.6f}"
            )
        if maximum_depth_m is not None:
            too_far = valid & (resized > maximum_depth_m)
            too_far_rate = float(np.mean(too_far))
            if too_far_rate > 0.01:
                raise RuntimeError(
                    f"VDA depth beyond {maximum_depth_m:.3f} m is not sparse "
                    f"at frame {frame_id}: rate={too_far_rate:.6f}"
                )
            resized[too_far] = 0.0
        resized[~valid] = 0.0
        output[frame_id] = resized

    valid_output = np.isfinite(output) & (output > 0.0)
    if not np.any(valid_output):
        raise RuntimeError("VDA depth contains no finite positive samples")
    maximum_m = float(np.max(output, where=valid_output, initial=0.0))
    maximum_uint16_m = np.iinfo(np.uint16).max / 1000.0
    if maximum_m > maximum_uint16_m:
        raise OverflowError(
            "VDA depth exceeds the uint16-millimeter representation: "
            f"{maximum_m:.3f} m > {maximum_uint16_m:.3f} m"
        )
    return output


def depth_m_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
    """Convert one already-sized metric depth frame to uint16 millimeters."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a two-dimensional frame")
    valid = np.isfinite(depth) & (depth > 0.0)
    if np.any(depth[valid] > np.iinfo(np.uint16).max / 1000.0):
        raise OverflowError(
            "Depth exceeds the uint16-millimeter representation"
        )
    millimeters = np.zeros(depth.shape, dtype=np.uint16)
    millimeters[valid] = np.clip(
        np.rint(depth[valid] * 1000.0),
        1,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    return millimeters
