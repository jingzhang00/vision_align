#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any
import warnings

# Stable defaults for large egocentric videos.  Callers can still override the
# corresponding CLI options below.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vision-align-mpl")

warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release.*",
)

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
INPAINT_DIR = WORKSPACE_ROOT / "Inpaint-Anything"
PROPAINTER_DIR = WORKSPACE_ROOT / "ProPainter"
PRETRAINED_DIR = INPAINT_DIR / "pretrained_models"
DEFAULT_SAM_CHECKPOINT = PRETRAINED_DIR / "sam3.pt"
DEFAULT_PROPAINTER_CHECKPOINTS = PRETRAINED_DIR / "propainter"

for dependency_root in (PROPAINTER_DIR, INPAINT_DIR):
    dependency = str(dependency_root)
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

import remove_hands as upstream  # noqa: E402
from sam3_video_track import IMAGE_EXTS  # noqa: E402


_ACTIVE_PROGRESS: Any | None = None


def _set_progress_stage(stage: str) -> None:
    if _ACTIVE_PROGRESS is not None:
        _ACTIVE_PROGRESS.set_postfix_str(stage, refresh=True)


def _advance_progress(amount: int = 1) -> None:
    if _ACTIVE_PROGRESS is not None:
        _ACTIVE_PROGRESS.update(amount)


def _disable_nested_progress() -> None:
    """Hide library bars; this wrapper renders one bar per episode."""

    def plain_iterable(iterable: Any, *args: Any, **kwargs: Any) -> Any:
        return iterable

    import propainter_video_inpaint
    from sam3.model.utils import sam2_utils

    propainter_video_inpaint.tqdm = plain_iterable
    sam2_utils.tqdm = plain_iterable


def _track_bidirectionally(
    predictor: Any,
    frame_dir: str,
    init_mask: np.ndarray,
    frame_idx: int,
    offload_to_cpu: bool,
) -> list[np.ndarray]:
    """Track one seed mask through all frames in both directions."""
    state = predictor.init_state(
        video_path=frame_dir,
        offload_video_to_cpu=offload_to_cpu,
        offload_state_to_cpu=offload_to_cpu,
    )
    object_id = 1
    predictor.add_new_mask(
        state,
        frame_idx=frame_idx,
        obj_id=object_id,
        mask=torch.as_tensor(init_mask > 0, dtype=torch.bool),
    )

    frame_count = int(state["num_frames"])
    per_frame: dict[int, np.ndarray] = {}

    def collect(*, reverse: bool, preflight: bool) -> None:
        _set_progress_stage("tracking")
        outputs = predictor.propagate_in_video(
            state,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=frame_count,
            reverse=reverse,
            propagate_preflight=preflight,
            tqdm_disable=True,
        )
        for index, object_ids, _, masks, _ in outputs:
            mask_index = (
                object_ids.index(object_id)
                if object_id in object_ids
                else 0
            )
            frame_index = int(index)
            is_new_frame = frame_index not in per_frame
            per_frame[frame_index] = (
                (masks[mask_index] > 0.0)
                .squeeze()
                .cpu()
                .numpy()
                .astype(bool)
            )
            if is_new_frame:
                _advance_progress()

    collect(reverse=False, preflight=True)
    if frame_idx > 0:
        collect(reverse=True, preflight=False)

    height = int(state["video_height"])
    width = int(state["video_width"])
    missing = sorted(set(range(frame_count)) - set(per_frame))
    if missing:
        raise RuntimeError(
            "Bidirectional SAM 3 tracking missed frame indices: "
            f"{missing[:10]}"
        )
    return [
        per_frame.get(index, np.zeros((height, width), dtype=bool))
        for index in range(frame_count)
    ]


def _seed_from_most_complete_probe(
    tracker: Any,
    frame_dir: str,
    target: str,
    confidence: float,
    segmentor: Any,
    offload: bool,
) -> list[np.ndarray] | None:
    """Pick the largest plausible text mask from several temporal probes."""
    frame_paths = sorted(
        path
        for path in Path(frame_dir).iterdir()
        if path.suffix in IMAGE_EXTS
    )
    if not frame_paths:
        return None

    fractions = (0.0, 0.1, 0.25, 0.5, 0.75)
    probe_indices = sorted(
        {
            min(int(len(frame_paths) * fraction), len(frame_paths) - 1)
            for fraction in fractions
        }
    )
    best: tuple[int, int, np.ndarray] | None = None

    for index in probe_indices:
        frame = np.asarray(Image.open(frame_paths[index]).convert("RGB"))
        segmentor.set_image(frame)
        try:
            masks, _, _ = segmentor.predict_text(
                target,
                confidence_threshold=confidence,
            )
        finally:
            segmentor.reset_image()
        if len(masks) == 0:
            continue

        union = np.any(masks, axis=0)
        candidate = (int(union.sum()), index, union)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None

    _, seed_index, seed_mask = best
    _set_progress_stage(
        f"seed {seed_index}/{len(frame_paths) - 1}, "
        f"coverage {seed_mask.mean():.1%}"
    )
    return _track_bidirectionally(
        tracker,
        frame_dir,
        seed_mask,
        seed_index,
        offload,
    )


_UPSTREAM_PROCESS_EPISODE = upstream.process_episode
_UPSTREAM_WRITE_OUTPUT = upstream._write_output
_UPSTREAM_WRITE_MASKS = upstream._write_masks


def _episode_frame_count(episode: Any) -> int:
    if not episode.is_video:
        return len(episode.frame_names)

    import cv2

    capture = cv2.VideoCapture(str(episode.source))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frame_count <= 0:
        import imageio.v2 as imageio_v2

        reader = imageio_v2.get_reader(str(episode.source))
        try:
            frame_count = int(reader.count_frames())
        finally:
            reader.close()
    if frame_count <= 0:
        raise ValueError(
            f"Could not determine frame count for {episode.source}"
        )
    return frame_count


def _inpaint_step_count(frame_count: int, chunk_size: int) -> int:
    if chunk_size <= 0 or frame_count <= chunk_size:
        return 1
    return (frame_count + chunk_size - 1) // chunk_size


def _process_episode_with_one_progress(
    models: Any,
    episode: Any,
    args: Any,
) -> dict[str, Any]:
    """Wrap one upstream episode in a single aggregate progress bar."""
    global _ACTIVE_PROGRESS

    frame_count = _episode_frame_count(episode)
    total = (
        frame_count
        + _inpaint_step_count(frame_count, args.chunk_size)
        + 1
        + int(args.save_masks)
    )
    with tqdm(
        total=total,
        desc=episode.name,
        unit="step",
        dynamic_ncols=True,
        leave=True,
    ) as progress:
        _ACTIVE_PROGRESS = progress
        try:
            entry = _UPSTREAM_PROCESS_EPISODE(models, episode, args)
            remaining = total - progress.n
            if remaining > 0:
                progress.update(remaining)
            progress.set_postfix_str(entry["status"], refresh=False)
            return entry
        except Exception:
            progress.set_postfix_str("error", refresh=False)
            raise
        finally:
            _ACTIVE_PROGRESS = None


def _inpaint_chunked_with_one_progress(
    painter: Any,
    frames: list[Image.Image],
    masks: list[Image.Image],
    args: Any,
) -> list[Image.Image]:
    """Run upstream ProPainter while updating only the episode bar."""
    frame_count = len(frames)
    _set_progress_stage("inpainting")
    if args.chunk_size <= 0 or frame_count <= args.chunk_size:
        result = upstream.inpaint_video_with_builded_propainter(
            painter,
            frames,
            masks,
            device=args.device,
            fp16=args.fp16,
            subvideo_length=args.subvideo_length,
        )
        _advance_progress()
        return result

    overlap = args.chunk_overlap
    output: list[Image.Image | None] = [None] * frame_count
    start = 0
    while start < frame_count:
        end = min(frame_count, start + args.chunk_size)
        lower = max(0, start - overlap)
        chunk = upstream.inpaint_video_with_builded_propainter(
            painter,
            frames[lower:end],
            masks[lower:end],
            device=args.device,
            fp16=args.fp16,
            subvideo_length=args.subvideo_length,
        )
        for offset, frame in enumerate(chunk):
            index = lower + offset
            if index >= start:
                output[index] = frame
        start = end
        _advance_progress()

    missing = [index for index, frame in enumerate(output) if frame is None]
    if missing:
        raise RuntimeError(
            f"ProPainter produced no output for frame indices {missing[:10]}"
        )
    return [frame for frame in output if frame is not None]


def _write_output_without_resizing(
    out_frames: list[Image.Image],
    episode: Any,
    destination: Path,
    args: Any,
) -> None:
    """Write video at its exact source dimensions without resizing frames."""
    _set_progress_stage("writing video")
    if not episode.is_video:
        _UPSTREAM_WRITE_OUTPUT(out_frames, episode, destination, args)
        _advance_progress()
        return

    import imageio
    import imageio.v2 as imageio_v2

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        metadata = imageio.v3.immeta(
            str(episode.source),
            exclude_applied=False,
        )
    except Exception:
        metadata = {}

    fps = metadata.get("fps", args.fps)
    source_size = metadata.get("size") or metadata.get("source_size")
    if source_size is None:
        reader = imageio_v2.get_reader(str(episode.source))
        try:
            source_frame = reader.get_data(0)
        finally:
            reader.close()
        source_size = (source_frame.shape[1], source_frame.shape[0])

    source_width, source_height = map(int, source_size)
    encoded_frames = []
    for index, frame in enumerate(out_frames):
        frame_array = np.asarray(frame)
        frame_size = (frame_array.shape[1], frame_array.shape[0])
        if frame_size != (source_width, source_height):
            raise ValueError(
                f"Output frame {index} has size {frame_size}, but source "
                f"video is {(source_width, source_height)}; refusing to resize."
            )
        encoded_frames.append(frame_array)

    imageio_v2.mimwrite(
        destination,
        encoded_frames,
        fps=fps,
        quality=args.video_quality,
        macro_block_size=1,
    )
    _advance_progress()


def _write_masks_with_one_progress(
    masks: list[Image.Image],
    destination: Path,
    output_format: str,
) -> None:
    _set_progress_stage("writing masks")
    _UPSTREAM_WRITE_MASKS(masks, destination, output_format)
    _advance_progress()


def main() -> None:
    """Install the wrapper hooks and delegate CLI parsing to upstream."""
    command_line = sys.argv[1:]

    def has_option(name: str) -> bool:
        return any(
            item == name or item.startswith(f"{name}=")
            for item in command_line
        )

    defaults = (
        ("--sam_ckpt", str(DEFAULT_SAM_CHECKPOINT)),
        (
            "--propainter_ckpt",
            str(DEFAULT_PROPAINTER_CHECKPOINTS),
        ),
        ("--target", "human arms and hands"),
        ("--text_confidence", "0.3"),
        ("--chunk_size", "20"),
        ("--chunk_overlap", "5"),
        ("--subvideo_length", "20"),
    )
    for option, value in defaults:
        if not has_option(option):
            sys.argv.extend([option, value])

    if not has_option("--exclude"):
        sys.argv.extend(
            [
                "--exclude",
                "_masks",
                "_plates",
                "comparison",
                "visualization",
            ]
        )

    for flag in ("--fp16", "--offload"):
        if not has_option(flag):
            sys.argv.append(flag)

    _disable_nested_progress()
    upstream._seed_masks = _seed_from_most_complete_probe
    upstream.process_episode = _process_episode_with_one_progress
    upstream._inpaint_chunked = _inpaint_chunked_with_one_progress
    upstream._write_output = _write_output_without_resizing
    upstream._write_masks = _write_masks_with_one_progress
    upstream.main()


if __name__ == "__main__":
    main()
