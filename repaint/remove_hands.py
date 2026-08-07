#!/usr/bin/env python3
"""Remove source human arms for both canonical VDA pipelines."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sys
import tempfile
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

for dependency_root in (WORKSPACE_ROOT, PROPAINTER_DIR, INPAINT_DIR):
    dependency = str(dependency_root)
    if dependency not in sys.path:
        sys.path.insert(0, dependency)

import remove_hands as upstream  # noqa: E402
from sam3_video_track import IMAGE_EXTS  # noqa: E402
from repaint.vda_pipeline_common import probe_video  # noqa: E402


_ACTIVE_PROGRESS: Any | None = None
_LAST_TRACKED_PRESENCE: list[bool] | None = None


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


def _tracked_target_present(score: float, mask: np.ndarray) -> bool:
    """Use SAM's native zero-logit decision boundary for target presence."""
    if not np.isfinite(score):
        raise ValueError("SAM object-presence score must be finite")
    return score > 0.0 and bool(np.asarray(mask, dtype=bool).any())


def _track_bidirectionally(
    predictor: Any,
    frame_dir: str,
    init_mask: np.ndarray,
    frame_idx: int,
    offload_to_cpu: bool,
) -> list[np.ndarray]:
    """Track one seed mask through all frames in both directions."""
    global _LAST_TRACKED_PRESENCE

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
    per_frame_presence: dict[int, bool] = {}
    height = int(state["video_height"])
    width = int(state["video_width"])

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
        for index, object_ids, _, masks, object_scores in outputs:
            frame_index = int(index)
            is_new_frame = frame_index not in per_frame
            if object_id not in object_ids:
                mask = np.zeros((height, width), dtype=bool)
                present = False
            else:
                mask_index = object_ids.index(object_id)
                mask = (
                    (masks[mask_index] > 0.0)
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )
                score = float(
                    object_scores[mask_index].detach().float().item()
                )
                present = _tracked_target_present(score, mask)
                if not present:
                    mask.fill(False)
            per_frame[frame_index] = mask
            per_frame_presence[frame_index] = present
            if is_new_frame:
                _advance_progress()

    collect(reverse=False, preflight=True)
    if frame_idx > 0:
        collect(reverse=True, preflight=False)

    missing = sorted(set(range(frame_count)) - set(per_frame))
    if missing:
        raise RuntimeError(
            "Bidirectional SAM 3 tracking missed frame indices: "
            f"{missing[:10]}"
        )
    _LAST_TRACKED_PRESENCE = [
        per_frame_presence[index] for index in range(frame_count)
    ]
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
    """Find the first target anywhere in the clip, then track both ways."""
    frame_paths = sorted(
        path
        for path in Path(frame_dir).iterdir()
        if path.suffix in IMAGE_EXTS
    )
    if not frame_paths:
        return None

    for index, frame_path in enumerate(frame_paths):
        frame = np.asarray(Image.open(frame_path).convert("RGB"))
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

        seed_mask = np.any(masks, axis=0)
        _set_progress_stage(
            f"seed {index}/{len(frame_paths) - 1}, "
            f"coverage {seed_mask.mean():.1%}"
        )
        return _track_bidirectionally(
            tracker,
            frame_dir,
            seed_mask,
            index,
            offload,
        )
    return None


_UPSTREAM_PROCESS_EPISODE = upstream.process_episode
_UPSTREAM_WRITE_OUTPUT = upstream._write_output
_UPSTREAM_WRITE_MASKS = upstream._write_masks
_UPSTREAM_AS_FRAME_FOLDER = upstream._as_frame_folder
_UPSTREAM_READ_FRAMES = upstream._read_frames


def _displayed_video_frame_count(path: Path) -> int:
    """Return decoded/displayed frames without container-duration padding."""

    return probe_video(path).frame_count


def _episode_frame_count(episode: Any) -> int:
    if not episode.is_video:
        return len(episode.frame_names)
    return _displayed_video_frame_count(Path(episode.source))


def _read_video_frames_without_cfr_padding(path: Path) -> list[Image.Image]:
    """Decode the MP4 display timeline without duplicating its final frame."""

    frames = list(_iter_video_display_frames(path))
    expected = _displayed_video_frame_count(path)
    if len(frames) != expected:
        raise ValueError(
            f"Display-timeline decode mismatch for {path}: "
            f"decoded={len(frames)}, expected={expected}"
        )
    return frames


def _iter_video_display_frames(path: Path) -> Any:
    """Yield frames decoded by PyAV with native presentation timing."""

    import av

    with av.open(str(path)) as container:
        streams = container.streams.video
        if len(streams) != 1:
            raise ValueError(
                f"Expected one video stream in {path}, got {len(streams)}"
            )
        for frame in container.decode(streams[0]):
            yield frame.to_image().convert("RGB")


def _validate_display_frame_count(
    path: Path,
    *,
    decoded: int,
) -> None:
    expected = _displayed_video_frame_count(path)
    if decoded != expected:
        raise ValueError(
            f"Display-timeline decode mismatch for {path}: "
            f"decoded={decoded}, expected={expected}"
        )


def _read_frames_on_display_timeline(episode: Any) -> list[Image.Image]:
    if not episode.is_video:
        return _UPSTREAM_READ_FRAMES(episode)
    return _read_video_frames_without_cfr_padding(Path(episode.source))


@contextmanager
def _as_display_frame_folder(video_path: str) -> Any:
    """Give SAM the same unpadded frame timeline used by the inpainter."""

    source = Path(video_path)
    if source.is_dir():
        with _UPSTREAM_AS_FRAME_FOLDER(video_path) as frame_dir:
            yield frame_dir
        return

    temporary = Path(tempfile.mkdtemp(prefix="ia_display_frames_"))
    try:
        frame_count = 0
        for index, frame in enumerate(
            _iter_video_display_frames(source)
        ):
            frame.save(
                temporary / f"{index:05d}.jpg",
                quality=95,
            )
            frame_count += 1
        _validate_display_frame_count(source, decoded=frame_count)
        yield str(temporary)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _source_video_fps(path: Path) -> float:
    """Read timing from the source video; never invent a fallback FPS."""

    return probe_video(path).fps


def _inpaint_step_count(frame_count: int, chunk_size: int) -> int:
    if chunk_size <= 0 or frame_count <= chunk_size:
        return 1
    return (frame_count + chunk_size - 1) // chunk_size


def _is_nymeria_dataset_root(path: str | Path | None) -> bool:
    if path is None:
        return False
    root = Path(path).expanduser().resolve()
    return (root / "manifest.json").is_file() or (
        (root / "meta/nymeria.json").is_file()
        and not (root / "meta/egodex.json").exists()
    )


def _process_episode_with_one_progress(
    models: Any,
    episode: Any,
    args: Any,
) -> dict[str, Any]:
    """Wrap one upstream episode in a single aggregate progress bar."""
    global _ACTIVE_PROGRESS, _LAST_TRACKED_PRESENCE

    frame_count = _episode_frame_count(episode)
    is_nymeria = _is_nymeria_dataset_root(
        getattr(args, "input_dir", None)
    )
    _LAST_TRACKED_PRESENCE = None
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
            if (
                entry.get("status") == "no_target"
                and is_nymeria
            ):
                destination = Path(args.output_dir) / (
                    episode.name + (".mp4" if episode.is_video else "")
                )
                if episode.is_video:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(episode.source, destination)
                else:
                    destination.mkdir(parents=True, exist_ok=True)
                    for frame_name in episode.frame_names:
                        shutil.copy2(
                            episode.source / frame_name,
                            destination / frame_name,
                        )
                entry.update(
                    status="passthrough_no_target",
                    output=str(destination),
                    frames=_episode_frame_count(episode),
                    passthrough=True,
                    note=(
                        "Nymeria continuation policy: no removable target was "
                        "detected, so the source was copied unchanged."
                    ),
                )
            if is_nymeria:
                if entry.get("status") == "passthrough_no_target":
                    visibility = [False] * int(entry["frames"])
                else:
                    visibility = _LAST_TRACKED_PRESENCE
                if visibility is None or len(visibility) != int(
                    entry.get("frames", -1)
                ):
                    raise RuntimeError(
                        "Nymeria SAM visibility timeline is missing or "
                        "does not match the episode"
                    )
                entry.update(
                    human_arm_visible=visibility,
                    human_arm_visible_frames=int(sum(visibility)),
                    human_arm_visibility_method=(
                        "SAM 3 tracked target with its native positive "
                        "object-presence logit and a nonempty mask"
                    ),
                )
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

    expected_frame_count = _episode_frame_count(episode)
    if len(out_frames) != expected_frame_count:
        raise ValueError(
            f"Repaint frame count differs from the source display timeline: "
            f"{len(out_frames)} != {expected_frame_count}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        metadata = imageio.v3.immeta(
            str(episode.source),
            exclude_applied=False,
        )
    except Exception:
        metadata = {}

    fps = metadata.get("fps")
    if fps is None or not np.isfinite(float(fps)) or float(fps) <= 0.0:
        fps = _source_video_fps(Path(episode.source))
    fps = float(fps)
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
    written_frame_count = _displayed_video_frame_count(destination)
    if written_frame_count != expected_frame_count:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Encoded video frame count changed for {destination}: "
            f"{expected_frame_count} -> {written_frame_count}"
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
    upstream._as_frame_folder = _as_display_frame_folder
    upstream._read_frames = _read_frames_on_display_timeline
    upstream.process_episode = _process_episode_with_one_progress
    upstream._inpaint_chunked = _inpaint_chunked_with_one_progress
    upstream._write_output = _write_output_without_resizing
    upstream._write_masks = _write_masks_with_one_progress
    upstream.main()


if __name__ == "__main__":
    main()
