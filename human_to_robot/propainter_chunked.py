from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage

from .video_io import (
    EncodedVideoWriter,
    source_encoding_profile,
    video_metadata,
)


BBox = tuple[int, int, int, int]
CommandRunner = Callable[[list[str], Path], None]


@dataclass(frozen=True)
class MaskFrameInfo:
    index: int
    path: Path
    bbox: BBox | None
    area: int


@dataclass(frozen=True)
class ChunkPlan:
    core_start: int
    core_end: int
    clip_start: int
    clip_end: int
    roi: BBox
    equivalent_frames: float

    @property
    def clip_length(self) -> int:
        return self.clip_end - self.clip_start


@dataclass(frozen=True)
class ChunkTransition:
    previous_chunk: int
    next_chunk: int
    start: int
    end: int


def _strict_numbered_masks(mask_root: Path, frame_count: int) -> list[Path]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not mask_root.is_dir():
        raise NotADirectoryError(f"Mask directory not found: {mask_root}")
    expected = [mask_root / f"{index:05d}.png" for index in range(frame_count)]
    actual = sorted(mask_root.glob("*.png"))
    if actual != expected:
        expected_names = {path.name for path in expected}
        actual_names = {path.name for path in actual}
        missing = sorted(expected_names - actual_names)[:5]
        extra = sorted(actual_names - expected_names)[:5]
        raise ValueError(
            f"Expected exactly {frame_count} numbered masks in {mask_root}; "
            f"missing={missing}, extra={extra}"
        )
    return expected


def _target_mask(path: Path, target_size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    width, height = target_size
    if mask.shape[1::-1] != target_size:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def load_mask_infos(
    mask_root: Path,
    *,
    frame_count: int,
    target_size: tuple[int, int],
) -> list[MaskFrameInfo]:
    width, height = target_size
    if width <= 0 or height <= 0:
        raise ValueError("Target width and height must be positive")
    infos: list[MaskFrameInfo] = []
    for index, path in enumerate(_strict_numbered_masks(mask_root, frame_count)):
        mask = _target_mask(path, target_size)
        points = cv2.findNonZero(mask.astype(np.uint8))
        bbox: BBox | None
        if points is None:
            bbox = None
        else:
            x, y, box_width, box_height = cv2.boundingRect(points)
            bbox = (x, y, x + box_width, y + box_height)
        infos.append(
            MaskFrameInfo(
                index=index,
                path=path,
                bbox=bbox,
                area=int(np.count_nonzero(mask)),
            )
        )
    return infos


def _active_windows(
    infos: Sequence[MaskFrameInfo],
    *,
    context_frames: int,
) -> list[tuple[int, int]]:
    active = [info.index for info in infos if info.bbox is not None]
    if not active:
        return []
    windows: list[tuple[int, int]] = []
    start = active[0]
    previous = active[0]
    merge_gap = 2 * context_frames
    for index in active[1:]:
        if index - previous - 1 > merge_gap:
            windows.append((start, previous + 1))
            start = index
        previous = index
    windows.append((start, previous + 1))
    return windows


def _aligned_axis(
    lower: int,
    upper: int,
    *,
    limit: int,
    margin: int,
    minimum_size: int,
    alignment: int,
) -> tuple[int, int]:
    expanded_lower = max(0, lower - margin)
    expanded_upper = min(limit, upper + margin)
    required = expanded_upper - expanded_lower
    size = max(required, min(minimum_size, limit))
    if size < limit:
        size = ((size + alignment - 1) // alignment) * alignment
    size = min(size, limit)
    start = (expanded_lower + expanded_upper - size) // 2
    start = max(0, min(start, limit - size))
    if start > expanded_lower:
        start = expanded_lower
    if start + size < expanded_upper:
        start = expanded_upper - size
    start = max(0, min(start, limit - size))
    return start, start + size


def _roi_for_range(
    infos: Sequence[MaskFrameInfo],
    *,
    start: int,
    end: int,
    target_size: tuple[int, int],
    margin: int,
    minimum_size: int = 64,
    alignment: int = 8,
) -> BBox:
    boxes = [
        info.bbox for info in infos[start:end] if info.bbox is not None
    ]
    if not boxes:
        raise ValueError(f"Frame range [{start}, {end}) has no non-empty mask")
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    width, height = target_size
    left, right = _aligned_axis(
        x0,
        x1,
        limit=width,
        margin=margin,
        minimum_size=minimum_size,
        alignment=alignment,
    )
    top, bottom = _aligned_axis(
        y0,
        y1,
        limit=height,
        margin=margin,
        minimum_size=minimum_size,
        alignment=alignment,
    )
    return left, top, right, bottom


def _candidate_plan(
    infos: Sequence[MaskFrameInfo],
    *,
    core_start: int,
    core_end: int,
    frame_count: int,
    target_size: tuple[int, int],
    context_frames: int,
    roi_margin: int,
) -> ChunkPlan:
    clip_start = max(0, core_start - context_frames)
    clip_end = min(frame_count, core_end + context_frames)
    roi = _roi_for_range(
        infos,
        start=clip_start,
        end=clip_end,
        target_size=target_size,
        margin=roi_margin,
    )
    left, top, right, bottom = roi
    canvas_area = target_size[0] * target_size[1]
    equivalent_frames = (
        (clip_end - clip_start)
        * (right - left)
        * (bottom - top)
        / canvas_area
    )
    return ChunkPlan(
        core_start=core_start,
        core_end=core_end,
        clip_start=clip_start,
        clip_end=clip_end,
        roi=roi,
        equivalent_frames=float(equivalent_frames),
    )


def _split_position(
    infos: Sequence[MaskFrameInfo],
    *,
    start: int,
    end: int,
    search_radius: int = 20,
) -> int:
    midpoint = (start + end) // 2
    first = max(start + 1, midpoint - search_radius)
    last = min(end - 1, midpoint + search_radius)
    if first > last:
        return midpoint
    return min(
        range(first, last + 1),
        key=lambda index: (
            infos[index - 1].area + infos[index].area,
            abs(index - midpoint),
            index,
        ),
    )


def plan_chunks(
    infos: Sequence[MaskFrameInfo],
    *,
    frame_count: int,
    target_size: tuple[int, int],
    context_frames: int = 20,
    roi_margin: int = 64,
    max_equivalent_frames: float = 80.0,
    max_clip_frames: int = 240,
) -> list[ChunkPlan]:
    if len(infos) != frame_count:
        raise ValueError(
            f"Mask info count {len(infos)} does not match {frame_count} frames"
        )
    if context_frames < 0:
        raise ValueError("context_frames cannot be negative")
    if roi_margin < 0:
        raise ValueError("roi_margin cannot be negative")
    if max_equivalent_frames <= 0:
        raise ValueError("max_equivalent_frames must be positive")
    if max_clip_frames <= 0:
        raise ValueError("max_clip_frames must be positive")

    chunks: list[ChunkPlan] = []

    def append_or_split(core_start: int, core_end: int) -> None:
        candidate = _candidate_plan(
            infos,
            core_start=core_start,
            core_end=core_end,
            frame_count=frame_count,
            target_size=target_size,
            context_frames=context_frames,
            roi_margin=roi_margin,
        )
        within_budget = (
            candidate.equivalent_frames <= max_equivalent_frames + 1e-9
            and candidate.clip_length <= max_clip_frames
        )
        if within_budget:
            chunks.append(candidate)
            return
        if core_end - core_start <= 1:
            for reduced_context in range(context_frames - 1, -1, -1):
                candidate = _candidate_plan(
                    infos,
                    core_start=core_start,
                    core_end=core_end,
                    frame_count=frame_count,
                    target_size=target_size,
                    context_frames=reduced_context,
                    roi_margin=roi_margin,
                )
                if (
                    candidate.equivalent_frames
                    <= max_equivalent_frames + 1e-9
                    and candidate.clip_length <= max_clip_frames
                ):
                    chunks.append(candidate)
                    return
            raise RuntimeError(
                "A single masked frame exceeds the ProPainter chunk budget; "
                "increase --subvideo-length or reduce the output resolution"
            )
        split = _split_position(
            infos,
            start=core_start,
            end=core_end,
        )
        append_or_split(core_start, split)
        append_or_split(split, core_end)

    for start, end in _active_windows(
        infos,
        context_frames=context_frames,
    ):
        append_or_split(start, end)

    chunks.sort(key=lambda chunk: chunk.core_start)
    validate_chunk_plan(
        infos,
        chunks,
        frame_count=frame_count,
        target_size=target_size,
        max_equivalent_frames=max_equivalent_frames,
        max_clip_frames=max_clip_frames,
    )
    return chunks


def validate_chunk_plan(
    infos: Sequence[MaskFrameInfo],
    chunks: Sequence[ChunkPlan],
    *,
    frame_count: int,
    target_size: tuple[int, int],
    max_equivalent_frames: float,
    max_clip_frames: int,
) -> None:
    active_coverage = np.zeros(frame_count, dtype=np.uint16)
    previous_core_end = 0
    width, height = target_size
    for chunk_index, chunk in enumerate(chunks):
        if not (
            0
            <= chunk.clip_start
            <= chunk.core_start
            < chunk.core_end
            <= chunk.clip_end
            <= frame_count
        ):
            raise ValueError(f"Invalid chunk bounds: {chunk}")
        if chunk_index and chunk.core_start < previous_core_end:
            raise ValueError("Chunk core ranges overlap")
        previous_core_end = chunk.core_end
        left, top, right, bottom = chunk.roi
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"Invalid chunk ROI: {chunk.roi}")
        if chunk.clip_length > max_clip_frames:
            raise ValueError(f"Chunk exceeds max clip length: {chunk}")
        if chunk.equivalent_frames > max_equivalent_frames + 1e-6:
            raise ValueError(f"Chunk exceeds pixel-frame budget: {chunk}")
        for info in infos[chunk.clip_start : chunk.clip_end]:
            if info.bbox is None:
                continue
            x0, y0, x1, y1 = info.bbox
            if not (left <= x0 and top <= y0 and right >= x1 and bottom >= y1):
                raise ValueError(
                    f"Chunk ROI {chunk.roi} excludes mask bbox {info.bbox}"
                )
        active_coverage[chunk.core_start : chunk.core_end] += 1

    for info in infos:
        expected = 1 if info.bbox is not None else 0
        actual = int(active_coverage[info.index])
        if expected and actual != 1:
            raise ValueError(
                f"Active frame {info.index} has core coverage {actual}, expected 1"
            )


def plan_transitions(
    chunks: Sequence[ChunkPlan],
    *,
    transition_frames: int,
) -> list[ChunkTransition]:
    if transition_frames < 0:
        raise ValueError("transition_frames cannot be negative")
    if transition_frames < 2:
        return []
    transitions: list[ChunkTransition] = []
    left_frames = transition_frames // 2
    right_frames = transition_frames - left_frames
    for next_index in range(1, len(chunks)):
        previous_index = next_index - 1
        previous = chunks[previous_index]
        following = chunks[next_index]
        boundary = previous.core_end
        if boundary != following.core_start:
            continue
        previous_span = min(
            left_frames,
            max(1, (previous.core_end - previous.core_start) // 3),
        )
        following_span = min(
            right_frames,
            max(1, (following.core_end - following.core_start) // 3),
        )
        start = max(
            previous.core_start,
            previous.clip_start,
            following.clip_start,
            boundary - previous_span,
        )
        end = min(
            previous.clip_end,
            following.clip_end,
            following.core_end,
            boundary + following_span,
        )
        if transitions and start < transitions[-1].end:
            start = transitions[-1].end
        if end - start < 2:
            continue
        transitions.append(
            ChunkTransition(
                previous_chunk=previous_index,
                next_chunk=next_index,
                start=start,
                end=end,
            )
        )
    return transitions


def compose_crop(
    source: np.ndarray,
    crop: np.ndarray,
    mask: np.ndarray,
    *,
    roi: BBox,
    mask_dilation: int,
) -> np.ndarray:
    left, top, right, bottom = roi
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source frame must be a BGR image")
    if mask.shape != source.shape[:2]:
        raise ValueError("Mask and source frame sizes differ")
    if crop.shape[:2] != (bottom - top, right - left):
        raise ValueError(
            f"Crop shape {crop.shape[:2]} does not match ROI {roi}"
        )
    replace = mask.astype(bool)
    if mask_dilation > 0:
        replace = scipy.ndimage.binary_dilation(
            replace, iterations=mask_dilation
        )
    replace_crop = replace[top:bottom, left:right]
    result = source.copy()
    target = result[top:bottom, left:right]
    target[replace_crop] = crop[replace_crop]
    return result


def _default_runner(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def _resize_frame(
    frame: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    if frame.shape[1::-1] == target_size:
        return frame
    shrinking = (
        target_size[0] <= frame.shape[1]
        and target_size[1] <= frame.shape[0]
    )
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(frame, target_size, interpolation=interpolation)


def _write_chunk_inputs(
    *,
    video: Path,
    infos: Sequence[MaskFrameInfo],
    chunk: ChunkPlan,
    target_size: tuple[int, int],
    frame_root: Path,
    mask_root: Path,
) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    seek_ok = capture.set(cv2.CAP_PROP_POS_FRAMES, chunk.clip_start)
    seek_position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    if not seek_ok or seek_position != chunk.clip_start:
        capture.release()
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not reopen video: {video}")
        for frame_index in range(chunk.clip_start):
            ok, _ = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(
                    f"Could not seek past source frame {frame_index}: {video}"
                )
    left, top, right, bottom = chunk.roi
    try:
        for local_index, global_index in enumerate(
            range(chunk.clip_start, chunk.clip_end)
        ):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not decode frame {global_index} from {video}"
                )
            frame = _resize_frame(frame, target_size)
            mask = _target_mask(infos[global_index].path, target_size)
            _write_png(
                frame_root / f"{local_index:05d}.png",
                frame[top:bottom, left:right],
            )
            _write_png(
                mask_root / f"{local_index:05d}.png",
                mask[top:bottom, left:right].astype(np.uint8) * 255,
            )
    finally:
        capture.release()


def _validate_saved_frames(
    frame_root: Path,
    *,
    expected_count: int,
    expected_size: tuple[int, int],
) -> None:
    expected = [frame_root / f"{index:04d}.png" for index in range(expected_count)]
    actual = sorted(frame_root.glob("*.png")) if frame_root.is_dir() else []
    if actual != expected:
        raise RuntimeError(
            f"ProPainter saved {len(actual)} frames, expected {expected_count}: "
            f"{frame_root}"
        )
    for path in (expected[0], expected[-1]):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[1::-1] != expected_size:
            raise RuntimeError(
                f"Invalid ProPainter frame size at {path}; "
                f"expected {expected_size}"
            )


def _run_chunk(
    *,
    chunk_index: int,
    chunk: ChunkPlan,
    video: Path,
    infos: Sequence[MaskFrameInfo],
    target_size: tuple[int, int],
    workspace: Path,
    propainter_root: Path,
    subvideo_length: int,
    neighbor_length: int,
    ref_stride: int,
    raft_iter: int,
    mask_dilation: int,
    save_fps: int,
    fp16: bool,
    runner: CommandRunner,
) -> Path:
    chunk_root = workspace / f"chunk_{chunk_index:04d}"
    input_frames = chunk_root / "input_frames"
    input_masks = chunk_root / "input_masks"
    output_root = chunk_root / "output"
    _write_chunk_inputs(
        video=video,
        infos=infos,
        chunk=chunk,
        target_size=target_size,
        frame_root=input_frames,
        mask_root=input_masks,
    )
    left, top, right, bottom = chunk.roi
    command = [
        sys.executable,
        str(propainter_root / "inference_propainter.py"),
        "--video",
        str(input_frames),
        "--mask",
        str(input_masks),
        "--output",
        str(output_root),
        "--width",
        str(right - left),
        "--height",
        str(bottom - top),
        "--subvideo_length",
        str(subvideo_length),
        "--neighbor_length",
        str(neighbor_length),
        "--ref_stride",
        str(ref_stride),
        "--raft_iter",
        str(raft_iter),
        "--mask_dilation",
        str(mask_dilation),
        "--save_fps",
        str(save_fps),
        "--save_frames",
    ]
    if fp16:
        command.append("--fp16")
    runner(command, propainter_root)
    saved_frames = output_root / input_frames.name / "frames"
    _validate_saved_frames(
        saved_frames,
        expected_count=chunk.clip_length,
        expected_size=(right - left, bottom - top),
    )
    return saved_frames


def _stream_final_video(
    *,
    video: Path,
    infos: Sequence[MaskFrameInfo],
    chunks: Sequence[ChunkPlan],
    chunk_frame_roots: Sequence[Path],
    destination: Path,
    target_size: tuple[int, int],
    frame_count: int,
    mask_dilation: int,
    transition_frames: int,
    hardware_encoding: bool,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.{time.time_ns()}.mp4"
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    encoding = source_encoding_profile(
        video,
        output_size=target_size,
        hardware_encoding=hardware_encoding,
    )
    try:
        writer = EncodedVideoWriter(temporary, encoding)
    except BaseException:
        capture.release()
        temporary.unlink(missing_ok=True)
        raise
    transitions = plan_transitions(
        chunks,
        transition_frames=transition_frames,
    )
    transition_by_frame = {
        frame_index: transition
        for transition in transitions
        for frame_index in range(transition.start, transition.end)
    }

    chunk_index = 0
    written = 0

    def compose_chunk(
        source: np.ndarray,
        mask: np.ndarray,
        *,
        selected_chunk: int,
        frame_index: int,
    ) -> np.ndarray:
        chunk = chunks[selected_chunk]
        local_index = frame_index - chunk.clip_start
        crop_path = (
            chunk_frame_roots[selected_chunk] / f"{local_index:04d}.png"
        )
        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop is None:
            raise RuntimeError(
                f"Could not read ProPainter frame for source frame "
                f"{frame_index}: {crop_path}"
            )
        return compose_crop(
            source,
            crop,
            mask,
            roi=chunk.roi,
            mask_dilation=mask_dilation,
        )

    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not decode source frame {frame_index}: {video}"
                )
            frame = _resize_frame(frame, target_size)
            while (
                chunk_index < len(chunks)
                and frame_index >= chunks[chunk_index].core_end
            ):
                chunk_index += 1
            if (
                chunk_index < len(chunks)
                and chunks[chunk_index].core_start
                <= frame_index
                < chunks[chunk_index].core_end
            ):
                mask = _target_mask(infos[frame_index].path, target_size)
                transition = transition_by_frame.get(frame_index)
                if transition is None:
                    frame = compose_chunk(
                        frame,
                        mask,
                        selected_chunk=chunk_index,
                        frame_index=frame_index,
                    )
                else:
                    previous = compose_chunk(
                        frame,
                        mask,
                        selected_chunk=transition.previous_chunk,
                        frame_index=frame_index,
                    )
                    following = compose_chunk(
                        frame,
                        mask,
                        selected_chunk=transition.next_chunk,
                        frame_index=frame_index,
                    )
                    weight = (
                        (frame_index - transition.start)
                        / (transition.end - transition.start - 1)
                    )
                    frame = cv2.addWeighted(
                        previous,
                        1.0 - weight,
                        following,
                        weight,
                        0.0,
                    )
            writer.write(frame)
            written += 1
        writer.close()
        capture.release()
    except BaseException:
        try:
            writer.close()
        except Exception:
            pass
        finally:
            capture.release()
            temporary.unlink(missing_ok=True)
        raise
    if written != frame_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Wrote {written} frames, expected {frame_count}")
    width, height, output_count, _ = video_metadata(temporary)
    if (width, height) != target_size or output_count != frame_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Invalid final inpaint video: size={(width, height)}, "
            f"frames={output_count}; expected size={target_size}, "
            f"frames={frame_count}"
        )
    os.replace(temporary, destination)
    return encoding.to_dict()


def run_chunked_propainter(
    *,
    video: Path,
    mask_root: Path,
    output_root: Path,
    propainter_root: Path,
    width: int,
    height: int,
    subvideo_length: int = 80,
    context_frames: int = 20,
    roi_margin: int = 64,
    max_clip_frames: int = 240,
    transition_frames: int = 16,
    neighbor_length: int = 10,
    ref_stride: int = 10,
    raft_iter: int = 20,
    mask_dilation: int = 4,
    fp16: bool = True,
    hardware_encoding: bool = True,
    runner: CommandRunner = _default_runner,
) -> dict[str, object]:
    video = video.resolve()
    mask_root = mask_root.resolve()
    output_root = output_root.resolve()
    propainter_root = propainter_root.resolve()
    if not (propainter_root / "inference_propainter.py").is_file():
        raise FileNotFoundError(f"ProPainter not found: {propainter_root}")
    if min(width, height) <= 0:
        raise ValueError("Output width and height must be positive")
    if subvideo_length < 20:
        raise ValueError("subvideo_length must be at least 20")
    if context_frames < 0 or roi_margin < 0:
        raise ValueError("Context and ROI margin cannot be negative")
    if transition_frames < 0:
        raise ValueError("transition_frames cannot be negative")
    if roi_margin < mask_dilation:
        raise ValueError("roi_margin must be at least mask_dilation")
    if max_clip_frames < 20:
        raise ValueError("max_clip_frames must be at least 20")

    _, _, frame_count, fps = video_metadata(video)
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"Invalid source video metadata: {video}")
    target_size = (width, height)
    reference_area = 1280 * 720
    max_equivalent_frames = (
        subvideo_length * reference_area / (width * height)
    )
    infos = load_mask_infos(
        mask_root,
        frame_count=frame_count,
        target_size=target_size,
    )
    chunks = plan_chunks(
        infos,
        frame_count=frame_count,
        target_size=target_size,
        context_frames=context_frames,
        roi_margin=roi_margin,
        max_equivalent_frames=max_equivalent_frames,
        max_clip_frames=max_clip_frames,
    )
    active_count = sum(info.bbox is not None for info in infos)
    transitions = plan_transitions(
        chunks,
        transition_frames=transition_frames,
    )
    destination_root = output_root / video.stem
    destination = destination_root / "inpaint_out.mp4"
    destination_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Chunked ProPainter: {active_count}/{frame_count} active frames, "
        f"{len(chunks)} chunk(s), canvas={width}x{height}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".propainter_chunks_",
        dir=destination_root,
    ) as temporary_directory:
        workspace = Path(temporary_directory)
        chunk_frame_roots: list[Path] = []
        for chunk_index, chunk in enumerate(chunks):
            left, top, right, bottom = chunk.roi
            print(
                f"Chunk {chunk_index + 1}/{len(chunks)}: "
                f"core=[{chunk.core_start},{chunk.core_end}), "
                f"clip=[{chunk.clip_start},{chunk.clip_end}), "
                f"roi={right - left}x{bottom - top}, "
                f"equivalent_frames={chunk.equivalent_frames:.1f}",
                flush=True,
            )
            chunk_frame_roots.append(
                _run_chunk(
                    chunk_index=chunk_index,
                    chunk=chunk,
                    video=video,
                    infos=infos,
                    target_size=target_size,
                    workspace=workspace,
                    propainter_root=propainter_root,
                    subvideo_length=subvideo_length,
                    neighbor_length=neighbor_length,
                    ref_stride=ref_stride,
                    raft_iter=raft_iter,
                    mask_dilation=mask_dilation,
                    save_fps=max(1, int(round(fps))),
                    fp16=fp16,
                    runner=runner,
                )
            )
        encoding = _stream_final_video(
            video=video,
            infos=infos,
            chunks=chunks,
            chunk_frame_roots=chunk_frame_roots,
            destination=destination,
            target_size=target_size,
            frame_count=frame_count,
            mask_dilation=mask_dilation,
            transition_frames=transition_frames,
            hardware_encoding=hardware_encoding,
        )

    report: dict[str, object] = {
        "source_video": str(video),
        "mask_root": str(mask_root),
        "output_video": str(destination),
        "frame_count": frame_count,
        "active_frame_count": active_count,
        "fps": fps,
        "resolution": [width, height],
        "encoding": encoding,
        "chunk_count": len(chunks),
        "parameters": {
            "subvideo_length": subvideo_length,
            "context_frames": context_frames,
            "roi_margin": roi_margin,
            "max_clip_frames": max_clip_frames,
            "transition_frames": transition_frames,
            "max_1280x720_equivalent_frames": subvideo_length,
            "max_canvas_equivalent_frames": max_equivalent_frames,
            "neighbor_length": neighbor_length,
            "ref_stride": ref_stride,
            "raft_iter": raft_iter,
            "hardware_encoding": hardware_encoding,
            "mask_dilation": mask_dilation,
            "fp16": fp16,
        },
        "chunks": [asdict(chunk) for chunk in chunks],
        "transitions": [asdict(transition) for transition in transitions],
    }
    report_path = destination_root / "chunked_propainter_report.json"
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary_report, report_path)
    print(
        f"Chunked ProPainter completed: {destination}",
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the unmodified ProPainter on temporal ROI chunks"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--propainter-root", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--subvideo-length", type=int, default=80)
    parser.add_argument("--context-frames", type=int, default=20)
    parser.add_argument("--roi-margin", type=int, default=64)
    parser.add_argument("--max-clip-frames", type=int, default=240)
    parser.add_argument("--transition-frames", type=int, default=16)
    parser.add_argument("--neighbor-length", type=int, default=10)
    parser.add_argument("--ref-stride", type=int, default=10)
    parser.add_argument("--raft-iter", type=int, default=20)
    parser.add_argument("--mask-dilation", type=int, default=4)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hardware-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVIDIA NVENC when available (default: true)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_chunked_propainter(
        video=args.video,
        mask_root=args.mask,
        output_root=args.output,
        propainter_root=args.propainter_root,
        width=args.width,
        height=args.height,
        subvideo_length=args.subvideo_length,
        context_frames=args.context_frames,
        roi_margin=args.roi_margin,
        max_clip_frames=args.max_clip_frames,
        transition_frames=args.transition_frames,
        neighbor_length=args.neighbor_length,
        ref_stride=args.ref_stride,
        raft_iter=args.raft_iter,
        mask_dilation=args.mask_dilation,
        fp16=args.fp16,
        hardware_encoding=args.hardware_encoding,
    )


if __name__ == "__main__":
    main()
