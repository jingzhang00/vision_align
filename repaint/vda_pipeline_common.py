"""Shared file, JSON, and video helpers for repaint."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str | None = None
    pixel_format: str | None = None
    duration_seconds: float | None = None

    def compatible_with(
        self, other: "VideoMetadata", *, fps_tolerance: float = 0.01
    ) -> bool:
        return (
            self.width == other.width
            and self.height == other.height
            and self.frame_count == other.frame_count
            and abs(self.fps - other.fps) <= fps_tolerance
        )


def _positive(value: object) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = map(float, text.split("/", 1))
        result = numerator / denominator if denominator else math.nan
    else:
        result = float(text)
    return result if math.isfinite(result) and result > 0.0 else None


def probe_video(path: Path) -> VideoMetadata:
    video = path.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {video}")
    stream = streams[0]
    fps = _positive(stream.get("avg_frame_rate")) or _positive(
        stream.get("r_frame_rate")
    )
    frames = _positive(stream.get("nb_read_frames")) or _positive(
        stream.get("nb_frames")
    )
    duration = _positive(stream.get("duration")) or _positive(
        payload.get("format", {}).get("duration")
    )
    width = _positive(stream.get("width"))
    height = _positive(stream.get("height"))
    if None in (fps, frames, width, height):
        raise ValueError(f"Incomplete video metadata: {video}")
    return VideoMetadata(
        width=int(width),
        height=int(height),
        fps=float(fps),
        frame_count=int(frames),
        codec=stream.get("codec_name"),
        pixel_format=stream.get("pix_fmt"),
        duration_seconds=duration,
    )


def natural_key(value: str) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def render_complete(video: Path) -> bool:
    """A render is complete only after Isaac wrote its same-stem report."""

    try:
        return (
            video.is_file()
            and video.stat().st_size > 0
            and int(read_json(video.with_suffix(".json"))["frame_count"]) > 0
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def remove_exact(path: Path, *, owner_root: Path | None = None) -> None:
    if owner_root is not None:
        owner = owner_root.resolve()
        target = path.resolve()
        if target != owner and not target.is_relative_to(owner):
            raise ValueError(f"Refusing to remove path outside {owner}: {path}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
