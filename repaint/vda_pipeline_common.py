"""Small shared primitives used by both canonical VDA pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any, Sequence


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
        self,
        other: "VideoMetadata",
        *,
        fps_tolerance: float = 0.01,
    ) -> bool:
        return (
            self.width == other.width
            and self.height == other.height
            and self.frame_count == other.frame_count
            and abs(self.fps - other.fps) <= fps_tolerance
        )


def _positive_number(value: object) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0.0:
            return None
        result = float(numerator) / denominator_value
    else:
        result = float(text)
    return result if math.isfinite(result) and result > 0.0 else None


def _positive_int(value: object) -> int | None:
    number = _positive_number(value)
    return None if number is None else int(number)


def probe_video(path: Path) -> VideoMetadata:
    """Probe decoded video geometry and cadence without fixed defaults."""

    video_path = path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    try:
        completed = subprocess.run(
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
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required but was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffprobe failed for {video_path}: {exc.stderr.strip()}"
        ) from exc

    payload = json.loads(completed.stdout)
    streams = [
        stream
        for stream in payload.get("streams", [])
        if stream.get("codec_type", "video") == "video"
    ]
    if len(streams) != 1:
        raise ValueError(
            f"{video_path} must contain exactly one video stream; "
            f"found {len(streams)}"
        )
    stream = streams[0]
    fps = _positive_number(stream.get("avg_frame_rate"))
    if fps is None:
        fps = _positive_number(stream.get("r_frame_rate"))
    frame_count = _positive_int(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _positive_int(stream.get("nb_frames"))
    duration = _positive_number(stream.get("duration"))
    if duration is None:
        duration = _positive_number(payload.get("format", {}).get("duration"))
    if frame_count is None and duration is not None and fps is not None:
        frame_count = int(round(duration * fps))
    width = _positive_int(stream.get("width"))
    height = _positive_int(stream.get("height"))
    if None in (fps, frame_count, width, height):
        raise ValueError(f"Could not probe complete video metadata: {video_path}")
    return VideoMetadata(
        width=int(width),
        height=int(height),
        fps=float(fps),
        frame_count=int(frame_count),
        codec=(
            None
            if stream.get("codec_name") is None
            else str(stream["codec_name"])
        ),
        pixel_format=(
            None
            if stream.get("pix_fmt") is None
            else str(stream["pix_fmt"])
        ),
        duration_seconds=duration,
    )


def natural_key(value: str) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def build_command(
    executable: Path,
    script: Path,
    *arguments: str | Path,
) -> tuple[str, ...]:
    executable = executable.expanduser()
    if not executable.is_absolute():
        executable = Path.cwd() / executable
    return (
        # Resolving a virtualenv launcher can silently select its base Python.
        str(executable.absolute()),
        str(script.resolve()),
        *(str(argument) for argument in arguments),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_previous_stages(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        stages = read_json(path).get("stages")
    except Exception:
        return {}
    if not isinstance(stages, list):
        return {}
    return {
        str(stage["stage_id"]): stage
        for stage in stages
        if isinstance(stage, dict) and "stage_id" in stage
    }


def remove_exact(path: Path, *, owner_root: Path | None = None) -> None:
    if owner_root is not None:
        if owner_root.is_symlink():
            raise ValueError(
                f"Refusing to remove through symlinked owner: {owner_root}"
            )
        owner = owner_root.resolve()
        parent = path.parent.resolve()
        if parent != owner and not parent.is_relative_to(owner):
            raise ValueError(
                f"Refusing to remove path outside {owner}: {path}"
            )
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_stage_process(
    *,
    command: Sequence[str],
    stage_id: str,
    stage_name: str,
    item_key: str,
    item_label: str,
    progress: Any,
    pipeline_started: float,
    poll_seconds: float,
    cwd: Path,
) -> None:
    """Run one pipeline stage and keep its aggregate progress bar current."""

    stage_started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
        start_new_session=True,
    )

    def terminate_process_group() -> None:
        def signal_group(group_signal: signal.Signals) -> None:
            while True:
                try:
                    os.killpg(process.pid, group_signal)
                    return
                except ProcessLookupError:
                    return
                except KeyboardInterrupt:
                    continue

        signal_group(signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            pass
        signal_group(signal.SIGKILL)
        while True:
            try:
                process.wait()
                return
            except KeyboardInterrupt:
                continue

    try:
        while True:
            try:
                return_code = process.wait(timeout=poll_seconds)
            except subprocess.TimeoutExpired:
                return_code = None
            now = time.perf_counter()
            progress.set_postfix(
                stage=stage_name,
                **{
                    item_key: item_label,
                    "stage_elapsed": format_elapsed(now - stage_started),
                    "total_elapsed": format_elapsed(now - pipeline_started),
                    "refresh": True,
                },
            )
            if return_code is not None:
                break
    except BaseException:
        terminate_process_group()
        raise
    if return_code != 0:
        terminate_process_group()
        raise RuntimeError(f"{stage_id} failed with exit code {return_code}")
