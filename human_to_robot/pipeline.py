from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from .lighting_profile import (
    aggregate_task_lighting_profiles,
    estimate_lighting_profile,
)
from .propainter_chunked import DEFAULT_MAX_CHUNK_PIXEL_FRAMES
from .trajectory import (
    FINGERTIP_GRASP_TARGET,
    hdf5_hand_target_availability,
    hdf5_hand_target_mode,
)
from .video_io import video_metadata


_SINGLE_EPISODE_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]"
)


@dataclass(frozen=True)
class Episode:
    task_dir: Path
    task_relative: Path
    task_name: str
    name: str
    video: Path
    hdf5: Path
    output_root: Path


@dataclass
class PreparedEpisode:
    episode: Episode
    frame_count: int
    inpaint_video: Path
    sam_root: Path
    g1_root: Path
    lighting_observation: dict[str, Any]
    prepare_seconds: float
    prepare_generation: int


@dataclass(frozen=True)
class IsaacRuntime:
    command: tuple[str, ...]
    environment: dict[str, str]
    description: str


class EpisodeInterrupted(RuntimeError):
    """A child process stopped before completing one episode."""


class EpisodeSkipped(RuntimeError):
    """An episode cannot produce an output for an expected data reason."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _is_interruption(error: BaseException) -> bool:
    return isinstance(
        error,
        (EpisodeInterrupted, KeyboardInterrupt, SystemExit),
    )


def _is_skip(error: BaseException) -> bool:
    return isinstance(error, EpisodeSkipped)


def _episode_error_status(error: BaseException) -> str:
    if _is_interruption(error):
        return "interrupted"
    if _is_skip(error):
        return "skipped"
    return "failed"


def _natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def discover_episodes(folder: Path, output: Path) -> list[Episode]:
    folder = folder.resolve()
    output = output.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Input folder not found: {folder}")

    pairs: list[tuple[Path, Path]] = []
    unpaired: list[Path] = []
    for video in folder.rglob("*.mp4"):
        hdf5 = video.with_suffix(".hdf5")
        if hdf5.is_file():
            pairs.append((video.resolve(), hdf5.resolve()))
        else:
            unpaired.append(video.resolve())
    if not pairs:
        detail = f"; unpaired videos: {unpaired[:5]}" if unpaired else ""
        raise FileNotFoundError(
            f"No same-name .mp4 + .hdf5 episodes found under {folder}{detail}"
        )

    episodes: list[Episode] = []
    seen_outputs: set[Path] = set()
    for video, hdf5 in pairs:
        task_dir = video.parent
        task_relative = (
            Path(folder.name)
            if task_dir == folder
            else task_dir.relative_to(folder)
        )
        task_output = output / task_relative
        episode_output = task_output / video.stem
        if episode_output in seen_outputs:
            raise ValueError(f"Duplicate episode output path: {episode_output}")
        seen_outputs.add(episode_output)
        episodes.append(
            Episode(
                task_dir=task_dir,
                task_relative=task_relative,
                task_name=task_dir.name,
                name=video.stem,
                video=video,
                hdf5=hdf5,
                output_root=episode_output,
            )
        )

    episodes.sort(
        key=lambda episode: (
            str(episode.task_relative),
            _natural_key(episode.name),
        )
    )
    return episodes


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _progress_timing(
    stage_label: str,
    stage_elapsed: float,
    elapsed_offset: float,
) -> str:
    timing = f"{stage_label}={_format_duration(stage_elapsed)}"
    if elapsed_offset > 0:
        timing += (
            f" total={_format_duration(elapsed_offset + stage_elapsed)}"
        )
    return timing


def _frame_progress_display(
    label: str,
    completed: int,
    total: int,
    rate: float | None,
) -> str:
    completed = min(max(0, completed), total)
    percent = 100.0 * completed / total if total > 0 else 0.0
    status = f"{label}={completed}/{total} ({percent:.1f}%)"
    if rate is None or rate <= 0 or completed >= total:
        return f"{status} eta={'00:00' if completed >= total else 'estimating'}"
    remaining = (total - completed) / rate
    return (
        f"{status} eta={_format_duration(remaining)} "
        f"rate={rate:.1f}fps"
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _file_fingerprint(path: Path, *, content_hash: bool = False) -> dict[str, object]:
    stat = path.stat()
    fingerprint: dict[str, object] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        fingerprint["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprint


def _python_tree_fingerprint(root: Path) -> dict[str, object]:
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Python source files found in {root}")
    digest = hashlib.sha256()
    total_size = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(content)
        total_size += len(content)
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "size": total_size,
        "sha256": digest.hexdigest(),
    }


def _load_pipeline_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"state_version": 1, "stages": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state_version": 1, "stages": {}}
    if state.get("state_version") != 1 or not isinstance(
        state.get("stages"), dict
    ):
        return {"state_version": 1, "stages": {}}
    return state


def _stage_complete(
    state: dict[str, Any],
    stage: str,
    signature: dict[str, object],
    *,
    ignored_inputs: frozenset[str] = frozenset(),
) -> bool:
    record = state.get("stages", {}).get(stage)
    return (
        isinstance(record, dict)
        and record.get("status") == "completed"
        and _signatures_match(
            record.get("signature"),
            signature,
            ignored_inputs=ignored_inputs,
        )
        and isinstance(record.get("generation"), int)
    )


def _signatures_match(
    recorded: object,
    expected: dict[str, object],
    *,
    ignored_inputs: frozenset[str] = frozenset(),
    ignored_top_level: frozenset[str] = frozenset(),
) -> bool:
    if not isinstance(recorded, dict):
        return False

    def normalized(signature: dict[str, object]) -> dict[str, object]:
        result = {
            key: value
            for key, value in signature.items()
            if key not in ignored_top_level
        }
        inputs = signature.get("inputs")
        if isinstance(inputs, dict):
            result["inputs"] = {
                name: fingerprint
                for name, fingerprint in inputs.items()
                if name not in ignored_inputs
            }
        parameters = signature.get("parameters")
        if isinstance(parameters, dict):
            result["parameters"] = dict(parameters)
        task_profile = signature.get("task_profile")
        if isinstance(task_profile, dict) and "sha256" in task_profile:
            stable_profile = dict(task_profile)
            stable_profile.pop("mtime_ns", None)
            result["task_profile"] = stable_profile
        return result

    return normalized(recorded) == normalized(expected)


def _signature_subset_matches(
    recorded: object,
    expected: dict[str, object],
    *,
    input_names: frozenset[str],
    parameter_names: frozenset[str],
) -> bool:
    if not isinstance(recorded, dict):
        return False
    if recorded.get("signature_version") != expected.get("signature_version"):
        return False
    recorded_inputs = recorded.get("inputs")
    expected_inputs = expected.get("inputs")
    recorded_parameters = recorded.get("parameters")
    expected_parameters = expected.get("parameters")
    if not all(
        isinstance(value, dict)
        for value in (
            recorded_inputs,
            expected_inputs,
            recorded_parameters,
            expected_parameters,
        )
    ):
        return False
    return all(
        recorded_inputs.get(name) == expected_inputs.get(name)
        for name in input_names
    ) and all(
        recorded_parameters.get(name) == expected_parameters.get(name)
        for name in parameter_names
    )


def _set_stage_state(
    path: Path,
    state: dict[str, Any],
    *,
    stage: str,
    status: str,
    signature: dict[str, object],
    generation: int | None = None,
    started_at_ns: int | None = None,
    finished_at_ns: int | None = None,
    elapsed_seconds: float | None = None,
    error: str | None = None,
) -> None:
    state["state_version"] = 1
    stages = state.setdefault("stages", {})
    record: dict[str, object] = {
        "status": status,
        "signature": signature,
    }
    if generation is not None:
        record["generation"] = generation
    if started_at_ns is not None:
        record["started_at_ns"] = started_at_ns
    if finished_at_ns is not None:
        record["finished_at_ns"] = finished_at_ns
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = elapsed_seconds
    if error is not None:
        record["error"] = error
    stages[stage] = record
    _write_json(path, state)


def _mark_running_stage_terminal(
    path: Path,
    *,
    stage: str,
    error: BaseException,
    elapsed_seconds: float,
) -> None:
    state = _load_pipeline_state(path)
    record = state.get("stages", {}).get(stage)
    if not isinstance(record, dict) or record.get("status") != "running":
        return
    signature = record.get("signature")
    if not isinstance(signature, dict):
        return
    message = f"{type(error).__name__}: {error}"
    if len(message) > 4000:
        message = f"{message[:3997]}..."
    _set_stage_state(
        path,
        state,
        stage=stage,
        status=_episode_error_status(error),
        signature=signature,
        started_at_ns=(
            int(record["started_at_ns"])
            if isinstance(record.get("started_at_ns"), int)
            else None
        ),
        finished_at_ns=time.time_ns(),
        elapsed_seconds=elapsed_seconds,
        error=message,
    )


def _task_label(episode: Episode) -> str:
    if episode.task_relative == Path("."):
        return episode.task_name
    return episode.task_relative.as_posix()


def _episode_label(episode: Episode) -> str:
    return f"{_task_label(episode)}/{episode.name}"


def _running_episode_report(episode: Episode) -> dict[str, object]:
    report: dict[str, object] = {
        "task": episode.task_name,
        "task_relative": episode.task_relative.as_posix(),
        "episode": episode.name,
        "status": "running",
        "video": str(episode.video),
        "hdf5": str(episode.hdf5),
        "renderer": "isaacsim",
    }
    _write_json(episode.output_root / "episode_report.json", report)
    return report


def _validate_file(path: Path, description: str, minimum_bytes: int = 1) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    if path.stat().st_size < minimum_bytes:
        raise ValueError(
            f"{description} is incomplete ({path.stat().st_size} bytes): {path}"
        )


def _probe_isaacsim_runtime(
    command: tuple[str, ...],
    *,
    environment: dict[str, str],
    cwd: Path,
) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [
                *command,
                "-c",
                (
                    "from isaacsim import SimulationApp; "
                    "print(SimulationApp.__module__)"
                ),
            ],
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"{type(error).__name__}: {error}"
    output = result.stdout.strip()
    return result.returncode == 0, output


def _resolve_isaacsim_runtime(
    args: argparse.Namespace,
    *,
    repo: Path,
) -> IsaacRuntime:
    base_environment = os.environ.copy()
    active_python = str(Path(sys.executable).absolute())
    candidates: list[tuple[str, tuple[str, ...], dict[str, str]]] = []
    requested_root = args.isaacsim_root
    environment_root = os.environ.get("ISAACSIM_ROOT")
    roots: list[tuple[str, Path]] = []
    if requested_root is not None:
        roots.append(("explicit", requested_root.expanduser().resolve()))
    if environment_root:
        roots.append(
            ("ISAACSIM_ROOT", Path(environment_root).expanduser().resolve())
        )
    roots.append(("default", Path.home() / "isaacsim"))

    seen_roots: set[Path] = set()
    for source, root in roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        launcher = root / "python.sh"
        if not launcher.is_file():
            if source == "explicit":
                raise FileNotFoundError(
                    f"Isaac Sim python launcher not found: {launcher}"
                )
            continue
        environment = base_environment.copy()
        environment["PYTHONEXE"] = active_python
        candidates.append(
            (
                f"Isaac Sim standalone at {root}",
                (str(launcher),),
                environment,
            )
        )

    candidates.append(
        (
            f"Python package in {active_python}",
            (active_python,),
            base_environment,
        )
    )
    failures: list[str] = []
    for description, command, environment in candidates:
        usable, output = _probe_isaacsim_runtime(
            command,
            environment=environment,
            cwd=repo,
        )
        if usable:
            return IsaacRuntime(
                command=command,
                environment=environment,
                description=description,
            )
        failures.append(f"{description}: {output or 'probe failed'}")
    raise RuntimeError(
        "No usable Isaac Sim runtime was found. Checked:\n"
        + "\n".join(failures)
    )


def _prepare_stage_signature(
    episode: Episode,
    *,
    args: argparse.Namespace,
    repo: Path,
    sam_checkpoint: Path,
    urdf: Path,
    propainter: Path,
    prompt_frame: int,
    source_size: tuple[int, int],
) -> dict[str, object]:
    input_paths: dict[str, Path] = {
        "video": episode.video,
        "hdf5": episode.hdf5,
        "sam_checkpoint": sam_checkpoint,
        "raft_weight": propainter / "weights/raft-things.pth",
        "flow_weight": propainter / "weights/recurrent_flow_completion.pth",
        "propainter_weight": propainter / "weights/ProPainter.pth",
        "pipeline_code": Path(__file__),
        "video_io_code": repo / "human_to_robot/video_io.py",
        "segment_code": repo / "human_to_robot/segment_sam3.py",
        "lighting_code": repo / "human_to_robot/lighting_profile.py",
        "propainter_engine_code": (
            propainter
            / "web-demos/hugging_face/inpainter/base_inpainter.py"
        ),
        "official_propainter_code": propainter / "inference_propainter.py",
        "chunked_propainter_code": (
            repo / "human_to_robot/propainter_chunked.py"
        ),
        "propainter_worker_code": (
            repo / "human_to_robot/propainter_worker.py"
        ),
        "urdf": urdf,
        "retarget_code": repo / "human_to_robot/retarget_g1_pinocchio.py",
        "trajectory_code": repo / "human_to_robot/trajectory.py",
        "scene_depth_code": repo / "human_to_robot/scene_depth_da3.py",
        "da3_config": repo / "models/DA3METRIC-LARGE/config.json",
        "da3_weights": (
            repo / "models/DA3METRIC-LARGE/model.safetensors"
        ),
        "da3_model_code": (
            repo
            / "Depth-Anything-3/src/depth_anything_3/model/da3.py"
        ),
        "da3_input_processor_code": (
            repo
            / "Depth-Anything-3/src/depth_anything_3/utils/io/"
            "input_processor.py"
        ),
    }
    inputs = {
        name: _file_fingerprint(path)
        for name, path in input_paths.items()
    }
    inputs["da3_source_tree"] = _python_tree_fingerprint(
        repo / "Depth-Anything-3/src/depth_anything_3"
    )
    return {
        "signature_version": 2,
        "inputs": inputs,
        "parameters": {
            "python": str(Path(sys.executable).resolve()),
            "debug_artifacts": args.debug,
            "hardware_encoding": args.hardware_encoding,
            "source_resolution": list(source_size),
            "propainter_mode": args.propainter_mode,
            "propainter_fp16": args.propainter_fp16,
            "propainter_subvideo_length": args.propainter_subvideo_length,
            "propainter_context_frames": args.propainter_context_frames,
            "propainter_roi_margin": args.propainter_roi_margin,
            "propainter_max_chunk_pixel_frames": (
                args.propainter_max_chunk_pixel_frames
            ),
            "propainter_max_clip_frames": args.propainter_max_clip_frames,
            "propainter_transition_frames": args.propainter_transition_frames,
            "sam_threshold": args.sam_threshold,
            "arm_prompt": args.arm_prompt,
            "arm_prompt_frame": prompt_frame,
            "lighting_sample_count": args.lighting_sample_count,
            "lighting_max_dimension": args.lighting_max_dimension,
        },
    }


def _finish_stage_signature(
    prepared: PreparedEpisode,
    *,
    args: argparse.Namespace,
    repo: Path,
    urdf: Path,
    task_profile_path: Path,
) -> dict[str, object]:
    input_paths = {
        "source_video": prepared.episode.video,
        "hdf5": prepared.episode.hdf5,
        "inpaint_video": prepared.inpaint_video,
        "trajectory": prepared.g1_root / "g1_trajectory.npz",
        "urdf": urdf,
        "scene_depth_report": (
            prepared.episode.output_root
            / "depth_anything3/scene_depth_report.json"
        ),
        "composite_code": repo / "human_to_robot/composite.py",
        "pipeline_code": Path(__file__),
        "video_io_code": repo / "human_to_robot/video_io.py",
        "isaac_renderer_code": repo / "human_to_robot/render_isaacsim.py",
    }
    return {
        "signature_version": 2,
        "prepare_generation": prepared.prepare_generation,
        "inputs": {
            name: _file_fingerprint(path)
            for name, path in input_paths.items()
        },
        "task_profile": _file_fingerprint(
            task_profile_path, content_hash=True
        ),
        "parameters": {
            "debug_artifacts": args.debug,
            "hardware_encoding": args.hardware_encoding,
            "visualize": args.visualize,
        },
    }


def _image_sequence_complete(
    directory: Path,
    expected_count: int,
    *,
    expected_size: tuple[int, int] | None = None,
) -> bool:
    if expected_count <= 0 or not directory.is_dir():
        return False
    first = directory / "00000.png"
    last = directory / f"{expected_count - 1:05d}.png"
    if (
        not first.is_file()
        or not last.is_file()
        or len(list(directory.glob("*.png"))) != expected_count
    ):
        return False
    if expected_size is not None:
        image = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[1::-1] != expected_size:
            return False
    return True


def _scene_depth_complete(
    directory: Path,
    report_path: Path,
    expected_count: int,
    *,
    expected_size: tuple[int, int],
    scene_video: Path,
) -> bool:
    if expected_count <= 0 or not directory.is_dir() or not report_path.is_file():
        return False
    paths = sorted(directory.glob("*.png"))
    expected_names = [
        f"{frame_id:05d}.png" for frame_id in range(expected_count)
    ]
    if [path.name for path in paths] != expected_names:
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        scene_stat = scene_video.stat()
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        report.get("schema_version") == 2
        and report.get("model") == "DA3METRIC-LARGE"
        and report.get("native_metric") is True
        and report.get("frame_count") == expected_count
        and report.get("source_resolution") == list(expected_size)
        and report.get("scene_video") == str(scene_video.resolve())
        and report.get("scene_video_fingerprint")
        == {
            "size": scene_stat.st_size,
            "mtime_ns": scene_stat.st_mtime_ns,
        }
    ):
        return False
    width, height = expected_size
    digest = hashlib.sha256()
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if (
            image is None
            or image.dtype != np.uint16
            or image.shape != (height, width)
        ):
            return False
        digest.update(path.name.encode("ascii"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            return False
    return digest.hexdigest() == report.get("depth_png_sha256")


def _video_complete(
    path: Path,
    expected_count: int,
    *,
    expected_size: tuple[int, int] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        width, height, count, _ = video_metadata(path)
    except (FileNotFoundError, ValueError):
        return False
    return count == expected_count and (
        expected_size is None or (width, height) == expected_size
    )


def _trajectory_complete(path: Path, expected_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as archive:
            return archive["qpos"].shape[0] == expected_count
    except (OSError, KeyError, ValueError):
        return False


def _auto_arm_prompt_frame(hdf5_path: Path, frame_count: int) -> int:
    with h5py.File(hdf5_path, "r") as handle:
        confidences = {}
        for side in ("left", "right"):
            key = f"confidences/{side}Hand"
            confidences[side] = (
                np.asarray(handle[key], dtype=np.float32)
                if key in handle
                else np.ones(frame_count, dtype=np.float32)
            )
        left = confidences["left"]
        right = confidences["right"]
    if left.shape != (frame_count,) or right.shape != (frame_count,):
        return frame_count // 2
    both = np.flatnonzero((left >= 0.5) & (right >= 0.5))
    if both.size:
        return int(both[np.argmin(np.abs(both - frame_count // 2))])
    return int(np.argmax(np.nan_to_num(left) + np.nan_to_num(right)))


def _log_tail(path: Path, line_count: int = 30) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_tree(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float,
) -> None:
    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process_group):
            break
        time.sleep(0.1)
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    progress: tqdm[Any],
    current_episode: str,
    environment: dict[str, str] | None = None,
    episode_started: float | None = None,
    elapsed_offset: float = 0.0,
    stage_label: str = "stage",
    frame_progress_pattern: str | None = None,
    frame_progress_label: str = "frames",
) -> float:
    started = time.perf_counter()
    active_started = episode_started if episode_started is not None else started
    progress_pattern = (
        re.compile(frame_progress_pattern)
        if frame_progress_pattern is not None
        else None
    )
    progress_baseline: tuple[int, float] | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        command_log_offset = log_path.stat().st_size
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name == "posix",
        )
        try:
            while process.poll() is None:
                now = time.perf_counter()
                stage_elapsed = now - active_started
                frame_status = (
                    f"{frame_progress_label}=starting eta=estimating"
                    if progress_pattern is not None
                    else ""
                )
                if progress_pattern is not None:
                    try:
                        with log_path.open("rb") as reader:
                            reader.seek(command_log_offset)
                            command_output = reader.read().decode(
                                "utf-8", errors="replace"
                            )
                    except OSError:
                        command_output = ""
                    matches = list(progress_pattern.finditer(command_output))
                    if matches:
                        completed = int(matches[-1].group(1))
                        total = int(matches[-1].group(2))
                        if progress_baseline is None:
                            progress_baseline = (completed, now)
                        baseline_count, baseline_time = progress_baseline
                        sample_seconds = now - baseline_time
                        sample_count = completed - baseline_count
                        rate = (
                            sample_count / sample_seconds
                            if sample_count > 0 and sample_seconds > 0
                            else None
                        )
                        frame_status = _frame_progress_display(
                            frame_progress_label,
                            completed,
                            total,
                            rate,
                        )
                timing = _progress_timing(
                    stage_label,
                    stage_elapsed,
                    elapsed_offset,
                )
                progress.set_postfix_str(
                    " ".join(
                        part
                        for part in (
                            f"current={current_episode}",
                            timing,
                            frame_status,
                        )
                        if part
                    ),
                    refresh=True,
                )
                time.sleep(1.0)
        except BaseException:
            _terminate_process_tree(process, grace_seconds=10.0)
            raise
        return_code = int(process.returncode or 0)
    elapsed = time.perf_counter() - started
    if return_code:
        _terminate_process_tree(process, grace_seconds=2.0)
        tail = _log_tail(log_path)
        error_type = (
            EpisodeInterrupted
            if return_code
            in {
                -signal.SIGINT,
                -signal.SIGTERM,
                128 + signal.SIGINT,
                128 + signal.SIGTERM,
            }
            else RuntimeError
        )
        outcome = (
            "was interrupted"
            if error_type is EpisodeInterrupted
            else "failed"
        )
        raise error_type(
            f"Episode {current_episode} {outcome}; "
            f"log: {log_path}\n{tail}"
        )
    return elapsed


def _task_output_root(base_output: Path, episode: Episode) -> Path:
    return (
        base_output
        if episode.task_relative == Path(".")
        else base_output / episode.task_relative
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError(f"Refusing to overwrite source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_episode_outputs(
    episode: Episode,
    *,
    synthetic_video: Path,
    output: Path,
) -> tuple[Path, Path]:
    task_output = _task_output_root(output.resolve(), episode)
    published_video = task_output / f"{episode.name}.mp4"
    published_hdf5 = task_output / f"{episode.name}.hdf5"
    sources_and_destinations = (
        (synthetic_video, published_video),
        (episode.hdf5, published_hdf5),
    )
    for source, destination in sources_and_destinations:
        if source.resolve() == destination.resolve():
            raise ValueError(f"Refusing to overwrite source file: {source}")
    for source, destination in sources_and_destinations:
        _atomic_copy(source, destination)
    return published_video, published_hdf5


def _propainter_invocation(
    *,
    args: argparse.Namespace,
    python: str,
    video: Path,
    mask_root: Path,
    output_root: Path,
    propainter: Path,
    repo: Path,
) -> tuple[list[str], Path]:
    if args.propainter_mode == "official":
        command = [
            python,
            str(propainter / "inference_propainter.py"),
            "--video",
            str(video),
            "--mask",
            str(mask_root),
            "--output",
            str(output_root),
            "--subvideo_length",
            str(args.propainter_subvideo_length),
        ]
        if args.propainter_fp16:
            command.append("--fp16")
        return command, propainter

    command = [
        python,
        "-m",
        "human_to_robot.propainter_chunked",
        "--video",
        str(video),
        "--mask",
        str(mask_root),
        "--output",
        str(output_root),
        "--propainter-root",
        str(propainter),
        "--subvideo-length",
        str(args.propainter_subvideo_length),
        "--context-frames",
        str(args.propainter_context_frames),
        "--roi-margin",
        str(args.propainter_roi_margin),
        "--max-chunk-pixel-frames",
        str(args.propainter_max_chunk_pixel_frames),
        "--max-clip-frames",
        str(args.propainter_max_clip_frames),
        "--transition-frames",
        str(args.propainter_transition_frames),
        "--fp16" if args.propainter_fp16 else "--no-fp16",
        (
            "--hardware-encoding"
            if args.hardware_encoding
            else "--no-hardware-encoding"
        ),
    ]
    return command, repo


def _prepare_episode(
    episode: Episode,
    *,
    args: argparse.Namespace,
    repo: Path,
    sam_checkpoint: Path,
    urdf: Path,
    propainter: Path,
    progress: tqdm[Any],
) -> PreparedEpisode:
    started = time.perf_counter()
    stage_started_at_ns = time.time_ns()
    output = episode.output_root
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "pipeline.log"
    _running_episode_report(episode)
    source_width, source_height, frame_count, source_fps = video_metadata(
        episode.video
    )
    if min(source_width, source_height, frame_count) <= 0 or source_fps <= 0:
        raise ValueError(f"Invalid source video metadata: {episode.video}")
    source_size = (source_width, source_height)
    hand_target_mode = hdf5_hand_target_mode(episode.hdf5)
    hand_targets = hdf5_hand_target_availability(episode.hdf5)
    missing_hands = [
        side for side, usable in hand_targets.items() if not usable
    ]
    if missing_hands:
        required = (
            "annotated wrist/thumb/index/middle 3D keypoints"
            if hand_target_mode == FINGERTIP_GRASP_TARGET
            else "explicit leftHand/rightHand rigid palm poses"
        )
        raise ValueError(
            f"Episode requires {required} for both hands; unusable: "
            f"{', '.join(missing_hands)} ({episode.hdf5})"
        )
    episode_label = _episode_label(episode)

    sam_root = output / "sam3"
    g1_root = output / "g1"
    propainter_root = output / "propainter"
    python = sys.executable
    prompt_frame = (
        args.arm_prompt_frame
        if args.arm_prompt_frame is not None
        else _auto_arm_prompt_frame(episode.hdf5, frame_count)
    )
    if not 0 <= prompt_frame < frame_count:
        raise ValueError(
            f"Arm prompt frame {prompt_frame} is outside episode "
            f"{episode.task_name}/{episode.name}"
        )
    state_path = output / "pipeline_state.json"
    prepare_signature = _prepare_stage_signature(
        episode,
        args=args,
        repo=repo,
        sam_checkpoint=sam_checkpoint,
        urdf=urdf,
        propainter=propainter,
        prompt_frame=prompt_frame,
        source_size=source_size,
    )
    pipeline_state = _load_pipeline_state(state_path)
    previous_prepare = pipeline_state.get("stages", {}).get("prepare", {})
    previous_generation = (
        previous_prepare.get("generation")
        if isinstance(previous_prepare, dict)
        else None
    )
    previous_signature = (
        previous_prepare.get("signature")
        if isinstance(previous_prepare, dict)
        else None
    )
    reuse_existing_prepare = (
        args.reuse
        and isinstance(previous_prepare, dict)
        and previous_prepare.get("status") == "completed"
        and isinstance(previous_generation, int)
    )
    _set_stage_state(
        state_path,
        pipeline_state,
        stage="prepare",
        status="running",
        signature=prepare_signature,
        started_at_ns=stage_started_at_ns,
    )
    prepare_changed = False

    arm_masks_ready = _image_sequence_complete(
        sam_root / "arm_masks", frame_count
    )
    reuse_sam = (
        reuse_existing_prepare
        and arm_masks_ready
        and _signature_subset_matches(
            previous_signature,
            prepare_signature,
            input_names=frozenset(
                {"video", "sam_checkpoint", "segment_code"}
            ),
            parameter_names=frozenset(
                {
                    "python",
                    "debug_artifacts",
                    "sam_threshold",
                    "arm_prompt",
                    "arm_prompt_frame",
                }
            ),
        )
    )
    sam_changed = not reuse_sam
    if sam_changed:
        prepare_changed = True
        sam_command = [
            python,
            "-m",
            "human_to_robot.segment_sam3",
            "--video",
            str(episode.video),
            "--checkpoint",
            str(sam_checkpoint),
            "--output",
            str(sam_root),
            "--arm-prompt",
            args.arm_prompt,
            "--arm-prompt-frame",
            str(prompt_frame),
            "--threshold",
            str(args.sam_threshold),
            (
                "--hardware-encoding"
                if args.hardware_encoding
                else "--no-hardware-encoding"
            ),
        ]
        if args.debug:
            sam_command.append("--debug-artifacts")
        _run_logged(
            sam_command,
            cwd=repo,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
            stage_label="prepare",
        )
        if not _image_sequence_complete(sam_root / "arm_masks", frame_count):
            raise RuntimeError(f"SAM3 arm masks are incomplete: {sam_root}")

    inpaint_video = propainter_root / episode.video.stem / "inpaint_out.mp4"
    if args.propainter_mode == "official":
        propainter_signature_inputs = frozenset(
            {
                "video",
                "raft_weight",
                "flow_weight",
                "propainter_weight",
                "official_propainter_code",
            }
        )
        propainter_signature_parameters = frozenset(
            {
                "python",
                "propainter_mode",
                "propainter_fp16",
                "propainter_subvideo_length",
            }
        )
    else:
        propainter_signature_inputs = frozenset(
            {
                "video",
                "raft_weight",
                "flow_weight",
                "propainter_weight",
                "video_io_code",
                "propainter_engine_code",
                "chunked_propainter_code",
                "propainter_worker_code",
            }
        )
        propainter_signature_parameters = frozenset(
            {
                "python",
                "hardware_encoding",
                "propainter_mode",
                "propainter_fp16",
                "source_resolution",
                "propainter_subvideo_length",
                "propainter_context_frames",
                "propainter_roi_margin",
                "propainter_max_chunk_pixel_frames",
                "propainter_max_clip_frames",
                "propainter_transition_frames",
            }
        )
    reuse_inpaint = (
        reuse_existing_prepare
        and not sam_changed
        and _video_complete(
            inpaint_video,
            frame_count,
            expected_size=source_size,
        )
        and _signature_subset_matches(
            previous_signature,
            prepare_signature,
            input_names=propainter_signature_inputs,
            parameter_names=propainter_signature_parameters,
        )
    )
    if not reuse_inpaint:
        prepare_changed = True
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", "/tmp/vision_align_matplotlib")
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        propainter_command, propainter_cwd = _propainter_invocation(
            args=args,
            python=python,
            video=episode.video,
            mask_root=sam_root / "arm_masks",
            output_root=propainter_root,
            propainter=propainter,
            repo=repo,
        )
        _run_logged(
            propainter_command,
            cwd=propainter_cwd,
            environment=environment,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
            stage_label="prepare",
        )
        if not _video_complete(
            inpaint_video,
            frame_count,
            expected_size=source_size,
        ):
            raise RuntimeError(
                f"ProPainter output is incomplete: {inpaint_video}"
            )

    inpaint_width, inpaint_height, inpaint_count, _ = video_metadata(
        inpaint_video
    )
    if (inpaint_width, inpaint_height, inpaint_count) != (
        source_width,
        source_height,
        frame_count,
    ):
        raise ValueError(
            f"ProPainter changed source video geometry for {episode.video}: "
            f"{(source_width, source_height, frame_count)} -> "
            f"{(inpaint_width, inpaint_height, inpaint_count)}"
        )
    trajectory_ready = _trajectory_complete(
        g1_root / "g1_trajectory.npz", frame_count
    )
    trajectory_signature_inputs = {
        "video",
        "hdf5",
        "urdf",
        "retarget_code",
        "trajectory_code",
    }
    reuse_trajectory = (
        reuse_existing_prepare
        and trajectory_ready
        and _signature_subset_matches(
            previous_signature,
            prepare_signature,
            input_names=frozenset(trajectory_signature_inputs),
            parameter_names=frozenset({"python"}),
        )
    )
    if not reuse_trajectory:
        prepare_changed = True
        skip_report_path = g1_root / "episode_skip.json"
        skip_report_path.unlink(missing_ok=True)
        retarget_command = [
            python,
            "-m",
            "human_to_robot.retarget_g1_pinocchio",
            "--video",
            str(episode.video),
            "--hdf5",
            str(episode.hdf5),
            "--urdf",
            str(urdf),
            "--output",
            str(g1_root),
        ]
        _run_logged(
            retarget_command,
            cwd=repo,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
            stage_label="prepare",
        )
        if skip_report_path.is_file():
            try:
                skip_report = json.loads(
                    skip_report_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"Invalid retarget skip report: {skip_report_path}"
                ) from error
            if (
                not isinstance(skip_report, dict)
                or skip_report.get("status") != "skipped"
            ):
                raise RuntimeError(
                    f"Invalid retarget skip report: {skip_report_path}"
                )
            reason_code = skip_report.get("reason_code")
            reason = skip_report.get("reason")
            if not isinstance(reason_code, str) or not isinstance(reason, str):
                raise RuntimeError(
                    f"Incomplete retarget skip report: {skip_report_path}"
                )
            raise EpisodeSkipped(reason, reason_code=reason_code)
        if not _trajectory_complete(
            g1_root / "g1_trajectory.npz", frame_count
        ):
            raise RuntimeError(f"G1 trajectory is incomplete: {g1_root}")

    scene_depth_root = output / "depth_anything3"
    scene_depth_frames = scene_depth_root / "scene_depth_mm"
    scene_depth_report = scene_depth_root / "scene_depth_report.json"
    scene_depth_ready = _scene_depth_complete(
        scene_depth_frames,
        scene_depth_report,
        frame_count,
        expected_size=source_size,
        scene_video=inpaint_video,
    )
    scene_depth_signature_inputs = trajectory_signature_inputs | {
        "scene_depth_code",
        "da3_config",
        "da3_weights",
        "da3_model_code",
        "da3_input_processor_code",
        "da3_source_tree",
    }
    reuse_scene_depth = (
        reuse_existing_prepare
        and reuse_inpaint
        and reuse_trajectory
        and scene_depth_ready
        and _signature_subset_matches(
            previous_signature,
            prepare_signature,
            input_names=frozenset(scene_depth_signature_inputs),
            parameter_names=frozenset(
                {
                    "python",
                    "source_resolution",
                }
            ),
        )
    )
    if not reuse_scene_depth:
        prepare_changed = True
        environment = os.environ.copy()
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        _run_logged(
            [
                python,
                "-m",
                "human_to_robot.scene_depth_da3",
                "--video",
                str(inpaint_video),
                "--trajectory",
                str(g1_root / "g1_trajectory.npz"),
                "--output",
                str(scene_depth_root),
            ],
            cwd=repo,
            environment=environment,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
            stage_label="prepare",
            frame_progress_pattern=(
                r"Depth Anything 3 processed (\d+)/(\d+)"
            ),
            frame_progress_label="depth",
        )
        if not _scene_depth_complete(
            scene_depth_frames,
            scene_depth_report,
            frame_count,
            expected_size=source_size,
            scene_video=inpaint_video,
        ):
            raise RuntimeError(
                f"Depth Anything 3 output is incomplete: {scene_depth_root}"
            )

    observation = estimate_lighting_profile(
        inpaint_video=inpaint_video,
        reliability_mask_root=sam_root / "arm_masks",
        sample_count=args.lighting_sample_count,
        max_dimension=args.lighting_max_dimension,
        write_artifacts=False,
    )
    prepare_generation = (
        int(previous_generation)
        if reuse_existing_prepare
        and not prepare_changed
        and isinstance(previous_generation, int)
        else time.time_ns()
    )
    _set_stage_state(
        state_path,
        pipeline_state,
        stage="prepare",
        status="completed",
        signature=prepare_signature,
        generation=prepare_generation,
        started_at_ns=stage_started_at_ns,
        finished_at_ns=time.time_ns(),
        elapsed_seconds=time.perf_counter() - started,
    )
    return PreparedEpisode(
        episode=episode,
        frame_count=frame_count,
        inpaint_video=inpaint_video,
        sam_root=sam_root,
        g1_root=g1_root,
        lighting_observation=observation,
        prepare_seconds=time.perf_counter() - started,
        prepare_generation=prepare_generation,
    )


def _render_episode(
    prepared: PreparedEpisode,
    *,
    args: argparse.Namespace,
    repo: Path,
    urdf: Path,
    isaac_runtime: IsaacRuntime | None,
    task_profile_path: Path,
    progress: tqdm[Any],
) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    episode = prepared.episode
    log_path = episode.output_root / "pipeline.log"
    episode_label = _episode_label(episode)
    render_width, render_height, source_count, _ = video_metadata(
        episode.video
    )
    inpaint_width, inpaint_height, inpaint_count, _ = video_metadata(
        prepared.inpaint_video
    )
    if (inpaint_width, inpaint_height, inpaint_count) != (
        render_width,
        render_height,
        source_count,
    ):
        raise ValueError(
            "Inpaint video must preserve the source resolution and frame "
            f"count: source={(render_width, render_height, source_count)}, "
            f"inpaint={(inpaint_width, inpaint_height, inpaint_count)}"
        )
    render_size = (render_width, render_height)
    render_root = episode.output_root / "g1_isaacsim"
    trajectory_path = prepared.g1_root / "g1_trajectory.npz"
    scene_depth_frames = (
        episode.output_root / "depth_anything3/scene_depth_mm"
    )
    final_root = episode.output_root / "final"
    final_video = final_root / "synthetic_g1.mp4"
    composite_report = final_root / "composite_report.json"
    finish_signature = _finish_stage_signature(
        prepared,
        args=args,
        repo=repo,
        urdf=urdf,
        task_profile_path=task_profile_path,
    )
    finish_stage = "finish"
    state_path = episode.output_root / "pipeline_state.json"
    pipeline_state = _load_pipeline_state(state_path)
    stages = pipeline_state.get("stages", {})
    previous_finish = stages.get(finish_stage, {})
    if not isinstance(previous_finish, dict):
        previous_finish = {}
    if not previous_finish:
        legacy_finish = stages.get("finish_isaacsim", {})
        if isinstance(legacy_finish, dict):
            previous_finish = legacy_finish
    previous_finish_generation = (
        previous_finish.get("generation")
        if isinstance(previous_finish, dict)
        else None
    )

    render_sequences_complete = all(
        _image_sequence_complete(
            render_root / directory,
            prepared.frame_count,
            expected_size=render_size,
        )
        for directory in ("robot_rgb", "robot_masks", "robot_depth_mm")
    )
    isaac_report = render_root / "isaacsim_render_report.json"
    final_complete = (
        _video_complete(
            final_video,
            prepared.frame_count,
            expected_size=render_size,
        )
        and composite_report.is_file()
    )
    render_complete = isaac_report.is_file() and (
        render_sequences_complete if args.debug else final_complete
    )
    reuse_render = (
        not args.visualize
        and args.debug
        and render_sequences_complete
        and isaac_report.is_file()
        and _signatures_match(
            (
                previous_finish.get("signature")
                if isinstance(previous_finish, dict)
                else None
            ),
            finish_signature,
            ignored_inputs=frozenset(
                {
                    "composite_code",
                    "inpaint_video",
                    "pipeline_code",
                    "video_io_code",
                    "scene_depth_report",
                }
            ),
            ignored_top_level=frozenset({"prepare_generation"}),
        )
    )
    reuse_finish = (
        not args.visualize
        and args.reuse
        and _stage_complete(pipeline_state, finish_stage, finish_signature)
        and render_complete
        and final_complete
    )

    if not reuse_finish:
        stage_started_at_ns = time.time_ns()
        stage_started = time.perf_counter()
        _set_stage_state(
            state_path,
            pipeline_state,
            stage=finish_stage,
            status="running",
            signature=finish_signature,
            started_at_ns=stage_started_at_ns,
        )
        if not reuse_render:
            if isaac_runtime is None:
                raise RuntimeError("Isaac Sim runtime was not initialized")
            isaac_command = [
                *isaac_runtime.command,
                "-m",
                "human_to_robot.render_isaacsim",
                "--trajectory",
                str(trajectory_path),
                "--urdf",
                str(urdf),
                "--video",
                str(episode.video),
                "--hdf5",
                str(episode.hdf5),
                "--lighting-profile",
                str(task_profile_path),
                "--scene-depths",
                str(scene_depth_frames),
                "--output",
                str(render_root),
                (
                    "--hardware-encoding"
                    if args.hardware_encoding
                    else "--no-hardware-encoding"
                ),
            ]
            if args.debug:
                isaac_command.append("--debug-artifacts")
            else:
                isaac_command.extend(
                    [
                        "--inpaint-video",
                        str(prepared.inpaint_video),
                        "--composite-output",
                        str(final_root),
                    ]
                )
            if args.visualize:
                isaac_command.append("--no-headless")
            _run_logged(
                isaac_command,
                cwd=repo,
                environment=isaac_runtime.environment,
                log_path=log_path,
                progress=progress,
                current_episode=episode_label,
                episode_started=started,
                elapsed_offset=prepared.prepare_seconds,
                stage_label="finish",
                frame_progress_pattern=(
                    r"Isaac Sim rendered (\d+)/(\d+)"
                ),
                frame_progress_label="render",
            )
            rendered_output_complete = (
                all(
                    _image_sequence_complete(
                        render_root / directory,
                        prepared.frame_count,
                        expected_size=render_size,
                    )
                    for directory in (
                        "robot_rgb",
                        "robot_masks",
                        "robot_depth_mm",
                    )
                )
                if args.debug
                else (
                    _video_complete(
                        final_video,
                        prepared.frame_count,
                        expected_size=render_size,
                    )
                    and composite_report.is_file()
                )
            )
            if not rendered_output_complete or not isaac_report.is_file():
                raise RuntimeError(
                    "Isaac Sim render/composite is incomplete: "
                    f"{render_root}"
                )

        if args.debug:
            composite_command = [
                sys.executable,
                "-m",
                "human_to_robot.composite",
                "--source-video",
                str(episode.video),
                "--inpaint-video",
                str(prepared.inpaint_video),
                "--g1-root",
                str(render_root),
                "--trajectory",
                str(trajectory_path),
                "--scene-depths",
                str(scene_depth_frames),
                "--lighting-profile",
                str(task_profile_path),
                "--output",
                str(final_root),
                (
                    "--hardware-encoding"
                    if args.hardware_encoding
                    else "--no-hardware-encoding"
                ),
                "--debug-artifacts",
            ]
            _run_logged(
                composite_command,
                cwd=repo,
                log_path=log_path,
                progress=progress,
                current_episode=episode_label,
                episode_started=started,
                elapsed_offset=prepared.prepare_seconds,
                stage_label="finish",
                frame_progress_pattern=r"Composite processed (\d+)/(\d+)",
                frame_progress_label="composite",
            )
        if not _video_complete(
            final_video,
            prepared.frame_count,
            expected_size=render_size,
        ) or not composite_report.is_file():
            raise RuntimeError(f"Composite output is incomplete: {final_root}")
    if reuse_finish:
        stage_started_at_ns = time.time_ns()
        stage_started = time.perf_counter()
        _set_stage_state(
            state_path,
            pipeline_state,
            stage=finish_stage,
            status="running",
            signature=finish_signature,
            started_at_ns=stage_started_at_ns,
        )
    published_video, published_hdf5 = _publish_episode_outputs(
        episode,
        synthetic_video=final_video,
        output=args.output,
    )
    finish_generation = (
        int(previous_finish_generation)
        if reuse_finish and isinstance(previous_finish_generation, int)
        else time.time_ns()
    )
    _set_stage_state(
        state_path,
        pipeline_state,
        stage=finish_stage,
        status="completed",
        signature=finish_signature,
        generation=finish_generation,
        started_at_ns=stage_started_at_ns,
        finished_at_ns=time.time_ns(),
        elapsed_seconds=time.perf_counter() - stage_started,
    )
    elapsed = time.perf_counter() - started
    artifacts: dict[str, object] = {
        "task": episode.task_name,
        "task_relative": episode.task_relative.as_posix(),
        "episode": episode.name,
        "status": "completed",
        "video": str(episode.video),
        "source_hdf5": str(episode.hdf5),
        "hdf5": str(published_hdf5),
        "renderer": "isaacsim",
        "synthetic_video": str(published_video),
        "elapsed_seconds": prepared.prepare_seconds + elapsed,
    }
    if args.debug:
        artifacts.update(
            {
                "task_lighting_profile": str(task_profile_path),
                "comparison_video": str(
                    final_root / "comparison_original_inpaint_g1.mp4"
                ),
                "inpaint_video": str(prepared.inpaint_video),
                "trajectory": str(
                    prepared.g1_root / "g1_trajectory.npz"
                ),
                "scene_depth_report": str(
                    episode.output_root
                    / "depth_anything3/scene_depth_report.json"
                ),
                "scene_depth_frames": str(
                    episode.output_root / "depth_anything3/scene_depth_mm"
                ),
            }
        )
    _write_json(episode.output_root / "episode_report.json", artifacts)
    return artifacts, elapsed


def _episode_error_report(
    episode: Episode,
    *,
    elapsed_seconds: float,
    error: BaseException,
) -> dict[str, object]:
    report: dict[str, object] = {
        "task": episode.task_name,
        "task_relative": episode.task_relative.as_posix(),
        "episode": episode.name,
        "status": _episode_error_status(error),
        "episode_path": str(episode.hdf5),
        "video": str(episode.video),
        "hdf5": str(episode.hdf5),
        "renderer": "isaacsim",
        "elapsed_seconds": elapsed_seconds,
        "log": str(episode.output_root / "pipeline.log"),
    }
    if isinstance(error, EpisodeSkipped):
        report["reason_code"] = error.reason_code
        report["reason"] = str(error)
    else:
        report["error"] = f"{type(error).__name__}: {error}"
    _write_json(episode.output_root / "episode_report.json", report)
    return report


def _persist_failure_diagnostics(
    episode: Episode,
    *,
    output: Path,
    report: dict[str, object],
) -> dict[str, object]:
    failure_root = (
        _task_output_root(output.resolve(), episode)
        / ".failures"
        / episode.name
    )
    failure_root.mkdir(parents=True, exist_ok=True)
    source_log = episode.output_root / "pipeline.log"
    published_log = failure_root / "pipeline.log"
    if source_log.is_file():
        _atomic_copy(source_log, published_log)
    else:
        published_log.write_text(
            f"{report.get('error', 'Episode failed')}\n",
            encoding="utf-8",
        )
    persisted = dict(report)
    persisted["log"] = str(published_log)
    report_path = failure_root / "episode_report.json"
    _write_json(report_path, persisted)
    tqdm.write(f"Failure diagnostics saved: {report_path}")
    return persisted


def _persist_skip_diagnostics(
    episode: Episode,
    *,
    output: Path,
    report: dict[str, object],
) -> dict[str, object]:
    skip_root = (
        _task_output_root(output.resolve(), episode)
        / ".skipped"
        / episode.name
    )
    skip_root.mkdir(parents=True, exist_ok=True)
    source_log = episode.output_root / "pipeline.log"
    published_log = skip_root / "pipeline.log"
    if source_log.is_file():
        _atomic_copy(source_log, published_log)
    else:
        published_log.write_text(
            f"{report.get('reason', 'Episode skipped')}\n",
            encoding="utf-8",
        )
    persisted = dict(report)
    persisted["log"] = str(published_log)
    report_path = skip_root / "episode_report.json"
    _write_json(report_path, persisted)
    tqdm.write(f"Skipped episode recorded: {report_path}")
    return persisted


def _record_episode_error(
    episode: Episode,
    *,
    elapsed_seconds: float,
    error: BaseException,
    output: Path,
    debug: bool,
) -> dict[str, object]:
    report = _episode_error_report(
        episode,
        elapsed_seconds=elapsed_seconds,
        error=error,
    )
    if debug:
        return report
    if _is_skip(error):
        return _persist_skip_diagnostics(
            episode,
            output=output,
            report=report,
        )
    return _persist_failure_diagnostics(
        episode,
        output=output,
        report=report,
    )


def _discard_episode_workspace(episode: Episode, *, debug: bool) -> None:
    if debug or not episode.output_root.exists():
        return
    try:
        shutil.rmtree(episode.output_root)
    except OSError as error:
        tqdm.write(
            f"Could not remove temporary data for {_episode_label(episode)}: "
            f"{type(error).__name__}: {error}"
        )


def _print_episode_error(
    episode: Episode,
    *,
    elapsed_seconds: float,
    error: BaseException,
    interrupted: bool = False,
) -> None:
    if interrupted:
        outcome = "Interrupted"
    elif _is_skip(error):
        outcome = "Skipped"
    else:
        outcome = "Failed"
    tqdm.write(
        f"{outcome} {_episode_label(episode)} after "
        f"{_format_duration(elapsed_seconds)}:\n"
        f"{type(error).__name__}: {error}"
    )


def _make_batch_report(
    *,
    folder: Path,
    output: Path,
    discovered_episode_count: int,
    discovered_task_count: int,
    episode_reports: list[dict[str, object]],
    elapsed_seconds: float,
    status: str,
    error: BaseException | None = None,
) -> dict[str, object]:
    completed = sum(
        report.get("status") == "completed" for report in episode_reports
    )
    failed = sum(
        report.get("status") in {"failed", "interrupted"}
        for report in episode_reports
    )
    skipped = sum(
        report.get("status") == "skipped" for report in episode_reports
    )
    skipped_paths = [
        str(report["episode_path"])
        for report in episode_reports
        if report.get("status") == "skipped"
        and isinstance(report.get("episode_path"), str)
    ]
    report: dict[str, object] = {
        "input_folder": str(folder),
        "output_root": str(output),
        "status": status,
        "renderer": "isaacsim",
        "task_count": discovered_task_count,
        "episode_count": discovered_episode_count,
        "completed_episode_count": completed,
        "skipped_episode_count": skipped,
        "skipped_episode_paths": skipped_paths,
        "failed_episode_count": failed,
        "pending_episode_count": max(
            0, discovered_episode_count - completed - skipped - failed
        ),
        "elapsed_seconds": elapsed_seconds,
        "episodes": episode_reports,
    }
    if error is not None:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def _run_pipeline(
    args: argparse.Namespace,
    *,
    folder: Path,
    output: Path,
    workspace_output: Path,
) -> dict[str, object]:
    batch_started = time.perf_counter()
    repo = Path(__file__).resolve().parent.parent
    sam_checkpoint = (repo / "models/sam3/sam3.pt").resolve()
    urdf = (
        repo
        / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
    ).resolve()
    propainter = (repo / "ProPainter").resolve()
    da3_model = (repo / "models/DA3METRIC-LARGE").resolve()
    da3_source = (repo / "Depth-Anything-3/src/depth_anything_3").resolve()
    isaac_runtime: IsaacRuntime | None = None
    episodes: list[Episode] = []
    task_groups: dict[Path, list[Episode]] = {}
    _write_json(
        workspace_output / "batch_report.json",
        _make_batch_report(
            folder=folder,
            output=output,
            discovered_episode_count=0,
            discovered_task_count=0,
            episode_reports=[],
            elapsed_seconds=0.0,
            status="running",
        ),
    )
    try:
        _validate_file(sam_checkpoint, "SAM3 checkpoint", 3_000_000_000)
        _validate_file(urdf, "G1 URDF")
        _validate_file(
            da3_model / "config.json",
            "DA3METRIC-LARGE config",
        )
        _validate_file(
            da3_model / "model.safetensors",
            "DA3METRIC-LARGE weights",
            1_000_000_000,
        )
        if not da3_source.is_dir():
            raise FileNotFoundError(
                f"Depth Anything 3 submodule source not found: {da3_source}"
            )
        isaac_runtime = _resolve_isaacsim_runtime(args, repo=repo)
        tqdm.write(f"Isaac runtime: {isaac_runtime.description}")
        if args.propainter_subvideo_length < 20:
            raise ValueError("--propainter-subvideo-length must be at least 20")
        if args.propainter_mode == "chunked":
            if args.propainter_context_frames < 0:
                raise ValueError(
                    "--propainter-context-frames cannot be negative"
                )
            if args.propainter_max_chunk_pixel_frames <= 0:
                raise ValueError(
                    "--propainter-max-chunk-pixel-frames must be positive"
                )
            if args.propainter_roi_margin < 0:
                raise ValueError("--propainter-roi-margin cannot be negative")
            if args.propainter_roi_margin < 4:
                raise ValueError(
                    "--propainter-roi-margin must be at least the 4-pixel "
                    "inpaint mask dilation"
                )
            if args.propainter_transition_frames < 0:
                raise ValueError(
                    "--propainter-transition-frames cannot be negative"
                )
            if args.propainter_max_clip_frames < 20:
                raise ValueError(
                    "--propainter-max-clip-frames must be at least 20"
                )
        for name in (
            "raft-things.pth",
            "recurrent_flow_completion.pth",
            "ProPainter.pth",
        ):
            _validate_file(
                propainter / "weights" / name, f"ProPainter weight {name}"
            )

        episodes = discover_episodes(folder, workspace_output)
        for episode in episodes:
            task_groups.setdefault(episode.task_relative, []).append(episode)
    except BaseException as error:
        initialization_report = _make_batch_report(
            folder=folder,
            output=output,
            discovered_episode_count=len(episodes),
            discovered_task_count=len(task_groups),
            episode_reports=[],
            elapsed_seconds=time.perf_counter() - batch_started,
            status=(
                "interrupted"
                if _is_interruption(error)
                else "failed"
            ),
            error=error,
        )
        _write_json(
            workspace_output / "batch_report.json", initialization_report
        )
        raise

    tqdm.write(
        f"Discovered {len(episodes)} episodes in {len(task_groups)} task(s); "
        "renderer=isaacsim"
    )
    batch_episode_reports: dict[
        tuple[Path, str],
        dict[str, object],
    ] = {}
    try:
        for task_episodes in task_groups.values():
            task_started = time.perf_counter()
            task_label = _task_label(task_episodes[0])
            task_root = _task_output_root(workspace_output, task_episodes[0])
            task_episode_reports: dict[str, dict[str, object]] = {}
            prepared_episodes: list[PreparedEpisode] = []
            task_profile_path: Path | None = None
            try:
                with tqdm(
                    total=len(task_episodes),
                    desc=f"{task_label} prepare",
                    unit="episode",
                    dynamic_ncols=True,
                    bar_format=(
                        _SINGLE_EPISODE_BAR_FORMAT
                        if len(task_episodes) == 1
                        else None
                    ),
                ) as progress:
                    for episode in task_episodes:
                        episode_started = time.perf_counter()
                        try:
                            prepared = _prepare_episode(
                                episode,
                                args=args,
                                repo=repo,
                                sam_checkpoint=sam_checkpoint,
                                urdf=urdf,
                                propainter=propainter,
                                progress=progress,
                            )
                            prepared_episodes.append(prepared)
                            duration = prepared.prepare_seconds
                        except (KeyboardInterrupt, SystemExit) as error:
                            duration = time.perf_counter() - episode_started
                            _mark_running_stage_terminal(
                                episode.output_root / "pipeline_state.json",
                                stage="prepare",
                                error=error,
                                elapsed_seconds=duration,
                            )
                            _print_episode_error(
                                episode,
                                elapsed_seconds=duration,
                                error=error,
                                interrupted=True,
                            )
                            _discard_episode_workspace(
                                episode,
                                debug=args.debug,
                            )
                            raise
                        except Exception as error:
                            duration = time.perf_counter() - episode_started
                            _mark_running_stage_terminal(
                                episode.output_root / "pipeline_state.json",
                                stage="prepare",
                                error=error,
                                elapsed_seconds=duration,
                            )
                            task_episode_reports[episode.name] = (
                                _record_episode_error(
                                    episode,
                                    elapsed_seconds=duration,
                                    error=error,
                                    output=args.output,
                                    debug=args.debug,
                                )
                            )
                            _print_episode_error(
                                episode,
                                elapsed_seconds=duration,
                                error=error,
                                interrupted=_is_interruption(error),
                            )
                            _discard_episode_workspace(
                                episode,
                                debug=args.debug,
                            )
                        progress.update(1)
                        progress.set_postfix_str(
                            f"last={_episode_label(episode)} "
                            f"prepare={_format_duration(duration)}",
                            refresh=True,
                        )

                if prepared_episodes:
                    task_profile_path = task_root / "lighting_profile.json"
                    try:
                        aggregate_task_lighting_profiles(
                            task_name=task_label,
                            episode_profiles=[
                                prepared.lighting_observation
                                for prepared in prepared_episodes
                            ],
                            output_path=task_profile_path,
                        )
                    except Exception as error:
                        for prepared in prepared_episodes:
                            episode = prepared.episode
                            task_episode_reports[episode.name] = (
                                _record_episode_error(
                                    episode,
                                    elapsed_seconds=prepared.prepare_seconds,
                                    error=error,
                                    output=args.output,
                                    debug=args.debug,
                                )
                            )
                            _print_episode_error(
                                episode,
                                elapsed_seconds=prepared.prepare_seconds,
                                error=error,
                            )
                            _discard_episode_workspace(
                                episode,
                                debug=args.debug,
                            )
                        prepared_episodes = []
                        task_profile_path = None

                if prepared_episodes and task_profile_path is not None:
                    with tqdm(
                        total=len(prepared_episodes),
                        desc=f"{task_label} finish (isaacsim)",
                        unit="episode",
                        dynamic_ncols=True,
                        bar_format=(
                            _SINGLE_EPISODE_BAR_FORMAT
                            if len(prepared_episodes) == 1
                            else None
                        ),
                    ) as progress:
                        for prepared in prepared_episodes:
                            episode = prepared.episode
                            finish_started = time.perf_counter()
                            try:
                                report, _ = _render_episode(
                                    prepared,
                                    args=args,
                                    repo=repo,
                                    urdf=urdf,
                                    isaac_runtime=isaac_runtime,
                                    task_profile_path=task_profile_path,
                                    progress=progress,
                                )
                                task_episode_reports[episode.name] = report
                                duration = float(report["elapsed_seconds"])
                                finish_duration = max(
                                    0.0,
                                    duration - prepared.prepare_seconds,
                                )
                                tqdm.write(
                                    f"Completed {_episode_label(episode)} in "
                                    f"{_format_duration(duration)} "
                                    f"(prepare="
                                    f"{_format_duration(prepared.prepare_seconds)}, "
                                    f"finish={_format_duration(finish_duration)})"
                                )
                            except (KeyboardInterrupt, SystemExit) as error:
                                duration = (
                                    prepared.prepare_seconds
                                    + time.perf_counter()
                                    - finish_started
                                )
                                finish_duration = (
                                    time.perf_counter() - finish_started
                                )
                                _mark_running_stage_terminal(
                                    episode.output_root
                                    / "pipeline_state.json",
                                    stage="finish",
                                    error=error,
                                    elapsed_seconds=(
                                        time.perf_counter() - finish_started
                                    ),
                                )
                                _print_episode_error(
                                    episode,
                                    elapsed_seconds=duration,
                                    error=error,
                                    interrupted=True,
                                )
                                raise
                            except Exception as error:
                                duration = (
                                    prepared.prepare_seconds
                                    + time.perf_counter()
                                    - finish_started
                                )
                                finish_duration = (
                                    time.perf_counter() - finish_started
                                )
                                _mark_running_stage_terminal(
                                    episode.output_root
                                    / "pipeline_state.json",
                                    stage="finish",
                                    error=error,
                                    elapsed_seconds=(
                                        time.perf_counter() - finish_started
                                    ),
                                )
                                task_episode_reports[episode.name] = (
                                    _record_episode_error(
                                        episode,
                                        elapsed_seconds=duration,
                                        error=error,
                                        output=args.output,
                                        debug=args.debug,
                                    )
                                )
                                _print_episode_error(
                                    episode,
                                    elapsed_seconds=duration,
                                    error=error,
                                    interrupted=_is_interruption(error),
                                )
                            finally:
                                _discard_episode_workspace(
                                    episode,
                                    debug=args.debug,
                                )
                            progress.update(1)
                            progress.set_postfix_str(
                                f"last={_episode_label(episode)} "
                                f"finish={_format_duration(finish_duration)} "
                                f"total={_format_duration(duration)}",
                                refresh=True,
                            )
            except BaseException as error:
                for episode in task_episodes:
                    if episode.name in task_episode_reports:
                        batch_episode_reports[
                            (episode.task_relative, episode.name)
                        ] = task_episode_reports[episode.name]
                raise

            ordered_reports = [
                task_episode_reports[episode.name]
                for episode in task_episodes
                if episode.name in task_episode_reports
            ]
            for episode in task_episodes:
                if episode.name in task_episode_reports:
                    batch_episode_reports[
                        (episode.task_relative, episode.name)
                    ] = task_episode_reports[episode.name]
            task_seconds = time.perf_counter() - task_started
            task_completed = sum(
                report.get("status") == "completed"
                for report in ordered_reports
            )
            task_skipped = sum(
                report.get("status") == "skipped"
                for report in ordered_reports
            )
            task_status = (
                "completed"
                if task_completed + task_skipped == len(task_episodes)
                else "failed"
            )
            tqdm.write(
                f"Task {task_label} {task_status}: "
                f"{task_completed} completed, {task_skipped} skipped, "
                f"{len(task_episodes)} total, "
                f"elapsed {_format_duration(task_seconds)}"
            )
            partial_batch_reports = [
                batch_episode_reports[
                    (episode.task_relative, episode.name)
                ]
                for episode in episodes
                if (episode.task_relative, episode.name)
                in batch_episode_reports
            ]
            _write_json(
                workspace_output / "batch_report.json",
                _make_batch_report(
                    folder=folder,
                    output=output,
                    discovered_episode_count=len(episodes),
                    discovered_task_count=len(task_groups),
                    episode_reports=partial_batch_reports,
                    elapsed_seconds=time.perf_counter() - batch_started,
                    status="running",
                ),
            )
    except BaseException as error:
        batch_seconds = time.perf_counter() - batch_started
        ordered_batch_reports = [
            batch_episode_reports[(episode.task_relative, episode.name)]
            for episode in episodes
            if (episode.task_relative, episode.name)
            in batch_episode_reports
        ]
        batch_report = _make_batch_report(
            folder=folder,
            output=output,
            discovered_episode_count=len(episodes),
            discovered_task_count=len(task_groups),
            episode_reports=ordered_batch_reports,
            elapsed_seconds=batch_seconds,
            status=(
                "interrupted" if _is_interruption(error) else "failed"
            ),
            error=error,
        )
        _write_json(workspace_output / "batch_report.json", batch_report)
        raise

    batch_seconds = time.perf_counter() - batch_started
    ordered_batch_reports = [
        batch_episode_reports[(episode.task_relative, episode.name)]
        for episode in episodes
        if (episode.task_relative, episode.name) in batch_episode_reports
    ]
    batch_status = (
        "completed"
        if len(ordered_batch_reports) == len(episodes)
        and all(
            report.get("status") in {"completed", "skipped"}
            for report in ordered_batch_reports
        )
        else "failed"
    )
    batch_report = _make_batch_report(
        folder=folder,
        output=output,
        discovered_episode_count=len(episodes),
        discovered_task_count=len(task_groups),
        episode_reports=ordered_batch_reports,
        elapsed_seconds=batch_seconds,
        status=batch_status,
    )
    _write_json(workspace_output / "batch_report.json", batch_report)
    tqdm.write(
        f"Batch {batch_status}: "
        f"{batch_report['completed_episode_count']} completed, "
        f"{batch_report['skipped_episode_count']} skipped, "
        f"{len(episodes)} total, "
        f"{len(task_groups)} tasks, "
        f"elapsed {_format_duration(batch_seconds)}"
    )
    return batch_report


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    folder = args.folder.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if getattr(args, "debug", False):
        return _run_pipeline(
            args,
            folder=folder,
            output=output,
            workspace_output=output,
        )
    with tempfile.TemporaryDirectory(
        prefix=".human_to_robot-",
        dir=output,
    ) as temporary:
        return _run_pipeline(
            args,
            folder=folder,
            output=output,
            workspace_output=Path(temporary),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize G1-aligned videos from paired MP4 and HDF5 episodes"
        )
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help=(
            "Task folder or dataset root; recursively pairs same-name "
            ".mp4 and .hdf5 episodes"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Keep masks, inpainted video, render frames, logs, and JSON "
            "reports; by default only each synthetic MP4 and the unchanged "
            "source HDF5 copy remain"
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Open the Isaac Sim window while rendering; always executes the "
            "render stage instead of reusing cached output"
        ),
    )
    parser.add_argument(
        "--isaacsim-root",
        type=Path,
        help=(
            "Isaac Sim standalone directory containing python.sh; default "
            "checks ISAACSIM_ROOT, ~/isaacsim, then the active Python package"
        ),
    )
    parser.add_argument(
        "--hardware-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use NVIDIA NVENC for generated videos when available; "
            "automatically fall back to software encoding (default: true)"
        ),
    )
    parser.add_argument(
        "--propainter-mode",
        choices=("official", "chunked"),
        default="official",
        help=(
            "Use the official full-frame whole-video ProPainter entry point "
            "(default), or the memory-bounded ROI chunk wrapper"
        ),
    )
    parser.add_argument(
        "--propainter-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use half precision after official RAFT inference (default: true)",
    )
    parser.add_argument(
        "--propainter-subvideo-length",
        type=int,
        default=80,
        help="Internal ProPainter memory-efficient inference window",
    )
    parser.add_argument(
        "--propainter-context-frames",
        type=int,
        default=10,
        help="Temporal context kept on each side of an inpaint chunk",
    )
    parser.add_argument(
        "--propainter-roi-margin",
        type=int,
        default=64,
        help="Target-canvas pixels added around each chunk's mask union",
    )
    parser.add_argument(
        "--propainter-max-chunk-pixel-frames",
        type=int,
        default=DEFAULT_MAX_CHUNK_PIXEL_FRAMES,
        help=(
            "Maximum clip-frame count multiplied by cropped ROI pixels; "
            "source and ROI resolution are accounted directly"
        ),
    )
    parser.add_argument(
        "--propainter-max-clip-frames",
        type=int,
        default=240,
        help="Hard temporal cap per cropped ProPainter invocation",
    )
    parser.add_argument(
        "--propainter-transition-frames",
        type=int,
        default=16,
        help="Frames blended across adjacent ProPainter chunk boundaries",
    )
    parser.add_argument("--sam-threshold", type=float, default=0.4)
    parser.add_argument("--arm-prompt", default="human arm")
    parser.add_argument(
        "--arm-prompt-frame",
        type=int,
        help="Optional fixed frame for every episode; default selects from HDF5",
    )

    parser.add_argument("--lighting-sample-count", type=int, default=16)
    parser.add_argument("--lighting-max-dimension", type=int, default=384)
    parser.add_argument(
        "--reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse complete episode intermediates when --debug keeps them "
            "(default: true)"
        ),
    )
    return parser


def main() -> None:
    report = run_pipeline(build_parser().parse_args())
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
