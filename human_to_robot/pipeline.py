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
from .trajectory import hdf5_action_keypoint_availability
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


def _is_interruption(error: BaseException) -> bool:
    return isinstance(
        error,
        (EpisodeInterrupted, KeyboardInterrupt, SystemExit),
    )


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
) -> bool:
    if not isinstance(recorded, dict):
        return False

    def normalized(signature: dict[str, object]) -> dict[str, object]:
        result = dict(signature)
        inputs = signature.get("inputs")
        if isinstance(inputs, dict):
            result["inputs"] = {
                name: fingerprint
                for name, fingerprint in inputs.items()
                if name not in ignored_inputs
            }
        parameters = signature.get("parameters")
        if isinstance(parameters, dict):
            stable_parameters = dict(parameters)
            if stable_parameters.get("backend") == "isaacsim":
                stable_parameters.pop("backend")
            result["parameters"] = stable_parameters
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
        status=(
            "interrupted"
            if _is_interruption(error)
            else "failed"
        ),
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
    object_prompt_frame: int | None,
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
        "propainter_code": propainter / "inference_propainter.py",
        "chunked_propainter_code": (
            repo / "human_to_robot/propainter_chunked.py"
        ),
        "urdf": urdf,
        "retarget_code": repo / "human_to_robot/retarget_g1_pinocchio.py",
    }
    return {
        "signature_version": 1,
        "inputs": {
            name: _file_fingerprint(path)
            for name, path in input_paths.items()
        },
        "parameters": {
            "python": str(Path(sys.executable).resolve()),
            "debug_artifacts": args.debug,
            "hardware_encoding": args.hardware_encoding,
            "width": args.width,
            "height": args.height,
            "propainter_subvideo_length": args.propainter_subvideo_length,
            "propainter_context_frames": args.propainter_context_frames,
            "propainter_roi_margin": args.propainter_roi_margin,
            "propainter_max_clip_frames": args.propainter_max_clip_frames,
            "propainter_transition_frames": args.propainter_transition_frames,
            "sam_threshold": args.sam_threshold,
            "arm_prompt": args.arm_prompt,
            "arm_prompt_frame": prompt_frame,
            "object_prompt": args.object_prompt,
            "object_prompt_frame": object_prompt_frame,
            "lighting_sample_count": args.lighting_sample_count,
            "lighting_max_dimension": args.lighting_max_dimension,
        },
    }


def _finish_stage_signature(
    prepared: PreparedEpisode,
    *,
    args: argparse.Namespace,
    repo: Path,
    task_profile_path: Path,
) -> dict[str, object]:
    input_paths = {
        "source_video": prepared.episode.video,
        "hdf5": prepared.episode.hdf5,
        "inpaint_video": prepared.inpaint_video,
        "trajectory": prepared.g1_root / "g1_trajectory.npz",
        "composite_code": repo / "human_to_robot/composite.py",
        "pipeline_code": Path(__file__),
        "video_io_code": repo / "human_to_robot/video_io.py",
        "isaac_renderer_code": repo / "human_to_robot/render_isaacsim.py",
    }
    return {
        "signature_version": 1,
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
            "object_prompt_enabled": args.object_prompt is not None,
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


def _mano_keypoints_complete(path: Path, expected_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as archive:
            return all(
                archive[f"{side}_keypoints_camera"].shape
                == (expected_count, 21, 3)
                and archive[f"{side}_confidence"].shape
                in {(expected_count,), (expected_count, 21)}
                for side in ("left", "right")
            )
    except (OSError, KeyError, ValueError):
        return False


def _auto_arm_prompt_frame(hdf5_path: Path, frame_count: int) -> int:
    with h5py.File(hdf5_path, "r") as handle:
        left = np.asarray(handle["confidences/leftHand"], dtype=np.float32)
        right = np.asarray(handle["confidences/rightHand"], dtype=np.float32)
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
    _, _, frame_count, _ = video_metadata(episode.video)
    if frame_count <= 0:
        raise ValueError(f"Episode has no video frames: {episode.video}")
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
    object_prompt_frame = (
        None
        if args.object_prompt is None
        else (
            args.object_prompt_frame
            if args.object_prompt_frame is not None
            else prompt_frame
        )
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
        object_prompt_frame=object_prompt_frame,
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
    object_masks_ready = (
        args.object_prompt is None
        or _image_sequence_complete(sam_root / "object_masks", frame_count)
    )
    reuse_sam = (
        reuse_existing_prepare
        and arm_masks_ready
        and object_masks_ready
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
                    "object_prompt",
                    "object_prompt_frame",
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
        if args.object_prompt is not None:
            if object_prompt_frame is None:
                raise RuntimeError("Object prompt frame was not resolved")
            sam_command.extend(
                [
                    "--object-prompt",
                    args.object_prompt,
                    "--object-prompt-frame",
                    str(object_prompt_frame),
                ]
            )
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
        if args.object_prompt is not None and not _image_sequence_complete(
            sam_root / "object_masks", frame_count
        ):
            raise RuntimeError(f"SAM3 object masks are incomplete: {sam_root}")

    inpaint_video = propainter_root / episode.video.stem / "inpaint_out.mp4"
    reuse_inpaint = (
        reuse_existing_prepare
        and not sam_changed
        and _video_complete(inpaint_video, frame_count)
        and _signature_subset_matches(
            previous_signature,
            prepare_signature,
            input_names=frozenset(
                {
                    "video",
                    "raft_weight",
                    "flow_weight",
                    "propainter_weight",
                    "video_io_code",
                    "propainter_code",
                    "chunked_propainter_code",
                }
            ),
            parameter_names=frozenset(
                {
                    "python",
                    "hardware_encoding",
                    "width",
                    "height",
                    "propainter_subvideo_length",
                    "propainter_context_frames",
                    "propainter_roi_margin",
                    "propainter_max_clip_frames",
                    "propainter_transition_frames",
                }
            ),
        )
    )
    if not reuse_inpaint:
        prepare_changed = True
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", "/tmp/vision_align_matplotlib")
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        _run_logged(
            [
                python,
                "-m",
                "human_to_robot.propainter_chunked",
                "--video",
                str(episode.video),
                "--mask",
                str(sam_root / "arm_masks"),
                "--output",
                str(propainter_root),
                "--propainter-root",
                str(propainter),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--subvideo-length",
                str(args.propainter_subvideo_length),
                "--context-frames",
                str(args.propainter_context_frames),
                "--roi-margin",
                str(args.propainter_roi_margin),
                "--max-clip-frames",
                str(args.propainter_max_clip_frames),
                "--transition-frames",
                str(args.propainter_transition_frames),
                (
                    "--hardware-encoding"
                    if args.hardware_encoding
                    else "--no-hardware-encoding"
                ),
            ],
            cwd=repo,
            environment=environment,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
            stage_label="prepare",
        )
        if not _video_complete(inpaint_video, frame_count):
            raise RuntimeError(
                f"ProPainter output is incomplete: {inpaint_video}"
            )

    _, _, inpaint_count, _ = video_metadata(inpaint_video)
    if inpaint_count != frame_count:
        raise ValueError(
            f"ProPainter frame count changed for {episode.video}: "
            f"{frame_count} -> {inpaint_count}"
        )
    hand_keypoints = hdf5_action_keypoint_availability(episode.hdf5)
    needs_mano = not all(hand_keypoints.values())
    mano_keypoints = g1_root / "mano_keypoints.npz"
    if needs_mano and not _mano_keypoints_complete(
        mano_keypoints, frame_count
    ):
        mano_root = (
            args.mano_root.expanduser().resolve()
            if args.mano_root is not None
            else (repo / "mano_v1_2").resolve()
        )
        wilor_weights = (
            args.wilor_weights.expanduser().resolve()
            if args.wilor_weights is not None
            else (repo / "models/wilor").resolve()
        )
        _validate_file(
            mano_root / "models/MANO_RIGHT.pkl",
            "MANO right-hand model",
        )
        estimator_environment = os.environ.copy()
        estimator_environment.setdefault(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1"
        )
        _run_logged(
            [
                python,
                "-m",
                "human_to_robot.estimate_mano_wilor",
                "--video",
                str(episode.video),
                "--mano-root",
                str(mano_root),
                "--weights",
                str(wilor_weights),
                "--output",
                str(mano_keypoints),
            ],
            cwd=repo,
            environment=estimator_environment,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
            stage_label="prepare",
            frame_progress_pattern=r"MANO estimated (\d+)/(\d+)",
            frame_progress_label="mano",
        )
        if not _mano_keypoints_complete(mano_keypoints, frame_count):
            missing_sides = [
                side for side, available in hand_keypoints.items() if not available
            ]
            raise RuntimeError(
                "MANO hand estimation is incomplete for "
                f"{', '.join(missing_sides)}: {mano_keypoints}"
            )
    trajectory_ready = _trajectory_complete(
        g1_root / "g1_trajectory.npz", frame_count
    )
    reuse_trajectory = (
        reuse_existing_prepare
        and trajectory_ready
        and _signature_subset_matches(
            previous_signature,
            prepare_signature,
            input_names=frozenset(
                {"video", "hdf5", "urdf", "retarget_code"}
            ),
            parameter_names=frozenset({"python"}),
        )
    )
    if not reuse_trajectory:
        prepare_changed = True
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
        if needs_mano:
            retarget_command.extend(
                ["--mano-keypoints", str(mano_keypoints)]
            )
        _run_logged(
            retarget_command,
            cwd=repo,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
            stage_label="prepare",
        )
        if not _trajectory_complete(
            g1_root / "g1_trajectory.npz", frame_count
        ):
            raise RuntimeError(f"G1 trajectory is incomplete: {g1_root}")

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
    render_width, render_height, _, _ = video_metadata(prepared.inpaint_video)
    render_size = (render_width, render_height)
    render_root = episode.output_root / "g1_isaacsim"
    trajectory_path = prepared.g1_root / "g1_trajectory.npz"
    final_root = episode.output_root / "final"
    final_video = final_root / "synthetic_g1.mp4"
    composite_report = final_root / "composite_report.json"
    finish_signature = _finish_stage_signature(
        prepared,
        args=args,
        repo=repo,
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

    render_complete = all(
        _image_sequence_complete(
            render_root / directory,
            prepared.frame_count,
            expected_size=render_size,
        )
        for directory in ("robot_rgb", "robot_masks", "robot_depth_mm")
    )
    isaac_report = render_root / "isaacsim_render_report.json"
    render_complete = render_complete and isaac_report.is_file()
    reuse_render = render_complete and _signatures_match(
        (
            previous_finish.get("signature")
            if isinstance(previous_finish, dict)
            else None
        ),
        finish_signature,
        ignored_inputs=frozenset(
            {"composite_code", "pipeline_code", "video_io_code"}
        ),
    )
    final_complete = (
        _video_complete(
            final_video,
            prepared.frame_count,
            expected_size=render_size,
        )
        and composite_report.is_file()
    )
    reuse_finish = (
        args.reuse
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
                "--output",
                str(render_root),
                "--width",
                str(render_width),
                "--height",
                str(render_height),
                (
                    "--hardware-encoding"
                    if args.hardware_encoding
                    else "--no-hardware-encoding"
                ),
            ]
            if args.debug:
                isaac_command.append("--debug-artifacts")
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
            if not all(
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
            ) or not isaac_report.is_file():
                raise RuntimeError(
                    f"Isaac Sim render is incomplete: {render_root}"
                )

        composite_command = [
            sys.executable,
            "-m",
            "human_to_robot.composite",
            "--source-video",
            str(episode.video),
            "--inpaint-video",
            str(prepared.inpaint_video),
            "--hdf5",
            str(episode.hdf5),
            "--g1-root",
            str(render_root),
            "--trajectory",
            str(trajectory_path),
            "--lighting-profile",
            str(task_profile_path),
            "--output",
            str(final_root),
            (
                "--hardware-encoding"
                if args.hardware_encoding
                else "--no-hardware-encoding"
            ),
        ]
        object_masks = prepared.sam_root / "object_masks"
        if args.object_prompt is not None and _image_sequence_complete(
            object_masks, prepared.frame_count
        ):
            composite_command.extend(["--object-masks", str(object_masks)])
        if args.debug:
            composite_command.append("--debug-artifacts")
        else:
            composite_command.append("--delete-input-frames")
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
        "task_lighting_profile": str(task_profile_path),
        "synthetic_video": str(published_video),
        "comparison_video": (
            str(final_root / "comparison_original_inpaint_g1.mp4")
            if args.debug
            else None
        ),
        "inpaint_video": str(prepared.inpaint_video),
        "trajectory": str(prepared.g1_root / "g1_trajectory.npz"),
        "elapsed_seconds": prepared.prepare_seconds + elapsed,
    }
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
        "status": "interrupted" if _is_interruption(error) else "failed",
        "video": str(episode.video),
        "hdf5": str(episode.hdf5),
        "renderer": "isaacsim",
        "elapsed_seconds": elapsed_seconds,
        "error": f"{type(error).__name__}: {error}",
        "log": str(episode.output_root / "pipeline.log"),
    }
    _write_json(episode.output_root / "episode_report.json", report)
    return report


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
    outcome = "Interrupted" if interrupted else "Failed"
    tqdm.write(
        f"{outcome} {_episode_label(episode)} after "
        f"{_format_duration(elapsed_seconds)}:\n"
        f"{type(error).__name__}: {error}"
    )


def _make_task_report(
    *,
    task_episodes: list[Episode],
    episode_reports: list[dict[str, object]],
    elapsed_seconds: float,
    lighting_profile_path: Path | None,
    lighting_profile: dict[str, Any] | None,
    status: str | None = None,
    error: BaseException | None = None,
) -> dict[str, object]:
    completed = sum(
        report.get("status") == "completed" for report in episode_reports
    )
    failed = sum(
        report.get("status") in {"failed", "interrupted"}
        for report in episode_reports
    )
    pending = max(0, len(task_episodes) - completed - failed)
    if status is None:
        status = "completed" if failed == 0 and pending == 0 else "failed"
    profile_summary: dict[str, object] | None = None
    if lighting_profile is not None:
        profile_summary = {
            "episode_count": lighting_profile["episode_count"],
            "confidence": lighting_profile["confidence"],
            "warnings": lighting_profile["warnings"],
            "isaac": lighting_profile["isaac"],
        }
    report: dict[str, object] = {
        "task": task_episodes[0].task_name,
        "task_relative": task_episodes[0].task_relative.as_posix(),
        "source_directory": str(task_episodes[0].task_dir),
        "status": status,
        "renderer": "isaacsim",
        "episode_count": len(task_episodes),
        "completed_episode_count": completed,
        "failed_episode_count": failed,
        "pending_episode_count": pending,
        "elapsed_seconds": elapsed_seconds,
        "episode_active_seconds": float(
            sum(float(item["elapsed_seconds"]) for item in episode_reports)
        ),
        "lighting_profile": (
            str(lighting_profile_path)
            if lighting_profile_path is not None
            and lighting_profile is not None
            else None
        ),
        "lighting_profile_summary": profile_summary,
        "episodes": episode_reports,
    }
    if error is not None:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def _make_batch_report(
    *,
    folder: Path,
    output: Path,
    discovered_episode_count: int,
    discovered_task_count: int,
    task_reports: list[dict[str, object]],
    elapsed_seconds: float,
    status: str,
    error: BaseException | None = None,
) -> dict[str, object]:
    completed = sum(
        int(report["completed_episode_count"]) for report in task_reports
    )
    failed = sum(int(report["failed_episode_count"]) for report in task_reports)
    report: dict[str, object] = {
        "input_folder": str(folder),
        "output_root": str(output),
        "status": status,
        "renderer": "isaacsim",
        "task_count": discovered_task_count,
        "reported_task_count": len(task_reports),
        "episode_count": discovered_episode_count,
        "completed_episode_count": completed,
        "failed_episode_count": failed,
        "pending_episode_count": max(
            0, discovered_episode_count - completed - failed
        ),
        "elapsed_seconds": elapsed_seconds,
        "tasks": task_reports,
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
            task_reports=[],
            elapsed_seconds=0.0,
            status="running",
        ),
    )
    try:
        _validate_file(sam_checkpoint, "SAM3 checkpoint", 3_000_000_000)
        _validate_file(urdf, "G1 URDF")
        isaac_runtime = _resolve_isaacsim_runtime(args, repo=repo)
        tqdm.write(f"Isaac runtime: {isaac_runtime.description}")
        if args.width <= 0 or args.height <= 0:
            raise ValueError("--width and --height must be positive")
        if args.propainter_subvideo_length < 20:
            raise ValueError("--propainter-subvideo-length must be at least 20")
        if args.propainter_context_frames < 0:
            raise ValueError("--propainter-context-frames cannot be negative")
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
            task_reports=[],
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
    task_reports: list[dict[str, object]] = []
    try:
        for task_relative, task_episodes in task_groups.items():
            task_started = time.perf_counter()
            task_label = _task_label(task_episodes[0])
            task_root = _task_output_root(workspace_output, task_episodes[0])
            task_episode_reports: dict[str, dict[str, object]] = {}
            prepared_episodes: list[PreparedEpisode] = []
            task_profile_path: Path | None = None
            task_profile: dict[str, Any] | None = None
            _write_json(
                task_root / "task_report.json",
                _make_task_report(
                    task_episodes=task_episodes,
                    episode_reports=[],
                    elapsed_seconds=0.0,
                    lighting_profile_path=None,
                    lighting_profile=None,
                    status="running",
                ),
            )
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
                                _episode_error_report(
                                    episode,
                                    elapsed_seconds=duration,
                                    error=error,
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
                        task_profile = aggregate_task_lighting_profiles(
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
                                _episode_error_report(
                                    episode,
                                    elapsed_seconds=prepared.prepare_seconds,
                                    error=error,
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
                                    _episode_error_report(
                                        episode,
                                        elapsed_seconds=duration,
                                        error=error,
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
                ordered_reports = [
                    task_episode_reports[episode.name]
                    for episode in task_episodes
                    if episode.name in task_episode_reports
                ]
                task_report = _make_task_report(
                    task_episodes=task_episodes,
                    episode_reports=ordered_reports,
                    elapsed_seconds=time.perf_counter() - task_started,
                    lighting_profile_path=task_profile_path,
                    lighting_profile=task_profile,
                    status=(
                        "interrupted"
                        if _is_interruption(error)
                        else "failed"
                    ),
                    error=error,
                )
                _write_json(task_root / "task_report.json", task_report)
                task_reports.append(task_report)
                raise

            ordered_reports = [
                task_episode_reports[episode.name]
                for episode in task_episodes
                if episode.name in task_episode_reports
            ]
            task_seconds = time.perf_counter() - task_started
            task_report = _make_task_report(
                task_episodes=task_episodes,
                episode_reports=ordered_reports,
                elapsed_seconds=task_seconds,
                lighting_profile_path=task_profile_path,
                lighting_profile=task_profile,
            )
            _write_json(task_root / "task_report.json", task_report)
            task_reports.append(task_report)
            tqdm.write(
                f"Task {task_label} {task_report['status']}: "
                f"{task_report['completed_episode_count']}/"
                f"{task_report['episode_count']} episodes, "
                f"elapsed {_format_duration(task_seconds)}"
            )
    except BaseException as error:
        batch_seconds = time.perf_counter() - batch_started
        batch_report = _make_batch_report(
            folder=folder,
            output=output,
            discovered_episode_count=len(episodes),
            discovered_task_count=len(task_groups),
            task_reports=task_reports,
            elapsed_seconds=batch_seconds,
            status=(
                "interrupted" if _is_interruption(error) else "failed"
            ),
            error=error,
        )
        _write_json(workspace_output / "batch_report.json", batch_report)
        raise

    batch_seconds = time.perf_counter() - batch_started
    batch_status = (
        "completed"
        if all(report["status"] == "completed" for report in task_reports)
        else "failed"
    )
    batch_report = _make_batch_report(
        folder=folder,
        output=output,
        discovered_episode_count=len(episodes),
        discovered_task_count=len(task_groups),
        task_reports=task_reports,
        elapsed_seconds=batch_seconds,
        status=batch_status,
    )
    _write_json(workspace_output / "batch_report.json", batch_report)
    tqdm.write(
        f"Batch {batch_status}: "
        f"{batch_report['completed_episode_count']}/{len(episodes)} episodes, "
        f"{len(task_reports)}/{len(task_groups)} tasks, "
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
        description="Folder-based paper §2.3 human-to-G1 synthesis"
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
        "--backend",
        dest="_legacy_backend",
        choices=("isaacsim",),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Keep masks, inpainted video, render frames, logs, and JSON "
            "reports; by default only each synthetic MP4 and copied HDF5 remain"
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
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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
        "--propainter-subvideo-length",
        type=int,
        default=80,
        help=(
            "Internal ProPainter window and 1280x720-equivalent chunk "
            "pixel budget"
        ),
    )
    parser.add_argument(
        "--propainter-context-frames",
        type=int,
        default=20,
        help="Temporal context kept on each side of an inpaint chunk",
    )
    parser.add_argument(
        "--propainter-roi-margin",
        type=int,
        default=64,
        help="Target-canvas pixels added around each chunk's mask union",
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
    parser.add_argument("--object-prompt")
    parser.add_argument("--object-prompt-frame", type=int)
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
