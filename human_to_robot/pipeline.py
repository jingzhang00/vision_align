from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
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
from .video_io import video_metadata


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
            Path(".") if task_dir == folder else task_dir.relative_to(folder)
        )
        task_output = output if task_relative == Path(".") else output / task_relative
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
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


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
) -> bool:
    record = state.get("stages", {}).get(stage)
    return (
        isinstance(record, dict)
        and record.get("status") == "completed"
        and record.get("signature") == signature
        and isinstance(record.get("generation"), int)
    )


def _set_stage_state(
    path: Path,
    state: dict[str, Any],
    *,
    stage: str,
    status: str,
    signature: dict[str, object],
    generation: int | None = None,
) -> None:
    state["state_version"] = 1
    stages = state.setdefault("stages", {})
    record: dict[str, object] = {
        "status": status,
        "signature": signature,
    }
    if generation is not None:
        record["generation"] = generation
    stages[stage] = record
    _write_json(path, state)


def _task_label(episode: Episode) -> str:
    if episode.task_relative == Path("."):
        return episode.task_name
    return episode.task_relative.as_posix()


def _episode_label(episode: Episode) -> str:
    return f"{_task_label(episode)}/{episode.name}"


def _running_episode_report(
    episode: Episode,
    *,
    backend: str,
) -> dict[str, object]:
    report: dict[str, object] = {
        "task": episode.task_name,
        "task_relative": episode.task_relative.as_posix(),
        "episode": episode.name,
        "status": "running",
        "video": str(episode.video),
        "hdf5": str(episode.hdf5),
        "backend": backend,
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


def _prepare_stage_signature(
    episode: Episode,
    *,
    args: argparse.Namespace,
    repo: Path,
    sam_checkpoint: Path,
    mjcf: Path,
    propainter: Path,
    prompt_frame: int,
    object_prompt_frame: int | None,
) -> dict[str, object]:
    input_paths = {
        "video": episode.video,
        "hdf5": episode.hdf5,
        "sam_checkpoint": sam_checkpoint,
        "mjcf": mjcf,
        "raft_weight": propainter / "weights/raft-things.pth",
        "flow_weight": propainter / "weights/recurrent_flow_completion.pth",
        "propainter_weight": propainter / "weights/ProPainter.pth",
        "pipeline_code": Path(__file__),
        "segment_code": repo / "human_to_robot/segment_sam3.py",
        "retarget_code": repo / "human_to_robot/retarget_g1.py",
        "lighting_code": repo / "human_to_robot/lighting_profile.py",
        "propainter_code": propainter / "inference_propainter.py",
    }
    return {
        "signature_version": 1,
        "inputs": {
            name: _file_fingerprint(path)
            for name, path in input_paths.items()
        },
        "parameters": {
            "python": str(Path(sys.executable).resolve()),
            "width": args.width,
            "height": args.height,
            "propainter_subvideo_length": args.propainter_subvideo_length,
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
    }
    if args.backend == "isaacsim":
        input_paths["isaac_renderer_code"] = (
            repo / "human_to_robot/render_isaacsim.py"
        )
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
            "backend": args.backend,
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
) -> float:
    started = time.perf_counter()
    active_started = episode_started if episode_started is not None else started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
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
                elapsed = elapsed_offset + time.perf_counter() - active_started
                progress.set_postfix_str(
                    f"current={current_episode} "
                    f"elapsed={_format_duration(elapsed)}",
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
        raise RuntimeError(
            f"Episode {current_episode} failed; log: {log_path}\n{tail}"
        )
    return elapsed


def _task_output_root(base_output: Path, episode: Episode) -> Path:
    return (
        base_output
        if episode.task_relative == Path(".")
        else base_output / episode.task_relative
    )


def _prepare_episode(
    episode: Episode,
    *,
    args: argparse.Namespace,
    repo: Path,
    sam_checkpoint: Path,
    mjcf: Path,
    propainter: Path,
    progress: tqdm[Any],
) -> PreparedEpisode:
    started = time.perf_counter()
    output = episode.output_root
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "pipeline.log"
    _running_episode_report(episode, backend=args.backend)
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
        mjcf=mjcf,
        propainter=propainter,
        prompt_frame=prompt_frame,
        object_prompt_frame=object_prompt_frame,
    )
    pipeline_state = _load_pipeline_state(state_path)
    reuse_prepare = args.reuse and _stage_complete(
        pipeline_state, "prepare", prepare_signature
    )
    previous_prepare = pipeline_state.get("stages", {}).get("prepare", {})
    previous_generation = (
        previous_prepare.get("generation")
        if isinstance(previous_prepare, dict)
        else None
    )
    _set_stage_state(
        state_path,
        pipeline_state,
        stage="prepare",
        status="running",
        signature=prepare_signature,
    )
    prepare_changed = False

    arm_masks_ready = _image_sequence_complete(
        sam_root / "arm_masks", frame_count
    )
    object_masks_ready = (
        args.object_prompt is None
        or _image_sequence_complete(sam_root / "object_masks", frame_count)
    )
    if not reuse_prepare or not (arm_masks_ready and object_masks_ready):
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
        ]
        if args.object_prompt is not None:
            assert object_prompt_frame is not None
            sam_command.extend(
                [
                    "--object-prompt",
                    args.object_prompt,
                    "--object-prompt-frame",
                    str(object_prompt_frame),
                ]
            )
        _run_logged(
            sam_command,
            cwd=repo,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
        )
        if not _image_sequence_complete(sam_root / "arm_masks", frame_count):
            raise RuntimeError(f"SAM3 arm masks are incomplete: {sam_root}")
        if args.object_prompt is not None and not _image_sequence_complete(
            sam_root / "object_masks", frame_count
        ):
            raise RuntimeError(f"SAM3 object masks are incomplete: {sam_root}")

    inpaint_video = propainter_root / episode.video.stem / "inpaint_out.mp4"
    if (
        not reuse_prepare
        or prepare_changed
        or not _video_complete(inpaint_video, frame_count)
    ):
        prepare_changed = True
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", "/tmp/vision_align_matplotlib")
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        _run_logged(
            [
                python,
                "inference_propainter.py",
                "--video",
                str(episode.video),
                "--mask",
                str(sam_root / "arm_masks"),
                "--output",
                str(propainter_root),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--fp16",
                "--subvideo_length",
                str(args.propainter_subvideo_length),
                "--neighbor_length",
                "10",
                "--ref_stride",
                "10",
                "--raft_iter",
                "20",
                "--mask_dilation",
                "4",
                "--save_fps",
                "30",
            ],
            cwd=propainter,
            environment=environment,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
        )
        if not _video_complete(inpaint_video, frame_count):
            raise RuntimeError(
                f"ProPainter output is incomplete: {inpaint_video}"
            )

    render_width, render_height, inpaint_count, _ = video_metadata(inpaint_video)
    if inpaint_count != frame_count:
        raise ValueError(
            f"ProPainter frame count changed for {episode.video}: "
            f"{frame_count} -> {inpaint_count}"
        )
    render_size = (render_width, render_height)
    trajectory_ready = _trajectory_complete(
        g1_root / "g1_trajectory.npz", frame_count
    )
    mujoco_images_ready = all(
        _image_sequence_complete(
            g1_root / directory,
            frame_count,
            expected_size=render_size,
        )
        for directory in ("robot_rgb", "robot_masks", "robot_depth_mm")
    )
    retarget_required = not trajectory_ready or (
        args.backend == "mujoco" and not mujoco_images_ready
    )
    if not reuse_prepare or prepare_changed or retarget_required:
        prepare_changed = True
        _run_logged(
            [
                python,
                "-m",
                "human_to_robot.retarget_g1",
                "--video",
                str(episode.video),
                "--hdf5",
                str(episode.hdf5),
                "--mjcf",
                str(mjcf),
                "--output",
                str(g1_root),
                "--render-width",
                str(render_width),
                "--render-height",
                str(render_height),
            ],
            cwd=repo,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
        )
        if not _trajectory_complete(
            g1_root / "g1_trajectory.npz", frame_count
        ):
            raise RuntimeError(f"G1 trajectory is incomplete: {g1_root}")
        if args.backend == "mujoco" and not all(
            _image_sequence_complete(
                g1_root / directory,
                frame_count,
                expected_size=render_size,
            )
            for directory in ("robot_rgb", "robot_masks", "robot_depth_mm")
        ):
            raise RuntimeError(f"MuJoCo render is incomplete: {g1_root}")

    observation = estimate_lighting_profile(
        inpaint_video=inpaint_video,
        reliability_mask_root=sam_root / "arm_masks",
        sample_count=args.lighting_sample_count,
        max_dimension=args.lighting_max_dimension,
        write_artifacts=False,
    )
    prepare_generation = (
        int(previous_generation)
        if reuse_prepare
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
    mjcf: Path,
    task_profile_path: Path,
    progress: tqdm[Any],
) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    episode = prepared.episode
    log_path = episode.output_root / "pipeline.log"
    episode_label = _episode_label(episode)
    render_width, render_height, _, _ = video_metadata(prepared.inpaint_video)
    render_size = (render_width, render_height)
    render_root = prepared.g1_root
    trajectory_path = prepared.g1_root / "g1_trajectory.npz"
    if args.backend == "isaacsim":
        render_root = episode.output_root / "g1_isaacsim"
    final_root = episode.output_root / f"final_{args.backend}"
    final_video = final_root / "synthetic_g1.mp4"
    composite_report = final_root / "composite_report.json"
    finish_signature = _finish_stage_signature(
        prepared,
        args=args,
        repo=repo,
        task_profile_path=task_profile_path,
    )
    finish_stage = f"finish_{args.backend}"
    state_path = episode.output_root / "pipeline_state.json"
    pipeline_state = _load_pipeline_state(state_path)

    render_complete = all(
        _image_sequence_complete(
            render_root / directory,
            prepared.frame_count,
            expected_size=render_size,
        )
        for directory in ("robot_rgb", "robot_masks", "robot_depth_mm")
    )
    if args.backend == "isaacsim":
        isaac_report = render_root / "isaacsim_render_report.json"
        render_complete = render_complete and isaac_report.is_file()
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
        _set_stage_state(
            state_path,
            pipeline_state,
            stage=finish_stage,
            status="running",
            signature=finish_signature,
        )
        if args.backend == "isaacsim":
            _run_logged(
                [
                    sys.executable,
                    "-m",
                    "human_to_robot.render_isaacsim",
                    "--trajectory",
                    str(trajectory_path),
                    "--mjcf",
                    str(mjcf),
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
                ],
                cwd=repo,
                log_path=log_path,
                progress=progress,
                current_episode=episode_label,
                episode_started=started,
                elapsed_offset=prepared.prepare_seconds,
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
            ) or not (render_root / "isaacsim_render_report.json").is_file():
                raise RuntimeError(
                    f"Isaac Sim render is incomplete: {render_root}"
                )
        elif not render_complete:
            raise RuntimeError(f"MuJoCo render is incomplete: {render_root}")

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
            "--lighting-profile",
            str(task_profile_path),
            "--output",
            str(final_root),
        ]
        object_masks = prepared.sam_root / "object_masks"
        if args.object_prompt is not None and _image_sequence_complete(
            object_masks, prepared.frame_count
        ):
            composite_command.extend(["--object-masks", str(object_masks)])
        _run_logged(
            composite_command,
            cwd=repo,
            log_path=log_path,
            progress=progress,
            current_episode=episode_label,
            episode_started=started,
            elapsed_offset=prepared.prepare_seconds,
        )
        if not _video_complete(
            final_video,
            prepared.frame_count,
            expected_size=render_size,
        ) or not composite_report.is_file():
            raise RuntimeError(f"Composite output is incomplete: {final_root}")
        _set_stage_state(
            state_path,
            pipeline_state,
            stage=finish_stage,
            status="completed",
            signature=finish_signature,
            generation=time.time_ns(),
        )

    elapsed = time.perf_counter() - started
    artifacts: dict[str, object] = {
        "task": episode.task_name,
        "task_relative": episode.task_relative.as_posix(),
        "episode": episode.name,
        "status": "completed",
        "video": str(episode.video),
        "hdf5": str(episode.hdf5),
        "backend": args.backend,
        "task_lighting_profile": str(task_profile_path),
        "synthetic_video": str(final_video),
        "comparison_video": str(
            final_root / "comparison_original_inpaint_g1.mp4"
        ),
        "inpaint_video": str(prepared.inpaint_video),
        "trajectory": str(prepared.g1_root / "g1_trajectory.npz"),
        "elapsed_seconds": prepared.prepare_seconds + elapsed,
    }
    _write_json(episode.output_root / "episode_report.json", artifacts)
    return artifacts, elapsed


def _failed_episode_report(
    episode: Episode,
    *,
    backend: str,
    elapsed_seconds: float,
    error: BaseException,
) -> dict[str, object]:
    report: dict[str, object] = {
        "task": episode.task_name,
        "task_relative": episode.task_relative.as_posix(),
        "episode": episode.name,
        "status": "failed",
        "video": str(episode.video),
        "hdf5": str(episode.hdf5),
        "backend": backend,
        "elapsed_seconds": elapsed_seconds,
        "error": f"{type(error).__name__}: {error}",
        "log": str(episode.output_root / "pipeline.log"),
    }
    _write_json(episode.output_root / "episode_report.json", report)
    return report


def _make_task_report(
    *,
    task_episodes: list[Episode],
    episode_reports: list[dict[str, object]],
    backend: str,
    elapsed_seconds: float,
    lighting_profile_path: Path | None,
    lighting_profile: dict[str, Any] | None,
    status: str | None = None,
    error: BaseException | None = None,
) -> dict[str, object]:
    completed = sum(
        report.get("status") == "completed" for report in episode_reports
    )
    failed = sum(report.get("status") == "failed" for report in episode_reports)
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
        "backend": backend,
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
    backend: str,
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
        "backend": backend,
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


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    batch_started = time.perf_counter()
    repo = Path(__file__).resolve().parent.parent
    folder = args.folder.resolve()
    output = args.output.resolve()
    sam_checkpoint = (repo / "models/sam3/sam3.pt").resolve()
    mjcf = (repo / "assets/unitree_description/mjcf/g1.xml").resolve()
    propainter = (repo / "ProPainter").resolve()
    episodes: list[Episode] = []
    task_groups: dict[Path, list[Episode]] = {}
    _write_json(
        output / "batch_report.json",
        _make_batch_report(
            folder=folder,
            output=output,
            backend=args.backend,
            discovered_episode_count=0,
            discovered_task_count=0,
            task_reports=[],
            elapsed_seconds=0.0,
            status="running",
        ),
    )
    try:
        _validate_file(sam_checkpoint, "SAM3 checkpoint", 3_000_000_000)
        _validate_file(mjcf, "G1 MJCF")
        if args.width <= 0 or args.height <= 0:
            raise ValueError("--width and --height must be positive")
        if args.propainter_subvideo_length < 20:
            raise ValueError("--propainter-subvideo-length must be at least 20")
        for name in (
            "raft-things.pth",
            "recurrent_flow_completion.pth",
            "ProPainter.pth",
        ):
            _validate_file(
                propainter / "weights" / name, f"ProPainter weight {name}"
            )

        episodes = discover_episodes(folder, output)
        for episode in episodes:
            task_groups.setdefault(episode.task_relative, []).append(episode)
    except BaseException as error:
        initialization_report = _make_batch_report(
            folder=folder,
            output=output,
            backend=args.backend,
            discovered_episode_count=len(episodes),
            discovered_task_count=len(task_groups),
            task_reports=[],
            elapsed_seconds=time.perf_counter() - batch_started,
            status=(
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed"
            ),
            error=error,
        )
        _write_json(output / "batch_report.json", initialization_report)
        raise

    tqdm.write(
        f"Discovered {len(episodes)} episodes in {len(task_groups)} task(s); "
        f"backend={args.backend}"
    )
    task_reports: list[dict[str, object]] = []
    try:
        for task_relative, task_episodes in task_groups.items():
            task_started = time.perf_counter()
            task_label = _task_label(task_episodes[0])
            task_root = _task_output_root(output, task_episodes[0])
            task_episode_reports: dict[str, dict[str, object]] = {}
            prepared_episodes: list[PreparedEpisode] = []
            task_profile_path: Path | None = None
            task_profile: dict[str, Any] | None = None
            _write_json(
                task_root / "task_report.json",
                _make_task_report(
                    task_episodes=task_episodes,
                    episode_reports=[],
                    backend=args.backend,
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
                ) as progress:
                    for episode in task_episodes:
                        episode_started = time.perf_counter()
                        try:
                            prepared = _prepare_episode(
                                episode,
                                args=args,
                                repo=repo,
                                sam_checkpoint=sam_checkpoint,
                                mjcf=mjcf,
                                propainter=propainter,
                                progress=progress,
                            )
                            prepared_episodes.append(prepared)
                            duration = prepared.prepare_seconds
                        except Exception as error:
                            duration = time.perf_counter() - episode_started
                            task_episode_reports[episode.name] = (
                                _failed_episode_report(
                                    episode,
                                    backend=args.backend,
                                    elapsed_seconds=duration,
                                    error=error,
                                )
                            )
                            tqdm.write(
                                f"Failed {_episode_label(episode)} after "
                                f"{_format_duration(duration)}; "
                                f"log: {episode.output_root / 'pipeline.log'}"
                            )
                        progress.update(1)
                        progress.set_postfix_str(
                            f"last={_episode_label(episode)} "
                            f"time={_format_duration(duration)}",
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
                                _failed_episode_report(
                                    episode,
                                    backend=args.backend,
                                    elapsed_seconds=prepared.prepare_seconds,
                                    error=error,
                                )
                            )
                        tqdm.write(
                            f"Failed to build lighting profile for {task_label}: "
                            f"{type(error).__name__}: {error}"
                        )
                        prepared_episodes = []
                        task_profile_path = None

                if prepared_episodes:
                    assert task_profile_path is not None
                    with tqdm(
                        total=len(prepared_episodes),
                        desc=f"{task_label} finish ({args.backend})",
                        unit="episode",
                        dynamic_ncols=True,
                    ) as progress:
                        for prepared in prepared_episodes:
                            episode = prepared.episode
                            finish_started = time.perf_counter()
                            try:
                                report, _ = _render_episode(
                                    prepared,
                                    args=args,
                                    repo=repo,
                                    mjcf=mjcf,
                                    task_profile_path=task_profile_path,
                                    progress=progress,
                                )
                                task_episode_reports[episode.name] = report
                                duration = float(report["elapsed_seconds"])
                                tqdm.write(
                                    f"Completed {_episode_label(episode)} in "
                                    f"{_format_duration(duration)}"
                                )
                            except Exception as error:
                                duration = (
                                    prepared.prepare_seconds
                                    + time.perf_counter()
                                    - finish_started
                                )
                                task_episode_reports[episode.name] = (
                                    _failed_episode_report(
                                        episode,
                                        backend=args.backend,
                                        elapsed_seconds=duration,
                                        error=error,
                                    )
                                )
                                tqdm.write(
                                    f"Failed {_episode_label(episode)} after "
                                    f"{_format_duration(duration)}; "
                                    f"log: {episode.output_root / 'pipeline.log'}"
                                )
                            progress.update(1)
                            progress.set_postfix_str(
                                f"last={_episode_label(episode)} "
                                f"time={_format_duration(duration)}",
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
                    backend=args.backend,
                    elapsed_seconds=time.perf_counter() - task_started,
                    lighting_profile_path=task_profile_path,
                    lighting_profile=task_profile,
                    status=(
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
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
                backend=args.backend,
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
            backend=args.backend,
            discovered_episode_count=len(episodes),
            discovered_task_count=len(task_groups),
            task_reports=task_reports,
            elapsed_seconds=batch_seconds,
            status=(
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            ),
            error=error,
        )
        _write_json(output / "batch_report.json", batch_report)
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
        backend=args.backend,
        discovered_episode_count=len(episodes),
        discovered_task_count=len(task_groups),
        task_reports=task_reports,
        elapsed_seconds=batch_seconds,
        status=batch_status,
    )
    _write_json(output / "batch_report.json", batch_report)
    tqdm.write(
        f"Batch {batch_status}: "
        f"{batch_report['completed_episode_count']}/{len(episodes)} episodes, "
        f"{len(task_reports)}/{len(task_groups)} tasks, "
        f"elapsed {_format_duration(batch_seconds)}"
    )
    return batch_report


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
        dest="backend",
        choices=("mujoco", "isaacsim"),
        default="mujoco",
        help="MuJoCo/Mink IK is always used; choose the image renderer",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--propainter-subvideo-length",
        type=int,
        default=80,
        help="Lower this for long videos when ProPainter approaches GPU memory limits",
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
        help="Reuse complete episode intermediates (default: true)",
    )
    return parser


def main() -> None:
    report = run_pipeline(build_parser().parse_args())
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
