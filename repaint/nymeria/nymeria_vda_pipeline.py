#!/usr/bin/env python3
"""Run the calibrated Nymeria-to-G1 Metric VDA/Isaac pipeline.

The route mirrors the current EgoDex repaint pipeline after adapting Nymeria's
native inputs:

    Aria MPS + XSens camera calibration
      -> Nymeria full-body GMR LeRobot (fixed-open hands)
      -> human-arm repaint
      -> metric Video Depth Anything RGB-D preparation
      -> source-sized Isaac robot-camera replay

Camera translation follows the EgoDex camera-relative-to-hip mapping.  Nymeria
does not contain articulated fingertips, so its calibrated scene coordinates
are kept unchanged instead of fabricating a semantic scene anchor.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
STREAM_ROOT = WORKSPACE_ROOT / "stream"
for import_root in (WORKSPACE_ROOT, STREAM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from repaint.nymeria.nymeria_adapter import (  # noqa: E402
    NymeriaClip,
    discover_nymeria_clips,
    load_motion_metadata,
    natural_key,
)
from repaint.nymeria.nymeria_to_lerobot import (  # noqa: E402
    VIDEO_KEY,
    load_source_episode,
)
from repaint.robot_camera import (  # noqa: E402
    BEINGBEYOND_HEAD_CAMERA_PROFILE,
)
from repaint.vda_depth import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_VDA_METRIC_SCALE,
)
from repaint.vda_pipeline_common import (  # noqa: E402
    VideoMetadata,
    build_command as _command,
    format_elapsed as _format_elapsed,
    load_previous_stages as _load_previous_stages,
    probe_video,
    read_json as _read_json,
    remove_exact as _remove_exact,
    run_stage_process as _run_process,
    utc_now as _utc_now,
    write_json_atomic as _write_json_atomic,
)


STOP_ORDER = {
    "alignment": 0,
    "retarget": 1,
    "repaint": 2,
    "prepare": 3,
    "render": 4,
}


@dataclass(frozen=True)
class SelectedClip:
    episode_index: int
    source: NymeriaClip
    video: VideoMetadata

    @property
    def stem(self) -> str:
        return self.source.stem

    @property
    def session(self) -> str:
        return self.source.session


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    camera_alignment: Path
    alignment_reports: Path
    lerobot: Path
    handless: Path
    prepared: Path
    final: Path
    logs: Path
    report: Path

    @classmethod
    def from_root(cls, root: Path) -> "PipelinePaths":
        return cls(
            root=root,
            camera_alignment=root / "camera_alignment",
            alignment_reports=root / "alignment_reports",
            lerobot=root / "lerobot",
            handless=root / "handless",
            prepared=root / "prepared",
            final=root / "final",
            logs=root / "logs",
            report=root / "pipeline_report.json",
        )


@dataclass(frozen=True)
class PipelineConfig:
    src_root: Path
    paths: PipelinePaths
    alignment_root: Path
    alignment_report_root: Path
    alignment_is_external: bool
    python_executable: Path
    resume: bool
    overwrite: bool
    headless: bool
    require_visible_hands: bool
    hardware_encoding: bool
    hand_eye_sample_fps: float
    scene_scale: float | None
    vda_checkpoint: Path | None
    stop_after: str
    poll_seconds: float


@dataclass(frozen=True)
class Stage:
    stage_id: str
    name: str
    command: tuple[str, ...] | None
    log_path: Path
    dependencies: tuple[str, ...] = ()
    session: str | None = None
    episode_index: int | None = None
    clip_stems: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        if self.session is not None:
            return self.session
        if self.episode_index is not None:
            return f"{self.episode_index:06d}"
        return "all"


@dataclass(frozen=True)
class PipelinePlan:
    config: PipelineConfig
    clips: tuple[SelectedClip, ...]
    stages: tuple[Stage, ...]


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path contains the other."""

    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _update_path_fingerprint(
    digest: Any,
    path: Path,
    *,
    hash_contents: bool = False,
) -> None:
    path = path.expanduser().resolve()
    digest.update(f"path:{path}\n".encode())
    if not path.exists() and not path.is_symlink():
        digest.update(b"missing\n")
        return
    candidates = [path]
    if path.is_dir():
        candidates.extend(sorted(path.rglob("*")))
    for candidate in candidates:
        relative = (
            "."
            if candidate == path
            else candidate.relative_to(path).as_posix()
        )
        stat = candidate.stat()
        kind = "dir" if candidate.is_dir() else "file"
        digest.update(
            (
                f"{relative}\0{kind}\0{stat.st_size}\0"
                f"{stat.st_mtime_ns}\n"
            ).encode()
        )
        if hash_contents and candidate.is_file():
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)


def _update_repaint_episode_fingerprint(
    digest: Any,
    plan: PipelinePlan,
    episode_index: int,
) -> None:
    paths = plan.config.paths
    info = _read_json(paths.lerobot / "meta/info.json")
    chunks_size = int(info.get("chunks_size", 1000))
    if chunks_size <= 0:
        raise ValueError("LeRobot chunks_size must be positive")
    relative_video = Path(str(info["video_path"]).format(
        video_key=VIDEO_KEY,
        chunk_index=episode_index // chunks_size,
        file_index=episode_index % chunks_size,
    ))
    source_video = (paths.lerobot / relative_video).resolve()
    output_video = paths.handless / relative_video
    _update_path_fingerprint(digest, output_video)

    manifest = _read_json(paths.handless / "manifest.json")
    entries = manifest.get("episodes")
    if not isinstance(entries, list):
        raise ValueError("Arm-removal manifest has no episode list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and Path(str(entry.get("source", ""))).expanduser().resolve()
        == source_video
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Episode {episode_index} has {len(matches)} arm-removal entries"
        )
    digest.update(
        json.dumps(
            matches[0],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def _stage_input_fingerprint(stage: Stage, plan: PipelinePlan) -> str:
    """Fingerprint commands plus inputs/outputs that make reuse safe."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            None if stage.command is None else list(stage.command),
            separators=(",", ":"),
        ).encode()
    )
    if stage.command is not None and len(stage.command) > 1:
        script = Path(stage.command[1])
        if script.is_file():
            _update_path_fingerprint(digest, script, hash_contents=True)

    paths = plan.config.paths
    if stage.name == "alignment":
        assert stage.session is not None
        session_root = plan.config.src_root / stage.session
        for source in (
            session_root / "metadata.json",
            session_root / "undistorted/camera_intrinsics.json",
            session_root
            / "recording_head/mps/slam/closed_loop_trajectory.csv",
            session_root
            / "recording_head/mps/slam/online_calibration.jsonl",
            session_root / "act_sum_208_clips",
            session_root / "act_sum_smpl",
            session_root / "act_sum_text",
            *session_root.glob(
                "undistorted_*_activity_summarization_clips"
            ),
        ):
            _update_path_fingerprint(digest, source)
        for stem in stage.clip_stems:
            _update_path_fingerprint(
                digest,
                plan.config.alignment_root
                / stage.session
                / f"{stem}.hdf5",
                hash_contents=True,
            )
        _update_path_fingerprint(
            digest,
            plan.config.alignment_report_root
            / stage.session
            / "visual_alignment_report.json",
            hash_contents=True,
        )
    elif stage.name == "retarget":
        for clip in plan.clips:
            for source in (
                clip.source.video_path,
                clip.source.motion_208_path,
                clip.source.text_path,
                clip.source.smpl_path,
            ):
                _update_path_fingerprint(digest, source)
        _update_path_fingerprint(digest, paths.lerobot)
    elif stage.name == "repaint":
        _update_path_fingerprint(digest, paths.lerobot)
        _update_path_fingerprint(digest, paths.handless)
    elif stage.name in {"prepare", "render"}:
        assert stage.episode_index is not None
        label = f"episode_{stage.episode_index:06d}"
        _update_path_fingerprint(digest, paths.prepared / label)
        if stage.name == "prepare":
            _update_path_fingerprint(digest, paths.lerobot / "meta")
            _update_repaint_episode_fingerprint(
                digest,
                plan,
                stage.episode_index,
            )
            _update_path_fingerprint(
                digest,
                plan.config.vda_checkpoint or DEFAULT_CHECKPOINT,
            )
        else:
            _update_path_fingerprint(digest, paths.final / label)
    return digest.hexdigest()


def _selected_clips(
    src_root: Path,
    *,
    sessions: set[str] | None,
    requested_clips: set[str] | None,
) -> tuple[SelectedClip, ...]:
    discovered = discover_nymeria_clips(src_root)
    suffix_to_stems: dict[str, list[str]] = {}
    for clip in discovered:
        suffix_to_stems.setdefault(clip.stem.rsplit("_", 1)[-1], []).append(
            clip.stem
        )

    resolved_requested: set[str] | None = None
    if requested_clips is not None:
        resolved_requested = set()
        available_stems = {clip.stem for clip in discovered}
        for requested in requested_clips:
            if requested in available_stems:
                resolved_requested.add(requested)
                continue
            matches = suffix_to_stems.get(requested, [])
            if sessions is not None:
                matches = [
                    stem
                    for stem in matches
                    if any(
                        clip.stem == stem and clip.session in sessions
                        for clip in discovered
                    )
                ]
            if len(matches) != 1:
                raise ValueError(
                    f"Clip selector {requested!r} resolved to {matches}; "
                    "use the full clip stem"
                )
            resolved_requested.add(matches[0])

    selected_sources = [
        clip
        for clip in discovered
        if (sessions is None or clip.session in sessions)
        and (
            resolved_requested is None
            or clip.stem in resolved_requested
        )
    ]
    if not selected_sources:
        raise ValueError("Session/clip filters selected no Nymeria clips")
    if sessions is not None:
        missing_sessions = sessions - {
            clip.session for clip in selected_sources
        }
        if missing_sessions:
            raise ValueError(
                f"Unknown or empty Nymeria sessions: {sorted(missing_sessions)}"
            )
    return tuple(
        SelectedClip(
            episode_index=index,
            source=clip,
            video=probe_video(clip.video_path),
        )
        for index, clip in enumerate(selected_sources)
    )


def _stage_enabled(stop_after: str, name: str) -> bool:
    return STOP_ORDER[name] <= STOP_ORDER[stop_after]


def build_plan(
    config: PipelineConfig,
    *,
    sessions: set[str] | None,
    requested_clips: set[str] | None,
) -> PipelinePlan:
    clips = _selected_clips(
        config.src_root,
        sessions=sessions,
        requested_clips=requested_clips,
    )
    grouped: dict[str, list[SelectedClip]] = {}
    for clip in clips:
        grouped.setdefault(clip.session, []).append(clip)

    stages: list[Stage] = []
    alignment_ids: list[str] = []
    for session in sorted(grouped, key=natural_key):
        session_clips = grouped[session]
        stage_id = f"alignment_{session}"
        alignment_ids.append(stage_id)
        command: tuple[str, ...] | None = None
        if not config.alignment_is_external:
            arguments: list[str | Path] = [
                "--episode-root",
                config.src_root / session,
                "--output",
                config.paths.alignment_reports,
                "--adapter-output",
                config.alignment_root,
                "--hand-eye-sample-fps",
                repr(config.hand_eye_sample_fps),
            ]
            for clip in session_clips:
                arguments.extend(["--clip", clip.stem])
            command = _command(
                config.python_executable,
                (
                    WORKSPACE_ROOT
                    / "repaint/nymeria/nymeria_camera_alignment.py"
                ),
                *arguments,
            )
        stages.append(
            Stage(
                stage_id=stage_id,
                name="alignment",
                command=command,
                log_path=config.paths.logs / f"00_{stage_id}.log",
                session=session,
                clip_stems=tuple(clip.stem for clip in session_clips),
            )
        )

    if _stage_enabled(config.stop_after, "retarget"):
        arguments = [
            "--src-root",
            config.src_root,
            "--output-root",
            config.paths.lerobot,
            "--camera-alignment-root",
            config.alignment_root,
            "--camera-alignment-report-root",
            config.alignment_report_root,
            "--no-overwrite",
        ]
        for session in sorted(grouped, key=natural_key):
            arguments.extend(["--session", session])
        for clip in clips:
            arguments.extend(["--clip", clip.stem])
        stages.append(
            Stage(
                stage_id="retarget",
                name="retarget",
                command=_command(
                    config.python_executable,
                    (
                        WORKSPACE_ROOT
                        / "repaint/nymeria/nymeria_to_lerobot.py"
                    ),
                    *arguments,
                ),
                log_path=config.paths.logs / "01_retarget.log",
                dependencies=tuple(alignment_ids),
            )
        )

    if _stage_enabled(config.stop_after, "repaint"):
        stages.append(
            Stage(
                stage_id="repaint",
                name="repaint",
                command=_command(
                    config.python_executable,
                    WORKSPACE_ROOT / "repaint/remove_hands.py",
                    "--input_dir",
                    config.paths.lerobot,
                    "--output_dir",
                    config.paths.handless,
                    "--min_coverage",
                    "0",
                ),
                log_path=config.paths.logs / "02_repaint.log",
                dependencies=("retarget",),
            )
        )

    if _stage_enabled(config.stop_after, "prepare"):
        for clip in clips:
            label = f"episode_{clip.episode_index:06d}"
            prepared_root = config.paths.prepared / label
            arguments = [
                "--dataset",
                config.paths.lerobot,
                "--source-root",
                config.src_root,
                "--removed-root",
                config.paths.handless,
                "--episode",
                str(clip.episode_index),
                "--output",
                prepared_root,
                "--vda-scale-failure-policy",
                "fixed",
            ]
            if config.scene_scale is not None:
                arguments.extend(
                    ["--scene-scale", repr(config.scene_scale)]
                )
            if config.vda_checkpoint is not None:
                arguments.extend(
                    ["--vda-checkpoint", config.vda_checkpoint]
                )
            stages.append(
                Stage(
                    stage_id=f"prepare_{clip.episode_index:06d}",
                    name="prepare",
                    command=_command(
                        config.python_executable,
                        WORKSPACE_ROOT / "repaint/rgbd_prepare.py",
                        *arguments,
                    ),
                    log_path=(
                        config.paths.logs
                        / f"03_prepare_{clip.episode_index:06d}.log"
                    ),
                    dependencies=("retarget", "repaint"),
                    episode_index=clip.episode_index,
                )
            )

    if _stage_enabled(config.stop_after, "render"):
        for clip in clips:
            label = f"episode_{clip.episode_index:06d}"
            output_video = config.paths.final / label / "g1_view.mp4"
            stages.append(
                Stage(
                    stage_id=f"render_{clip.episode_index:06d}",
                    name="render",
                    command=_command(
                        config.python_executable,
                        WORKSPACE_ROOT / "repaint/vda_scene_isaac.py",
                        "--prepared",
                        config.paths.prepared / label,
                        "--output",
                        output_video,
                        "--capture-view",
                        "robot-camera",
                        "--scene-alignment",
                        "none",
                        "--scene-leveling",
                        "none",
                        (
                            "--headless"
                            if config.headless
                            else "--no-headless"
                        ),
                    ),
                    log_path=(
                        config.paths.logs
                        / f"04_render_{clip.episode_index:06d}.log"
                    ),
                    dependencies=(f"prepare_{clip.episode_index:06d}",),
                    episode_index=clip.episode_index,
                )
            )
    return PipelinePlan(config=config, clips=clips, stages=tuple(stages))


def _clip_by_index(plan: PipelinePlan, episode_index: int) -> SelectedClip:
    if not 0 <= episode_index < len(plan.clips):
        raise IndexError(episode_index)
    clip = plan.clips[episode_index]
    if clip.episode_index != episode_index:
        raise ValueError("Selected clip indices are not contiguous")
    return clip


def _episode_rows(root: Path) -> dict[int, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    result = {int(row["episode_index"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("LeRobot episode metadata contains duplicate indices")
    return result


def _video_path(
    root: Path,
    info: dict[str, Any],
    row: dict[str, Any],
) -> Path:
    return root / str(info["video_path"]).format(
        video_key=VIDEO_KEY,
        chunk_index=int(row[f"videos/{VIDEO_KEY}/chunk_index"]),
        file_index=int(row[f"videos/{VIDEO_KEY}/file_index"]),
    )


def _validate_alignment(stage: Stage, plan: PipelinePlan) -> None:
    assert stage.session is not None
    selected = [
        clip for clip in plan.clips if clip.session == stage.session
    ]
    if tuple(clip.stem for clip in selected) != stage.clip_stems:
        raise ValueError("Alignment stage clip selection changed")

    report_path = (
        plan.config.alignment_report_root
        / stage.session
        / "visual_alignment_report.json"
    )
    report = _read_json(report_path)
    if report.get("status") not in {
        "prepared_only",
        "completed",
    }:
        raise ValueError(
            f"Nymeria alignment report is not complete: "
            f"{report.get('status')!r}"
        )
    hand_eye = report.get("hand_eye")
    if not isinstance(hand_eye, dict):
        raise ValueError("Nymeria alignment report has no hand-eye audit")
    if float(hand_eye.get("p95_rotation_error_degrees", math.inf)) > 10.0:
        raise ValueError("Nymeria hand-eye p95 rotation residual exceeds 10 deg")
    if float(hand_eye.get("p95_translation_error_m", math.inf)) > 0.10:
        raise ValueError("Nymeria hand-eye p95 translation residual exceeds 10 cm")
    if float(hand_eye.get("maximum_time_error_us", math.inf)) > 5000.0:
        raise ValueError("Nymeria hand-eye timestamp residual exceeds 5 ms")

    prepared = report.get("prepared")
    if not isinstance(prepared, list):
        raise ValueError("Nymeria alignment report has no prepared clip list")
    prepared_stems = {
        str(item["clip"])
        for item in prepared
        if isinstance(item, dict) and "clip" in item
    }
    expected_stems = {clip.stem for clip in selected}
    if not expected_stems.issubset(prepared_stems):
        raise ValueError(
            "Camera alignment did not prepare every selected clip: "
            f"missing={sorted(expected_stems - prepared_stems)}"
        )
    for clip in selected:
        source = load_source_episode(
            clip.source,
            camera_alignment_root=plan.config.alignment_root,
        )
        if source.frame_count != clip.video.frame_count:
            raise ValueError(
                f"{clip.stem} alignment frame count differs from source video"
            )


def _validate_retarget(plan: PipelinePlan) -> None:
    root = plan.config.paths.lerobot
    info = _read_json(root / "meta/info.json")
    manifest = _read_json(root / "meta/nymeria.json")
    quality_gate = manifest.get("camera_alignment_quality_gate", {})
    if not isinstance(quality_gate, dict) or quality_gate.get("status") != "passed":
        raise ValueError("Nymeria LeRobot camera alignment gate did not pass")
    report_entries = quality_gate.get("hand_eye_quality_reports")
    if not isinstance(report_entries, list):
        raise ValueError("Nymeria LeRobot has no hand-eye report provenance")
    expected_sessions = {clip.session for clip in plan.clips}
    recorded_reports: dict[str, dict[str, Any]] = {}
    for item in report_entries:
        if not isinstance(item, dict):
            raise ValueError("Invalid hand-eye report provenance entry")
        report_path = Path(str(item.get("path", ""))).resolve()
        if report_path.name != "visual_alignment_report.json":
            raise ValueError("Invalid hand-eye report provenance path")
        session = report_path.parent.name
        expected_path = (
            plan.config.alignment_report_root
            / session
            / "visual_alignment_report.json"
        ).resolve()
        if report_path != expected_path or session in recorded_reports:
            raise ValueError("Hand-eye report provenance does not match config")
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        if hashlib.sha256(report_path.read_bytes()).hexdigest() != item.get(
            "sha256"
        ):
            raise ValueError(f"Hand-eye report changed after retarget: {session}")
        recorded_reports[session] = item
    if set(recorded_reports) != expected_sessions:
        raise ValueError("Hand-eye report provenance has incorrect sessions")
    if "observation.camera_pose" not in info.get("features", {}):
        raise ValueError("Nymeria LeRobot has no observation.camera_pose")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(plan.clips):
        raise ValueError("Nymeria LeRobot manifest episode count is incorrect")
    if int(info.get("total_episodes", -1)) != len(plan.clips):
        raise ValueError("Nymeria LeRobot info episode count is incorrect")
    rows = _episode_rows(root)
    if set(rows) != set(range(len(plan.clips))):
        raise ValueError("Nymeria LeRobot episode indices are not contiguous")

    for clip, episode in zip(plan.clips, episodes, strict=True):
        if int(episode.get("episode_index", -1)) != clip.episode_index:
            raise ValueError("Nymeria manifest episode order changed")
        if episode.get("source_stem") != clip.stem:
            raise ValueError("Nymeria manifest source stem changed")
        alignment = episode.get("camera_alignment")
        if not isinstance(alignment, dict) or alignment.get("status") != "validated":
            raise ValueError(f"{clip.stem} camera alignment is not validated")
        adapter_path = Path(str(alignment.get("adapter_hdf5", ""))).resolve()
        expected_adapter = (
            plan.config.alignment_root
            / clip.session
            / f"{clip.stem}.hdf5"
        ).resolve()
        if adapter_path != expected_adapter or not adapter_path.is_file():
            raise ValueError(f"{clip.stem} camera adapter path changed")
        digest = hashlib.sha256()
        with adapter_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != alignment.get("adapter_hdf5_sha256"):
            raise ValueError(f"{clip.stem} camera adapter content changed")
        if alignment.get("hand_eye_quality_report") != recorded_reports[
            clip.session
        ]:
            raise ValueError(f"{clip.stem} hand-eye provenance changed")
        metrics = alignment.get("metrics", {})
        if float(metrics.get("basis_rotation_error_p95_degrees", math.inf)) > 10.0:
            raise ValueError(f"{clip.stem} camera/body basis residual is too high")
        if int(metrics.get("scene_scale_application_count", -1)) != 1:
            raise ValueError(f"{clip.stem} scene scale was not applied exactly once")
        scale = float(
            alignment["source_to_robot_camera_translation_scale"]
        )
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{clip.stem} has an invalid camera scale")

        row = rows[clip.episode_index]
        length = int(row["length"])
        if length != clip.video.frame_count:
            raise ValueError(f"{clip.stem} LeRobot length differs from source")
        data_path = root / str(info["data_path"]).format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        table = pq.read_table(
            data_path,
            columns=[
                "observation.state",
                "action",
                "episode_index",
                "frame_index",
                "timestamp",
                "observation.root_pose",
                "observation.camera_pose",
                "observation.camera_intrinsics",
            ],
        )
        episode_column = np.asarray(
            table["episode_index"].to_pylist(),
            dtype=np.int64,
        )
        selected = np.flatnonzero(episode_column == clip.episode_index)
        if len(selected) != length:
            raise ValueError(f"{clip.stem} parquet row count is incorrect")
        frame_indices = np.asarray(
            table["frame_index"].to_pylist(),
            dtype=np.int64,
        )[selected]
        if not np.array_equal(frame_indices, np.arange(length)):
            raise ValueError(f"{clip.stem} frame indices are not contiguous")
        joint_count = len(manifest.get("joint_order", ()))
        for column_name, width in (
            ("observation.state", joint_count),
            ("action", joint_count),
            ("observation.root_pose", 7),
            ("observation.camera_pose", 7),
            ("observation.camera_intrinsics", 9),
        ):
            values = np.asarray(
                table[column_name].to_pylist(),
                dtype=np.float64,
            )[selected]
            if (
                width <= 0
                or values.shape != (length, width)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"{clip.stem} {column_name} has an invalid shape/value"
                )
        timestamps = np.asarray(
            table["timestamp"].to_pylist(),
            dtype=np.float64,
        )[selected]
        if (
            timestamps.shape != (length,)
            or not np.isfinite(timestamps).all()
            or not np.isclose(timestamps[0], 0.0, atol=1e-5)
            or (length > 1 and np.any(np.diff(timestamps) <= 0.0))
        ):
            raise ValueError(f"{clip.stem} timestamps are invalid")
        root_pose = np.asarray(
            table["observation.root_pose"].to_pylist(),
            dtype=np.float64,
        )[selected]
        root_quaternion_norm = np.linalg.norm(root_pose[:, 3:], axis=1)
        if not np.allclose(root_quaternion_norm, 1.0, atol=1e-4):
            raise ValueError(f"{clip.stem} root quaternions are not unit length")
        camera_pose = np.asarray(
            table["observation.camera_pose"].to_pylist(),
            dtype=np.float64,
        )[selected]
        if camera_pose.shape != (length, 7) or not np.isfinite(camera_pose).all():
            raise ValueError(f"{clip.stem} camera pose is invalid")
        quaternion_norm = np.linalg.norm(camera_pose[:, 3:], axis=1)
        if not np.allclose(quaternion_norm, 1.0, atol=1e-4):
            raise ValueError(f"{clip.stem} camera quaternions are not unit length")
        intrinsics = np.asarray(
            table["observation.camera_intrinsics"].to_pylist(),
            dtype=np.float64,
        )[selected].reshape(length, 3, 3)
        if (
            not np.allclose(intrinsics, intrinsics[:1], atol=1e-6)
            or np.any(intrinsics[:, 0, 0] <= 0.0)
            or np.any(intrinsics[:, 1, 1] <= 0.0)
            or not np.allclose(
                intrinsics[:, 2],
                np.array([0.0, 0.0, 1.0])[None],
                atol=1e-6,
            )
        ):
            raise ValueError(f"{clip.stem} camera intrinsics are invalid")
        copied_video = _video_path(root, info, row)
        if not probe_video(copied_video).compatible_with(clip.video):
            raise ValueError(f"{clip.stem} LeRobot video differs from source")


def _validate_repaint(plan: PipelinePlan) -> None:
    info = _read_json(plan.config.paths.lerobot / "meta/info.json")
    rows = _episode_rows(plan.config.paths.lerobot)
    failed, entries = _repaint_manifest(plan)
    for clip in plan.clips:
        if clip.episode_index in failed:
            continue
        output = _video_path(
            plan.config.paths.handless,
            info,
            rows[clip.episode_index],
        )
        if not output.is_file():
            raise FileNotFoundError(
                f"Arm removal did not produce {clip.stem}: {output}"
            )
        if not probe_video(output).compatible_with(clip.video):
            raise ValueError(f"{clip.stem} handless video differs from source")
        entry = entries[clip.episode_index]
        visibility = entry.get("human_arm_visible")
        if (
            not isinstance(visibility, list)
            or len(visibility) != clip.video.frame_count
            or not all(isinstance(value, bool) for value in visibility)
        ):
            raise ValueError(
                f"{clip.stem} has no valid per-frame human-arm visibility"
            )
        if int(entry.get("human_arm_visible_frames", -1)) != sum(
            visibility
        ):
            raise ValueError(
                f"{clip.stem} human-arm visibility count is inconsistent"
            )


def _repaint_manifest(
    plan: PipelinePlan,
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, Any]]]:
    info = _read_json(plan.config.paths.lerobot / "meta/info.json")
    rows = _episode_rows(plan.config.paths.lerobot)
    manifest = _read_json(plan.config.paths.handless / "manifest.json")
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or len(entries) != len(plan.clips):
        raise ValueError("Arm-removal manifest episode count is incorrect")

    failures: dict[int, dict[str, str]] = {}
    matched_entries: dict[int, dict[str, Any]] = {}
    successful = {"ok", "skipped", "passthrough_no_target"}
    for clip in plan.clips:
        source_video = _video_path(
            plan.config.paths.lerobot,
            info,
            rows[clip.episode_index],
        ).resolve()
        matches = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and Path(str(entry.get("source", ""))).expanduser().resolve()
            == source_video
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{clip.stem} has {len(matches)} arm-removal entries"
            )
        entry = matches[0]
        matched_entries[clip.episode_index] = entry
        status = str(entry.get("status", ""))
        if status not in successful:
            failures[clip.episode_index] = {
                "status": status or "unknown",
                "error": str(
                    entry.get("error")
                    or entry.get("note")
                    or f"remove_hands status={status!r}"
                ),
            }
            continue
        expected_output = _video_path(
            plan.config.paths.handless,
            info,
            rows[clip.episode_index],
        ).resolve()
        if Path(str(entry.get("output", ""))).expanduser().resolve() != (
            expected_output
        ):
            raise ValueError(
                f"{clip.stem} arm-removal output path is incorrect"
            )
    return failures, matched_entries


def _numbered_png_count(path: Path) -> int:
    files = sorted(path.glob("*.png"))
    expected = [f"{index:05d}.png" for index in range(len(files))]
    if [file.name for file in files] != expected:
        raise ValueError(f"{path} is not a contiguous numbered PNG sequence")
    return len(files)


def _validate_prepared(stage: Stage, plan: PipelinePlan) -> None:
    assert stage.episode_index is not None
    clip = _clip_by_index(plan, stage.episode_index)
    root = plan.config.paths.prepared / f"episode_{clip.episode_index:06d}"
    report = _read_json(root / "prepare_report.json")
    if report.get("source_dataset_kind") != "nymeria":
        raise ValueError("Prepared episode does not identify Nymeria")
    if int(report.get("frame_count", -1)) != clip.video.frame_count:
        raise ValueError("Prepared frame count differs from source")
    if tuple(report.get("source_size", ())) != (
        clip.video.width,
        clip.video.height,
    ):
        raise ValueError("Prepared source size differs from source")
    depth = report.get("depth")
    if not isinstance(depth, dict):
        raise ValueError("Prepared report has no depth audit")
    if int(depth.get("scene_scale_application_count", -1)) != 1:
        raise ValueError("Prepared scene scale application count is not one")
    if int(depth.get("vda_metric_scale_application_count", -1)) != 1:
        raise ValueError("Prepared VDA scale application count is not one")
    vda_metric_scale = float(depth.get("vda_metric_scale", math.nan))
    depth_scene_scale = float(depth.get("scene_scale", math.nan))
    effective_scale = float(
        depth.get("effective_raw_depth_to_robot_scale", math.nan)
    )
    if not math.isfinite(vda_metric_scale) or vda_metric_scale <= 0.0:
        raise ValueError("Prepared VDA metric scale is invalid")
    if not math.isfinite(depth_scene_scale) or depth_scene_scale <= 0.0:
        raise ValueError("Prepared depth scene scale is invalid")
    if (
        not math.isfinite(effective_scale)
        or effective_scale <= 0.0
        or not math.isclose(
            effective_scale,
            vda_metric_scale * depth_scene_scale,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Prepared effective depth scale is inconsistent")
    scale_calibration = depth.get("vda_scale_calibration")
    if not isinstance(scale_calibration, dict):
        raise ValueError("Prepared report has no VDA scale calibration")
    if scale_calibration.get("requested_mode") != "auto":
        raise ValueError("Nymeria VDA scale was not requested automatically")
    resolved_scale_mode = str(scale_calibration.get("resolved_mode", ""))
    if resolved_scale_mode not in {"auto", "fixed_fallback"}:
        raise ValueError("Prepared VDA scale mode is invalid")
    if scale_calibration.get("scale_excludes_scene_scale") is not True:
        raise ValueError("Prepared VDA scale incorrectly includes scene scale")
    calibration_scale = float(
        scale_calibration.get("resolved_vda_metric_scale", math.nan)
    )
    if not math.isclose(
        calibration_scale,
        vda_metric_scale,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Prepared VDA scale audit disagrees with depth report")
    decision = scale_calibration.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("Prepared VDA scale has no quality decision")
    decision_reasons = decision.get("reasons")
    if not isinstance(decision_reasons, list):
        raise ValueError("Prepared VDA scale decision reasons are invalid")
    if resolved_scale_mode == "auto":
        if decision.get("status") != "validated" or decision_reasons:
            raise ValueError("Automatic VDA scale did not pass quality gates")
    else:
        fallback_reasons = scale_calibration.get("fallback_reason")
        if (
            decision.get("status") == "validated"
            or not isinstance(fallback_reasons, list)
            or not fallback_reasons
            or fallback_reasons != decision_reasons
            or not math.isclose(
                vda_metric_scale,
                DEFAULT_VDA_METRIC_SCALE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Prepared fixed VDA fallback audit is invalid")
    camera = report.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("Prepared report has no camera audit")
    camera_delta = float(
        camera.get("max_position_delta_from_recorded_m", math.inf)
    )
    if not math.isfinite(camera_delta) or camera_delta < 0.0:
        raise ValueError("Prepared camera delta is invalid")
    if (
        camera.get("scene_scale_source") == "retarget metadata"
        and camera_delta > 1e-5
    ):
        raise ValueError("Prepared and recorded camera poses disagree")
    expected_scale_source = (
        "retarget metadata"
        if plan.config.scene_scale is None
        else "explicit CLI override"
    )
    if camera.get("scene_scale_source") != expected_scale_source:
        raise ValueError("Prepared scene scale source differs from config")
    prepared_scale = float(camera.get("scene_scale", math.nan))
    if not math.isfinite(prepared_scale) or prepared_scale <= 0.0:
        raise ValueError("Prepared scene scale is invalid")
    if not math.isclose(
        depth_scene_scale,
        prepared_scale,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Prepared depth/camera scene scales disagree")
    if (
        plan.config.scene_scale is not None
        and not math.isclose(
            prepared_scale,
            plan.config.scene_scale,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Prepared scene scale override differs from config")
    if _numbered_png_count(root / "rgb") != clip.video.frame_count:
        raise ValueError("Prepared RGB frame count differs from source")
    if _numbered_png_count(root / "depth_mm") != clip.video.frame_count:
        raise ValueError("Prepared depth frame count differs from source")
    replacement = report.get("arm_replacement")
    if (
        not isinstance(replacement, dict)
        or replacement.get("manual_review") is not False
        or replacement.get("per_dataset_threshold_tuning") is not False
        or int(replacement.get("frame_count", -1))
        != clip.video.frame_count
    ):
        raise ValueError("Prepared arm-replacement audit is invalid")
    with np.load(root / "episode_data.npz", allow_pickle=False) as data:
        required = {
            "fps",
            "source_size",
            "mesh_size",
            "frame_indices",
            "human_arm_visible",
            "robot_arm_render",
            "source_dataset_kind",
            "joint_names",
            "joint_positions",
            "root_pose_xyzw",
            "camera_pose_cv_xyzw",
            "intrinsic",
            "mesh_intrinsic",
            "g1_urdf",
            "source_camera_pose_cv_xyzw",
            "recorded_camera_pose_cv_xyzw",
            "source_hdf5",
            "scene_scale",
            "vda_metric_scale",
            "vda_scale_mode",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Prepared NPZ is missing {sorted(missing)}")
        frame_count = clip.video.frame_count
        if str(np.asarray(data["source_dataset_kind"]).item()) != "nymeria":
            raise ValueError("Prepared NPZ does not identify Nymeria")
        fps = float(np.asarray(data["fps"]).item())
        if not math.isclose(fps, clip.video.fps, abs_tol=0.01):
            raise ValueError("Prepared FPS differs from source")
        source_size = tuple(
            int(value) for value in np.asarray(data["source_size"]).tolist()
        )
        if source_size != (clip.video.width, clip.video.height):
            raise ValueError("Prepared NPZ source size differs from source")
        mesh_size = np.asarray(data["mesh_size"], dtype=np.int64)
        if mesh_size.shape != (2,) or np.any(mesh_size <= 1):
            raise ValueError("Prepared mesh size is invalid")
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
        if not np.array_equal(frame_indices, np.arange(frame_count)):
            raise ValueError("Prepared frame indices are not contiguous")
        timelines = {}
        for key in (
            "human_arm_visible",
            "robot_arm_render",
        ):
            raw = np.asarray(data[key])
            if (
                raw.shape != (frame_count,)
                or raw.dtype.kind not in "biu"
                or not np.isin(raw, (0, 1)).all()
            ):
                raise ValueError(f"Prepared {key} timeline is invalid")
            timelines[key] = raw.astype(bool)
        if not np.array_equal(
            timelines["robot_arm_render"],
            timelines["human_arm_visible"],
        ):
            raise ValueError("Prepared robot-arm render decision is invalid")
        for key, report_key in (
            ("human_arm_visible", "human_arm_visible_frames"),
            ("robot_arm_render", "robot_arm_render_frames"),
        ):
            if int(replacement.get(report_key, -1)) != int(
                timelines[key].sum()
            ):
                raise ValueError(
                    f"Prepared {key} report count is inconsistent"
                )
        if int(replacement.get("view_only_frames", -1)) != int(
            (~timelines["robot_arm_render"]).sum()
        ):
            raise ValueError("Prepared view-only frame count is inconsistent")
        joint_names = np.asarray(data["joint_names"])
        if (
            joint_names.ndim != 1
            or len(joint_names) == 0
            or len(set(joint_names.tolist())) != len(joint_names)
        ):
            raise ValueError("Prepared joint names are invalid")
        arrays = {
            "joint_positions": (
                np.asarray(data["joint_positions"], dtype=np.float64),
                (frame_count, len(joint_names)),
            ),
            "root_pose_xyzw": (
                np.asarray(data["root_pose_xyzw"], dtype=np.float64),
                (frame_count, 7),
            ),
            "camera_pose_cv_xyzw": (
                np.asarray(data["camera_pose_cv_xyzw"], dtype=np.float64),
                (frame_count, 7),
            ),
        }
        for key, (values, expected_shape) in arrays.items():
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise ValueError(f"Prepared {key} is invalid")
        for key in ("root_pose_xyzw", "camera_pose_cv_xyzw"):
            quaternion_norm = np.linalg.norm(arrays[key][0][:, 3:], axis=1)
            if not np.allclose(quaternion_norm, 1.0, atol=1e-4):
                raise ValueError(f"Prepared {key} quaternion is invalid")
        for key in ("intrinsic", "mesh_intrinsic"):
            intrinsic = np.asarray(data[key], dtype=np.float64)
            if (
                intrinsic.shape != (3, 3)
                or not np.isfinite(intrinsic).all()
                or intrinsic[0, 0] <= 0.0
                or intrinsic[1, 1] <= 0.0
            ):
                raise ValueError(f"Prepared {key} is invalid")
        urdf_value = np.asarray(data["g1_urdf"])
        if urdf_value.shape != () or not Path(
            str(urdf_value.item())
        ).expanduser().resolve().is_file():
            raise ValueError("Prepared G1 URDF is invalid")
        npz_scale = float(np.asarray(data["scene_scale"]).item())
        if not math.isclose(
            npz_scale,
            prepared_scale,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Prepared NPZ/report scene scales disagree")
        npz_vda_scale = float(np.asarray(data["vda_metric_scale"]).item())
        if not math.isclose(
            npz_vda_scale,
            vda_metric_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Prepared NPZ/report VDA scales disagree")
        npz_scale_mode = str(np.asarray(data["vda_scale_mode"]).item())
        if npz_scale_mode != resolved_scale_mode:
            raise ValueError("Prepared NPZ/report VDA scale modes disagree")


def _validate_render(stage: Stage, plan: PipelinePlan) -> None:
    assert stage.episode_index is not None
    clip = _clip_by_index(plan, stage.episode_index)
    output = (
        plan.config.paths.final
        / f"episode_{clip.episode_index:06d}"
        / "g1_view.mp4"
    )
    rendered = probe_video(output)
    if not rendered.compatible_with(clip.video):
        raise ValueError(
            "Final G1 video timing or resolution differs from source: "
            f"{asdict(rendered)} != {asdict(clip.video)}"
        )
    report = _read_json(output.with_suffix(".json"))
    if report.get("capture_view") != "robot-camera":
        raise ValueError("Final video is not a robot-camera capture")
    if int(report.get("frame_count", -1)) != clip.video.frame_count:
        raise ValueError("Final report frame count differs from source")
    capture_camera = report.get("capture_camera")
    if not isinstance(capture_camera, dict) or (
        capture_camera.get("profile") != BEINGBEYOND_HEAD_CAMERA_PROFILE
    ):
        raise ValueError("Final render does not use the BeingBeyond camera")
    articulation = report.get("articulation_commit")
    if not isinstance(articulation, dict) or not bool(
        articulation.get("strict_validation_passed")
    ):
        raise ValueError("Isaac articulation commit validation did not pass")
    if (
        int(articulation.get("verified_application_count", -1))
        != clip.video.frame_count
    ):
        raise ValueError("Isaac articulation commit count differs from source")
    render_sync = articulation.get("render_sync_validation")
    if not isinstance(render_sync, dict) or not bool(render_sync.get("passed")):
        raise ValueError("Isaac render synchronization did not pass")
    robot = report.get("robot")
    visual_visibility = (
        robot.get("visual_visibility")
        if isinstance(robot, dict)
        else None
    )
    prepared_report = _read_json(
        plan.config.paths.prepared
        / f"episode_{clip.episode_index:06d}"
        / "prepare_report.json"
    )
    replacement = prepared_report.get("arm_replacement", {})
    if (
        not isinstance(visual_visibility, dict)
        or int(visual_visibility.get("visible_frames", -1))
        != int(replacement.get("robot_arm_render_frames", -2))
        or int(visual_visibility.get("hidden_frames", -1))
        != int(replacement.get("view_only_frames", -2))
    ):
        raise ValueError("Final robot visual-visibility audit is inconsistent")
    alignment = report.get("scene_alignment")
    if (
        not isinstance(alignment, dict)
        or alignment.get("method") != "disabled"
        or alignment.get("requested_mode") != "none"
        or int(alignment.get("application_count", -1)) != 0
    ):
        raise ValueError(
            "Nymeria final render must disable scene alignment"
        )
    translation = np.asarray(
        alignment.get("world_translation_xyz_m"),
        dtype=np.float64,
    )
    if (
        translation.shape != (3,)
        or not np.isfinite(translation).all()
        or not np.array_equal(translation, np.zeros(3, dtype=np.float64))
    ):
        raise ValueError("Disabled Nymeria scene alignment must be identity")
    operation_order = report.get("operation_order")
    if operation_order != []:
        raise ValueError("Nymeria scene-placement order is invalid")
    leveling = report.get("scene_leveling")
    if (
        not isinstance(leveling, dict)
        or leveling.get("status") != "disabled"
        or leveling.get("requested_mode") != "none"
        or int(leveling.get("application_count", -1)) != 0
    ):
        raise ValueError("Nymeria final render must disable scene leveling")


def validate_stage(stage: Stage, plan: PipelinePlan) -> tuple[bool, str]:
    try:
        if stage.name == "alignment":
            _validate_alignment(stage, plan)
        elif stage.name == "retarget":
            _validate_retarget(plan)
        elif stage.name == "repaint":
            _validate_repaint(plan)
        elif stage.name == "prepare":
            _validate_prepared(stage, plan)
        elif stage.name == "render":
            _validate_render(stage, plan)
        else:
            raise ValueError(f"Unknown stage {stage.name!r}")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "validated"


def _clear_stage(stage: Stage, plan: PipelinePlan) -> None:
    paths = plan.config.paths
    if stage.name == "alignment":
        if plan.config.alignment_is_external:
            raise RuntimeError(
                "External camera alignment failed validation and is read-only"
            )
        assert stage.session is not None
        _remove_exact(plan.config.alignment_root / stage.session)
        _remove_exact(paths.alignment_reports / stage.session)
    elif stage.name == "retarget":
        _remove_exact(paths.lerobot)
    elif stage.name == "repaint":
        _remove_exact(paths.handless)
    elif stage.name == "prepare":
        assert stage.episode_index is not None
        _remove_exact(
            paths.prepared / f"episode_{stage.episode_index:06d}"
        )
    elif stage.name == "render":
        assert stage.episode_index is not None
        _remove_exact(paths.final / f"episode_{stage.episode_index:06d}")


def _run_stage(
    stage: Stage,
    progress: tqdm[Any],
    *,
    pipeline_started: float,
    poll_seconds: float,
) -> None:
    assert stage.command is not None
    _run_process(
        command=stage.command,
        log_path=stage.log_path,
        stage_id=stage.stage_id,
        stage_name=stage.name,
        item_key="item",
        item_label=stage.label,
        progress=progress,
        pipeline_started=pipeline_started,
        poll_seconds=poll_seconds,
        cwd=WORKSPACE_ROOT,
    )


def _new_report(plan: PipelinePlan) -> dict[str, Any]:
    config = plan.config
    return {
        "schema_version": 2,
        "pipeline": (
            "nymeria_camera_alignment -> nymeria full-body GMR LeRobot -> "
            "arm repaint -> metric VDA RGB-D -> Isaac robot camera"
        ),
        "status": "running",
        "source_root": str(config.src_root),
        "output_root": str(config.paths.root),
        "camera_alignment_root": str(config.alignment_root),
        "camera_alignment_report_root": str(
            config.alignment_report_root
        ),
        "camera_alignment_source": (
            "external_read_only"
            if config.alignment_is_external
            else "generated_by_this_pipeline"
        ),
        "camera_mapping": (
            "EgoDex camera-relative-to-hip mapping onto current GMR root; "
            "GMR wrist scale applied once"
        ),
        "hand_policy": (
            "51-DoF G1 with 29 retargeted body joints and 22 fixed-open hand "
            "joints; no synthetic Nymeria fingertips"
        ),
        "visibility_policy": (
            "robot camera is active on every frame; SAM 3 native target "
            "presence directly selects automatic arm replacement and G1 "
            "visual visibility, otherwise the original RGB is kept and G1 "
            "visuals are hidden"
        ),
        "scene_alignment_policy": (
            "support-plane leveling and wrist-derived scene translation are "
            "disabled; calibrated Nymeria scene coordinates stay unchanged"
        ),
        "video_metadata_policy": (
            "ffprobe decoded/displayed frame count, source FPS and source "
            "resolution; container-declared GOP preroll is excluded"
        ),
        "stop_after": config.stop_after,
        "resume": config.resume,
        "overwrite": config.overwrite,
        "selected_clips": [
            {
                "episode_index": clip.episode_index,
                "session": clip.session,
                "source_stem": clip.stem,
                "source_video": str(clip.source.video_path),
                "video": asdict(clip.video),
            }
            for clip in plan.clips
        ],
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "elapsed_seconds": 0.0,
        "skipped_episodes": [],
        "stages": [
            {
                "stage_id": stage.stage_id,
                "name": stage.name,
                "session": stage.session,
                "episode_index": stage.episode_index,
                "status": "pending",
                "command": (
                    None if stage.command is None else list(stage.command)
                ),
                "log": (
                    None if stage.command is None else str(stage.log_path)
                ),
                "dependencies": list(stage.dependencies),
                "input_fingerprint": None,
                "validation": None,
                "started_at": None,
                "finished_at": None,
                "elapsed_seconds": None,
            }
            for stage in plan.stages
        ],
    }


def _initialize_output(plan: PipelinePlan) -> None:
    config = plan.config
    paths = config.paths
    owned_targets = [
        paths.lerobot,
        paths.handless,
        paths.prepared,
        paths.final,
        paths.logs,
        paths.report,
    ]
    if not config.alignment_is_external:
        owned_targets.extend(
            [paths.camera_alignment, paths.alignment_reports]
        )
    if config.overwrite:
        for target in owned_targets:
            _remove_exact(target)
    elif not config.resume:
        existing = [target for target in owned_targets if target.exists()]
        if existing:
            raise FileExistsError(
                "Pipeline outputs already exist while resume is disabled: "
                + ", ".join(str(path) for path in existing)
            )
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)


def execute(plan: PipelinePlan) -> dict[str, Any]:
    _initialize_output(plan)
    previous_stages = _load_previous_stages(
        plan.config.paths.report
    )
    report = _new_report(plan)
    report_by_id = {
        str(stage["stage_id"]): stage for stage in report["stages"]
    }
    reusable_statuses = {
        "completed",
        "reused",
        "completed_with_skips",
        "reused_with_skips",
    }
    for stage_id, entry in report_by_id.items():
        previous = previous_stages.get(stage_id, {})
        if previous.get("status") not in reusable_statuses:
            continue
        for key in (
            "status",
            "input_fingerprint",
            "validation",
            "started_at",
            "finished_at",
            "elapsed_seconds",
        ):
            entry[key] = previous.get(key)
    pipeline_started = time.perf_counter()
    changed: set[str] = set()
    skipped_by_episode: dict[int, dict[str, Any]] = {}
    dependents: dict[str, list[str]] = {}
    for planned_stage in plan.stages:
        for dependency in planned_stage.dependencies:
            dependents.setdefault(dependency, []).append(
                planned_stage.stage_id
            )

    def invalidate_stage_and_dependents(stage_id: str) -> None:
        pending = [stage_id]
        invalidated: set[str] = set()
        while pending:
            current = pending.pop()
            if current in invalidated:
                continue
            invalidated.add(current)
            pending.extend(dependents.get(current, ()))
        for invalidated_id in invalidated:
            entry = report_by_id[invalidated_id]
            entry["input_fingerprint"] = None
            if invalidated_id != stage_id:
                entry.update(
                    status="invalidated",
                    started_at=None,
                    finished_at=None,
                    elapsed_seconds=None,
                    validation=f"dependency {stage_id} will rerun",
                )

    def record_skipped_episode(
        episode_index: int,
        stage: Stage,
        error: str,
        *,
        failure_status: str = "exception",
    ) -> None:
        clip = _clip_by_index(plan, episode_index)
        source = clip.source
        session_root = plan.config.src_root / clip.session
        skipped_by_episode.setdefault(
            episode_index,
            {
                "episode_index": episode_index,
                "session": clip.session,
                "source_stem": clip.stem,
                "source_path": str(source.video_path),
                "input_paths": {
                    "video": str(source.video_path),
                    "motion_208": str(source.motion_208_path),
                    "text": str(source.text_path),
                    "smpl": str(source.smpl_path),
                    "session_metadata": str(source.session_metadata_path),
                    "camera_intrinsics": str(
                        source.camera_intrinsics_path
                    ),
                    "closed_loop_trajectory": str(
                        session_root
                        / "recording_head/mps/slam/"
                        "closed_loop_trajectory.csv"
                    ),
                    "online_calibration": str(
                        session_root
                        / "recording_head/mps/slam/"
                        "online_calibration.jsonl"
                    ),
                    "camera_alignment": str(
                        plan.config.alignment_root
                        / clip.session
                        / f"{clip.stem}.hdf5"
                    ),
                },
                "status": "skipped",
                "failure_status": failure_status,
                "failed_stage": stage.stage_id,
                "error": error,
                "log": (
                    None
                    if stage.command is None
                    else str(stage.log_path)
                ),
                "stage_paths": {
                    "lerobot_root": str(plan.config.paths.lerobot),
                    "handless_root": str(plan.config.paths.handless),
                    "prepared_episode": str(
                        plan.config.paths.prepared
                        / f"episode_{episode_index:06d}"
                    ),
                    "final_video": str(
                        plan.config.paths.final
                        / f"episode_{episode_index:06d}"
                        / "g1_view.mp4"
                    ),
                },
                "recorded_at": _utc_now(),
            },
        )
        report["skipped_episodes"] = list(skipped_by_episode.values())

    _write_json_atomic(plan.config.paths.report, report)

    with tqdm(
        total=len(plan.stages),
        desc="Nymeria VDA",
        unit="stage",
        dynamic_ncols=True,
    ) as progress:
        try:
            for stage in plan.stages:
                entry = report_by_id[stage.stage_id]
                if (
                    stage.episode_index is not None
                    and stage.episode_index in skipped_by_episode
                ):
                    cleanup_error = None
                    try:
                        _clear_stage(stage, plan)
                    except Exception as exc:
                        cleanup_error = f"{type(exc).__name__}: {exc}"
                        skipped_by_episode[stage.episode_index].setdefault(
                            "cleanup_errors",
                            [],
                        ).append(
                            {
                                "stage": stage.stage_id,
                                "error": cleanup_error,
                            }
                        )
                    now = _utc_now()
                    failed_stage = skipped_by_episode[
                        stage.episode_index
                    ]["failed_stage"]
                    entry.update(
                        status="skipped",
                        started_at=now,
                        finished_at=now,
                        elapsed_seconds=0.0,
                        input_fingerprint=None,
                        validation=(
                            f"episode skipped after {failed_stage} failed"
                            + (
                                f"; cleanup failed: {cleanup_error}"
                                if cleanup_error
                                else ""
                            )
                        ),
                    )
                    progress.update(1)
                    report["updated_at"] = now
                    report["elapsed_seconds"] = (
                        time.perf_counter() - pipeline_started
                    )
                    _write_json_atomic(plan.config.paths.report, report)
                    continue

                dependencies_changed = any(
                    dependency in changed for dependency in stage.dependencies
                )
                previous = previous_stages.get(stage.stage_id, {})
                previously_complete = (
                    previous.get("status") in reusable_statuses
                )
                input_fingerprint: str | None = None
                inputs_unchanged = False
                if dependencies_changed:
                    valid, reason = False, "dependency changed"
                elif stage.command is None:
                    input_fingerprint = _stage_input_fingerprint(stage, plan)
                    entry["input_fingerprint"] = input_fingerprint
                    inputs_unchanged = (
                        previous.get("input_fingerprint")
                        == input_fingerprint
                    )
                    valid, reason = validate_stage(stage, plan)
                elif plan.config.resume and previously_complete:
                    try:
                        input_fingerprint = _stage_input_fingerprint(
                            stage,
                            plan,
                        )
                    except Exception as exc:
                        if stage.episode_index is None:
                            raise
                        error = f"{type(exc).__name__}: {exc}"
                        now = _utc_now()
                        entry.update(
                            status="failed",
                            started_at=now,
                            finished_at=now,
                            elapsed_seconds=0.0,
                            input_fingerprint=None,
                            validation=error,
                        )
                        record_skipped_episode(
                            stage.episode_index,
                            stage,
                            error,
                        )
                        report.update(
                            updated_at=now,
                            elapsed_seconds=(
                                time.perf_counter() - pipeline_started
                            ),
                        )
                        _write_json_atomic(
                            plan.config.paths.report,
                            report,
                        )
                        progress.update(1)
                        continue
                    entry["input_fingerprint"] = input_fingerprint
                    inputs_unchanged = (
                        previous.get("input_fingerprint")
                        == input_fingerprint
                    )
                    if inputs_unchanged:
                        valid, reason = validate_stage(stage, plan)
                    else:
                        valid, reason = False, "input fingerprint changed"
                else:
                    valid = False
                    reason = (
                        "resume disabled"
                        if not plan.config.resume
                        else "previous stage is incomplete"
                    )
                may_reuse = (
                    plan.config.resume
                    and not dependencies_changed
                    and valid
                    and previously_complete
                    and inputs_unchanged
                )
                if may_reuse:
                    repaint_failures = (
                        _repaint_manifest(plan)[0]
                        if stage.name == "repaint"
                        else {}
                    )
                    for episode_index, failure in repaint_failures.items():
                        record_skipped_episode(
                            episode_index,
                            stage,
                            failure["error"],
                            failure_status=failure["status"],
                        )
                    now = _utc_now()
                    entry.update(
                        status=(
                            "reused_with_skips"
                            if repaint_failures
                            else "reused"
                        ),
                        started_at=now,
                        finished_at=now,
                        elapsed_seconds=0.0,
                        validation=(
                            f"validated with {len(repaint_failures)} "
                            "skipped episode(s)"
                            if repaint_failures
                            else reason
                        ),
                    )
                    progress.update(1)
                    report["updated_at"] = now
                    report["elapsed_seconds"] = (
                        time.perf_counter() - pipeline_started
                    )
                    _write_json_atomic(plan.config.paths.report, report)
                    continue
                if stage.command is None:
                    if not valid:
                        error = RuntimeError(
                            f"{stage.stage_id} external validation failed: "
                            f"{reason}"
                        )
                        now = _utc_now()
                        entry.update(
                            status="failed",
                            started_at=now,
                            finished_at=now,
                            elapsed_seconds=0.0,
                            validation=str(error),
                        )
                        report.update(
                            status="failed",
                            failed_stage=stage.stage_id,
                            error=f"RuntimeError: {error}",
                            updated_at=now,
                            elapsed_seconds=(
                                time.perf_counter() - pipeline_started
                            ),
                        )
                        _write_json_atomic(
                            plan.config.paths.report,
                            report,
                        )
                        raise error
                    invalidate_stage_and_dependents(stage.stage_id)
                    changed.add(stage.stage_id)
                    report["updated_at"] = _utc_now()
                    _write_json_atomic(plan.config.paths.report, report)
                    now = _utc_now()
                    entry.update(
                        status="completed",
                        started_at=now,
                        finished_at=now,
                        elapsed_seconds=0.0,
                        input_fingerprint=_stage_input_fingerprint(
                            stage,
                            plan,
                        ),
                        validation=reason,
                    )
                    progress.update(1)
                    report["updated_at"] = now
                    report["elapsed_seconds"] = (
                        time.perf_counter() - pipeline_started
                    )
                    _write_json_atomic(plan.config.paths.report, report)
                    continue

                invalidate_stage_and_dependents(stage.stage_id)
                changed.add(stage.stage_id)
                stage_started = time.perf_counter()
                entry.update(
                    status="running",
                    started_at=_utc_now(),
                    validation=(
                        "dependency changed"
                        if dependencies_changed
                        else reason
                    ),
                )
                _write_json_atomic(plan.config.paths.report, report)
                try:
                    _clear_stage(stage, plan)
                    _run_stage(
                        stage,
                        progress,
                        pipeline_started=pipeline_started,
                        poll_seconds=plan.config.poll_seconds,
                    )
                    valid, reason = validate_stage(stage, plan)
                    if not valid:
                        raise RuntimeError(
                            f"{stage.stage_id} output validation failed: "
                            f"{reason}"
                        )
                    completed_fingerprint = _stage_input_fingerprint(
                        stage,
                        plan,
                    )
                    repaint_failures = (
                        _repaint_manifest(plan)[0]
                        if stage.name == "repaint"
                        else {}
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    entry.update(
                        status="failed",
                        finished_at=_utc_now(),
                        elapsed_seconds=time.perf_counter() - stage_started,
                        validation=str(exc),
                    )
                    if stage.episode_index is not None:
                        record_skipped_episode(
                            stage.episode_index,
                            stage,
                            error,
                        )
                        report.update(
                            updated_at=_utc_now(),
                            elapsed_seconds=(
                                time.perf_counter() - pipeline_started
                            ),
                        )
                        _write_json_atomic(
                            plan.config.paths.report,
                            report,
                        )
                        progress.update(1)
                        continue
                    report.update(
                        status="failed",
                        failed_stage=stage.stage_id,
                        error=error,
                        updated_at=_utc_now(),
                        elapsed_seconds=time.perf_counter() - pipeline_started,
                    )
                    _write_json_atomic(plan.config.paths.report, report)
                    raise
                for episode_index, failure in repaint_failures.items():
                    record_skipped_episode(
                        episode_index,
                        stage,
                        failure["error"],
                        failure_status=failure["status"],
                    )
                entry.update(
                    status=(
                        "completed_with_skips"
                        if repaint_failures
                        else "completed"
                    ),
                    finished_at=_utc_now(),
                    elapsed_seconds=time.perf_counter() - stage_started,
                    input_fingerprint=completed_fingerprint,
                    validation=(
                        f"validated with {len(repaint_failures)} "
                        "skipped episode(s)"
                        if repaint_failures
                        else reason
                    ),
                )
                report["updated_at"] = _utc_now()
                report["elapsed_seconds"] = (
                    time.perf_counter() - pipeline_started
                )
                _write_json_atomic(plan.config.paths.report, report)
                progress.update(1)

            report.update(
                status=(
                    "completed_with_skips"
                    if skipped_by_episode
                    else "completed"
                ),
                finished_at=_utc_now(),
                updated_at=_utc_now(),
                elapsed_seconds=time.perf_counter() - pipeline_started,
            )
            if plan.config.stop_after == "render":
                completed_render_indices = {
                    stage.episode_index
                    for stage in plan.stages
                    if stage.name == "render"
                    and report_by_id[stage.stage_id]["status"]
                    in {"completed", "reused"}
                }
                report["final_videos"] = [
                    str(
                        plan.config.paths.final
                        / f"episode_{clip.episode_index:06d}"
                        / "g1_view.mp4"
                    )
                    for clip in plan.clips
                    if clip.episode_index in completed_render_indices
                ]
            _write_json_atomic(plan.config.paths.report, report)
            return report
        except BaseException:
            if report.get("status") == "running":
                report.update(
                    status="interrupted",
                    updated_at=_utc_now(),
                    elapsed_seconds=time.perf_counter() - pipeline_started,
                )
                _write_json_atomic(plan.config.paths.report, report)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrated Nymeria -> full-body G1 -> handless Metric VDA -> "
            "source-sized Isaac robot-camera pipeline."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Nymeria dataset root containing session directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Pipeline-owned output directory; must be disjoint from the "
            "dataset root."
        ),
    )
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument(
        "--clip",
        action="append",
        default=None,
        help="Exact full stem or an unambiguous cNNN suffix; repeat as needed.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hardware-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    src_root = args.dataset_root.expanduser().resolve()
    if not src_root.is_dir():
        raise NotADirectoryError(src_root)
    output_root = args.output_root.expanduser().resolve()
    if output_root in {WORKSPACE_ROOT, WORKSPACE_ROOT / "nymeria"}:
        raise ValueError(
            "--output-root must be a dedicated pipeline subdirectory, not "
            "the workspace or the shared nymeria directory"
        )
    if _paths_overlap(output_root, src_root):
        raise ValueError(
            "--output-root and --dataset-root must be disjoint; pipeline "
            "cleanup must never touch source data"
        )
    # Keep the virtual-environment entry point intact.  uv creates
    # ``.venv/bin/python`` as a symlink to its managed base interpreter;
    # resolving that symlink would make child stages lose .venv site-packages.
    executable = Path(sys.executable).absolute()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    paths = PipelinePaths.from_root(output_root)
    return PipelineConfig(
        src_root=src_root,
        paths=paths,
        alignment_root=paths.camera_alignment,
        alignment_report_root=paths.alignment_reports,
        alignment_is_external=False,
        python_executable=executable,
        resume=True,
        overwrite=False,
        headless=bool(args.headless),
        require_visible_hands=False,
        hardware_encoding=bool(args.hardware_encoding),
        hand_eye_sample_fps=15.0,
        scene_scale=None,
        vda_checkpoint=None,
        stop_after="render",
        poll_seconds=0.5,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = _config_from_args(args)
    plan = build_plan(
        config,
        sessions=set(args.session) if args.session else None,
        requested_clips=set(args.clip) if args.clip else None,
    )
    execute(plan)


if __name__ == "__main__":
    main()
