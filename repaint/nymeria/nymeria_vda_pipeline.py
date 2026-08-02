#!/usr/bin/env python3
"""Run the calibrated Nymeria-to-G1 Metric VDA/Isaac pipeline.

The route mirrors the current EgoDex repaint pipeline after adapting Nymeria's
native inputs:

    raw Aria MPS camera trajectory
      -> Nymeria full-body GMR LeRobot (fixed-open hands)
      -> human-arm repaint
      -> metric Video Depth Anything RGB-D preparation
      -> source-sized Isaac robot-camera replay

The first five frames fit one fixed Aria-world-to-G1-world camera transform.
The original camera rotation trajectory is preserved without a rejection
threshold, and Isaac scene alignment is disabled because retargeting already
anchors the camera.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from io import StringIO
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

_write_notice = tqdm.write


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
from repaint.lighting_profile import load_lighting_profile  # noqa: E402
with redirect_stdout(StringIO()):
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
    load_previous_stages as _load_previous_stages,
    probe_video,
    read_json as _read_json,
    remove_exact as _remove_exact,
    run_stage_process as _run_process,
    total_progress_bar_format,
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
    scene_scale: float | None
    vda_checkpoint: Path | None
    stop_after: str
    poll_seconds: float
    retain_intermediates: bool = False


@dataclass(frozen=True)
class Stage:
    stage_id: str
    name: str
    command: tuple[str, ...] | None
    dependencies: tuple[str, ...] = ()
    session: str | None = None
    episode_index: int | None = None
    clip_stems: tuple[str, ...] = ()


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


@lru_cache(maxsize=32)
def _file_content_digest(
    path: str,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> bytes:
    del size, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


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
            digest.update(
                _file_content_digest(
                    str(candidate),
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            )


def _repaint_episode_video_paths(
    plan: PipelinePlan,
    episode_index: int,
) -> tuple[Path, Path]:
    paths = plan.config.paths
    info = _read_json(paths.lerobot / "meta/info.json")
    rows = _episode_rows(paths.lerobot)
    try:
        row = rows[episode_index]
    except KeyError as exc:
        raise ValueError(
            f"LeRobot has no metadata row for episode {episode_index}"
        ) from exc
    return (
        _video_path(paths.lerobot, info, row),
        _video_path(paths.handless, info, row),
    )


def _update_repaint_episode_fingerprint(
    digest: Any,
    plan: PipelinePlan,
    episode_index: int,
) -> None:
    paths = plan.config.paths
    source_video, output_video = _repaint_episode_video_paths(
        plan,
        episode_index,
    )
    source_video = source_video.resolve()
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


def _stage_input_fingerprint(
    stage: Stage,
    plan: PipelinePlan,
    *,
    include_pruned_outputs: bool = True,
) -> str:
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
        if include_pruned_outputs:
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
        if include_pruned_outputs:
            _update_path_fingerprint(digest, paths.lerobot)
            _update_path_fingerprint(digest, paths.handless)
    elif stage.name in {"prepare", "render"}:
        assert stage.episode_index is not None
        label = f"episode_{stage.episode_index:06d}"
        if include_pruned_outputs:
            _update_path_fingerprint(digest, paths.prepared / label)
        if stage.name == "prepare":
            if include_pruned_outputs:
                _update_path_fingerprint(digest, paths.lerobot / "meta")
                _update_repaint_episode_fingerprint(
                    digest,
                    plan,
                    stage.episode_index,
                )
            if plan.clips:
                clip = _clip_by_index(plan, stage.episode_index)
                _update_path_fingerprint(
                    digest,
                    clip.source.video_path.parents[1]
                    / "recording_head/mps/slam/semidense_points.csv.gz",
                )
            _update_path_fingerprint(
                digest,
                plan.config.vda_checkpoint or DEFAULT_CHECKPOINT,
                hash_contents=True,
            )
        else:
            final_root = paths.final / label
            _update_path_fingerprint(digest, final_root)
            _update_path_fingerprint(
                digest,
                final_root / "g1_view.json",
                hash_contents=True,
            )
    return digest.hexdigest()


def _retained_render_fingerprint(
    render_stage: Stage,
    plan: PipelinePlan,
) -> str:
    """Receipt for pruned stages; retained LeRobot is checked separately."""

    assert render_stage.name == "render"
    assert render_stage.episode_index is not None
    related = [
        stage
        for stage in plan.stages
        if stage.name == "repaint"
        or (
            stage.name == "prepare"
            and stage.episode_index == render_stage.episode_index
        )
    ]
    if [stage.name for stage in related] != ["repaint", "prepare"]:
        raise ValueError("Render retention dependencies are incomplete")
    digest = hashlib.sha256()
    for stage in (*related, render_stage):
        digest.update(stage.stage_id.encode())
        digest.update(
            _stage_input_fingerprint(
                stage,
                plan,
                include_pruned_outputs=False,
            ).encode()
        )
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
                    dependencies=("retarget", "repaint"),
                    episode_index=clip.episode_index,
                )
            )
            if not _stage_enabled(config.stop_after, "render"):
                continue
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
    resolved_root = root.resolve()
    path = (root / str(info["video_path"]).format(
        video_key=VIDEO_KEY,
        chunk_index=int(row[f"videos/{VIDEO_KEY}/chunk_index"]),
        file_index=int(row[f"videos/{VIDEO_KEY}/file_index"]),
    )).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError(f"Video path escapes its dataset root: {path}")
    return path


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
    trajectory = report.get("camera_trajectory")
    if (
        not isinstance(trajectory, dict)
        or trajectory.get("uses_208d_pose") is not False
        or trajectory.get("rotation_rejection_threshold_degrees") is not None
    ):
        raise ValueError(
            "Nymeria alignment is not a threshold-free raw camera trajectory"
        )

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


def _validate_retarget(
    plan: PipelinePlan,
    *,
    allow_pruned_alignment: bool = False,
) -> None:
    root = plan.config.paths.lerobot
    info = _read_json(root / "meta/info.json")
    manifest = _read_json(root / "meta/nymeria.json")
    quality_gate = manifest.get("camera_alignment_quality_gate", {})
    if not isinstance(quality_gate, dict) or quality_gate.get("status") != "passed":
        raise ValueError("Nymeria LeRobot camera alignment gate did not pass")
    report_entries = quality_gate.get("camera_trajectory_reports")
    if not isinstance(report_entries, list):
        raise ValueError("Nymeria LeRobot has no camera trajectory provenance")
    expected_sessions = {clip.session for clip in plan.clips}
    recorded_reports: dict[str, dict[str, Any]] = {}
    for item in report_entries:
        if not isinstance(item, dict):
            raise ValueError("Invalid camera trajectory provenance entry")
        report_path = Path(str(item.get("path", ""))).resolve()
        if report_path.name != "visual_alignment_report.json":
            raise ValueError("Invalid camera trajectory provenance path")
        session = report_path.parent.name
        expected_path = (
            plan.config.alignment_report_root
            / session
            / "visual_alignment_report.json"
        ).resolve()
        if report_path != expected_path or session in recorded_reports:
            raise ValueError("Camera trajectory provenance does not match config")
        if report_path.is_file():
            if hashlib.sha256(report_path.read_bytes()).hexdigest() != item.get(
                "sha256"
            ):
                raise ValueError(
                    f"Camera trajectory report changed after retarget: {session}"
                )
        elif not allow_pruned_alignment:
            raise FileNotFoundError(report_path)
        recorded_reports[session] = item
    if set(recorded_reports) != expected_sessions:
        raise ValueError("Camera trajectory provenance has incorrect sessions")
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
        if adapter_path != expected_adapter or (
            not allow_pruned_alignment and not adapter_path.is_file()
        ):
            raise ValueError(f"{clip.stem} camera adapter path changed")
        if adapter_path.is_file():
            digest = hashlib.sha256()
            with adapter_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != alignment.get("adapter_hdf5_sha256"):
                raise ValueError(f"{clip.stem} camera adapter content changed")
        if alignment.get("camera_trajectory_report") != recorded_reports[
            clip.session
        ]:
            raise ValueError(f"{clip.stem} camera trajectory provenance changed")
        metrics = alignment.get("metrics", {})
        if (
            metrics.get("mapping")
            != "raw Aria camera to G1 camera, fixed first-five-frame similarity"
            or metrics.get("source_camera_rotation_rejection_threshold_degrees")
            is not None
        ):
            raise ValueError(f"{clip.stem} camera mapping policy changed")
        if int(metrics.get("scene_scale_application_count", -1)) != 1:
            raise ValueError(f"{clip.stem} scene scale was not applied exactly once")
        scale = float(
            alignment["source_to_robot_camera_translation_scale"]
        )
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{clip.stem} has an invalid camera scale")
        rotation = np.asarray(
            alignment.get("source_to_robot_rotation"),
            dtype=np.float64,
        )
        translation = np.asarray(
            alignment.get("source_to_robot_translation"),
            dtype=np.float64,
        )
        if (
            rotation.shape != (3, 3)
            or translation.shape != (3,)
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
        ):
            raise ValueError(f"{clip.stem} has an invalid fixed camera transform")

        row = rows[clip.episode_index]
        length = int(row["length"])
        if length != clip.video.frame_count:
            raise ValueError(f"{clip.stem} LeRobot length differs from source")
        if int(metrics.get("fit_window_frames", -1)) != min(5, length):
            raise ValueError(f"{clip.stem} did not use the first five camera frames")
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
    lighting_path = root / "lighting_profile.json"
    lighting_profile = load_lighting_profile(lighting_path)
    if lighting_profile is None:
        raise ValueError("Prepared lighting profile is missing")
    lighting_report = report.get("lighting")
    if (
        not isinstance(lighting_report, dict)
        or Path(str(lighting_report.get("profile", ""))).resolve()
        != lighting_path.resolve()
    ):
        raise ValueError("Prepared lighting audit is invalid")
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
    if (
        scale_calibration.get("method")
        != "aria-mps-semidense-background-depth"
    ):
        raise ValueError("Nymeria VDA scale did not use Aria semidense points")
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
            "lighting_profile",
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
        npz_lighting_path = Path(
            str(np.asarray(data["lighting_profile"]).item())
        ).expanduser().resolve()
        if npz_lighting_path != lighting_path.resolve():
            raise ValueError("Prepared NPZ/report lighting profiles disagree")


def _validate_render(
    stage: Stage,
    plan: PipelinePlan,
    *,
    allow_pruned_prepared: bool = False,
) -> None:
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
    lighting = report.get("lighting")
    if (
        not isinstance(lighting, dict)
        or lighting.get("applied") is not True
        or int(lighting.get("profile_version", -1)) != 1
        or not isinstance(lighting.get("dome"), dict)
        or not isinstance(lighting.get("key"), dict)
    ):
        raise ValueError("Final Isaac lighting profile was not applied")
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
    prepared_report_path = (
        plan.config.paths.prepared
        / f"episode_{clip.episode_index:06d}"
        / "prepare_report.json"
    )
    if not isinstance(visual_visibility, dict):
        raise ValueError("Final robot visual-visibility audit is inconsistent")
    visible_frames = int(visual_visibility.get("visible_frames", -1))
    hidden_frames = int(visual_visibility.get("hidden_frames", -1))
    if prepared_report_path.is_file():
        replacement = _read_json(prepared_report_path).get(
            "arm_replacement",
            {},
        )
        if (
            visible_frames
            != int(replacement.get("robot_arm_render_frames", -2))
            or hidden_frames != int(replacement.get("view_only_frames", -2))
        ):
            raise ValueError(
                "Final robot visual-visibility audit is inconsistent"
            )
    elif not allow_pruned_prepared:
        raise FileNotFoundError(prepared_report_path)
    elif (
        visible_frames < 0
        or hidden_frames < 0
        or visible_frames + hidden_frames != clip.video.frame_count
    ):
        raise ValueError("Final robot visual-visibility totals are invalid")
    alignment = report.get("scene_alignment")
    if (
        not isinstance(alignment, dict)
        or alignment.get("method") != "disabled"
        or alignment.get("requested_mode") != "none"
        or int(alignment.get("application_count", -1)) != 0
    ):
        raise ValueError("Nymeria final render must disable scene alignment")
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
        _remove_exact(
            plan.config.alignment_root / stage.session,
            owner_root=paths.root,
        )
        _remove_exact(
            paths.alignment_reports / stage.session,
            owner_root=paths.root,
        )
    elif stage.name == "retarget":
        _remove_exact(paths.lerobot, owner_root=paths.root)
    elif stage.name == "repaint":
        _remove_exact(paths.handless, owner_root=paths.root)
    elif stage.name == "prepare":
        assert stage.episode_index is not None
        _remove_exact(
            paths.prepared / f"episode_{stage.episode_index:06d}",
            owner_root=paths.root,
        )
    elif stage.name == "render":
        assert stage.episode_index is not None
        _remove_exact(
            paths.final / f"episode_{stage.episode_index:06d}",
            owner_root=paths.root,
        )


def _clear_episode_intermediates(
    plan: PipelinePlan,
    episode_index: int,
) -> None:
    targets = [
        plan.config.paths.prepared / f"episode_{episode_index:06d}",
    ]
    errors: list[str] = []
    try:
        _, handless_video = _repaint_episode_video_paths(
            plan,
            episode_index,
        )
        targets.append(handless_video)
    except Exception as exc:
        errors.append(f"handless path: {type(exc).__name__}: {exc}")
    if not plan.config.alignment_is_external:
        clip = _clip_by_index(plan, episode_index)
        targets.append(
            plan.config.alignment_root
            / clip.session
            / f"{clip.stem}.hdf5"
        )
    for target in targets:
        try:
            _remove_exact(target, owner_root=plan.config.paths.root)
        except Exception as exc:
            errors.append(f"{target}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _run_stage(
    stage: Stage,
    progress: tqdm[Any],
    *,
    poll_seconds: float,
) -> None:
    assert stage.command is not None
    _run_process(
        command=stage.command,
        stage_id=stage.stage_id,
        progress=progress,
        poll_seconds=poll_seconds,
        cwd=WORKSPACE_ROOT,
    )


def _clip_report(clip: SelectedClip) -> dict[str, Any]:
    return {
        "episode_index": clip.episode_index,
        "session": clip.session,
        "source_stem": clip.stem,
        "source_video": str(clip.source.video_path),
        "video": asdict(clip.video),
    }


def _completion_path(plan: PipelinePlan, episode_index: int) -> Path:
    return (
        plan.config.paths.final
        / f"episode_{episode_index:06d}"
        / "completion.json"
    )


def _write_completion(plan: PipelinePlan, episode_index: int) -> None:
    _write_json_atomic(
        _completion_path(plan, episode_index),
        {
            "schema_version": 1,
            "status": "completed",
            "source_episode": _clip_report(
                _clip_by_index(plan, episode_index)
            ),
            "updated_at": _utc_now(),
        },
    )


def _completion_matches(
    plan: PipelinePlan,
    episode_index: int,
    previous_report: dict[str, Any],
) -> bool:
    expected = _clip_report(_clip_by_index(plan, episode_index))
    try:
        if (
            _read_json(_completion_path(plan, episode_index)).get(
                "source_episode"
            )
            == expected
        ):
            return True
    except Exception:
        pass
    previous_sources = previous_report.get("selected_clips")
    return isinstance(previous_sources, list) and expected in previous_sources


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
            "raw Aria camera mapped to the exact G1 head camera by one "
            "fixed first-five-frame similarity; GMR wrist scale applied once"
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
            "support-plane leveling and Isaac scene alignment disabled; "
            "retargeting already anchors the raw Aria camera to G1 world"
        ),
        "video_metadata_policy": (
            "ffprobe decoded/displayed frame count, source FPS and source "
            "resolution; container-declared GOP preroll is excluded"
        ),
        "stop_after": config.stop_after,
        "resume": config.resume,
        "overwrite": config.overwrite,
        "retain_intermediates": config.retain_intermediates,
        "episode_failure_policy": (
            "repaint manifest failures are terminal skips on resume; "
            "prepare/render exceptions are retried on the next run"
        ),
        "intermediate_retention": (
            "alignment, LeRobot, handless, prepared, and final artifacts are "
            "retained"
            if config.retain_intermediates
            else (
                "handless, prepared, and pipeline-owned alignment artifacts "
                "are removed after render; only LeRobot, final outputs, and "
                "this report are retained; failed prepare/render inputs are "
                "retained for retry"
            )
        ),
        "failed_episodes_file": str(
            config.paths.root / "failed_episodes.json"
        ),
        "selected_clips": [_clip_report(clip) for clip in plan.clips],
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
                "dependencies": list(stage.dependencies),
                "input_fingerprint": None,
                "retained_input_fingerprint": None,
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
        paths.root / "logs",
        paths.report,
        paths.root / "failed_episodes.json",
    ]
    if not config.alignment_is_external:
        owned_targets.extend(
            [paths.camera_alignment, paths.alignment_reports]
        )
    if config.overwrite:
        for target in owned_targets:
            _remove_exact(target, owner_root=paths.root)
    elif not config.resume:
        existing = [target for target in owned_targets if target.exists()]
        if existing:
            raise FileExistsError(
                "Pipeline outputs already exist while resume is disabled: "
                + ", ".join(str(path) for path in existing)
            )
    paths.root.mkdir(parents=True, exist_ok=True)
    _remove_exact(paths.root / "logs", owner_root=paths.root)


def execute(plan: PipelinePlan) -> dict[str, Any]:
    _file_content_digest.cache_clear()
    previous_report: dict[str, Any] = {}
    if plan.config.resume and not plan.config.overwrite:
        try:
            previous_report = _read_json(plan.config.paths.report)
        except Exception:
            pass
    _initialize_output(plan)
    try:
        previous_failures = _read_json(
            plan.config.paths.root / "failed_episodes.json"
        ).get("failed_episodes", [])
    except Exception:
        previous_failures = []
    if not isinstance(previous_failures, list):
        previous_failures = []
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
        if (
            previous.get("status") not in reusable_statuses
            or previous.get("command") != entry["command"]
        ):
            continue
        for key in (
            "status",
            "input_fingerprint",
            "retained_input_fingerprint",
            "validation",
            "started_at",
            "finished_at",
            "elapsed_seconds",
        ):
            entry[key] = previous.get(key)
    commands_unchanged = bool(previous_stages) and all(
        previous_stages.get(stage.stage_id, {}).get("command")
        == (None if stage.command is None else list(stage.command))
        for stage in plan.stages
    )
    retained_lerobot_valid = False
    retained_render_indices: set[int] = set()
    if plan.config.resume and commands_unchanged:
        retarget_stage = next(
            (stage for stage in plan.stages if stage.name == "retarget"),
            None,
        )
        try:
            alignment_inputs_unchanged = all(
                previous_stages.get(stage.stage_id, {}).get(
                    "retained_input_fingerprint"
                )
                == _stage_input_fingerprint(
                    stage,
                    plan,
                    include_pruned_outputs=False,
                )
                for stage in plan.stages
                if stage.name == "alignment"
            )
            retarget_inputs_unchanged = bool(
                retarget_stage is not None
                and previous_stages.get(retarget_stage.stage_id, {}).get(
                    "input_fingerprint"
                )
                == _stage_input_fingerprint(retarget_stage, plan)
            )
        except Exception:
            alignment_inputs_unchanged = False
            retarget_inputs_unchanged = False
        if (
            retarget_stage is not None
            and alignment_inputs_unchanged
            and retarget_inputs_unchanged
            and previous_stages.get(
                retarget_stage.stage_id,
                {},
            ).get("status")
            in reusable_statuses
        ):
            try:
                _validate_retarget(
                    plan,
                    allow_pruned_alignment=(
                        not plan.config.alignment_is_external
                    ),
                )
                retained_lerobot_valid = True
            except Exception:
                pass
        if retained_lerobot_valid:
            for stage in plan.stages:
                if (
                    stage.name != "render"
                    or stage.episode_index is None
                    or previous_stages.get(stage.stage_id, {}).get("status")
                    not in reusable_statuses
                ):
                    continue
                try:
                    if previous_stages.get(stage.stage_id, {}).get(
                        "retained_input_fingerprint"
                    ) != _retained_render_fingerprint(stage, plan):
                        continue
                    _validate_render(
                        stage,
                        plan,
                        allow_pruned_prepared=True,
                    )
                except Exception:
                    continue
                retained_render_indices.add(stage.episode_index)
    if plan.config.resume:
        for stage in plan.stages:
            if (
                stage.name != "render"
                or stage.episode_index is None
                or stage.episode_index in retained_render_indices
                or not _completion_matches(
                    plan,
                    stage.episode_index,
                    previous_report,
                )
            ):
                continue
            try:
                _validate_render(stage, plan, allow_pruned_prepared=True)
            except Exception:
                continue
            retained_render_indices.add(stage.episode_index)
            _write_completion(plan, stage.episode_index)
    all_episodes_retained = len(retained_render_indices) == len(plan.clips)
    retained_stage_ids = {
        stage.stage_id
        for stage in plan.stages
        if (
            stage.name in {"alignment", "retarget", "repaint"}
            and all_episodes_retained
        )
        or (
            retained_lerobot_valid
            and stage.name in {"alignment", "retarget"}
        )
        or (
            stage.name in {"prepare", "render"}
            and stage.episode_index in retained_render_indices
        )
    }
    pipeline_started = time.perf_counter()
    changed: set[str] = set()
    skipped_by_episode: dict[int, dict[str, Any]] = {}
    if commands_unchanged:
        previous_skips = previous_report.get("skipped_episodes")
        if not isinstance(previous_skips, list):
            previous_skips = []
        for skipped in previous_skips:
            if (
                not isinstance(skipped, dict)
                or skipped.get("failed_stage") != "repaint"
            ):
                continue
            try:
                episode_index = int(skipped["episode_index"])
                clip = _clip_by_index(plan, episode_index)
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if (
                skipped.get("session") == clip.session
                and skipped.get("source_stem") == clip.stem
            ):
                skipped_by_episode[episode_index] = dict(skipped)
    report["skipped_episodes"] = list(skipped_by_episode.values())
    for episode_index in sorted(retained_render_indices):
        _write_notice(
            f"[SKIPPED] episode={_clip_by_index(plan, episode_index).stem} "
            "reason=already_completed"
        )
    for episode_index in sorted(skipped_by_episode):
        skipped = skipped_by_episode[episode_index]
        _write_notice(
            f"[SKIPPED] episode={_clip_by_index(plan, episode_index).stem} "
            f"reason=previous_failure stage={skipped['failed_stage']}"
        )
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
            if invalidated_id in retained_stage_ids:
                continue
            entry = report_by_id[invalidated_id]
            entry["input_fingerprint"] = None
            entry["retained_input_fingerprint"] = None
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
        is_new = episode_index not in skipped_by_episode
        skipped = skipped_by_episode.setdefault(
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
        if is_new:
            _write_notice(
                f"[FAILED] episode={clip.stem} stage={stage.name}; "
                "remaining_stages=skipped"
            )
        if not plan.config.retain_intermediates:
            try:
                if stage.name in {"prepare", "render"}:
                    _clear_stage(stage, plan)
                else:
                    _clear_episode_intermediates(plan, episode_index)
            except Exception as exc:
                skipped.setdefault("cleanup_errors", []).append(
                    {
                        "stage": stage.stage_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        report["skipped_episodes"] = list(skipped_by_episode.values())

    def write_global_failure(stage: Stage) -> None:
        affected = len(stage.clip_stems) or sum(
            clip.episode_index not in retained_render_indices
            for clip in plan.clips
        )
        scope = (
            f"session={stage.session}"
            if stage.session is not None
            else "scope=all"
        )
        _write_notice(
            f"[FAILED] {scope} stage={stage.name} "
            f"affected_episodes={affected}"
        )

    def cleanup_completed_episode(stage: Stage) -> None:
        assert stage.episode_index is not None
        if plan.config.retain_intermediates:
            return
        try:
            _clear_episode_intermediates(plan, stage.episode_index)
        except Exception as exc:
            report.setdefault("cleanup_errors", []).append(
                {
                    "stage": stage.stage_id,
                    "episode_index": stage.episode_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            report["updated_at"] = _utc_now()
            persist_report()

    def persist_report() -> None:
        _write_json_atomic(plan.config.paths.report, report)
        failures_by_episode: dict[int, dict[str, Any]] = {}
        if report["status"] not in {"completed", "completed_with_skips"}:
            for failure in previous_failures:
                if not isinstance(failure, dict):
                    continue
                try:
                    episode_index = int(failure["episode_index"])
                    clip = _clip_by_index(plan, episode_index)
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
                if (
                    failure.get("session") == clip.session
                    and failure.get("source_stem") == clip.stem
                ):
                    failures_by_episode[episode_index] = failure
        failures_by_episode.update(skipped_by_episode)
        failures = [
            failures_by_episode[index]
            for index in sorted(failures_by_episode)
        ]
        _write_json_atomic(
            plan.config.paths.root / "failed_episodes.json",
            {
                "schema_version": 1,
                "source_root": str(plan.config.src_root),
                "updated_at": report["updated_at"],
                "failed_episode_count": len(failures),
                "failed_episodes": failures,
            },
        )

    persist_report()

    with tqdm(
        total=len(plan.stages),
        desc="Nymeria VDA",
        unit="stage",
        dynamic_ncols=True,
        bar_format=total_progress_bar_format(),
    ) as progress:
        try:
            for stage in plan.stages:
                entry = report_by_id[stage.stage_id]
                retained_stage = (
                    stage.name in {"alignment", "retarget", "repaint"}
                    and all_episodes_retained
                ) or (
                    retained_lerobot_valid
                    and stage.name in {"alignment", "retarget"}
                ) or (
                    stage.name in {"prepare", "render"}
                    and stage.episode_index in retained_render_indices
                )
                if retained_stage:
                    now = _utc_now()
                    previous = previous_stages.get(stage.stage_id, {})
                    entry.update(
                        status="reused",
                        started_at=now,
                        finished_at=now,
                        elapsed_seconds=0.0,
                        input_fingerprint=previous.get(
                            "input_fingerprint"
                        ),
                        validation=(
                            "retained output validated; intermediate artifact "
                            "was intentionally pruned"
                        ),
                    )
                    progress.update(1)
                    report["updated_at"] = now
                    report["elapsed_seconds"] = (
                        time.perf_counter() - pipeline_started
                    )
                    persist_report()
                    if stage.name == "render":
                        assert stage.episode_index is not None
                        _write_completion(plan, stage.episode_index)
                        cleanup_completed_episode(stage)
                    continue
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
                        retained_input_fingerprint=None,
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
                    persist_report()
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
                            retained_input_fingerprint=None,
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
                        persist_report()
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
                    repaint_failures = {
                        episode_index: failure
                        for episode_index, failure in repaint_failures.items()
                        if episode_index not in retained_render_indices
                    }
                    for episode_index, failure in repaint_failures.items():
                        record_skipped_episode(
                            episode_index,
                            stage,
                            failure["error"],
                            failure_status=failure["status"],
                        )
                    now = _utc_now()
                    retained_fingerprint = entry.get(
                        "retained_input_fingerprint"
                    )
                    if stage.name == "alignment":
                        retained_fingerprint = _stage_input_fingerprint(
                            stage,
                            plan,
                            include_pruned_outputs=False,
                        )
                    elif stage.name == "render":
                        retained_fingerprint = (
                            _retained_render_fingerprint(stage, plan)
                        )
                    entry.update(
                        status=(
                            "reused_with_skips"
                            if repaint_failures
                            else "reused"
                        ),
                        started_at=now,
                        finished_at=now,
                        elapsed_seconds=0.0,
                        retained_input_fingerprint=retained_fingerprint,
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
                    persist_report()
                    if stage.name == "render":
                        assert stage.episode_index is not None
                        _write_completion(plan, stage.episode_index)
                        cleanup_completed_episode(stage)
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
                        persist_report()
                        write_global_failure(stage)
                        raise error
                    invalidate_stage_and_dependents(stage.stage_id)
                    changed.add(stage.stage_id)
                    report["updated_at"] = _utc_now()
                    persist_report()
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
                        retained_input_fingerprint=(
                            _stage_input_fingerprint(
                                stage,
                                plan,
                                include_pruned_outputs=False,
                            )
                        ),
                        validation=reason,
                    )
                    progress.update(1)
                    report["updated_at"] = now
                    report["elapsed_seconds"] = (
                        time.perf_counter() - pipeline_started
                    )
                    persist_report()
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
                persist_report()
                try:
                    _clear_stage(stage, plan)
                    _run_stage(
                        stage,
                        progress,
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
                    retained_fingerprint = None
                    if stage.name == "alignment":
                        retained_fingerprint = _stage_input_fingerprint(
                            stage,
                            plan,
                            include_pruned_outputs=False,
                        )
                    elif stage.name == "render":
                        retained_fingerprint = (
                            _retained_render_fingerprint(stage, plan)
                        )
                    repaint_failures = (
                        _repaint_manifest(plan)[0]
                        if stage.name == "repaint"
                        else {}
                    )
                    repaint_failures = {
                        episode_index: failure
                        for episode_index, failure in repaint_failures.items()
                        if episode_index not in retained_render_indices
                    }
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
                        persist_report()
                        progress.update(1)
                        continue
                    report.update(
                        status="failed",
                        failed_stage=stage.stage_id,
                        error=error,
                        updated_at=_utc_now(),
                        elapsed_seconds=time.perf_counter() - pipeline_started,
                    )
                    persist_report()
                    write_global_failure(stage)
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
                    retained_input_fingerprint=retained_fingerprint,
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
                persist_report()
                progress.update(1)
                if stage.name == "render":
                    assert stage.episode_index is not None
                    _write_completion(plan, stage.episode_index)
                    cleanup_completed_episode(stage)

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
                report["successful_episode_count"] = len(
                    report["final_videos"]
                )
            report["skipped_episode_count"] = len(skipped_by_episode)
            for stage in plan.stages:
                if stage.name == "alignment":
                    try:
                        retained_fingerprint = _stage_input_fingerprint(
                            stage,
                            plan,
                            include_pruned_outputs=False,
                        )
                    except Exception:
                        retained_fingerprint = None
                    report_by_id[stage.stage_id][
                        "retained_input_fingerprint"
                    ] = retained_fingerprint
            persist_report()
            cleanup_targets = (
                []
                if plan.config.retain_intermediates
                else [plan.config.paths.root / "logs"]
            )
            retryable_failure = any(
                report_by_id[skipped["failed_stage"]]["name"]
                in {"prepare", "render"}
                for skipped in skipped_by_episode.values()
            )
            if (
                not plan.config.retain_intermediates
                and plan.config.stop_after == "render"
                and not retryable_failure
            ):
                cleanup_targets.extend(
                    [
                        plan.config.paths.handless,
                        plan.config.paths.prepared,
                    ]
                )
                if not plan.config.alignment_is_external:
                    cleanup_targets.extend(
                        [
                            plan.config.paths.camera_alignment,
                            plan.config.paths.alignment_reports,
                        ]
                    )
            for target in cleanup_targets:
                try:
                    _remove_exact(
                        target,
                        owner_root=plan.config.paths.root,
                    )
                except Exception as exc:
                    report.setdefault("cleanup_errors", []).append(
                        {
                            "stage": "finalize",
                            "path": str(target),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if report.get("cleanup_errors"):
                report["updated_at"] = _utc_now()
                persist_report()
            progress.bar_format = total_progress_bar_format(0.0)
            progress.refresh()
            return report
        except BaseException:
            if report.get("status") == "running":
                report.update(
                    status="interrupted",
                    updated_at=_utc_now(),
                    elapsed_seconds=time.perf_counter() - pipeline_started,
                )
                persist_report()
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
        scene_scale=None,
        vda_checkpoint=None,
        stop_after="render",
        poll_seconds=0.5,
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = _config_from_args(args)
        plan = build_plan(
            config,
            sessions=set(args.session) if args.session else None,
            requested_clips=set(args.clip) if args.clip else None,
        )
        execute(plan)
    except Exception as exc:
        lines = str(exc).splitlines()
        _write_notice(
            f"[FAILED] pipeline reason="
            f"{lines[-1] if lines else type(exc).__name__}"
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
