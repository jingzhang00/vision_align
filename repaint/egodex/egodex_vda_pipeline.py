#!/usr/bin/env python3
"""Run the canonical EgoDex-to-G1 Video Depth Anything pipeline.

The orchestrator deliberately keeps all video timing and image geometry out of
task configuration.  Source metadata is probed from every MP4, the LeRobot and
prepared stages retain it, and the final Isaac recording is checked against it.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Iterable, Sequence

from tqdm import tqdm

_write_notice = tqdm.write

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.robot_camera import BEINGBEYOND_HEAD_CAMERA_PROFILE  # noqa: E402
from repaint.lighting_profile import load_lighting_profile  # noqa: E402
from repaint.scene_hand_alignment import (  # noqa: E402
    PALM_ALIGNMENT_METHOD,
    PALM_ALIGNMENT_STAGE,
)
from repaint.vda_pipeline_common import (  # noqa: E402
    VideoMetadata,
    build_command as _stage_command,
    load_previous_stages as _load_previous_stages,
    natural_key as _natural_key,
    probe_video,
    read_json as _read_json_object,
    remove_exact as _remove_exact_path,
    run_stage_process as _run_process,
    total_progress_bar_format,
    utc_now as _utc_now,
    write_json_atomic as _write_json_atomic,
)

DEFAULT_TASKS = (
    "basic_pick_place",
    "flip_coin",
    "insert_remove_usb",
    "pour",
)
VIDEO_KEY = "observation.images.ego"
LEROBOT_CHUNKS_SIZE = 1000
BEINGBEYOND_REFERENCE_SIZE = (640, 480)
BEINGBEYOND_RESET_GUARD_X_M = 0.005
SCENE_GEOMETRY_OPERATION_ORDER = (
    PALM_ALIGNMENT_STAGE,
)


@dataclass(frozen=True)
class SourceEpisode:
    """One naturally ordered same-stem EgoDex HDF5/MP4 pair."""

    episode_index: int
    stem: str
    hdf5_path: Path
    video_path: Path
    video: VideoMetadata


@dataclass(frozen=True)
class TaskPaths:
    root: Path
    lerobot: Path
    handless: Path
    prepared: Path
    final: Path
    report: Path

    @classmethod
    def from_root(cls, root: Path) -> "TaskPaths":
        return cls(
            root=root,
            lerobot=root / "lerobot",
            handless=root / "handless",
            prepared=root / "prepared",
            final=root / "final",
            report=root / "pipeline_report.json",
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Inputs which may vary without encoding video geometry or timing."""

    dataset_root: Path
    output_root: Path
    python_executable: Path
    resume: bool = True
    overwrite: bool = False
    headless: bool = True
    retain_intermediates: bool = False
    scene_scale: float | None = None
    vda_checkpoint: Path | None = None
    poll_seconds: float = 0.5


@dataclass(frozen=True)
class Stage:
    """One resumable child command in a task pipeline."""

    stage_id: str
    name: str
    episode_index: int | None
    command: tuple[str, ...]
    artifact_paths: tuple[Path, ...]
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskPlan:
    task: str
    source_root: Path
    paths: TaskPaths
    episodes: tuple[SourceEpisode, ...]
    stages: tuple[Stage, ...]


def discover_source_episodes(source_root: Path) -> tuple[SourceEpisode, ...]:
    """Discover same-stem pairs using the retargeter's natural ordering."""

    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"EgoDex task directory does not exist: {root}")
    hdf5_paths = sorted(
        set(root.rglob("*.hdf5")),
        key=lambda path: _natural_key(str(path.relative_to(root))),
    )
    if not hdf5_paths:
        raise FileNotFoundError(f"No EgoDex HDF5 files found under {root}")

    episodes: list[SourceEpisode] = []
    missing: list[Path] = []
    for episode_index, hdf5_path in enumerate(hdf5_paths):
        video_path = hdf5_path.with_suffix(".mp4")
        if not video_path.is_file():
            missing.append(video_path)
            continue
        stem = hdf5_path.relative_to(root).with_suffix("").as_posix()
        episodes.append(
            SourceEpisode(
                episode_index=episode_index,
                stem=stem,
                hdf5_path=hdf5_path,
                video_path=video_path,
                video=probe_video(video_path),
            )
        )
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Each EgoDex HDF5 requires a same-stem MP4. Missing:\n"
            f"{formatted}"
        )
    return tuple(episodes)


def build_task_plan(task: str, config: PipelineConfig) -> TaskPlan:
    """Build the canonical route and its output contract for one task."""

    if (
        not task
        or task in {".", ".."}
        or Path(task).name != task
        or "/" in task
        or "\\" in task
    ):
        raise ValueError(f"Task must be one safe directory name, got {task!r}")
    source_root = (config.dataset_root / task).expanduser().resolve()
    episodes = discover_source_episodes(source_root)
    task_paths = TaskPaths.from_root(
        (config.output_root / task).expanduser().resolve()
    )
    python = config.python_executable
    stages: list[Stage] = []

    retarget_command = _stage_command(
        python,
        WORKSPACE_ROOT / "repaint/egodex/egodex_to_robot.py",
        "--src-root",
        source_root,
        "--output-root",
        task_paths.lerobot,
        "--no-overwrite",
    )
    stages.append(
        Stage(
            stage_id="retarget",
            name="retarget",
            episode_index=None,
            command=retarget_command,
            artifact_paths=(
                task_paths.lerobot / "meta/info.json",
                task_paths.lerobot / "meta/egodex.json",
            ),
        )
    )

    repaint_command = _stage_command(
        python,
        WORKSPACE_ROOT / "repaint/remove_hands.py",
        "--input_dir",
        task_paths.lerobot,
        "--output_dir",
        task_paths.handless,
    )
    stages.append(
        Stage(
            stage_id="repaint",
            name="repaint",
            episode_index=None,
            command=repaint_command,
            artifact_paths=tuple(
                (
                    task_paths.handless
                    / "videos"
                    / VIDEO_KEY
                    / f"chunk-{episode.episode_index // LEROBOT_CHUNKS_SIZE:03d}"
                    / f"file-{episode.episode_index % LEROBOT_CHUNKS_SIZE:03d}.mp4"
                )
                for episode in episodes
            ),
            dependencies=("retarget",),
        )
    )

    for episode in episodes:
        label = f"episode_{episode.episode_index:06d}"
        prepared_root = task_paths.prepared / label
        final_root = task_paths.final / label
        prepare_arguments: list[str | Path] = [
            "--dataset",
            task_paths.lerobot,
            "--source-root",
            source_root,
            "--removed-root",
            task_paths.handless,
            "--episode",
            str(episode.episode_index),
            "--output",
            prepared_root,
        ]
        if config.scene_scale is not None:
            prepare_arguments.extend(
                ["--scene-scale", repr(float(config.scene_scale))]
            )
        if config.vda_checkpoint is not None:
            prepare_arguments.extend(
                ["--vda-checkpoint", config.vda_checkpoint]
            )
        prepare_id = f"prepare_{episode.episode_index:06d}"
        stages.append(
            Stage(
                stage_id=prepare_id,
                name="prepare",
                episode_index=episode.episode_index,
                command=_stage_command(
                    python,
                    WORKSPACE_ROOT / "repaint/rgbd_prepare.py",
                    *prepare_arguments,
                ),
                artifact_paths=(
                    prepared_root / "episode_data.npz",
                    prepared_root / "prepare_report.json",
                ),
                dependencies=("retarget", "repaint"),
            )
        )

        output_video = final_root / episode.video_path.name
        render_arguments: list[str | Path] = [
            "--prepared",
            prepared_root,
            "--output",
            output_video,
            "--capture-view",
            "robot-camera",
            "--scene-alignment",
            "initial-palms",
            "--scene-leveling",
            "none",
            "--headless" if config.headless else "--no-headless",
        ]
        stages.append(
            Stage(
                stage_id=f"render_{episode.episode_index:06d}",
                name="render",
                episode_index=episode.episode_index,
                command=_stage_command(
                    python,
                    WORKSPACE_ROOT / "repaint/vda_scene_isaac.py",
                    *render_arguments,
                ),
                artifact_paths=(
                    output_video,
                    output_video.with_suffix(".json"),
                ),
                dependencies=(prepare_id,),
            )
        )

    return TaskPlan(
        task=task,
        source_root=source_root,
        paths=task_paths,
        episodes=episodes,
        stages=tuple(stages),
    )


def _episode_for_stage(stage: Stage, plan: TaskPlan) -> SourceEpisode:
    if stage.episode_index is None:
        raise ValueError(f"{stage.stage_id} is not episode-specific")
    return plan.episodes[stage.episode_index]


def _final_video_path(plan: TaskPlan, episode_index: int) -> Path:
    episode = plan.episodes[episode_index]
    return (
        plan.paths.final
        / f"episode_{episode_index:06d}"
        / episode.video_path.name
    )


def _validate_retarget(plan: TaskPlan) -> None:
    info = _read_json_object(plan.paths.lerobot / "meta/info.json")
    manifest = _read_json_object(plan.paths.lerobot / "meta/egodex.json")
    if int(manifest.get("schema_version", -1)) < 6:
        raise ValueError(
            "LeRobot EgoDex manifest predates complete-O6 GMR and must be "
            "regenerated"
        )
    if manifest.get("hand_target_adapter") != (
        "tracked Hand pose -> fixed side-specific local-axis map -> "
        "GMR wrist target"
    ):
        raise ValueError(
            "LeRobot EgoDex manifest has an unsupported hand target adapter"
        )
    gmr_robot_model = manifest.get("gmr_robot_model")
    if not isinstance(gmr_robot_model, dict) or gmr_robot_model.get(
        "kind"
    ) != "main_with_hand_new.urdf:floating_o6":
        raise ValueError(
            "LeRobot EgoDex manifest was not solved with the complete O6 URDF"
        )
    if (
        gmr_robot_model.get("floating_root") is not True
        or gmr_robot_model.get("fixed_link_fusion") is not False
        or int(gmr_robot_model.get("robot_joint_count", -1)) != 51
        or int(gmr_robot_model.get("o6_hand_joint_count", -1)) != 22
    ):
        raise ValueError(
            "LeRobot EgoDex manifest has an invalid complete-O6 GMR model"
        )
    mimic_constraints = gmr_robot_model.get("o6_mimic_constraints")
    if not isinstance(mimic_constraints, dict) or int(
        mimic_constraints.get("equation_count", -1)
    ) != 10:
        raise ValueError(
            "LeRobot EgoDex manifest lacks validated O6 mimic constraints"
        )
    collision_avoidance = gmr_robot_model.get(
        "cross_hand_collision_avoidance"
    )
    if (
        not isinstance(collision_avoidance, dict)
        or collision_avoidance.get("enabled") is not False
        or int(collision_avoidance.get("geometry_pairs", -1)) != 0
        or collision_avoidance.get("solver") != "disabled"
    ):
        raise ValueError(
            "LeRobot EgoDex manifest still enables cross-hand collision "
            "avoidance"
        )
    manifest_episodes = manifest.get("episodes")
    if not isinstance(manifest_episodes, list):
        raise ValueError("LeRobot EgoDex manifest has no episodes list")
    if int(info.get("total_episodes", -1)) != len(plan.episodes):
        raise ValueError("LeRobot total_episodes disagrees with the source task")
    if len(manifest_episodes) != len(plan.episodes):
        raise ValueError("LeRobot manifest episode count disagrees with source")
    chunks_size = int(info.get("chunks_size", 1000))
    if chunks_size <= 0:
        raise ValueError("LeRobot chunks_size must be positive")

    for source, recorded in zip(plan.episodes, manifest_episodes, strict=True):
        if int(recorded.get("episode_index", -1)) != source.episode_index:
            raise ValueError("LeRobot episode indices are not naturally ordered")
        if str(recorded.get("source_stem")) != source.stem:
            raise ValueError("LeRobot source_stem differs from the source task")
        if int(recorded.get("frame_count", -1)) != source.video.frame_count:
            raise ValueError("LeRobot frame_count differs from source MP4")
        chunk_index = source.episode_index // chunks_size
        file_index = source.episode_index % chunks_size
        copied_video = (
            plan.paths.lerobot
            / "videos"
            / VIDEO_KEY
            / f"chunk-{chunk_index:03d}"
            / f"file-{file_index:03d}.mp4"
        )
        copied_meta = probe_video(copied_video)
        if not copied_meta.compatible_with(source.video):
            raise ValueError(
                f"LeRobot video metadata changed for episode "
                f"{source.episode_index}"
            )


def _validate_repaint(plan: TaskPlan) -> None:
    info = _read_json_object(plan.paths.lerobot / "meta/info.json")
    chunks_size = int(info.get("chunks_size", LEROBOT_CHUNKS_SIZE))
    if chunks_size <= 0:
        raise ValueError("LeRobot chunks_size must be positive")
    failed = _repaint_manifest_failures(plan)
    for source in plan.episodes:
        if source.episode_index in failed:
            continue
        output = (
            plan.paths.handless
            / "videos"
            / VIDEO_KEY
            / f"chunk-{source.episode_index // chunks_size:03d}"
            / f"file-{source.episode_index % chunks_size:03d}.mp4"
        )
        output_meta = probe_video(output)
        if not output_meta.compatible_with(source.video):
            raise ValueError(
                f"Handless video metadata changed for {source.stem}: "
                f"{asdict(output_meta)} != {asdict(source.video)}"
            )


def _repaint_manifest_failures(
    plan: TaskPlan,
) -> dict[int, dict[str, Any]]:
    info = _read_json_object(plan.paths.lerobot / "meta/info.json")
    chunks_size = int(info.get("chunks_size", LEROBOT_CHUNKS_SIZE))
    if chunks_size <= 0:
        raise ValueError("LeRobot chunks_size must be positive")
    manifest = _read_json_object(plan.paths.handless / "manifest.json")
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or len(entries) != len(plan.episodes):
        raise ValueError("Arm-removal manifest episode count is incorrect")

    failures: dict[int, dict[str, Any]] = {}
    successful = {"ok", "skipped", "passthrough_no_target"}
    for source in plan.episodes:
        input_video = (
            plan.paths.lerobot
            / "videos"
            / VIDEO_KEY
            / f"chunk-{source.episode_index // chunks_size:03d}"
            / f"file-{source.episode_index % chunks_size:03d}.mp4"
        ).resolve()
        matches = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and Path(str(entry.get("source", ""))).expanduser().resolve()
            == input_video
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{source.stem} has {len(matches)} arm-removal entries"
            )
        entry = matches[0]
        status = str(entry.get("status", ""))
        if status not in successful:
            failures[source.episode_index] = {
                "status": status or "unknown",
                "error": str(
                    entry.get("error")
                    or entry.get("note")
                    or f"remove_hands status={status!r}"
                ),
            }
            continue
        expected_output = (
            plan.paths.handless
            / "videos"
            / VIDEO_KEY
            / f"chunk-{source.episode_index // chunks_size:03d}"
            / f"file-{source.episode_index % chunks_size:03d}.mp4"
        ).resolve()
        if Path(str(entry.get("output", ""))).expanduser().resolve() != (
            expected_output
        ):
            raise ValueError(
                f"{source.stem} arm-removal output path is incorrect"
            )
    return failures


def _numbered_png_count(path: Path) -> int:
    if not path.is_dir():
        raise FileNotFoundError(path)
    indices = sorted(
        int(item.stem)
        for item in path.iterdir()
        if item.is_file()
        and item.suffix.lower() == ".png"
        and item.stem.isdigit()
    )
    if indices != list(range(len(indices))):
        raise ValueError(f"PNG frames in {path} are not contiguous from zero")
    return len(indices)


def _validate_prepared(stage: Stage, plan: TaskPlan) -> None:
    import numpy as np

    source = _episode_for_stage(stage, plan)
    root = plan.paths.prepared / f"episode_{source.episode_index:06d}"
    archive_path = root / "episode_data.npz"
    prepare_report_path = root / "prepare_report.json"
    prepare_report = _read_json_object(prepare_report_path)
    lighting_path = root / "lighting_profile.json"
    lighting_profile = load_lighting_profile(lighting_path)
    if lighting_profile is None:
        raise ValueError("Prepared lighting profile is missing")
    lighting_report = prepare_report.get("lighting")
    if (
        not isinstance(lighting_report, dict)
        or Path(str(lighting_report.get("profile", ""))).resolve()
        != lighting_path.resolve()
    ):
        raise ValueError("Prepared lighting audit is invalid")
    with np.load(archive_path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        source_size = tuple(
            int(value) for value in np.asarray(archive["source_size"]).tolist()
        )
        fps = float(archive["fps"])
    if len(frame_indices) != source.video.frame_count:
        raise ValueError("Prepared frame count differs from source MP4")
    if not np.array_equal(
        frame_indices,
        np.arange(source.video.frame_count, dtype=np.int64),
    ):
        raise ValueError("Prepared frame_indices are not the full source episode")
    if source_size != (source.video.width, source.video.height):
        raise ValueError("Prepared source_size differs from source MP4")
    if abs(fps - source.video.fps) > 0.01:
        raise ValueError("Prepared FPS differs from source MP4")
    if _numbered_png_count(root / "rgb") != source.video.frame_count:
        raise ValueError("Prepared RGB frame count differs from source MP4")
    if _numbered_png_count(root / "depth_mm") != source.video.frame_count:
        raise ValueError("Prepared depth frame count differs from source MP4")


def _validate_render(stage: Stage, plan: TaskPlan) -> None:
    source = _episode_for_stage(stage, plan)
    output_video = _final_video_path(plan, source.episode_index)
    output_report = output_video.with_suffix(".json")
    rendered = probe_video(output_video)
    if not rendered.compatible_with(source.video):
        raise ValueError(
            "Final G1 video timing or resolution differs from source: "
            f"{asdict(rendered)} != {asdict(source.video)}"
        )
    report = _read_json_object(output_report)
    if int(report.get("schema_version", -1)) < 3:
        raise ValueError(
            "Final Isaac report predates mesh/capture camera synchronization"
        )
    if str(report.get("capture_view")) != "robot-camera":
        raise ValueError("Final Isaac report is not a robot-camera capture")
    if not str(report.get("mesh_pose_source", "")).startswith(
        "prepared virtual source-camera pose backprojection rigidly following"
    ):
        raise ValueError("Final virtual camera did not follow fixed extrinsics")
    if int(report.get("frame_count", -1)) != source.video.frame_count:
        raise ValueError("Final Isaac report frame_count differs from source")
    lighting = report.get("lighting")
    if (
        not isinstance(lighting, dict)
        or lighting.get("applied") is not True
        or int(lighting.get("profile_version", -1)) != 1
        or not isinstance(lighting.get("dome"), dict)
        or not isinstance(lighting.get("key"), dict)
    ):
        raise ValueError("Final Isaac lighting profile was not applied")
    output_size = (
        source.video.width,
        source.video.height,
    )
    if tuple(report.get("output_size", ())) != output_size:
        raise ValueError("Final Isaac report output_size differs from source")
    if abs(float(report.get("fps", -1.0)) - source.video.fps) > 0.01:
        raise ValueError("Final Isaac report FPS differs from source")

    # The active BeingBeyond camera is natively 4:3. Preserve that aspect
    # ratio, then center-letterbox it into the source-sized output instead of
    # independently scaling K along x/y.
    unit_count = min(output_size[0] // 4, output_size[1] // 3)
    if unit_count <= 0:
        raise ValueError("Source video is too small for a 4:3 camera capture")
    content_size = (4 * unit_count, 3 * unit_count)
    horizontal_padding = output_size[0] - content_size[0]
    vertical_padding = output_size[1] - content_size[1]
    expected_offsets = {
        "left": horizontal_padding // 2,
        "top": vertical_padding // 2,
        "right": horizontal_padding - horizontal_padding // 2,
        "bottom": vertical_padding - vertical_padding // 2,
    }
    if tuple(report.get("render_size", ())) != content_size:
        raise ValueError(
            "Final Isaac report render_size is not the exact 4:3 content size"
        )
    letterbox = report.get("letterbox")
    if not isinstance(letterbox, dict):
        raise ValueError("Final Isaac report has no letterbox geometry")
    if tuple(letterbox.get("output_size", ())) != output_size:
        raise ValueError("Final Isaac letterbox output_size differs from source")
    if tuple(letterbox.get("content_size", ())) != content_size:
        raise ValueError("Final Isaac letterbox content_size is not exact 4:3")
    if letterbox.get("offsets_px") != expected_offsets:
        raise ValueError("Final Isaac letterbox offsets are not centered")

    capture_camera = report.get("capture_camera")
    if not isinstance(capture_camera, dict):
        raise ValueError("Final Isaac report has no capture_camera metadata")
    if capture_camera.get("profile") != BEINGBEYOND_HEAD_CAMERA_PROFILE:
        raise ValueError(
            "Final Isaac report does not use the exact BeingBeyond profile"
        )
    if tuple(capture_camera.get("reference_size", ())) != (
        BEINGBEYOND_REFERENCE_SIZE
    ):
        raise ValueError("Final Isaac report has the wrong camera reference size")
    if tuple(capture_camera.get("content_size", ())) != content_size:
        raise ValueError("Final Isaac camera content_size differs from render")
    if tuple(capture_camera.get("output_size", ())) != output_size:
        raise ValueError("Final Isaac camera output_size differs from source")
    if not bool(capture_camera.get("reset_guard_applied")):
        raise ValueError("BeingBeyond 5 mm reset guard was not applied")
    if (
        abs(
            float(capture_camera.get("reset_guard_forward_x_m", -1.0))
            - BEINGBEYOND_RESET_GUARD_X_M
        )
        > 1e-9
    ):
        raise ValueError("BeingBeyond reset guard distance is incorrect")
    hand_projection = capture_camera.get("hand_projection")
    if not isinstance(hand_projection, dict):
        raise ValueError("Final Isaac report has no hand projection diagnostic")
    if int(hand_projection.get("frame_count", -1)) != source.video.frame_count:
        raise ValueError("Hand projection diagnostic frame_count differs")

    articulation_commit = report.get("articulation_commit")
    if not isinstance(articulation_commit, dict):
        raise ValueError("Final Isaac report has no articulation commit audit")
    if int(articulation_commit.get("schema_version", -1)) < 2:
        raise ValueError(
            "Final Isaac report predates render-synchronized articulation "
            "validation"
        )
    if not bool(articulation_commit.get("strict_validation_passed")):
        raise ValueError("Isaac trajectory commit validation did not pass")
    if (
        int(articulation_commit.get("verified_application_count", -1))
        != source.video.frame_count
    ):
        raise ValueError(
            "Isaac trajectory commit count differs from source frame count"
        )
    render_sync = articulation_commit.get("render_sync_validation")
    if not isinstance(render_sync, dict) or not bool(
        render_sync.get("passed")
    ):
        raise ValueError("Isaac PhysX-to-Fabric render sync did not pass")
    if not bool(render_sync.get("fabric_update_requested")):
        raise ValueError("Isaac articulation was not explicitly synced to Fabric")
    step_delta = render_sync.get("physics_step_index_delta")
    if (
        not isinstance(step_delta, dict)
        or float(step_delta.get("min", 0.0)) <= 0.0
    ):
        raise ValueError(
            "Isaac physics step did not advance before every rendered frame"
        )

    scene_leveling = report.get("scene_leveling")
    if not isinstance(scene_leveling, dict):
        raise ValueError("Final Isaac report has no scene-leveling audit")
    if tuple(report.get("operation_order", ())) != (
        SCENE_GEOMETRY_OPERATION_ORDER
    ):
        raise ValueError(
            "Final Isaac scene operation_order must contain only "
            "tracked-Hand palm placement"
        )
    if (
        scene_leveling.get("method") != "disabled"
        or scene_leveling.get("requested_mode") != "none"
        or scene_leveling.get("status") != "disabled"
        or int(scene_leveling.get("application_count", -1)) != 0
    ):
        raise ValueError("Final Isaac scene leveling must be disabled")

    leveling_linear = scene_leveling.get("linear_world")
    try:
        linear = tuple(
            tuple(float(value) for value in row)
            for row in leveling_linear
        )
    except (TypeError, ValueError):
        linear = ()
    if (
        len(linear) != 3
        or any(len(row) != 3 for row in linear)
        or not all(math.isfinite(value) for row in linear for value in row)
    ):
        raise ValueError("Scene-leveling linear_world is not a finite 3x3 matrix")
    identity = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    if any(
        abs(linear[row][column] - identity[row][column]) > 1e-9
        for row in range(3)
        for column in range(3)
    ):
        raise ValueError("Disabled scene leveling must use the identity affine")

    scene_alignment = report.get("scene_alignment")
    if not isinstance(scene_alignment, dict):
        raise ValueError("Final Isaac report has no scene-alignment audit")
    if int(scene_alignment.get("schema_version", -1)) < 2:
        raise ValueError("Final Isaac scene-alignment audit schema is obsolete")
    if scene_alignment.get("method") != PALM_ALIGNMENT_METHOD:
        raise ValueError(
            "Final Isaac scene did not use tracked-Hand palm alignment"
        )
    if int(scene_alignment.get("pipeline_order", -1)) != 1:
        raise ValueError("Tracked-palm placement is not pipeline operation 1")
    if int(scene_alignment.get("application_count", -1)) != 1:
        raise ValueError("Scene alignment must apply one constant translation")
    if (
        scene_alignment.get("fits_rotation") is not False
        or scene_alignment.get("fits_scale") is not False
        or scene_alignment.get("fits_linear_leveling") is not False
    ):
        raise ValueError(
            "Tracked-palm scene alignment must fit only a translation"
        )
    translation = scene_alignment.get("world_translation_xyz_m")
    if (
        not isinstance(translation, (list, tuple))
        or len(translation) != 3
    ):
        raise ValueError("Scene alignment has no constant XYZ translation")
    try:
        translation_xyz = tuple(float(value) for value in translation)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Scene alignment XYZ translation is not numeric"
        ) from exc
    if not all(math.isfinite(value) for value in translation_xyz):
        raise ValueError("Scene alignment XYZ translation is not finite")
    try:
        reported_translation_z = float(
            scene_alignment.get("world_translation_z_m", float("nan"))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Scene alignment world_translation_z_m is not numeric"
        ) from exc
    if (
        not math.isfinite(reported_translation_z)
        or abs(reported_translation_z - translation_xyz[2]) > 1e-9
    ):
        raise ValueError(
            "Scene alignment world_translation_z_m disagrees with its "
            "constant translation"
        )
    anchor_semantics = scene_alignment.get("anchor_semantics")
    if not isinstance(anchor_semantics, dict):
        raise ValueError("Scene alignment has no anchor-semantics audit")
    if (
        anchor_semantics.get("matches_retarget_tracked_hand_palm_targets")
        is not True
    ):
        raise ValueError(
            "Scene alignment does not use the retarget tracked-Hand palms"
        )
    if anchor_semantics.get("uses_hand_base_origins") is not False:
        raise ValueError("Scene alignment must not use hand-base origins")
    if anchor_semantics.get("uses_source_wrist_origins") is not True:
        raise ValueError("Scene alignment must use tracked Hand wrist roots")
    if anchor_semantics.get("uses_fingertips") is not False:
        raise ValueError("Scene alignment must not use fingertip anchors")
    height_policy = scene_alignment.get("height_policy")
    if not isinstance(height_policy, str):
        raise ValueError("Scene alignment has no explicit height policy")
    normalized_height_policy = height_policy.lower()
    if "same 3-d palm residual" not in normalized_height_policy or (
        "no fixed" not in normalized_height_policy
    ):
        raise ValueError(
            "Scene alignment height policy must use the same 3-D palm "
            "residual with no fixed scene height"
        )

    alignment_affine = scene_alignment.get("scene_world_affine")
    if not isinstance(alignment_affine, dict):
        raise ValueError(
            "Scene alignment does not record its preceding scene affine"
        )
    try:
        alignment_linear = tuple(
            tuple(float(value) for value in row)
            for row in alignment_affine.get("linear", ())
        )
    except (TypeError, ValueError):
        alignment_linear = ()
    if (
        len(alignment_linear) != 3
        or any(len(row) != 3 for row in alignment_linear)
        or any(
            abs(alignment_linear[row][column] - linear[row][column]) > 1e-9
            for row in range(3)
            for column in range(3)
        )
    ):
        raise ValueError(
            "Scene alignment did not use the identity scene affine"
        )
    leveling_pivot = scene_leveling.get("pivot_world_xyz_m")
    alignment_pivot = alignment_affine.get("pivot_world_xyz_m")
    try:
        leveling_pivot_xyz = tuple(float(value) for value in leveling_pivot)
        alignment_pivot_xyz = tuple(float(value) for value in alignment_pivot)
    except (TypeError, ValueError):
        leveling_pivot_xyz = ()
        alignment_pivot_xyz = ()
    if (
        len(leveling_pivot_xyz) != 3
        or len(alignment_pivot_xyz) != 3
        or not all(math.isfinite(value) for value in leveling_pivot_xyz)
        or not all(math.isfinite(value) for value in alignment_pivot_xyz)
        or any(
            abs(alignment_pivot_xyz[axis] - leveling_pivot_xyz[axis]) > 1e-9
            for axis in range(3)
        )
    ):
        raise ValueError(
            "Scene alignment did not use the disabled-leveling pivot"
        )
    if alignment_affine.get("application_order") != (
        "source camera -> raw scene world -> scene affine -> "
        "palm translation"
    ):
        raise ValueError(
            "Scene alignment affine application order is not explicit"
        )

    scene_placement = report.get("scene_placement")
    if not isinstance(scene_placement, dict):
        raise ValueError("Final Isaac report has no scene-placement audit")
    if tuple(scene_placement.get("operation_order", ())) != (
        SCENE_GEOMETRY_OPERATION_ORDER
    ):
        raise ValueError(
            "Scene-placement operation order differs from the top-level "
            "anchor-only contract"
        )
    if int(scene_placement.get("application_count", -1)) != 1:
        raise ValueError("Scene placement did not apply exactly one anchor")
    try:
        placement_translation = tuple(
            float(value)
            for value in scene_placement.get(
                "anchor_translation_world_xyz_m",
                (),
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Scene-placement translation is not numeric") from exc
    if (
        len(placement_translation) != 3
        or not all(math.isfinite(value) for value in placement_translation)
        or any(
            abs(placement_translation[axis] - translation_xyz[axis]) > 1e-9
            for axis in range(3)
        )
    ):
        raise ValueError(
            "Scene-placement translation differs from scene alignment"
        )


def validate_stage_artifacts(stage: Stage, plan: TaskPlan) -> tuple[bool, str]:
    """Validate a stage's complete output, returning a resume-safe reason."""

    try:
        if stage.name == "retarget":
            _validate_retarget(plan)
        elif stage.name == "repaint":
            _validate_repaint(plan)
        elif stage.name == "prepare":
            _validate_prepared(stage, plan)
        elif stage.name == "render":
            _validate_render(stage, plan)
        else:
            raise ValueError(f"Unknown pipeline stage {stage.name!r}")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "validated"


def _clear_stage_output(stage: Stage, plan: TaskPlan) -> None:
    if stage.name == "retarget":
        targets = (plan.paths.lerobot,)
    elif stage.name == "repaint":
        targets = (plan.paths.handless,)
    elif stage.name == "prepare":
        assert stage.episode_index is not None
        targets = (
            plan.paths.prepared
            / f"episode_{stage.episode_index:06d}",
        )
    elif stage.name == "render":
        assert stage.episode_index is not None
        final_video = _final_video_path(plan, stage.episode_index)
        targets = (
            final_video,
            final_video.with_suffix(".json"),
        )
    else:
        raise ValueError(f"Unknown stage {stage.name!r}")
    for target in targets:
        _remove_exact_path(target, owner_root=plan.paths.root)


def _clear_episode_intermediates(
    plan: TaskPlan,
    episode_index: int,
) -> None:
    targets = [
        plan.paths.prepared / f"episode_{episode_index:06d}",
    ]
    errors: list[str] = []
    try:
        info = _read_json_object(plan.paths.lerobot / "meta/info.json")
        chunks_size = int(info.get("chunks_size", LEROBOT_CHUNKS_SIZE))
        if chunks_size <= 0:
            raise ValueError("LeRobot chunks_size must be positive")
        targets.append(
            plan.paths.handless
            / "videos"
            / VIDEO_KEY
            / f"chunk-{episode_index // chunks_size:03d}"
            / f"file-{episode_index % chunks_size:03d}.mp4"
        )
    except Exception as exc:
        errors.append(f"handless path: {type(exc).__name__}: {exc}")
    for target in targets:
        try:
            _remove_exact_path(target, owner_root=plan.paths.root)
        except Exception as exc:
            errors.append(f"{target}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(command)


def _source_report(episode: SourceEpisode) -> dict[str, Any]:
    return {
        "episode_index": episode.episode_index,
        "source_stem": episode.stem,
        "hdf5": str(episode.hdf5_path),
        "video": str(episode.video_path),
        "video_metadata": asdict(episode.video),
    }


def _completion_path(plan: TaskPlan, episode_index: int) -> Path:
    return (
        plan.paths.final
        / f"episode_{episode_index:06d}"
        / "completion.json"
    )


def _write_completion(plan: TaskPlan, episode_index: int) -> None:
    _write_json_atomic(
        _completion_path(plan, episode_index),
        {
            "schema_version": 1,
            "status": "completed",
            "source_episode": _source_report(plan.episodes[episode_index]),
            "updated_at": _utc_now(),
        },
    )


def _completion_matches(
    plan: TaskPlan,
    episode_index: int,
    previous_report: dict[str, Any],
) -> bool:
    expected = _source_report(plan.episodes[episode_index])
    try:
        if (
            _read_json_object(_completion_path(plan, episode_index)).get(
                "source_episode"
            )
            == expected
        ):
            return True
    except Exception:
        pass
    previous_sources = previous_report.get("source_episodes")
    return isinstance(previous_sources, list) and expected in previous_sources


def _new_report(plan: TaskPlan, config: PipelineConfig) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "pipeline": (
            "egodex_to_robot -> remove_hands -> rgbd_prepare(VDA) "
            "-> vda_scene_isaac(robot-camera)"
        ),
        "task": plan.task,
        "status": "running",
        "source_root": str(plan.source_root),
        "task_output_root": str(plan.paths.root),
        "output_layout": {
            "lerobot": str(plan.paths.lerobot),
            "handless": str(plan.paths.handless),
            "prepared": str(plan.paths.prepared / "episode_XXXXXX"),
            "final_video": str(
                plan.paths.final
                / "episode_XXXXXX/<source-video-basename>"
            ),
            "failed_episodes": str(
                plan.paths.root / "failed_episodes.json"
            ),
        },
        "intermediate_retention": (
            "LeRobot, handless, prepared, and final artifacts are retained"
            if config.retain_intermediates
            else (
                "handless and prepared episode artifacts are removed after "
                "render; only LeRobot, final outputs, and this report are "
                "retained; failed prepare/render inputs are retained for retry"
            )
        ),
        "video_metadata_policy": {
            "source": "ffprobe each source MP4",
            "fps": "inherited from source; never task-hardcoded",
            "frame_count": "inherited from source; never task-hardcoded",
            "source_resolution": "inherited from source MP4",
            "final_resolution": "prepared source_size, verified against source",
            "robot_camera_profile": BEINGBEYOND_HEAD_CAMERA_PROFILE,
            "robot_camera_geometry": (
                "fixed 640x480 4:3 intrinsic uniformly scaled and centered "
                "letterboxed into the source-sized output"
            ),
            "mesh_projection_camera": (
                "prepared virtual source camera rigidly following the G1 "
                "capture camera through one fixed initial extrinsic"
            ),
            "trajectory_commit_validation": (
                "every G1 joint/root write is read back from Isaac"
            ),
            "fixed_render_size_argument": False,
            "fixed_fps_argument": False,
        },
        "source_episodes": [
            _source_report(episode) for episode in plan.episodes
        ],
        "resume": config.resume,
        "overwrite": config.overwrite,
        "retain_intermediates": config.retain_intermediates,
        "episode_failure_policy": (
            "repaint manifest failures are terminal skips on resume; "
            "prepare/render exceptions are retried on the next run"
        ),
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "elapsed_seconds": 0.0,
        "skipped_episodes": [],
        "stages": [],
    }


def _stage_report(stage: Stage) -> dict[str, Any]:
    return {
        "stage_id": stage.stage_id,
        "name": stage.name,
        "episode_index": stage.episode_index,
        "status": "pending",
        "command": list(stage.command),
        "command_shell_escaped": _command_text(stage.command),
        "artifacts": [str(path) for path in stage.artifact_paths],
        "dependencies": list(stage.dependencies),
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": None,
        "validation": None,
        "last_successful_command": None,
    }


def _report_stage_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(stage["stage_id"]): stage
        for stage in report["stages"]
    }


def _run_stage_process(
    stage: Stage,
    progress: tqdm[Any],
    *,
    poll_seconds: float,
) -> None:
    """Run one child while refreshing the aggregate bar."""

    _run_process(
        command=stage.command,
        stage_id=stage.stage_id,
        progress=progress,
        poll_seconds=poll_seconds,
        cwd=WORKSPACE_ROOT,
    )


def execute_task(
    plan: TaskPlan,
    config: PipelineConfig,
    *,
    progress: tqdm[Any] | None = None,
) -> dict[str, Any]:
    """Execute or resume one task on the caller's aggregate progress bar."""

    previous_report: dict[str, Any] = {}
    if config.resume and not config.overwrite:
        try:
            previous_report = _read_json_object(plan.paths.report)
        except Exception:
            pass
    if config.overwrite:
        _remove_exact_path(plan.paths.root)
    elif plan.paths.root.exists() and not config.resume:
        raise FileExistsError(
            f"Task output already exists and resume is disabled: "
            f"{plan.paths.root}"
        )
    _remove_exact_path(
        plan.paths.root / "logs",
        owner_root=plan.paths.root,
    )
    try:
        previous_failures = _read_json_object(
            plan.paths.root / "failed_episodes.json"
        ).get("failed_episodes", [])
    except Exception:
        previous_failures = []
    if not isinstance(previous_failures, list):
        previous_failures = []

    previous_stages = _load_previous_stages(plan.paths.report)
    report = _new_report(plan, config)
    report["stages"] = [_stage_report(stage) for stage in plan.stages]
    report_by_id = _report_stage_map(report)
    reusable_statuses = {
        "completed",
        "reused",
        "completed_with_skips",
        "reused_with_skips",
    }
    for stage_id, entry in report_by_id.items():
        previous = previous_stages.get(stage_id, {})
        if (
            previous.get("status") in reusable_statuses
            and previous.get("command") == entry["command"]
        ):
            for key in (
                "status",
                "validation",
                "started_at",
                "finished_at",
                "elapsed_seconds",
            ):
                entry[key] = previous.get(key)
        last_command = previous.get("last_successful_command")
        if (
            last_command is None
            and previous.get("status") in reusable_statuses
        ):
            last_command = previous.get("command")
        entry["last_successful_command"] = last_command
    commands_unchanged = bool(previous_stages) and all(
        (
            previous_stages.get(stage.stage_id, {}).get(
                "last_successful_command"
            )
            or previous_stages.get(stage.stage_id, {}).get("command")
        )
        == list(stage.command)
        for stage in plan.stages
    )
    retained_render_indices: set[int] = set()
    if config.resume and commands_unchanged:
        retarget_stage = next(
            (stage for stage in plan.stages if stage.name == "retarget"),
            None,
        )
        retarget_valid = bool(
            retarget_stage is not None
            and previous_stages.get(
                retarget_stage.stage_id,
                {},
            ).get("status")
            in reusable_statuses
            and validate_stage_artifacts(retarget_stage, plan)[0]
        )
        if retarget_valid:
            for stage in plan.stages:
                if (
                    stage.name == "render"
                    and stage.episode_index is not None
                    and previous_stages.get(stage.stage_id, {}).get("status")
                    in reusable_statuses
                    and validate_stage_artifacts(stage, plan)[0]
                ):
                    retained_render_indices.add(stage.episode_index)
    if config.resume:
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
                or not validate_stage_artifacts(stage, plan)[0]
            ):
                continue
            retained_render_indices.add(stage.episode_index)
            _write_completion(plan, stage.episode_index)
    all_episodes_retained = len(retained_render_indices) == len(plan.episodes)
    retained_stage_ids = {
        stage.stage_id
        for stage in plan.stages
        if (
            stage.name in {"retarget", "repaint"}
            and all_episodes_retained
        )
        or (
            stage.name in {"prepare", "render"}
            and stage.episode_index in retained_render_indices
        )
    }
    task_started = time.perf_counter()
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
                if not 0 <= episode_index < len(plan.episodes):
                    continue
                episode = plan.episodes[episode_index]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if skipped.get("source_stem") == episode.stem:
                skipped_by_episode[episode_index] = dict(skipped)
    report["skipped_episodes"] = list(skipped_by_episode.values())
    for episode_index in sorted(retained_render_indices):
        _write_notice(
            f"[SKIPPED] episode={plan.task}/"
            f"{plan.episodes[episode_index].stem} "
            "reason=already_completed"
        )
    for episode_index in sorted(skipped_by_episode):
        skipped = skipped_by_episode[episode_index]
        _write_notice(
            f"[SKIPPED] episode={plan.task}/"
            f"{plan.episodes[episode_index].stem} "
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
            entry["last_successful_command"] = None
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
        source = plan.episodes[episode_index]
        is_new = episode_index not in skipped_by_episode
        skipped = skipped_by_episode.setdefault(
            episode_index,
            {
                **_source_report(source),
                "status": "skipped",
                "failure_status": failure_status,
                "failed_stage": stage.stage_id,
                "error": error,
                "stage_paths": {
                    "lerobot_root": str(plan.paths.lerobot),
                    "handless_root": str(plan.paths.handless),
                    "prepared_episode": str(
                        plan.paths.prepared
                        / f"episode_{episode_index:06d}"
                    ),
                    "final_video": str(
                        _final_video_path(plan, episode_index)
                    ),
                },
                "recorded_at": _utc_now(),
            },
        )
        if is_new:
            _write_notice(
                f"[FAILED] episode={plan.task}/{source.stem} "
                f"stage={stage.name}; "
                "remaining_stages=skipped"
            )
        if not config.retain_intermediates:
            try:
                if stage.name in {"prepare", "render"}:
                    _clear_stage_output(stage, plan)
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
        affected = sum(
            source.episode_index not in retained_render_indices
            for source in plan.episodes
        )
        _write_notice(
            f"[FAILED] task={plan.task} stage={stage.name} "
            f"affected_episodes={affected}"
        )

    def cleanup_completed_episode(stage: Stage) -> None:
        assert stage.episode_index is not None
        if config.retain_intermediates:
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
        _write_json_atomic(plan.paths.report, report)
        failures_by_episode: dict[int, dict[str, Any]] = {}
        if report["status"] not in {"completed", "completed_with_skips"}:
            for failure in previous_failures:
                if not isinstance(failure, dict):
                    continue
                try:
                    episode_index = int(failure["episode_index"])
                    if not 0 <= episode_index < len(plan.episodes):
                        continue
                    source = plan.episodes[episode_index]
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
                if failure.get("source_stem") == source.stem:
                    failures_by_episode[episode_index] = failure
        failures_by_episode.update(skipped_by_episode)
        failures = [
            failures_by_episode[index]
            for index in sorted(failures_by_episode)
        ]
        _write_json_atomic(
            plan.paths.root / "failed_episodes.json",
            {
                "schema_version": 1,
                "task": plan.task,
                "updated_at": report["updated_at"],
                "failed_episode_count": len(failures),
                "failed_episodes": failures,
            },
        )

    persist_report()

    progress_context = (
        nullcontext(progress)
        if progress is not None
        else tqdm(
            total=len(plan.stages),
            desc="EgoDex VDA",
            unit="stage",
            dynamic_ncols=True,
            leave=True,
            bar_format=total_progress_bar_format(),
        )
    )
    with progress_context as progress:
        assert progress is not None
        try:
            for stage in plan.stages:
                stage_entry = report_by_id[stage.stage_id]
                retained_stage = (
                    stage.name in {"retarget", "repaint"}
                    and all_episodes_retained
                ) or (
                    stage.name in {"prepare", "render"}
                    and stage.episode_index in retained_render_indices
                )
                if retained_stage:
                    now = _utc_now()
                    stage_entry.update(
                        status="reused",
                        started_at=now,
                        finished_at=now,
                        elapsed_seconds=0.0,
                        last_successful_command=list(stage.command),
                        validation=(
                            "retained final validated; intermediate artifact "
                            "was intentionally pruned"
                        ),
                    )
                    progress.update(1)
                    report["updated_at"] = now
                    report["elapsed_seconds"] = (
                        time.perf_counter() - task_started
                    )
                    persist_report()
                    if stage.name == "render":
                        _write_completion(plan, stage.episode_index)
                        cleanup_completed_episode(stage)
                    continue
                if (
                    stage.episode_index is not None
                    and stage.episode_index in skipped_by_episode
                ):
                    cleanup_error = None
                    try:
                        _clear_stage_output(stage, plan)
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
                    stage_entry.update(
                        status="skipped",
                        started_at=now,
                        finished_at=now,
                        elapsed_seconds=0.0,
                        last_successful_command=None,
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
                        time.perf_counter() - task_started
                    )
                    persist_report()
                    continue

                dependencies_changed = any(
                    dependency in changed for dependency in stage.dependencies
                )
                if config.resume and not dependencies_changed:
                    valid, validation_reason = validate_stage_artifacts(
                        stage,
                        plan,
                    )
                else:
                    valid = False
                    validation_reason = (
                        "dependency changed"
                        if dependencies_changed
                        else "resume disabled"
                    )
                may_reuse = (
                    config.resume
                    and not dependencies_changed
                    and valid
                    and stage_entry["last_successful_command"]
                    == list(stage.command)
                )
                if may_reuse:
                    repaint_failures = (
                        _repaint_manifest_failures(plan)
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
                    stage_entry.update(
                        status=(
                            "reused_with_skips"
                            if repaint_failures
                            else "reused"
                        ),
                        started_at=now,
                        finished_at=now,
                        elapsed_seconds=0.0,
                        last_successful_command=list(stage.command),
                        validation=(
                            f"validated with {len(repaint_failures)} "
                            "skipped episode(s)"
                            if repaint_failures
                            else validation_reason
                        ),
                    )
                    progress.update(1)
                    report["updated_at"] = _utc_now()
                    report["elapsed_seconds"] = (
                        time.perf_counter() - task_started
                    )
                    persist_report()
                    if stage.name == "render":
                        assert stage.episode_index is not None
                        _write_completion(plan, stage.episode_index)
                        cleanup_completed_episode(stage)
                    continue

                invalidate_stage_and_dependents(stage.stage_id)
                changed.add(stage.stage_id)
                stage_started = time.perf_counter()
                stage_entry.update(
                    status="running",
                    started_at=_utc_now(),
                    validation=(
                        "dependency changed"
                        if dependencies_changed
                        else validation_reason
                    ),
                )
                report["updated_at"] = _utc_now()
                persist_report()
                try:
                    _clear_stage_output(stage, plan)
                    _run_stage_process(
                        stage,
                        progress,
                        poll_seconds=config.poll_seconds,
                    )
                    valid, validation_reason = validate_stage_artifacts(
                        stage,
                        plan,
                    )
                    if not valid:
                        raise RuntimeError(
                            f"{stage.stage_id} output validation failed: "
                            f"{validation_reason}"
                        )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    stage_entry.update(
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
                                time.perf_counter() - task_started
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
                        elapsed_seconds=time.perf_counter() - task_started,
                    )
                    persist_report()
                    write_global_failure(stage)
                    raise

                repaint_failures = (
                    _repaint_manifest_failures(plan)
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
                stage_entry.update(
                    status=(
                        "completed_with_skips"
                        if repaint_failures
                        else "completed"
                    ),
                    finished_at=_utc_now(),
                    elapsed_seconds=time.perf_counter() - stage_started,
                    last_successful_command=list(stage.command),
                    validation=(
                        f"validated with {len(repaint_failures)} "
                        "skipped episode(s)"
                        if repaint_failures
                        else validation_reason
                    ),
                )
                report["updated_at"] = _utc_now()
                report["elapsed_seconds"] = time.perf_counter() - task_started
                persist_report()
                progress.update(1)
                if stage.name == "render":
                    assert stage.episode_index is not None
                    _write_completion(plan, stage.episode_index)
                    cleanup_completed_episode(stage)

            final_videos = []
            for episode in plan.episodes:
                render_entry = report_by_id[
                    f"render_{episode.episode_index:06d}"
                ]
                if render_entry["status"] not in {"completed", "reused"}:
                    continue
                video_path = _final_video_path(
                    plan,
                    episode.episode_index,
                )
                final_videos.append(
                    {
                        "episode_index": episode.episode_index,
                        "source_stem": episode.stem,
                        "path": str(video_path),
                        "video_metadata": asdict(probe_video(video_path)),
                    }
                )
            report.update(
                status=(
                    "completed_with_skips"
                    if skipped_by_episode
                    else "completed"
                ),
                finished_at=_utc_now(),
                updated_at=_utc_now(),
                elapsed_seconds=time.perf_counter() - task_started,
                final_videos=final_videos,
                successful_episode_count=len(final_videos),
                skipped_episode_count=len(skipped_by_episode),
            )
            persist_report()
            cleanup_targets = (
                []
                if config.retain_intermediates
                else [plan.paths.root / "logs"]
            )
            if not config.retain_intermediates and not any(
                report_by_id[skipped["failed_stage"]]["name"]
                in {"prepare", "render"}
                for skipped in skipped_by_episode.values()
            ):
                cleanup_targets.extend(
                    [plan.paths.handless, plan.paths.prepared]
                )
            for target in cleanup_targets:
                try:
                    _remove_exact_path(
                        target,
                        owner_root=plan.paths.root,
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
            return report
        except BaseException:
            if report.get("status") == "running":
                report.update(
                    status="interrupted",
                    updated_at=_utc_now(),
                    elapsed_seconds=time.perf_counter() - task_started,
                )
                persist_report()
            raise


def run_pipeline(tasks: Iterable[str], config: PipelineConfig) -> list[dict[str, Any]]:
    """Run tasks sequentially because Isaac and GPU models are process-global."""

    plans = [build_task_plan(task, config) for task in tasks]
    reports: list[dict[str, Any]] = []
    with tqdm(
        total=sum(len(plan.stages) for plan in plans),
        desc="EgoDex VDA",
        unit="stage",
        dynamic_ncols=True,
        leave=True,
        bar_format=total_progress_bar_format(),
    ) as progress:
        for plan in plans:
            reports.append(execute_task(plan, config, progress=progress))
        progress.bar_format = total_progress_bar_format(0.0)
        progress.refresh()
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical EgoDex -> G1 LeRobot -> handless repaint -> metric VDA "
            "-> source-sized G1-camera Isaac pipeline. Source FPS, frame "
            "count, and video dimensions are always discovered automatically."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Dataset directory names under --dataset-root.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE_ROOT / "dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "egodex",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Isaac batch mode window visibility (default: headless).",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    executable = Path(sys.executable).expanduser()
    if not executable.is_absolute():
        executable = Path.cwd() / executable
    executable = executable.absolute()
    if not executable.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {executable}")
    return PipelineConfig(
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        python_executable=executable,
        resume=True,
        overwrite=False,
        headless=bool(args.headless),
        scene_scale=None,
        vda_checkpoint=None,
        poll_seconds=0.5,
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = _config_from_args(args)
        run_pipeline(args.tasks, config)
    except Exception as exc:
        lines = str(exc).splitlines()
        _write_notice(
            f"[FAILED] pipeline reason="
            f"{lines[-1] if lines else type(exc).__name__}"
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
