#!/usr/bin/env python3
"""Retarget cut Nymeria clips to native 51-DoF G1 trajectories.

Body motion is solved by upstream GMR's 29-DoF Unitree G1 configuration.  The
22 hand joints in ``main_with_hand_new.urdf`` are held at their neutral/open
URDF values because the available Nymeria clips contain rigid wrist/palm pose
but no articulated finger observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
STREAM_ROOT = WORKSPACE_ROOT / "stream"
for import_root in (WORKSPACE_ROOT, STREAM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import h5py  # noqa: E402
import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from tqdm import tqdm  # noqa: E402

from general_motion_retargeting import (  # noqa: E402
    GeneralMotionRetargeting as GMR,
)
from repaint.g1_hand import (  # noqa: E402
    build_pinocchio_g1,
)
from repaint.nymeria.nymeria_adapter import (  # noqa: E402
    DisplayVideoInfo,
    MotionMetadata,
    NymeriaClip,
    ResampledMotion,
    discover_nymeria_clips,
    load_motion_metadata,
    load_participant_height_m,
    load_scaled_intrinsic,
    natural_key,
    probe_display_video,
    resample_motion_to_video,
)
from repaint.robot_camera import (  # noqa: E402
    HEAD_CAMERA_PARENT_FRAME,
    retarget_scene_camera_poses_cv,
    robot_frame_poses_world,
)
from repaint.vda_pipeline_common import (  # noqa: E402
    remove_exact,
    write_json_atomic,
)


ROBOT_FORMAT = "nymeria-g1-trajectory-v1"
G1_URDF = (
    WORKSPACE_ROOT / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
)
SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
Y_UP_TO_Z_UP = Rotation.from_quat([0.5, 0.5, 0.5, 0.5])
MAXIMUM_MOTION_BOUNDARY_CLAMP_SECONDS = 0.25
MAXIMUM_CAMERA_TIMELINE_DRIFT_SECONDS = 0.02
@dataclass(frozen=True)
class NymeriaCameraAlignment:
    hdf5_path: Path
    camera_to_world: np.ndarray
    hip_to_world: np.ndarray
    spine3_to_world: np.ndarray
    motion_timestamps_seconds: np.ndarray


@dataclass
class NymeriaSourceEpisode:
    clip: NymeriaClip
    video: DisplayVideoInfo
    task: str
    participant_height_m: float
    intrinsic: np.ndarray
    resampled: ResampledMotion
    camera_alignment: NymeriaCameraAlignment

    @property
    def frame_count(self) -> int:
        return self.video.frame_count


@dataclass
class NymeriaRetargetedEpisode:
    source: NymeriaSourceEpisode
    state: np.ndarray
    root_pose: np.ndarray
    camera_pose: np.ndarray
    source_camera_pose_in_spine3: np.ndarray
    source_camera_rotation_world: np.ndarray
    source_spine3_to_robot_torso_rotation: np.ndarray
    source_to_robot_world_rotation: np.ndarray
    scene_scale: float
    camera_alignment_metrics: dict[str, Any]
    metrics: dict[str, float]
    elapsed_seconds: float


def _decode_hdf5_text(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_transform_sequence(
    values: np.ndarray,
    *,
    frame_count: int,
    label: str,
) -> np.ndarray:
    transforms = np.asarray(values, dtype=np.float64)
    if transforms.shape != (frame_count, 4, 4):
        raise ValueError(
            f"{label} must have shape {(frame_count, 4, 4)}, "
            f"got {transforms.shape}"
        )
    if not np.isfinite(transforms).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(
        transforms[:, 3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=1e-5,
    ):
        raise ValueError(f"{label} contains invalid homogeneous transforms")
    determinants = np.linalg.det(transforms[:, :3, :3])
    if not np.allclose(determinants, 1.0, atol=1e-3):
        raise ValueError(f"{label} contains non-rigid rotations")
    orthogonality = np.einsum(
        "nij,nkj->nik",
        transforms[:, :3, :3],
        transforms[:, :3, :3],
    )
    if not np.allclose(
        orthogonality,
        np.eye(3, dtype=np.float64)[None],
        atol=1e-4,
    ):
        raise ValueError(f"{label} contains non-orthonormal rotations")
    return transforms


def _declared_alignment_source_video(handle: h5py.File) -> Path | None:
    candidates: list[str] = []
    if "source_video" in handle.attrs:
        candidates.append(_decode_hdf5_text(handle.attrs["source_video"]))
    if "nymeria" in handle and "source_video" in handle["nymeria"].attrs:
        candidates.append(
            _decode_hdf5_text(handle["nymeria"].attrs["source_video"])
        )
    unique = {value for value in candidates if value}
    if len(unique) > 1:
        raise ValueError(
            "Camera alignment HDF5 declares multiple source videos: "
            f"{sorted(unique)}"
        )
    if not unique:
        return None
    return Path(unique.pop()).expanduser().resolve()


def load_camera_alignment(
    clip: NymeriaClip,
    *,
    alignment_root: Path,
    alignment_schema: int,
    frame_count: int,
    intrinsic: np.ndarray,
    motion: MotionMetadata,
    display_timestamps: np.ndarray,
) -> NymeriaCameraAlignment:
    if (
        isinstance(alignment_schema, bool)
        or not isinstance(alignment_schema, int)
        or alignment_schema != 9
    ):
        raise ValueError(
            f"Unsupported Nymeria camera alignment schema: {alignment_schema}"
        )
    path = (
        alignment_root.expanduser().resolve()
        / clip.session
        / f"{clip.stem}.hdf5"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Nymeria camera alignment is missing for {clip.stem}: {path}"
        )
    with h5py.File(path, "r") as handle:
        declared_source = _declared_alignment_source_video(handle)
        if (
            declared_source is not None
            and declared_source != clip.video_path.resolve()
        ):
            raise ValueError(
                f"{path} declares source video {declared_source}, expected "
                f"{clip.video_path.resolve()}"
            )
        aligned_intrinsic = np.asarray(
            handle["camera/intrinsic"],
            dtype=np.float64,
        )
        camera_to_world = _validate_transform_sequence(
            handle["transforms/camera"],
            frame_count=frame_count,
            label=f"{path}:transforms/camera",
        )
        hip_to_world = _validate_transform_sequence(
            handle["transforms/hip"],
            frame_count=frame_count,
            label=f"{path}:transforms/hip",
        )
        if "transforms/spine3" not in handle:
            raise ValueError(f"{path} schema 9 requires transforms/spine3")
        spine3_to_world = _validate_transform_sequence(
            handle["transforms/spine3"],
            frame_count=frame_count,
            label=f"{path}:transforms/spine3",
        )
        body_times = np.asarray(
            handle["nymeria/body_time_seconds"],
            dtype=np.float64,
        )
        source_frame_indices = np.asarray(
            handle["nymeria/source_full_video_frame"],
            dtype=np.int64,
        )
    if aligned_intrinsic.shape != (3, 3):
        raise ValueError(f"{path}:camera/intrinsic must have shape (3, 3)")
    if not np.allclose(aligned_intrinsic, intrinsic, rtol=0.0, atol=1e-5):
        raise ValueError(
            f"{path} camera intrinsic differs from the source clip intrinsic"
        )
    if body_times.shape != (frame_count,):
        raise ValueError(
            f"{path} body_time_seconds must have shape {(frame_count,)}"
        )
    if (
        not np.isfinite(body_times).all()
        or (len(body_times) > 1 and np.any(np.diff(body_times) <= 0.0))
    ):
        raise ValueError(f"{path} has an invalid body timestamp timeline")
    display_timestamps = np.asarray(display_timestamps, dtype=np.float64)
    if display_timestamps.shape != (frame_count,):
        raise ValueError(
            f"Displayed timestamps must have shape {(frame_count,)}"
        )
    relative_body_times = body_times - body_times[0]
    relative_display_times = display_timestamps - display_timestamps[0]
    timeline_drift = float(
        np.max(np.abs(relative_body_times - relative_display_times))
    )
    if timeline_drift > MAXIMUM_CAMERA_TIMELINE_DRIFT_SECONDS:
        raise ValueError(
            f"{path} body/video timeline drift is {timeline_drift:.6f}s, "
            f"above {MAXIMUM_CAMERA_TIMELINE_DRIFT_SECONDS:.3f}s"
        )
    if source_frame_indices.shape != (frame_count,):
        raise ValueError(
            f"{path} source_full_video_frame must have shape {(frame_count,)}"
        )
    if len(source_frame_indices) > 1 and not np.array_equal(
        np.diff(source_frame_indices),
        np.ones(frame_count - 1, dtype=np.int64),
    ):
        raise ValueError(f"{path} source video frame mapping is not contiguous")

    motion_timestamps = body_times - float(motion.actual_start_time)
    maximum_motion_time = (motion.sample_count - 1) / float(motion.fps)
    leading_clamp = max(0.0, -float(motion_timestamps[0]))
    trailing_clamp = max(
        0.0,
        float(motion_timestamps[-1]) - maximum_motion_time,
    )
    if max(leading_clamp, trailing_clamp) > (
        MAXIMUM_MOTION_BOUNDARY_CLAMP_SECONDS + 1e-9
    ):
        raise ValueError(
            f"{path} requires too much motion boundary clamping: "
            f"leading={leading_clamp:.6f}s, trailing={trailing_clamp:.6f}s"
        )
    return NymeriaCameraAlignment(
        hdf5_path=path,
        camera_to_world=camera_to_world,
        hip_to_world=hip_to_world,
        spine3_to_world=spine3_to_world,
        motion_timestamps_seconds=motion_timestamps,
    )


def _motion_hip_rotations_z_up(motion: ResampledMotion) -> np.ndarray:
    return (
        Y_UP_TO_Z_UP
        * motion.root_rotation
        * motion.relative_joint_rotations[0]
    ).as_matrix()


def load_source_episode(
    clip: NymeriaClip,
    *,
    camera_alignment_root: Path,
    camera_alignment_schema: int,
) -> NymeriaSourceEpisode:
    video = probe_display_video(clip.video_path)
    metadata = load_motion_metadata(clip)
    motion = np.load(clip.motion_208_path)
    intrinsic = load_scaled_intrinsic(
        clip,
        video_width=video.width,
        video_height=video.height,
    )
    alignment = load_camera_alignment(
        clip,
        alignment_root=camera_alignment_root,
        alignment_schema=camera_alignment_schema,
        frame_count=video.frame_count,
        intrinsic=intrinsic,
        motion=metadata,
        display_timestamps=video.display_timestamps,
    )
    resampled = resample_motion_to_video(
        motion,
        motion_fps=metadata.fps,
        display_timestamps=video.display_timestamps,
        motion_timestamps=alignment.motion_timestamps_seconds,
        maximum_boundary_clamp_seconds=MAXIMUM_MOTION_BOUNDARY_CLAMP_SECONDS,
    )
    return NymeriaSourceEpisode(
        clip=clip,
        video=video,
        task=metadata.action_text,
        participant_height_m=load_participant_height_m(clip),
        intrinsic=intrinsic,
        resampled=resampled,
        camera_alignment=alignment,
    )


def load_camera_alignment_reports(
    clips: Iterable[NymeriaClip],
    *,
    report_root: Path,
) -> dict[str, dict[str, Any]]:
    """Validate raw Aria camera reports and return compact provenance."""

    clips_by_session: dict[str, set[str]] = {}
    for clip in clips:
        clips_by_session.setdefault(clip.session, set()).add(clip.stem)
    reports: dict[str, dict[str, Any]] = {}
    for session, expected_stems in sorted(
        clips_by_session.items(),
        key=lambda item: natural_key(item[0]),
    ):
        path = (
            report_root.expanduser().resolve()
            / session
            / "visual_alignment_report.json"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Nymeria camera trajectory report is missing: {path}"
            )
        raw = path.read_bytes()
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise TypeError(f"Expected a JSON object in {path}")
        if report.get("status") not in {"prepared_only", "completed"}:
            raise ValueError(
                f"{path} is not complete: status={report.get('status')!r}"
            )
        schema_version = report.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 9
        ):
            raise ValueError(f"{path} has an unsupported schema version")
        adapter_contract = report.get("adapter_contract")
        if (
            not isinstance(adapter_contract, dict)
            or "transforms/spine3"
            not in adapter_contract.get("body_transform_datasets", ())
            or adapter_contract.get("spine3_transform")
            != (
                "per-frame aligned_head_to_world @ "
                "inverse(motion_head_to_world) @ motion_spine3_to_world"
            )
        ):
            raise ValueError(f"{path} does not declare per-frame spine3")
        report_episode = report.get("episode")
        if report_episode is not None and str(report_episode) != session:
            raise ValueError(
                f"{path} declares episode {report_episode!r}, "
                f"expected {session!r}"
            )
        camera_trajectory = report.get("camera_trajectory")
        if (
            not isinstance(camera_trajectory, dict)
            or camera_trajectory.get("uses_208d_pose") is not False
            or camera_trajectory.get("rotation_rejection_threshold_degrees")
            is not None
        ):
            raise ValueError(f"{path} is not a raw Aria camera trajectory")
        prepared = report.get("prepared")
        if not isinstance(prepared, list):
            raise ValueError(f"{path} has no prepared clip list")
        prepared_stems = {
            str(item["clip"])
            for item in prepared
            if isinstance(item, dict) and "clip" in item
        }
        missing = expected_stems - prepared_stems
        if missing:
            raise ValueError(
                f"{path} does not cover selected clips: {sorted(missing)}"
            )
        reports[session] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "schema_version": schema_version,
            "status": str(report["status"]),
            "camera_trajectory": camera_trajectory,
        }
    return reports


def iter_smplx_frames(
    motion: ResampledMotion,
) -> Iterable[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Yield GMR frames and include displayed frame zero."""

    if len(SMPL_JOINT_NAMES) != len(motion.relative_joint_rotations):
        raise ValueError("Nymeria joint count does not match the SMPL-X mapping")
    for frame_index in range(motion.frame_count):
        root_rotation = motion.root_rotation[frame_index]
        root_position = motion.root_position_y_up[frame_index]
        frame: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for joint_index, joint_name in enumerate(SMPL_JOINT_NAMES):
            relative_position = motion.relative_joint_positions[
                frame_index,
                joint_index,
            ]
            world_position_y_up = (
                root_position + root_rotation.apply(relative_position)
            )
            world_rotation_y_up = (
                root_rotation
                * motion.relative_joint_rotations[joint_index][frame_index]
            )
            world_position_z_up = Y_UP_TO_Z_UP.apply(world_position_y_up)
            world_rotation_z_up = Y_UP_TO_Z_UP * world_rotation_y_up
            frame[joint_name] = (
                world_position_z_up.astype(np.float64),
                world_rotation_z_up.as_quat(
                    scalar_first=True,
                ).astype(np.float64),
            )
        yield frame


def _mapped_camera_poses(
    *,
    camera_to_world: np.ndarray,
    hip_to_world: np.ndarray,
    spine3_to_world: np.ndarray,
    motion_hip_rotations_z_up: np.ndarray,
    robot_torso_poses_world: np.ndarray,
    scene_scale: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Apply source spine3-to-head-camera motion to the G1 torso."""

    if not np.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError("scene_scale must be positive and finite")
    camera = np.asarray(camera_to_world, dtype=np.float64)
    hip = np.asarray(hip_to_world, dtype=np.float64)
    spine3 = np.asarray(spine3_to_world, dtype=np.float64)
    motion_hip_rotation = np.asarray(
        motion_hip_rotations_z_up,
        dtype=np.float64,
    )
    robot_torso = np.asarray(robot_torso_poses_world, dtype=np.float64)
    if (
        camera.ndim != 3
        or camera.shape[1:] != (4, 4)
        or hip.shape != camera.shape
        or spine3.shape != camera.shape
        or motion_hip_rotation.shape != (len(camera), 3, 3)
        or robot_torso.shape != (len(camera), 7)
        or not np.isfinite(camera).all()
        or not np.isfinite(hip).all()
        or not np.isfinite(spine3).all()
        or not np.isfinite(motion_hip_rotation).all()
        or not np.isfinite(robot_torso).all()
    ):
        raise ValueError("Aria, aligned body, motion hip, and torso poses must align")
    if len(camera) == 0:
        raise ValueError("Camera alignment requires at least one frame")
    source_to_robot_world_rotation = (
        motion_hip_rotation[0] @ hip[0, :3, :3].T
    )
    (
        camera_pose,
        source_camera_pose_in_spine3,
        source_camera_rotation_world,
        source_spine3_to_robot_torso_rotation,
    ) = retarget_scene_camera_poses_cv(
        source_camera_to_world=camera,
        source_spine3_to_world=spine3,
        source_to_robot_world_rotation=source_to_robot_world_rotation,
        robot_torso_pose_world_xyzw=robot_torso,
        scene_scale=scene_scale,
    )
    camera_spine3_distance = np.linalg.norm(
        source_camera_pose_in_spine3[:, :3], axis=1
    )
    metrics = {
        "source_camera_to_spine3_distance_median_m": float(
            np.median(camera_spine3_distance)
        ),
        "source_camera_to_spine3_distance_p95_m": float(
            np.quantile(camera_spine3_distance, 0.95)
        ),
        "scene_camera_position_source": "g1_torso_fk_plus_scaled_spine3_offset",
        "scene_camera_rotation_source": "mapped_raw_aria_world_rotation",
    }
    return (
        camera_pose.astype(np.float32),
        source_camera_pose_in_spine3.astype(np.float32),
        source_camera_rotation_world.astype(np.float32),
        source_spine3_to_robot_torso_rotation,
        source_to_robot_world_rotation,
        metrics,
    )


def merge_body_with_open_hands(
    model: pin.Model,
    retargeter: GMR,
    body_qpos: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Map the 29 GMR joints and keep the 22 URDF hand joints neutral."""

    joint_names = list(model.names[1:])
    state = np.repeat(
        pin.neutral(model)[None],
        len(body_qpos),
        axis=0,
    )
    mapped_body: list[str] = []
    fixed_hand: list[str] = []
    for joint_name in joint_names:
        pin_joint_id = model.getJointId(joint_name)
        state_index = model.idx_qs[pin_joint_id]
        gmr_joint_id = mj.mj_name2id(
            retargeter.model,
            mj.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if gmr_joint_id < 0:
            fixed_hand.append(joint_name)
            continue
        qpos_index = int(retargeter.model.jnt_qposadr[gmr_joint_id])
        state[:, state_index] = body_qpos[:, qpos_index]
        mapped_body.append(joint_name)

    if len(mapped_body) != 29 or len(fixed_hand) != 22:
        raise ValueError(
            "Expected a 29-body + 22-hand split for the 51-DoF G1 URDF, "
            f"got body={len(mapped_body)}, hand={len(fixed_hand)}"
        )
    if state.shape[1] != 51:
        raise ValueError(f"Expected 51-DoF state, got {state.shape}")
    if np.any(state < model.lowerPositionLimit - 1e-6) or np.any(
        state > model.upperPositionLimit + 1e-6
    ):
        raise ValueError("Retargeted state exceeds G1 URDF joint limits")
    return (
        state.astype(np.float32),
        joint_names,
        mapped_body,
        fixed_hand,
    )


def retarget_source_episode(
    source: NymeriaSourceEpisode,
    *,
    model: pin.Model,
) -> tuple[NymeriaRetargetedEpisode, list[str], list[str], list[str]]:
    started = time.perf_counter()
    retargeter = GMR(
        actual_human_height=source.participant_height_m,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
    )
    body_qpos = np.empty(
        (source.frame_count, retargeter.model.nq),
        dtype=np.float64,
    )
    task_errors = np.empty(source.frame_count, dtype=np.float64)
    # A simple for-loop is intentional: GMR is stateful and frame zero must be
    # solved, unlike the old viewer loop which incremented before retargeting.
    for frame_index, frame in enumerate(iter_smplx_frames(source.resampled)):
        body_qpos[frame_index] = retargeter.retarget(frame)
        task_errors[frame_index] = float(retargeter.error2())

    state, joint_names, body_names, hand_names = merge_body_with_open_hands(
        model,
        retargeter,
        body_qpos,
    )
    root_pose = np.concatenate(
        [
            body_qpos[:, :3],
            body_qpos[:, 3:7][:, [1, 2, 3, 0]],
        ],
        axis=1,
    ).astype(np.float32)
    scene_scale = float(retargeter.human_scale_table["spine3"])
    robot_torso_poses = robot_frame_poses_world(
        urdf_path=G1_URDF,
        joint_names=tuple(joint_names),
        joint_positions=state,
        root_pose_xyzw=root_pose,
        frame_name=HEAD_CAMERA_PARENT_FRAME,
    )
    (
        camera_pose,
        source_camera_pose_in_spine3,
        source_camera_rotation_world,
        source_spine3_to_robot_torso_rotation,
        source_to_robot_world_rotation,
        camera_alignment_metrics,
    ) = _mapped_camera_poses(
        camera_to_world=source.camera_alignment.camera_to_world,
        hip_to_world=source.camera_alignment.hip_to_world,
        spine3_to_world=source.camera_alignment.spine3_to_world,
        motion_hip_rotations_z_up=_motion_hip_rotations_z_up(
            source.resampled
        ),
        robot_torso_poses_world=robot_torso_poses,
        scene_scale=scene_scale,
    )
    joint_steps = (
        np.abs(np.diff(state.astype(np.float64), axis=0))
        if len(state) > 1
        else np.zeros((1, state.shape[1]), dtype=np.float64)
    )
    metrics = {
        "gmr_task_error_mean": float(np.mean(task_errors)),
        "gmr_task_error_max": float(np.max(task_errors)),
        "max_joint_step_rad": float(np.max(joint_steps)),
        "mean_joint_step_rad": float(np.mean(joint_steps)),
        "motion_first_sample_index": float(
            source.resampled.source_sample_positions[0]
        ),
        "motion_last_sample_index": float(
            source.resampled.source_sample_positions[-1]
        ),
    }
    return (
        NymeriaRetargetedEpisode(
            source=source,
            state=state,
            root_pose=root_pose,
            camera_pose=camera_pose,
            source_camera_pose_in_spine3=source_camera_pose_in_spine3,
            source_camera_rotation_world=source_camera_rotation_world,
            source_spine3_to_robot_torso_rotation=(
                source_spine3_to_robot_torso_rotation
            ),
            source_to_robot_world_rotation=source_to_robot_world_rotation,
            scene_scale=scene_scale,
            camera_alignment_metrics=camera_alignment_metrics,
            metrics=metrics,
            elapsed_seconds=time.perf_counter() - started,
        ),
        joint_names,
        body_names,
        hand_names,
    )


def _commit_output(temporary: Path, output: Path) -> None:
    backup: Path | None = None
    if output.exists() or output.is_symlink():
        backup = output.parent / (
            f".{output.name}.backup-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        output.rename(backup)
    try:
        temporary.rename(output)
    except Exception:
        if backup is not None:
            backup.rename(output)
        raise
    else:
        if backup is not None:
            remove_exact(backup)


def write_robot_trajectories(
    episodes: list[NymeriaRetargetedEpisode],
    *,
    joint_names: list[str],
    body_joint_names: list[str],
    fixed_hand_joint_names: list[str],
    output_root: Path,
    overwrite: bool,
    session_timings: dict[str, float],
    alignment_reports: dict[str, dict[str, Any]],
) -> None:
    """Write the smallest native format consumed by the repaint pipeline."""

    if not episodes:
        raise ValueError("No episodes were retargeted")
    if (output_root.exists() or output_root.is_symlink()) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_root}")
    expected_sessions = {episode.source.clip.session for episode in episodes}
    if set(alignment_reports) != expected_sessions:
        raise ValueError(
            "Validated camera reports are required for every session"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / (
        f".{output_root.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    temporary.mkdir(parents=False, exist_ok=False)
    model = build_pinocchio_g1(G1_URDF)
    neutral = pin.neutral(model)
    joint_to_index = {
        name: model.idx_qs[model.getJointId(name)] for name in joint_names
    }
    manifest_episodes: list[dict[str, Any]] = []
    try:
        for episode_index, episode in enumerate(episodes):
            source = episode.source
            episode_relative = Path("episodes") / f"episode_{episode_index:06d}"
            episode_root = temporary / episode_relative
            episode_root.mkdir(parents=True)
            video_relative = episode_relative / source.clip.video_path.name
            trajectory_relative = episode_relative / "trajectory.npz"
            shutil.copy2(source.clip.video_path, temporary / video_relative)
            arrays = {
                "joint_positions": episode.state.astype(np.float32),
                "root_pose_xyzw": episode.root_pose.astype(np.float32),
                "camera_pose_cv_xyzw": episode.camera_pose.astype(np.float32),
                "source_camera_pose_in_spine3_xyzw": (
                    episode.source_camera_pose_in_spine3.astype(np.float32)
                ),
                "source_camera_rotation_world_xyzw": (
                    episode.source_camera_rotation_world.astype(np.float32)
                ),
                "camera_intrinsic": source.intrinsic.astype(np.float64),
                "frame_indices": np.arange(source.frame_count, dtype=np.int64),
                "timestamps": source.video.display_timestamps.astype(np.float64),
            }
            np.savez_compressed(temporary / trajectory_relative, **arrays)

            alignment = {
                "status": "validated",
                "adapter_hdf5": str(source.camera_alignment.hdf5_path),
                "adapter_hdf5_sha256": _sha256_file(
                    source.camera_alignment.hdf5_path
                ),
                "source_spine3_to_robot_torso_rotation": (
                    episode.source_spine3_to_robot_torso_rotation.tolist()
                ),
                "source_to_robot_world_rotation": (
                    episode.source_to_robot_world_rotation.tolist()
                ),
                "world_basis_source": (
                    "frame0_aligned_human_hip_world_to_208d_gmr_world"
                ),
                "world_basis_reference_frame": 0,
                "local_basis_source": (
                    "frame0_source_spine3_axes_mapped_through_initial_"
                    "human_hip_world_basis_into_g1_torso_fk"
                ),
                "local_basis_reference_frame": 0,
                "scene_camera_position_anchor": "g1_torso_fk",
                "scene_camera_rotation_anchor": "raw_aria_world",
                "scene_scale": episode.scene_scale,
                "scene_scale_source": "gmr_spine3_scale",
                "metrics": episode.camera_alignment_metrics,
            }
            manifest_episodes.append(
                {
                    "episode_index": episode_index,
                    "session": source.clip.session,
                    "source_stem": source.clip.stem,
                    "source_video": str(source.clip.video_path),
                    "source_motion_208": str(source.clip.motion_208_path),
                    "source_smpl_metadata": str(source.clip.smpl_path),
                    "source_text": str(source.clip.text_path),
                    "video_path": str(video_relative),
                    "trajectory_path": str(trajectory_relative),
                    "task": source.task,
                    "participant_height_m": source.participant_height_m,
                    "frame_count": source.frame_count,
                    "fps": source.video.fps,
                    "retarget_metrics": episode.metrics,
                    "camera_alignment": alignment,
                    "retarget_elapsed_seconds": episode.elapsed_seconds,
                }
            )

        write_json_atomic(
            temporary / "manifest.json",
            {
                "schema_version": 4,
                "format": ROBOT_FORMAT,
                "robot": "unitree_g1_with_hands",
                "robot_urdf": str(G1_URDF),
                "retargeter": "general_motion_retargeting",
                "joint_names": joint_names,
                "body_joint_names": body_joint_names,
                "hands": {
                    "status": "fixed_open",
                    "joint_names": fixed_hand_joint_names,
                    "joint_values_rad": [
                        float(neutral[joint_to_index[name]])
                        for name in fixed_hand_joint_names
                    ],
                },
                "camera_alignment_quality_gate": {
                    "status": "passed",
                    "camera_trajectory_reports": [
                        alignment_reports[session]
                        for session in sorted(alignment_reports, key=natural_key)
                    ],
                },
                "session_elapsed_seconds": session_timings,
                "episodes": manifest_episodes,
            },
        )
        _commit_output(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def retarget_clips(
    clips: list[NymeriaClip],
    *,
    model: pin.Model,
    camera_alignment_root: Path,
    camera_alignment_schemas: dict[str, int],
) -> tuple[
    list[NymeriaRetargetedEpisode],
    list[str],
    list[str],
    list[str],
    dict[str, float],
]:
    grouped: dict[str, list[NymeriaClip]] = {}
    for clip in clips:
        grouped.setdefault(clip.session, []).append(clip)
    if set(camera_alignment_schemas) != set(grouped):
        raise ValueError("Camera alignment report schemas do not match sessions")
    retargeted: list[NymeriaRetargetedEpisode] = []
    joint_names: list[str] | None = None
    body_names: list[str] | None = None
    hand_names: list[str] | None = None
    timings: dict[str, float] = {}
    for session in sorted(grouped, key=natural_key):
        session_started = time.perf_counter()
        session_clips = sorted(
            grouped[session],
            key=lambda item: natural_key(item.stem),
        )
        progress = tqdm(
            session_clips,
            desc=f"Nymeria {session}",
            unit="task",
            dynamic_ncols=True,
        )
        for clip in progress:
            clip_started = time.perf_counter()
            progress.set_postfix_str(f"{clip.stem}: load/time-map")
            source = load_source_episode(
                clip,
                camera_alignment_root=camera_alignment_root,
                camera_alignment_schema=camera_alignment_schemas[session],
            )
            progress.set_postfix_str(f"{clip.stem}: GMR 29+open-hand 22")
            converted, current_joint, current_body, current_hand = (
                retarget_source_episode(source, model=model)
            )
            if joint_names is None:
                joint_names = current_joint
                body_names = current_body
                hand_names = current_hand
            elif (
                current_joint != joint_names
                or current_body != body_names
                or current_hand != hand_names
            ):
                raise ValueError("G1 joint schema changed between Nymeria clips")
            retargeted.append(converted)
            progress.set_postfix_str(
                f"{clip.stem}: {time.perf_counter() - clip_started:.1f}s"
            )
        timings[session] = time.perf_counter() - session_started
        progress.close()
    assert joint_names is not None
    assert body_names is not None
    assert hand_names is not None
    return retargeted, joint_names, body_names, hand_names, timings


def run_conversion(
    *,
    src_root: Path,
    output_root: Path,
    sessions: set[str] | None = None,
    clip_stems: set[str] | None = None,
    camera_alignment_root: Path,
    camera_alignment_report_root: Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not G1_URDF.is_file():
        raise FileNotFoundError(f"G1 URDF not found: {G1_URDF}")
    clips = discover_nymeria_clips(src_root)
    if sessions is not None:
        clips = [clip for clip in clips if clip.session in sessions]
    if clip_stems is not None:
        clips = [clip for clip in clips if clip.stem in clip_stems]
    if not clips:
        raise ValueError("Session/clip filters selected no Nymeria clips")
    report_root = (
        camera_alignment_root
        if camera_alignment_report_root is None
        else camera_alignment_report_root
    )
    alignment_reports = load_camera_alignment_reports(
        clips,
        report_root=report_root,
    )
    model = build_pinocchio_g1(G1_URDF)
    (
        retargeted,
        joint_names,
        body_names,
        hand_names,
        session_timings,
    ) = retarget_clips(
        clips,
        model=model,
        camera_alignment_root=camera_alignment_root,
        camera_alignment_schemas={
            session: int(report["schema_version"])
            for session, report in alignment_reports.items()
        },
    )
    write_started = time.perf_counter()
    write_robot_trajectories(
        retargeted,
        joint_names=joint_names,
        body_joint_names=body_names,
        fixed_hand_joint_names=hand_names,
        output_root=output_root,
        overwrite=overwrite,
        session_timings=session_timings,
        alignment_reports=alignment_reports,
    )
    write_seconds = time.perf_counter() - write_started
    return {
        "status": "completed",
        "output_root": str(output_root),
        "episodes": len(retargeted),
        "frames": int(sum(item.source.frame_count for item in retargeted)),
        "sessions": sorted({item.source.clip.session for item in retargeted}),
        "session_elapsed_seconds": session_timings,
        "write_elapsed_seconds": write_seconds,
        "total_elapsed_seconds": time.perf_counter() - started,
        "camera_alignment_quality_gate": "passed",
        "camera_alignment_root": str(camera_alignment_root),
        "camera_alignment_report_root": str(report_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget cut Nymeria clips to a source-timed 51-DoF G1 "
            "trajectory directory."
        )
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=WORKSPACE_ROOT / "dataset/nymeria",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "nymeria/robot",
    )
    parser.add_argument(
        "--camera-alignment-root",
        type=Path,
        required=True,
        help=(
            "Root containing <session>/<clip>.hdf5 raw Aria camera "
            "trajectories used by the position-anchored scene camera."
        ),
    )
    parser.add_argument(
        "--camera-alignment-report-root",
        type=Path,
        help=(
            "Root containing <session>/visual_alignment_report.json. "
            "Required for raw-camera provenance and defaults to "
            "--camera-alignment-root."
        ),
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Optional exact session directory name; repeat to select several.",
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=None,
        help="Optional exact clip stem; repeat to select several.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Atomically replace an existing output (default: enabled).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_conversion(
        src_root=args.src_root.resolve(),
        output_root=args.output_root.resolve(),
        sessions=set(args.session) if args.session else None,
        clip_stems=set(args.clip) if args.clip else None,
        camera_alignment_root=(
            args.camera_alignment_root.resolve()
        ),
        camera_alignment_report_root=(
            None
            if args.camera_alignment_report_root is None
            else args.camera_alignment_report_root.resolve()
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
