#!/usr/bin/env python3
"""Retarget cut Nymeria clips to a 51-DoF G1 LeRobot v3 dataset.

Body motion is solved by upstream GMR's 29-DoF Unitree G1 configuration.  The
22 hand joints in ``main_with_hand_new.urdf`` are held at their neutral/open
URDF values because the available Nymeria clips contain rigid wrist/palm pose
but no articulated finger observation.  That limitation is written into the
dataset manifest rather than hidden behind synthetic finger motion.
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

import cv2  # noqa: E402
import h5py  # noqa: E402
import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pinocchio as pin  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from tqdm import tqdm  # noqa: E402

from general_motion_retargeting import (  # noqa: E402
    GeneralMotionRetargeting as GMR,
)
from human_to_robot.retarget_g1_pinocchio import (  # noqa: E402
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


VIDEO_KEY = "observation.images.ego"
DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_TEMPLATE = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
EPISODES_PATH_TEMPLATE = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)
CHUNK_SIZE = 1000
DATA_FILE_SIZE_MB = 100
VIDEO_FILE_SIZE_MB = 200
ACTION_OFFSET = 1
IMAGE_STAT_SAMPLES = 12
G1_URDF = (
    WORKSPACE_ROOT / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
)
STAT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
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
MAXIMUM_ALIGNMENT_P95_ROTATION_ERROR_DEGREES = 10.0
MAXIMUM_ALIGNMENT_ROTATION_ERROR_DEGREES = 30.0
MAXIMUM_HAND_EYE_P95_ROTATION_ERROR_DEGREES = 10.0
MAXIMUM_HAND_EYE_P95_TRANSLATION_ERROR_M = 0.10
MAXIMUM_HAND_EYE_TIME_ERROR_US = 5_000.0
MAXIMUM_CAMERA_TIMELINE_DRIFT_SECONDS = 0.02


@dataclass(frozen=True)
class NymeriaCameraAlignment:
    hdf5_path: Path
    camera_to_world: np.ndarray
    hip_to_world: np.ndarray
    body_times_seconds: np.ndarray
    source_frame_indices: np.ndarray
    motion_timestamps_seconds: np.ndarray
    leading_clamp_seconds: float
    trailing_clamp_seconds: float


@dataclass
class NymeriaSourceEpisode:
    clip: NymeriaClip
    video: DisplayVideoInfo
    motion: MotionMetadata
    task: str
    participant_height_m: float
    intrinsic: np.ndarray
    resampled: ResampledMotion
    camera_alignment: NymeriaCameraAlignment | None = None

    @property
    def frame_count(self) -> int:
        return self.video.frame_count


@dataclass
class NymeriaRetargetedEpisode:
    source: NymeriaSourceEpisode
    state: np.ndarray
    action: np.ndarray
    root_pose: np.ndarray
    camera_pose: np.ndarray | None
    camera_intrinsics: np.ndarray
    source_to_robot_rotation: np.ndarray | None
    source_to_robot_scale: float | None
    camera_alignment_metrics: dict[str, float] | None
    metrics: dict[str, float]
    elapsed_seconds: float


def _decode_hdf5_text(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    frame_count: int,
    intrinsic: np.ndarray,
    motion: MotionMetadata,
    display_timestamps: np.ndarray,
) -> NymeriaCameraAlignment:
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
        body_times_seconds=body_times,
        source_frame_indices=source_frame_indices,
        motion_timestamps_seconds=motion_timestamps,
        leading_clamp_seconds=leading_clamp,
        trailing_clamp_seconds=trailing_clamp,
    )


def load_source_episode(
    clip: NymeriaClip,
    *,
    camera_alignment_root: Path | None = None,
) -> NymeriaSourceEpisode:
    video = probe_display_video(clip.video_path)
    metadata = load_motion_metadata(clip)
    motion = np.load(clip.motion_208_path)
    intrinsic = load_scaled_intrinsic(
        clip,
        video_width=video.width,
        video_height=video.height,
    )
    alignment = (
        None
        if camera_alignment_root is None
        else load_camera_alignment(
            clip,
            alignment_root=camera_alignment_root,
            frame_count=video.frame_count,
            intrinsic=intrinsic,
            motion=metadata,
            display_timestamps=video.display_timestamps,
        )
    )
    resampled = resample_motion_to_video(
        motion,
        motion_fps=metadata.fps,
        display_timestamps=video.display_timestamps,
        motion_timestamps=(
            None if alignment is None else alignment.motion_timestamps_seconds
        ),
        maximum_boundary_clamp_seconds=(
            0.0
            if alignment is None
            else MAXIMUM_MOTION_BOUNDARY_CLAMP_SECONDS
        ),
    )
    return NymeriaSourceEpisode(
        clip=clip,
        video=video,
        motion=metadata,
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
    """Validate session hand-eye reports and return compact provenance."""

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
                f"Nymeria hand-eye quality report is missing: {path}"
            )
        raw = path.read_bytes()
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise TypeError(f"Expected a JSON object in {path}")
        if report.get("status") not in {"prepared_only", "completed"}:
            raise ValueError(
                f"{path} is not complete: status={report.get('status')!r}"
            )
        report_episode = report.get("episode")
        if report_episode is not None and str(report_episode) != session:
            raise ValueError(
                f"{path} declares episode {report_episode!r}, "
                f"expected {session!r}"
            )
        hand_eye = report.get("hand_eye")
        if not isinstance(hand_eye, dict):
            raise ValueError(f"{path} has no hand-eye quality audit")
        metrics = {
            "p95_rotation_error_degrees": float(
                hand_eye.get("p95_rotation_error_degrees", np.inf)
            ),
            "p95_translation_error_m": float(
                hand_eye.get("p95_translation_error_m", np.inf)
            ),
            "maximum_time_error_us": float(
                hand_eye.get("maximum_time_error_us", np.inf)
            ),
        }
        if (
            not all(np.isfinite(list(metrics.values())))
            or metrics["p95_rotation_error_degrees"]
            > MAXIMUM_HAND_EYE_P95_ROTATION_ERROR_DEGREES
            or metrics["p95_translation_error_m"]
            > MAXIMUM_HAND_EYE_P95_TRANSLATION_ERROR_M
            or metrics["maximum_time_error_us"]
            > MAXIMUM_HAND_EYE_TIME_ERROR_US
        ):
            raise ValueError(
                f"{path} hand-eye quality gate failed: {metrics}"
            )
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
            "status": str(report["status"]),
            "metrics": metrics,
            "thresholds": {
                "maximum_p95_rotation_error_degrees": (
                    MAXIMUM_HAND_EYE_P95_ROTATION_ERROR_DEGREES
                ),
                "maximum_p95_translation_error_m": (
                    MAXIMUM_HAND_EYE_P95_TRANSLATION_ERROR_M
                ),
                "maximum_time_error_us": MAXIMUM_HAND_EYE_TIME_ERROR_US,
            },
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


def _mean_proper_rotation(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not len(values):
        raise ValueError("Rotation samples must have shape [N,3,3]")
    u, _, vt = np.linalg.svd(np.mean(values, axis=0))
    correction = np.eye(3, dtype=np.float64)
    correction[2, 2] = np.linalg.det(u @ vt)
    result = u @ correction @ vt
    if not np.allclose(result @ result.T, np.eye(3), atol=1e-6):
        raise ValueError("Mean source-to-robot rotation is not orthonormal")
    if np.linalg.det(result) < 0.999999:
        raise ValueError("Mean source-to-robot rotation is left-handed")
    return result


def estimate_source_to_robot_rotation(
    source: NymeriaSourceEpisode,
) -> tuple[np.ndarray, dict[str, float]]:
    alignment = source.camera_alignment
    if alignment is None:
        raise ValueError("Camera alignment is required to estimate the basis")
    pelvis_rotation_208 = (
        Y_UP_TO_Z_UP
        * source.resampled.root_rotation
        * source.resampled.relative_joint_rotations[0]
    ).as_matrix()
    hip_rotation_source = alignment.hip_to_world[:, :3, :3]
    candidates = np.einsum(
        "nij,nkj->nik",
        pelvis_rotation_208,
        hip_rotation_source,
    )
    source_to_robot = _mean_proper_rotation(candidates)
    residual = Rotation.from_matrix(
        np.einsum(
            "ji,njk->nik",
            source_to_robot,
            candidates,
        )
    ).magnitude()
    residual_degrees = np.rad2deg(residual)
    metrics = {
        "basis_rotation_error_median_degrees": float(
            np.median(residual_degrees)
        ),
        "basis_rotation_error_p95_degrees": float(
            np.quantile(residual_degrees, 0.95)
        ),
        "basis_rotation_error_max_degrees": float(
            np.max(residual_degrees)
        ),
        "motion_leading_clamp_seconds": (
            alignment.leading_clamp_seconds
        ),
        "motion_trailing_clamp_seconds": (
            alignment.trailing_clamp_seconds
        ),
        "motion_clamped_frame_count": float(
            np.count_nonzero(
                (alignment.motion_timestamps_seconds < 0.0)
                | (
                    alignment.motion_timestamps_seconds
                    > (
                        source.motion.sample_count - 1
                    )
                    / source.motion.fps
                )
            )
        ),
    }
    if (
        metrics["basis_rotation_error_p95_degrees"]
        > MAXIMUM_ALIGNMENT_P95_ROTATION_ERROR_DEGREES
        or metrics["basis_rotation_error_max_degrees"]
        > MAXIMUM_ALIGNMENT_ROTATION_ERROR_DEGREES
    ):
        raise ValueError(
            "Nymeria 208-D pelvis and calibrated XSens hip do not admit a "
            "stable source-to-robot basis: "
            f"p95={metrics['basis_rotation_error_p95_degrees']:.3f} deg, "
            f"max={metrics['basis_rotation_error_max_degrees']:.3f} deg"
        )
    return source_to_robot, metrics


def _mapped_camera_poses(
    *,
    root_positions: np.ndarray,
    camera_to_world: np.ndarray,
    hip_to_world: np.ndarray,
    source_to_robot_rotation: np.ndarray,
    source_to_robot_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if not np.isfinite(source_to_robot_scale) or source_to_robot_scale <= 0.0:
        raise ValueError("source_to_robot_scale must be positive and finite")
    relative_source = (
        camera_to_world[:, :3, 3] - hip_to_world[:, :3, 3]
    )
    mapped_relative = (
        source_to_robot_scale
        * (source_to_robot_rotation @ relative_source.T).T
    )
    mapped_position = np.asarray(root_positions, dtype=np.float64) + (
        mapped_relative
    )
    mapped_rotation = np.einsum(
        "ij,njk->nik",
        source_to_robot_rotation,
        camera_to_world[:, :3, :3],
    )
    mapped_quaternion = Rotation.from_matrix(mapped_rotation).as_quat()
    for frame_index in range(1, len(mapped_quaternion)):
        if (
            np.dot(
                mapped_quaternion[frame_index - 1],
                mapped_quaternion[frame_index],
            )
            < 0.0
        ):
            mapped_quaternion[frame_index] *= -1.0
    source_distance = np.linalg.norm(relative_source, axis=1)
    mapped_distance = np.linalg.norm(mapped_relative, axis=1)
    if (
        not np.isfinite(mapped_position).all()
        or not np.isfinite(mapped_quaternion).all()
    ):
        raise ValueError("Mapped Nymeria camera pose contains non-finite values")
    median_source_distance = float(np.median(source_distance))
    if not 0.03 <= median_source_distance <= 1.0:
        raise ValueError(
            "Calibrated Nymeria camera-to-hip distance is implausible: "
            f"median={median_source_distance:.4f} m"
        )
    metrics = {
        "source_camera_to_hip_distance_median_m": median_source_distance,
        "source_camera_to_hip_distance_p05_m": float(
            np.quantile(source_distance, 0.05)
        ),
        "source_camera_to_hip_distance_p95_m": float(
            np.quantile(source_distance, 0.95)
        ),
        "robot_camera_to_root_distance_median_m": float(
            np.median(mapped_distance)
        ),
        "source_to_robot_camera_translation_scale": float(
            source_to_robot_scale
        ),
        "scene_scale_application_count": 1.0,
    }
    pose = np.concatenate([mapped_position, mapped_quaternion], axis=1)
    return pose.astype(np.float32), metrics


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
    action_indices = np.minimum(
        np.arange(source.frame_count) + ACTION_OFFSET,
        source.frame_count - 1,
    )
    action = state[action_indices].copy()
    root_pose = np.concatenate(
        [
            body_qpos[:, :3],
            body_qpos[:, 3:7][:, [1, 2, 3, 0]],
        ],
        axis=1,
    ).astype(np.float32)
    camera_pose: np.ndarray | None = None
    source_to_robot_rotation: np.ndarray | None = None
    source_to_robot_scale: float | None = None
    camera_alignment_metrics: dict[str, float] | None = None
    if source.camera_alignment is not None:
        source_to_robot_rotation, basis_metrics = (
            estimate_source_to_robot_rotation(source)
        )
        source_to_robot_scale = float(
            np.mean(
                [
                    retargeter.human_scale_table["left_wrist"],
                    retargeter.human_scale_table["right_wrist"],
                ]
            )
        )
        camera_pose, camera_metrics = _mapped_camera_poses(
            root_positions=body_qpos[:, :3],
            camera_to_world=source.camera_alignment.camera_to_world,
            hip_to_world=source.camera_alignment.hip_to_world,
            source_to_robot_rotation=source_to_robot_rotation,
            source_to_robot_scale=source_to_robot_scale,
        )
        camera_alignment_metrics = {
            **basis_metrics,
            **camera_metrics,
        }
    intrinsics = np.repeat(
        source.intrinsic.reshape(1, 9),
        source.frame_count,
        axis=0,
    ).astype(np.float32)
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
            action=action,
            root_pose=root_pose,
            camera_pose=camera_pose,
            camera_intrinsics=intrinsics,
            source_to_robot_rotation=source_to_robot_rotation,
            source_to_robot_scale=source_to_robot_scale,
            camera_alignment_metrics=camera_alignment_metrics,
            metrics=metrics,
            elapsed_seconds=time.perf_counter() - started,
        ),
        joint_names,
        body_names,
        hand_names,
    )


def validate_uniform_video_schema(
    episodes: list[NymeriaRetargetedEpisode],
) -> None:
    if not episodes:
        raise ValueError("No episodes were retargeted")
    first = episodes[0].source.video
    if not np.isclose(first.fps, round(first.fps), atol=1e-6):
        raise ValueError(
            f"LeRobot v3 requires integral dataset FPS; source has {first.fps}"
        )
    for episode in episodes[1:]:
        video = episode.source.video
        if not (
            video.width == first.width
            and video.height == first.height
            and np.isclose(video.fps, first.fps, atol=1e-6)
            and video.codec == first.codec
            and video.pixel_format == first.pixel_format
        ):
            raise ValueError(
                "All clips in one LeRobot dataset must share source-derived "
                "resolution, FPS, codec, and pixel format. "
                f"{episode.source.clip.stem} differs from the first clip."
            )


def _arrow_column(values: np.ndarray) -> pa.Array:
    values = np.asarray(values)
    if values.ndim == 1:
        if np.issubdtype(values.dtype, np.floating):
            return pa.array(values, type=pa.float32())
        if np.issubdtype(values.dtype, np.integer):
            return pa.array(values, type=pa.int64())
    elif values.ndim == 2:
        if np.issubdtype(values.dtype, np.floating):
            value_type = pa.float32()
            normalized = values.astype(np.float32, copy=False)
        elif np.issubdtype(values.dtype, np.integer):
            value_type = pa.int64()
            normalized = values.astype(np.int64, copy=False)
        else:
            raise TypeError(f"Unsupported vector dtype: {values.dtype}")
        return pa.array(
            normalized.tolist(),
            type=pa.list_(value_type, list_size=values.shape[1]),
        )
    raise TypeError(f"Unsupported parquet column shape/dtype: {values.shape}")


def _write_data_parquet(
    columns: dict[str, np.ndarray],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [_arrow_column(values) for values in columns.values()],
        names=list(columns),
    )
    pq.write_table(table, path, compression="snappy", use_dictionary=True)


def _numeric_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    floating = values.astype(np.float64)
    result: dict[str, list[Any]] = {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(floating, axis=0).tolist(),
        "std": np.std(floating, axis=0).tolist(),
        "count": [int(len(values))],
    }
    for quantile in STAT_QUANTILES:
        result[f"q{int(quantile * 100):02d}"] = np.quantile(
            floating,
            quantile,
            axis=0,
        ).tolist()
    return result


def _video_stats(
    path: Path,
    *,
    frame_count: int,
) -> dict[str, list[Any]]:
    wanted = set(
        np.linspace(
            0,
            frame_count - 1,
            min(frame_count, IMAGE_STAT_SAMPLES),
        ).round().astype(int).tolist()
    )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not decode {path}")
    samples: list[np.ndarray] = []
    decoded = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if decoded in wanted:
            height, width = frame_bgr.shape[:2]
            scale = min(1.0, 256.0 / max(height, width))
            if scale < 1.0:
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (
                        max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            samples.append(rgb.reshape(-1, 3).astype(np.float32) / 255.0)
        decoded += 1
    capture.release()
    if decoded != frame_count or len(samples) != len(wanted):
        raise ValueError(
            f"Displayed decode mismatch for {path}: decoded={decoded}, "
            f"expected={frame_count}"
        )
    pixels = np.concatenate(samples).astype(np.float64)

    def image_shape(value: np.ndarray) -> list[list[list[float]]]:
        return value.reshape(3, 1, 1).tolist()

    result: dict[str, Any] = {
        "min": image_shape(np.min(pixels, axis=0)),
        "max": image_shape(np.max(pixels, axis=0)),
        "mean": image_shape(np.mean(pixels, axis=0)),
        "std": image_shape(np.std(pixels, axis=0)),
        "count": [len(samples)],
    }
    for quantile in STAT_QUANTILES:
        result[f"q{int(quantile * 100):02d}"] = image_shape(
            np.quantile(pixels, quantile, axis=0)
        )
    return result


def _aggregate_feature_stats(
    feature_stats: list[dict[str, list[Any]]],
) -> dict[str, list[Any]]:
    counts = np.asarray(
        [float(item["count"][0]) for item in feature_stats],
        dtype=np.float64,
    )
    means = np.stack(
        [np.asarray(item["mean"], dtype=np.float64) for item in feature_stats]
    )
    variances = np.stack(
        [
            np.square(np.asarray(item["std"], dtype=np.float64))
            for item in feature_stats
        ]
    )
    weight_shape = (len(counts),) + (1,) * (means.ndim - 1)
    weights = counts.reshape(weight_shape)
    total = float(np.sum(counts))
    mean = np.sum(means * weights, axis=0) / total
    variance = np.sum(
        (variances + np.square(means - mean)) * weights,
        axis=0,
    ) / total
    result: dict[str, Any] = {
        "min": np.min(
            np.stack([np.asarray(item["min"]) for item in feature_stats]),
            axis=0,
        ).tolist(),
        "max": np.max(
            np.stack([np.asarray(item["max"]) for item in feature_stats]),
            axis=0,
        ).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(variance, 0)).tolist(),
        "count": [int(total)],
    }
    for key in ("q01", "q10", "q50", "q90", "q99"):
        quantiles = np.stack(
            [np.asarray(item[key], dtype=np.float64) for item in feature_stats]
        )
        result[key] = (np.sum(quantiles * weights, axis=0) / total).tolist()
    return result


def _aggregate_stats(
    episodes: list[dict[str, dict[str, list[Any]]]],
) -> dict[str, dict[str, list[Any]]]:
    feature_names = sorted(
        {feature for episode in episodes for feature in episode}
    )
    return {
        feature: _aggregate_feature_stats(
            [episode[feature] for episode in episodes if feature in episode]
        )
        for feature in feature_names
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


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
            _remove_path(backup)


def _feature_schema(
    *,
    joint_names: list[str],
    video: DisplayVideoInfo,
    include_camera_pose: bool,
) -> dict[str, dict[str, Any]]:
    fps = int(round(video.fps))
    pose_names = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    intrinsic_names = [
        "k00",
        "k01",
        "k02",
        "k10",
        "k11",
        "k12",
        "k20",
        "k21",
        "k22",
    ]
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(joint_names)],
            "names": joint_names,
            "fps": fps,
        },
        "action": {
            "dtype": "float32",
            "shape": [len(joint_names)],
            "names": joint_names,
            "fps": fps,
        },
        "observation.root_pose": {
            "dtype": "float32",
            "shape": [7],
            "names": pose_names,
            "fps": fps,
        },
        "observation.camera_intrinsics": {
            "dtype": "float32",
            "shape": [9],
            "names": intrinsic_names,
            "fps": fps,
        },
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [video.height, video.width, 3],
            "names": ["height", "width", "channels"],
            "fps": fps,
            "info": {
                "video.height": video.height,
                "video.width": video.width,
                "video.codec": video.codec,
                "video.pix_fmt": video.pixel_format,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": video.has_audio,
                "video.is_depth_map": False,
            },
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if include_camera_pose:
        features["observation.camera_pose"] = {
            "dtype": "float32",
            "shape": [7],
            "names": pose_names,
            "fps": fps,
        }
    return features


def _episode_columns(
    episode: NymeriaRetargetedEpisode,
    *,
    episode_index: int,
    task_index: int,
    global_index: int,
) -> dict[str, np.ndarray]:
    length = episode.source.frame_count
    columns = {
        "observation.state": episode.state,
        "action": episode.action,
        "observation.root_pose": episode.root_pose,
        "observation.camera_intrinsics": episode.camera_intrinsics,
        "timestamp": episode.source.video.display_timestamps.astype(np.float32),
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, episode_index, dtype=np.int64),
        "index": np.arange(
            global_index,
            global_index + length,
            dtype=np.int64,
        ),
        "task_index": np.full(length, task_index, dtype=np.int64),
    }
    if episode.camera_pose is not None:
        columns["observation.camera_pose"] = episode.camera_pose
    return columns


def write_lerobot_v3(
    episodes: list[NymeriaRetargetedEpisode],
    *,
    joint_names: list[str],
    body_joint_names: list[str],
    fixed_hand_joint_names: list[str],
    output_root: Path,
    overwrite: bool,
    session_timings: dict[str, float],
    alignment_reports: dict[str, dict[str, Any]] | None = None,
) -> None:
    validate_uniform_video_schema(episodes)
    if (output_root.exists() or output_root.is_symlink()) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / (
        f".{output_root.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    temporary.mkdir(parents=False, exist_ok=False)
    task_to_index: dict[str, int] = {}
    for episode in episodes:
        task_to_index.setdefault(episode.source.task, len(task_to_index))

    first_video = episodes[0].source.video
    camera_pose_availability = {
        episode.camera_pose is not None for episode in episodes
    }
    if len(camera_pose_availability) != 1:
        raise ValueError(
            "Camera alignment must be available for every episode or none"
        )
    include_camera_pose = camera_pose_availability == {True}
    if include_camera_pose:
        expected_sessions = {
            episode.source.clip.session for episode in episodes
        }
        if alignment_reports is None or set(alignment_reports) != expected_sessions:
            raise ValueError(
                "Validated hand-eye reports are required for every "
                "camera-aligned Nymeria session"
            )
    elif alignment_reports is not None:
        raise ValueError(
            "Hand-eye reports were supplied without camera alignment"
        )
    features = _feature_schema(
        joint_names=joint_names,
        video=first_video,
        include_camera_pose=include_camera_pose,
    )
    episode_rows: list[dict[str, Any]] = []
    all_stats: list[dict[str, dict[str, list[Any]]]] = []
    manifest_episodes: list[dict[str, Any]] = []
    global_index = 0
    try:
        for episode_index, episode in enumerate(episodes):
            source = episode.source
            task_index = task_to_index[source.task]
            chunk_index = episode_index // CHUNK_SIZE
            file_index = episode_index % CHUNK_SIZE
            columns = _episode_columns(
                episode,
                episode_index=episode_index,
                task_index=task_index,
                global_index=global_index,
            )
            data_relative = DATA_PATH_TEMPLATE.format(
                chunk_index=chunk_index,
                file_index=file_index,
            )
            _write_data_parquet(columns, temporary / data_relative)
            video_relative = VIDEO_PATH_TEMPLATE.format(
                video_key=VIDEO_KEY,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            video_destination = temporary / video_relative
            video_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.clip.video_path, video_destination)

            stats = {
                feature: _numeric_stats(values)
                for feature, values in columns.items()
            }
            stats[VIDEO_KEY] = _video_stats(
                source.clip.video_path,
                frame_count=source.frame_count,
            )
            all_stats.append(stats)
            length = source.frame_count
            row: dict[str, Any] = {
                "episode_index": episode_index,
                "tasks": [source.task],
                "length": length,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
                f"videos/{VIDEO_KEY}/chunk_index": chunk_index,
                f"videos/{VIDEO_KEY}/file_index": file_index,
                f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
                f"videos/{VIDEO_KEY}/to_timestamp": (
                    source.video.display_duration
                ),
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for feature, feature_stats in stats.items():
                for stat_name in STAT_KEYS:
                    row[f"stats/{feature}/{stat_name}"] = feature_stats[
                        stat_name
                    ]
            episode_rows.append(row)
            manifest_episodes.append(
                {
                    "episode_index": episode_index,
                    "session": source.clip.session,
                    "source_stem": source.clip.stem,
                    "source_video": str(source.clip.video_path),
                    "source_motion_208": str(source.clip.motion_208_path),
                    "source_smpl_metadata": str(source.clip.smpl_path),
                    "source_text": str(source.clip.text_path),
                    "task": source.task,
                    "participant_height_m": source.participant_height_m,
                    "video": {
                        "width": source.video.width,
                        "height": source.video.height,
                        "fps": source.video.fps,
                        "frame_rate": source.video.frame_rate,
                        "display_frame_count": source.video.frame_count,
                        "declared_encoded_frame_count": (
                            source.video.declared_frame_count
                        ),
                        "discarded_gop_preroll_frame_count": (
                            source.video.discarded_preroll_frames
                        ),
                        "display_duration_seconds": (
                            source.video.display_duration
                        ),
                    },
                    "motion": {
                        "fps": source.motion.fps,
                        "sample_count": source.motion.sample_count,
                        "actual_start_time": source.motion.actual_start_time,
                        "actual_end_time": source.motion.actual_end_time,
                        "resampling": (
                            "linear positions + quaternion SLERP at ffprobe "
                            "display timestamps"
                        ),
                        "root_delta_integration": (
                            "sample i displacement reaches sample i; sample "
                            "zero is the clip reset and is not shifted to i=1"
                        ),
                    },
                    "retarget_metrics": episode.metrics,
                    "camera_alignment": (
                        None
                        if source.camera_alignment is None
                        else {
                            "status": "validated",
                            "adapter_hdf5": str(
                                source.camera_alignment.hdf5_path
                            ),
                            "adapter_hdf5_sha256": _sha256_file(
                                source.camera_alignment.hdf5_path
                            ),
                            "source_to_robot_rotation": (
                                episode.source_to_robot_rotation.tolist()
                            ),
                            "source_to_robot_camera_translation_scale": (
                                episode.source_to_robot_scale
                            ),
                            "mapping": (
                                "EgoDex camera-relative-to-hip mapping onto "
                                "the current GMR root pose"
                            ),
                            "camera_pose_convention": (
                                "T_RobotWorld_CameraOpenCV as "
                                "[x,y,z,qx,qy,qz,qw]"
                            ),
                            "metrics": episode.camera_alignment_metrics,
                            "hand_eye_quality_report": (
                                alignment_reports[source.clip.session]
                            ),
                            "source_body_time_seconds": {
                                "first": float(
                                    source.camera_alignment
                                    .body_times_seconds[0]
                                ),
                                "last": float(
                                    source.camera_alignment
                                    .body_times_seconds[-1]
                                ),
                            },
                            "source_full_video_frame": {
                                "first": int(
                                    source.camera_alignment
                                    .source_frame_indices[0]
                                ),
                                "last": int(
                                    source.camera_alignment
                                    .source_frame_indices[-1]
                                ),
                            },
                        }
                    ),
                    "retarget_elapsed_seconds": episode.elapsed_seconds,
                }
            )
            global_index += length

        episodes_path = temporary / EPISODES_PATH_TEMPLATE.format(
            chunk_index=0,
            file_index=0,
        )
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(episode_rows),
            episodes_path,
            compression="snappy",
            use_dictionary=True,
        )
        task_frame = pd.DataFrame(
            {"task_index": list(task_to_index.values())},
            index=pd.Index(list(task_to_index), name="task"),
        )
        task_path = temporary / "meta/tasks.parquet"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_frame.to_parquet(task_path)
        _write_json(
            temporary / "meta/info.json",
            {
                "codebase_version": "v3.0",
                "robot_type": "unitree_g1_with_hands",
                "total_episodes": len(episodes),
                "total_frames": global_index,
                "total_tasks": len(task_to_index),
                "chunks_size": CHUNK_SIZE,
                "data_files_size_in_mb": DATA_FILE_SIZE_MB,
                "video_files_size_in_mb": VIDEO_FILE_SIZE_MB,
                "fps": int(round(first_video.fps)),
                "splits": {"train": f"0:{len(episodes)}"},
                "data_path": DATA_PATH_TEMPLATE,
                "video_path": VIDEO_PATH_TEMPLATE,
                "features": features,
            },
        )
        _write_json(temporary / "meta/stats.json", _aggregate_stats(all_stats))
        manifest_model = build_pinocchio_g1(G1_URDF)
        neutral = pin.neutral(manifest_model)
        joint_to_index = {
            name: manifest_model.idx_qs[manifest_model.getJointId(name)]
            for name in joint_names
        }
        _write_json(
            temporary / "meta/nymeria.json",
            {
                "schema_version": 5,
                "source_format": (
                    "Nymeria displayed MP4 + same-stem 208-D motion/text/SMPL"
                ),
                "robot": "unitree_g1_with_hands",
                "robot_urdf": str(G1_URDF),
                "retargeter": "general_motion_retargeting",
                "body_solver": "GMR 29-DoF unitree_g1",
                "joint_order": joint_names,
                "body_joint_names": body_joint_names,
                "hands": {
                    "status": "fixed_open",
                    "reason": (
                        "Nymeria clip bundle has no articulated finger "
                        "observations; rigid wrists must not be presented as "
                        "finger ground truth."
                    ),
                    "joint_names": fixed_hand_joint_names,
                    "joint_values_rad": [
                        float(neutral[joint_to_index[name]])
                        for name in fixed_hand_joint_names
                    ],
                },
                "video_geometry_and_fps": (
                    "derived per clip from ffprobe; no configured resolution "
                    "or FPS"
                ),
                "motion_fps": (
                    "derived per clip from same-stem SMPL mocap_frame_rate"
                ),
                "action_offset_frames": ACTION_OFFSET,
                "camera_alignment_quality_gate": (
                    {
                        "status": "passed",
                        "method": (
                            "Aria MPS + XSens hand-eye adapter, then EgoDex "
                            "camera-relative-to-hip mapping onto GMR root"
                        ),
                        "observation_camera_pose": True,
                        "source_to_robot_scale_application_count": 1,
                        "hand_eye_quality_reports": [
                            alignment_reports[session]
                            for session in sorted(
                                alignment_reports,
                                key=natural_key,
                            )
                        ],
                        "fingertip_policy": (
                            "no synthetic fingertips; Isaac scene alignment "
                            "must use none or a future rigid-palm mode"
                        ),
                    }
                    if include_camera_pose
                    else {
                        "status": (
                            "blocked_for_robot_relative_scene_placement"
                        ),
                        "available": [
                            "calibrated undistorted pinhole intrinsics",
                            "source RGB display timestamps",
                        ],
                        "missing": (
                            "a reproducible, verified Aria-camera-to-"
                            "retargeted-G1 body/world alignment adapter"
                        ),
                        "policy": (
                            "observation.camera_pose is omitted instead of "
                            "fabricating a transform; VDA/Isaac robot-relative "
                            "scene placement must not run until this gate passes"
                        ),
                    }
                ),
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
    camera_alignment_root: Path | None = None,
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
    camera_alignment_root: Path | None = None,
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
    if camera_alignment_root is None:
        if camera_alignment_report_root is not None:
            raise ValueError(
                "camera_alignment_report_root requires camera_alignment_root"
            )
        alignment_reports = None
    else:
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
    )
    validate_uniform_video_schema(retargeted)
    write_started = time.perf_counter()
    write_lerobot_v3(
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
        "camera_alignment_quality_gate": (
            "passed"
            if camera_alignment_root is not None
            else "blocked_for_robot_relative_scene_placement"
        ),
        "camera_alignment_root": (
            None
            if camera_alignment_root is None
            else str(camera_alignment_root)
        ),
        "camera_alignment_report_root": (
            None
            if camera_alignment_root is None
            else str(
                camera_alignment_root
                if camera_alignment_report_root is None
                else camera_alignment_report_root
            )
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget cut Nymeria clips to a source-timed 51-DoF G1 "
            "LeRobot v3 dataset."
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
        default=WORKSPACE_ROOT / "nymeria/lerobot_v3",
    )
    parser.add_argument(
        "--camera-alignment-root",
        type=Path,
        help=(
            "Root containing <session>/<clip>.hdf5 Aria/XSens calibration "
            "adapters. When supplied, observation.camera_pose is generated "
            "with the EgoDex camera-relative-to-hip mapping."
        ),
    )
    parser.add_argument(
        "--camera-alignment-report-root",
        type=Path,
        help=(
            "Root containing <session>/visual_alignment_report.json. "
            "Required for the hand-eye quality gate and defaults to "
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
            None
            if args.camera_alignment_root is None
            else args.camera_alignment_root.resolve()
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
