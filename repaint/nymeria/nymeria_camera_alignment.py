"""Align Nymeria 208-D clip motion to recording-level Project Aria MPS.

The MP4/208-D/SMPL bundle owns the body timeline.  Recording-level MPS owns
the camera trajectory and camera/device extrinsic.  Hand-eye calibration uses
all usable clips in a session, but relative motions never cross clip
boundaries and alignment HDF5 files are written only for requested clips.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.hand_trajectory import (  # noqa: E402
    HAND_TARGET_MODE_ATTRIBUTE,
    RIGID_PALM_TARGET,
)
from repaint.vda_pipeline_common import read_json as _read_json  # noqa: E402
from repaint.nymeria.nymeria_adapter import (  # noqa: E402
    DisplayVideoInfo,
    MotionMetadata,
    NymeriaClip,
    ResampledMotion,
    discover_nymeria_clips,
    frame0_anchored_spine3,
    load_motion_metadata,
    load_participant_height_m,
    load_scaled_intrinsic,
    natural_key,
    probe_display_video,
    resample_motion_to_video,
)


CAMERA_CALIBRATION_RELATIVE_PATH = Path(
    "recording_head/mps/slam/online_calibration.jsonl"
)
CAMERA_TRAJECTORY_RELATIVE_PATH = Path(
    "recording_head/mps/slam/closed_loop_trajectory.csv"
)
MAXIMUM_MOTION_BOUNDARY_CLAMP_SECONDS = 0.25
HAND_EYE_SAMPLE_FPS = 15.0
MAXIMUM_HAND_EYE_MEDIAN_ROTATION_ERROR_DEGREES = 5.0
VIDEO_ROTATION_CLOCKWISE_DEGREES = 90.0
CAMERA_RGB_FROM_VIDEO_CAMERA = np.eye(4, dtype=np.float64)
CAMERA_RGB_FROM_VIDEO_CAMERA[:3, :3] = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
MOTION_BODY_JOINTS = (
    ("hip", 0),
    ("neck4", 15),
    ("leftArm", 16),
    ("rightArm", 17),
    ("leftForearm", 18),
    ("rightForearm", 19),
    ("leftHand", 20),
    ("rightHand", 21),
    ("spine3", 9),
)
HEAD_BODY_INDEX = tuple(name for name, _ in MOTION_BODY_JOINTS).index("neck4")


@dataclass(frozen=True)
class CameraCalibration:
    intrinsic: np.ndarray
    source_device_from_camera: np.ndarray
    device_from_video_camera: np.ndarray
    source_model: str
    source_parameters: np.ndarray
    source_record_count: int


@dataclass(frozen=True)
class HandEyeResult:
    device_from_motion_head: np.ndarray
    sample_count: int
    median_rotation_error_degrees: float
    p95_rotation_error_degrees: float
    median_translation_error_m: float
    p95_translation_error_m: float
    maximum_time_error_us: float
    pose_count: int = 0
    clip_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": (
                "AX=XB SO(3)xR3 hand-eye solve using Project Aria "
                "closed-loop device poses and 208-D head poses; relative "
                "motions are formed only within each clip"
            ),
            "sample_count": self.sample_count,
            "relative_motion_pair_count": self.sample_count,
            "pose_count": self.pose_count,
            "calibration_clip_count": self.clip_count,
            "T_Device_MotionHead": self.device_from_motion_head.tolist(),
            "median_rotation_error_degrees": (
                self.median_rotation_error_degrees
            ),
            "p95_rotation_error_degrees": self.p95_rotation_error_degrees,
            "median_translation_error_m": self.median_translation_error_m,
            "p95_translation_error_m": self.p95_translation_error_m,
            "maximum_time_error_us": self.maximum_time_error_us,
        }


@dataclass(frozen=True)
class SlamTrajectory:
    timestamps_us: np.ndarray
    translations: np.ndarray
    quaternions_wxyz: np.ndarray
    quality: np.ndarray


@dataclass(frozen=True)
class ClipMotion:
    clip: NymeriaClip
    video: DisplayVideoInfo
    metadata: MotionMetadata
    resampled: ResampledMotion
    body_transforms: np.ndarray
    body_times_seconds: np.ndarray
    aria_times_us: np.ndarray


@dataclass(frozen=True)
class PreparedClip:
    clip: NymeriaClip
    adapter_hdf5: Path
    frame_count: int
    body_time_start: float
    body_time_end: float
    motion_time_start: float
    motion_time_end: float
    maximum_slam_time_error_us: float
    minimum_slam_quality: float
    low_quality_slam_pose_count: int
    maximum_camera_translation_m: float
    maximum_camera_rotation_degrees: float
    maximum_hand_displacement_m: float


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def load_intrinsic(path: str | Path) -> np.ndarray:
    path = Path(path)
    payload = _read_json(path)
    pinhole = payload.get("undistorted", payload)
    try:
        intrinsic = np.asarray(pinhole["K"], dtype=np.float64)
        resolution = tuple(int(value) for value in pinhole["resolution"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing pinhole K/resolution in {path}") from error
    if intrinsic.shape != (3, 3) or len(resolution) != 2:
        raise ValueError(f"Invalid pinhole calibration in {path}")
    if not np.isfinite(intrinsic).all() or min(resolution) <= 0:
        raise ValueError(f"Non-finite pinhole calibration in {path}")
    if not np.allclose(
        intrinsic[2],
        np.array([0.0, 0.0, 1.0]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(f"K has an invalid homogeneous row in {path}")
    return intrinsic


def _matrix_from_wxyz_translation(
    quaternion_wxyz: Sequence[float],
    translation: Sequence[float],
) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected WXYZ quaternion, got {quaternion.shape}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        quaternion[[1, 2, 3, 0]]
    ).as_matrix()
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def load_recording_camera(
    calibration_path: str | Path,
    intrinsics_path: str | Path,
    *,
    label: str = "camera-rgb",
) -> CameraCalibration:
    calibration_path = Path(calibration_path)
    try:
        handle = calibration_path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Aria calibration not found: {calibration_path}"
        ) from None

    reference_transform: np.ndarray | None = None
    reference_model: str | None = None
    reference_parameters: np.ndarray | None = None
    record_count = 0
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {calibration_path}:{line_number}: "
                    f"{error}"
                ) from error
            selected = next(
                (
                    camera
                    for camera in record.get("CameraCalibrations", [])
                    if camera.get("Label") == label
                ),
                None,
            )
            if selected is None:
                raise ValueError(
                    f"Camera {label!r} is absent at "
                    f"{calibration_path}:{line_number}"
                )
            transform = selected.get("T_Device_Camera", {})
            unit_quaternion = transform.get("UnitQuaternion")
            translation = transform.get("Translation")
            if (
                not isinstance(unit_quaternion, list)
                or len(unit_quaternion) != 2
                or not isinstance(unit_quaternion[1], list)
                or translation is None
            ):
                raise ValueError(
                    "T_Device_Camera has an unsupported layout at "
                    f"{calibration_path}:{line_number}"
                )
            current_transform = _matrix_from_wxyz_translation(
                [unit_quaternion[0], *unit_quaternion[1]],
                translation,
            )
            projection = selected.get("Projection", {})
            current_model = str(projection.get("Name", ""))
            current_parameters = np.asarray(
                projection.get("Params", []),
                dtype=np.float64,
            )
            if not current_model or current_parameters.ndim != 1:
                raise ValueError(
                    f"Invalid RGB projection at "
                    f"{calibration_path}:{line_number}"
                )
            if reference_transform is None:
                reference_transform = current_transform
                reference_model = current_model
                reference_parameters = current_parameters
            elif (
                current_model != reference_model
                or current_parameters.shape != reference_parameters.shape
                or not np.allclose(
                    current_parameters,
                    reference_parameters,
                    rtol=0.0,
                    atol=1e-9,
                )
                or not np.allclose(
                    current_transform,
                    reference_transform,
                    rtol=0.0,
                    atol=1e-10,
                )
            ):
                raise ValueError(
                    "Time-varying recording_head RGB calibration is not "
                    f"supported; first change is at {calibration_path}:"
                    f"{line_number}"
                )
            record_count += 1
    if (
        reference_transform is None
        or reference_model is None
        or reference_parameters is None
    ):
        raise ValueError(f"Aria calibration is empty: {calibration_path}")

    intrinsic = load_intrinsic(intrinsics_path)
    return CameraCalibration(
        intrinsic=intrinsic,
        source_device_from_camera=reference_transform,
        device_from_video_camera=(
            reference_transform @ CAMERA_RGB_FROM_VIDEO_CAMERA
        ),
        source_model=reference_model,
        source_parameters=reference_parameters,
        source_record_count=record_count,
    )


def load_slam_trajectory(path: str | Path) -> SlamTrajectory:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Aria closed-loop trajectory not found: {path}")
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 3, 4, 5, 9, 6, 7, 8, 19),
        dtype=np.float64,
    )
    if data.ndim == 1:
        data = data[None]
    if data.shape[0] < 2 or data.shape[1] != 9:
        raise ValueError(f"Invalid Aria closed-loop trajectory: {path}")
    timestamps = data[:, 0].astype(np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            f"Aria trajectory timestamps are not increasing: {path}"
        )
    quaternions = data[:, 4:8]
    norms = np.linalg.norm(quaternions, axis=1)
    if (
        not np.isfinite(data).all()
        or np.any(norms < 0.1)
        or np.any(data[:, 8] < 0.0)
    ):
        raise ValueError(f"Aria trajectory contains invalid values: {path}")
    quaternions /= norms[:, None]
    return SlamTrajectory(
        timestamps_us=timestamps,
        translations=data[:, 1:4],
        quaternions_wxyz=quaternions,
        quality=data[:, 8],
    )


def nearest_slam_poses(
    trajectory: SlamTrajectory,
    target_timestamps_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = np.asarray(target_timestamps_us, dtype=np.float64)
    if targets.ndim != 1 or len(targets) == 0 or np.any(np.diff(targets) < 0):
        raise ValueError("SLAM target timestamps must be sorted")
    timestamps = trajectory.timestamps_us
    if targets[0] < timestamps[0] or targets[-1] > timestamps[-1]:
        raise ValueError(
            "Requested timestamps fall outside the Aria SLAM trajectory: "
            f"{targets[0]:.0f}..{targets[-1]:.0f} vs "
            f"{timestamps[0]}..{timestamps[-1]}"
        )
    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 1, len(timestamps) - 1)
    left = right - 1
    selected = np.where(
        np.abs(timestamps[left] - targets)
        <= np.abs(timestamps[right] - targets),
        left,
        right,
    )
    return (
        timestamps[selected],
        trajectory.translations[selected],
        trajectory.quaternions_wxyz[selected],
        trajectory.quality[selected],
    )


def _relative_transforms(
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    clip_lengths: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(rotations)
    lengths = (
        (frame_count,)
        if clip_lengths is None
        else tuple(int(length) for length in clip_lengths)
    )
    if any(length < 2 for length in lengths) or sum(lengths) != frame_count:
        raise ValueError(
            "Hand-eye clip lengths must cover all poses with at least two "
            "poses per segment"
        )
    rotation_pairs: list[np.ndarray] = []
    translation_pairs: list[np.ndarray] = []
    start = 0
    for length in lengths:
        stop = start + length
        segment_rotations = rotations[start:stop]
        segment_translations = translations[start:stop]
        rotation_pairs.append(
            np.einsum(
                "nji,njk->nik",
                segment_rotations[:-1],
                segment_rotations[1:],
            )
        )
        translation_pairs.append(
            np.einsum(
                "nji,nj->ni",
                segment_rotations[:-1],
                segment_translations[1:] - segment_translations[:-1],
            )
        )
        start = stop
    return np.concatenate(rotation_pairs), np.concatenate(translation_pairs)


def solve_hand_eye(
    device_rotations: np.ndarray,
    device_translations: np.ndarray,
    motion_rotations: np.ndarray,
    motion_translations: np.ndarray,
    *,
    clip_lengths: Sequence[int] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve ``A X = X B`` without constructing cross-clip motion pairs."""

    arrays = (
        device_rotations,
        device_translations,
        motion_rotations,
        motion_translations,
    )
    frame_count = len(device_rotations)
    if any(len(array) != frame_count for array in arrays):
        raise ValueError("Hand-eye pose arrays must have equal lengths")
    device_r, device_t = _relative_transforms(
        device_rotations,
        device_translations,
        clip_lengths=clip_lengths,
    )
    motion_r, motion_t = _relative_transforms(
        motion_rotations,
        motion_translations,
        clip_lengths=clip_lengths,
    )
    if len(device_r) < 20:
        raise ValueError("Hand-eye solve requires at least 20 motion pairs")

    device_logs = Rotation.from_matrix(device_r).as_rotvec().T
    motion_logs = Rotation.from_matrix(motion_r).as_rotvec().T
    matrix_u, singular_values, matrix_vh = np.linalg.svd(
        motion_logs @ device_logs.T
    )
    rotation = matrix_vh.T @ matrix_u.T
    if np.linalg.det(rotation) < 0:
        matrix_vh[-1] *= -1.0
        rotation = matrix_vh.T @ matrix_u.T
    if singular_values[-1] < 1e-4:
        raise ValueError(
            "Head motion does not sufficiently excite all axes for hand-eye "
            f"calibration; singular values={singular_values.tolist()}"
        )

    jacobian = (device_r - np.eye(3)).reshape(-1, 3)
    residual = (
        np.einsum("ij,nj->ni", rotation, motion_t) - device_t
    ).reshape(-1)
    translation = np.linalg.lstsq(jacobian, residual, rcond=None)[0]

    left_rotations = np.einsum("nij,jk->nik", device_r, rotation)
    right_rotations = np.einsum("ij,njk->nik", rotation, motion_r)
    rotation_errors = Rotation.from_matrix(
        np.einsum(
            "nji,njk->nik",
            left_rotations,
            right_rotations,
        )
    ).magnitude()
    left_translations = (
        np.einsum("nij,j->ni", device_r, translation) + device_t
    )
    right_translations = (
        np.einsum("ij,nj->ni", rotation, motion_t) + translation
    )
    translation_errors = np.linalg.norm(
        left_translations - right_translations,
        axis=1,
    )
    rotation_degrees = np.degrees(rotation_errors)
    metrics = {
        "median_rotation_error_degrees": float(
            np.median(rotation_degrees)
        ),
        "p95_rotation_error_degrees": float(
            np.quantile(rotation_degrees, 0.95)
        ),
        "median_translation_error_m": float(
            np.median(translation_errors)
        ),
        "p95_translation_error_m": float(
            np.quantile(translation_errors, 0.95)
        ),
    }
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform, metrics


def motion_body_transforms(resampled: ResampledMotion) -> np.ndarray:
    """Build pelvis/spine3/head/arm transforms directly from 208-D motion."""

    frame_count = resampled.frame_count
    root_rotation = resampled.root_rotation
    transforms = np.repeat(
        np.eye(4, dtype=np.float64)[None, None],
        frame_count,
        axis=0,
    )
    transforms = np.repeat(
        transforms,
        len(MOTION_BODY_JOINTS),
        axis=1,
    )
    for output_index, (_, joint_index) in enumerate(MOTION_BODY_JOINTS):
        relative_position = resampled.relative_joint_positions[
            :,
            joint_index,
        ]
        transforms[:, output_index, :3, 3] = (
            resampled.root_position_y_up
            + root_rotation.apply(relative_position)
        )
        transforms[:, output_index, :3, :3] = (
            root_rotation
            * resampled.relative_joint_rotations[joint_index]
        ).as_matrix()
    return transforms


def load_clip_motion(clip: NymeriaClip) -> ClipMotion:
    video = probe_display_video(clip.video_path)
    metadata = load_motion_metadata(clip)
    display_times = video.display_timestamps
    motion_times = (
        metadata.csv_start_time
        - metadata.actual_start_time
        + display_times
    )
    motion = np.load(clip.motion_208_path, mmap_mode="r")
    resampled = resample_motion_to_video(
        motion,
        motion_fps=metadata.fps,
        display_timestamps=display_times,
        motion_timestamps=motion_times,
        maximum_boundary_clamp_seconds=(
            MAXIMUM_MOTION_BOUNDARY_CLAMP_SECONDS
        ),
    )
    return ClipMotion(
        clip=clip,
        video=video,
        metadata=metadata,
        resampled=resampled,
        body_transforms=motion_body_transforms(resampled),
        body_times_seconds=metadata.csv_start_time + display_times,
        aria_times_us=(
            metadata.csv_original_start_time + display_times
        )
        * 1.0e6,
    )


def _sample_indices(
    display_timestamps: np.ndarray,
    *,
    sample_fps: float,
) -> np.ndarray:
    if len(display_timestamps) < 2:
        return np.empty(0, dtype=np.int64)
    duration = float(display_timestamps[-1])
    targets = np.arange(
        0.0,
        duration + 0.5 / sample_fps,
        1.0 / sample_fps,
        dtype=np.float64,
    )
    right = np.searchsorted(display_timestamps, targets, side="left")
    right = np.clip(right, 0, len(display_timestamps) - 1)
    left = np.clip(right - 1, 0, len(display_timestamps) - 1)
    indices = np.where(
        np.abs(display_timestamps[left] - targets)
        <= np.abs(display_timestamps[right] - targets),
        left,
        right,
    )
    return np.unique(indices)


def estimate_session_hand_eye(
    *,
    clips: Sequence[ClipMotion],
    slam: SlamTrajectory,
    sample_fps: float,
) -> tuple[HandEyeResult, tuple[str, ...]]:
    device_rotations: list[np.ndarray] = []
    device_translations: list[np.ndarray] = []
    motion_rotations: list[np.ndarray] = []
    motion_translations: list[np.ndarray] = []
    segment_lengths: list[int] = []
    used_clips: list[str] = []
    time_errors: list[np.ndarray] = []

    for clip in clips:
        indices = _sample_indices(
            clip.video.display_timestamps,
            sample_fps=sample_fps,
        )
        if len(indices) < 2:
            continue
        (
            selected_times,
            selected_translations,
            selected_quaternions,
            qualities,
        ) = nearest_slam_poses(slam, clip.aria_times_us[indices])
        valid_positions = np.flatnonzero(qualities >= 0.5)
        runs = np.split(
            valid_positions,
            np.flatnonzero(np.diff(valid_positions) > 1) + 1,
        )
        clip_used = False
        for run in runs:
            if len(run) < 2:
                continue
            selected_indices = indices[run]
            device_rotations.append(
                Rotation.from_quat(
                    selected_quaternions[run][:, [1, 2, 3, 0]]
                ).as_matrix()
            )
            device_translations.append(selected_translations[run])
            head = clip.body_transforms[
                selected_indices,
                HEAD_BODY_INDEX,
            ]
            motion_rotations.append(head[:, :3, :3])
            motion_translations.append(head[:, :3, 3])
            segment_lengths.append(len(run))
            time_errors.append(
                np.abs(
                    selected_times[run]
                    - clip.aria_times_us[selected_indices]
                )
            )
            clip_used = True
        if clip_used:
            used_clips.append(clip.clip.stem)

    if not segment_lengths:
        raise ValueError("No session clip has usable paired head/device poses")
    transform, metrics = solve_hand_eye(
        np.concatenate(device_rotations),
        np.concatenate(device_translations),
        np.concatenate(motion_rotations),
        np.concatenate(motion_translations),
        clip_lengths=segment_lengths,
    )
    result = HandEyeResult(
        device_from_motion_head=transform,
        sample_count=sum(length - 1 for length in segment_lengths),
        pose_count=sum(segment_lengths),
        clip_count=len(used_clips),
        maximum_time_error_us=float(np.max(np.concatenate(time_errors))),
        **metrics,
    )
    if (
        result.maximum_time_error_us > 2_000.0
        or result.median_rotation_error_degrees
        > MAXIMUM_HAND_EYE_MEDIAN_ROTATION_ERROR_DEGREES
        or result.p95_rotation_error_degrees > 15.0
        or result.median_translation_error_m > 0.02
        or result.p95_translation_error_m > 0.10
    ):
        raise ValueError(
            "Nymeria session hand-eye calibration failed residual checks: "
            f"{result.as_dict()}"
        )
    return result, tuple(used_clips)


def world_space_parts(
    motion_parts: np.ndarray,
    world_from_device: np.ndarray,
    *,
    camera: CameraCalibration,
    hand_eye: HandEyeResult,
    head_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        motion_parts.ndim != 4
        or motion_parts.shape[2:] != (4, 4)
        or not 0 <= head_index < motion_parts.shape[1]
    ):
        raise ValueError(
            "Expected [frame, part, 4, 4] motion transforms and valid head "
            "index"
        )
    if world_from_device.shape != (len(motion_parts), 4, 4):
        raise ValueError(
            "Expected one [4, 4] Aria device pose for every motion frame"
        )
    motion_head_from_world = np.linalg.inv(
        motion_parts[:, head_index]
    )
    device_from_motion_head = np.repeat(
        hand_eye.device_from_motion_head[None],
        len(motion_parts),
        axis=0,
    )
    world_from_motion_world = np.einsum(
        "nij,njk,nkl->nil",
        world_from_device,
        device_from_motion_head,
        motion_head_from_world,
    )
    aligned_parts = np.einsum(
        "nij,nkjl->nkil",
        world_from_motion_world,
        motion_parts,
    )
    camera_to_world = np.einsum(
        "nij,jk->nik",
        world_from_device,
        camera.device_from_video_camera,
    )
    return aligned_parts, camera_to_world


def _fixed_base_motion_metrics(
    *,
    camera_to_world: np.ndarray,
    left_hands: np.ndarray,
    right_hands: np.ndarray,
) -> dict[str, float]:
    reference_world_to_camera = np.linalg.inv(camera_to_world[0])
    camera_motion = np.einsum(
        "ij,njk->nik",
        reference_world_to_camera,
        camera_to_world,
    )
    left_reference = np.einsum(
        "ij,njk->nik",
        reference_world_to_camera,
        left_hands,
    )
    right_reference = np.einsum(
        "ij,njk->nik",
        reference_world_to_camera,
        right_hands,
    )
    camera_translation = np.linalg.norm(camera_motion[:, :3, 3], axis=1)
    camera_rotation = np.degrees(
        Rotation.from_matrix(camera_motion[:, :3, :3]).magnitude()
    )
    left_displacement = np.linalg.norm(
        left_reference[:, :3, 3] - left_reference[0, :3, 3],
        axis=1,
    )
    right_displacement = np.linalg.norm(
        right_reference[:, :3, 3] - right_reference[0, :3, 3],
        axis=1,
    )
    return {
        "maximum_camera_translation_m": float(np.max(camera_translation)),
        "maximum_camera_rotation_degrees": float(np.max(camera_rotation)),
        "maximum_hand_displacement_m": float(
            max(np.max(left_displacement), np.max(right_displacement))
        ),
    }


def _world_from_device(
    slam: SlamTrajectory,
    target_times_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_times, translations, quaternions, qualities = (
        nearest_slam_poses(slam, target_times_us)
    )
    time_errors = np.abs(selected_times - target_times_us)
    if np.max(time_errors) > 2_000.0:
        raise ValueError(
            "Aria pose timestamp gap exceeds 2 ms: "
            f"{np.max(time_errors):.1f} us"
        )
    transforms = np.repeat(
        np.eye(4, dtype=np.float64)[None],
        len(target_times_us),
        axis=0,
    )
    transforms[:, :3, :3] = Rotation.from_quat(
        quaternions[:, [1, 2, 3, 0]]
    ).as_matrix()
    transforms[:, :3, 3] = translations
    return transforms, time_errors, qualities


def _validate_body_geometry(
    body_transforms: dict[str, np.ndarray],
    *,
    participant_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_arm_lengths: list[float] = []
    for side in ("left", "right"):
        shoulder = body_transforms[f"{side}Arm"][:, :3, 3]
        elbow = body_transforms[f"{side}Forearm"][:, :3, 3]
        hand = body_transforms[f"{side}Hand"][:, :3, 3]
        lengths = np.linalg.norm(shoulder - elbow, axis=1) + np.linalg.norm(
            elbow - hand,
            axis=1,
        )
        valid = np.isfinite(lengths) & (lengths > 1e-4)
        if np.count_nonzero(valid) < 2:
            raise ValueError(
                f"Nymeria {side} arm has fewer than two valid samples"
            )
        source_arm_lengths.append(float(np.median(lengths[valid])))
    arm_to_height = np.asarray(source_arm_lengths) / participant_height_m
    if np.any((arm_to_height < 0.20) | (arm_to_height > 0.50)):
        raise ValueError(
            "Nymeria 208-D arm length is inconsistent with participant "
            f"height: lengths={source_arm_lengths}, "
            f"height={participant_height_m:.3f} m"
        )
    return np.asarray(source_arm_lengths), arm_to_height


def write_adapter_hdf5(
    path: Path,
    *,
    clip_motion: ClipMotion,
    camera: CameraCalibration,
    camera_to_world: np.ndarray,
    aligned_parts: np.ndarray,
    participant_height_m: float,
) -> None:
    frame_count = clip_motion.video.frame_count
    if (
        aligned_parts.shape
        != (frame_count, len(MOTION_BODY_JOINTS), 4, 4)
        or camera_to_world.shape != (frame_count, 4, 4)
    ):
        raise ValueError("Adapter body arrays have inconsistent shapes")
    body_transforms = {
        name: aligned_parts[:, index]
        for index, (name, _) in enumerate(MOTION_BODY_JOINTS)
    }
    source_arm_lengths, arm_to_height = _validate_body_geometry(
        body_transforms,
        participant_height_m=participant_height_m,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs[HAND_TARGET_MODE_ATTRIBUTE] = RIGID_PALM_TARGET
            handle.attrs["llm_description"] = (
                clip_motion.metadata.action_text
            )
            handle.attrs["source_video"] = str(clip_motion.clip.video_path)
            handle.attrs["source_frame_start"] = 0
            handle.attrs["camera_alignment_contract"] = (
                "raw Aria world-from-OpenCV-camera plus 208-D body poses "
                "registered into the same world"
            )
            handle.attrs["human_to_robot_body_alignment_source"] = (
                "nymeria_208d_calibrated_upper_body"
            )
            handle.create_dataset("camera/intrinsic", data=camera.intrinsic)
            handle.create_dataset("transforms/camera", data=camera_to_world)
            for name, transforms in body_transforms.items():
                handle.create_dataset(f"transforms/{name}", data=transforms)

            metadata = handle.create_group("nymeria")
            metadata.attrs["source_video"] = str(
                clip_motion.clip.video_path
            )
            metadata.attrs["source_motion_208"] = str(
                clip_motion.clip.motion_208_path
            )
            metadata.attrs["source_smpl_metadata"] = str(
                clip_motion.clip.smpl_path
            )
            metadata.attrs["hand_pose"] = (
                "208-D rigid wrist pose; articulated finger pose unavailable"
            )
            metadata.attrs["hand_tracking_validity"] = (
                "finite 208-D wrist transforms; independent of RGB "
                "projection visibility"
            )
            metadata.attrs["camera_frame"] = (
                "Aria MPS world; transforms/camera is "
                "T_World_CameraUpright(t)"
            )
            metadata.attrs["action_text"] = (
                clip_motion.metadata.action_text
            )
            metadata.attrs["participant_height_m"] = participant_height_m
            metadata.attrs["source_arm_lengths_m"] = source_arm_lengths
            metadata.attrs["arm_length_to_height_ratio"] = arm_to_height
            metadata.attrs["body_alignment"] = (
                "208-D pelvis, spine3, head, shoulder, elbow, and wrist "
                "transforms registered into the Aria MPS world"
            )
            metadata.attrs["csv_start_time"] = (
                clip_motion.metadata.csv_start_time
            )
            metadata.attrs["actual_start_time"] = (
                clip_motion.metadata.actual_start_time
            )
            metadata.attrs["csv_original_start_time"] = (
                clip_motion.metadata.csv_original_start_time
            )
            metadata.create_dataset(
                "body_time_seconds",
                data=clip_motion.body_times_seconds,
            )
            metadata.create_dataset(
                "motion_time_seconds",
                data=(
                    clip_motion.body_times_seconds
                    - clip_motion.metadata.actual_start_time
                ),
            )
            metadata.create_dataset(
                "aria_time_seconds",
                data=clip_motion.aria_times_us / 1.0e6,
            )
            metadata.create_dataset(
                "video_display_timestamp_seconds",
                data=clip_motion.video.display_timestamps,
            )
            metadata.create_dataset(
                "source_full_video_frame",
                data=np.arange(frame_count, dtype=np.int64),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_clip(
    *,
    clip_motion: ClipMotion,
    adapter_root: Path,
    camera: CameraCalibration,
    slam: SlamTrajectory,
    hand_eye: HandEyeResult,
    participant_height_m: float,
) -> PreparedClip:
    intrinsic = load_scaled_intrinsic(
        clip_motion.clip,
        video_width=clip_motion.video.width,
        video_height=clip_motion.video.height,
    )
    clip_camera = replace(
        camera,
        intrinsic=intrinsic,
    )
    world_from_device, time_errors, qualities = _world_from_device(
        slam,
        clip_motion.aria_times_us,
    )
    aligned_parts, camera_to_world = world_space_parts(
        clip_motion.body_transforms,
        world_from_device,
        camera=clip_camera,
        hand_eye=hand_eye,
        head_index=HEAD_BODY_INDEX,
    )
    spine3_index = tuple(
        name for name, _ in MOTION_BODY_JOINTS
    ).index("spine3")
    aligned_parts[:, spine3_index] = frame0_anchored_spine3(
        aligned_head_to_world=aligned_parts[:, HEAD_BODY_INDEX],
        motion_head_to_world=clip_motion.body_transforms[:, HEAD_BODY_INDEX],
        motion_spine3_to_world=clip_motion.body_transforms[:, spine3_index],
    )
    aligned_by_name = {
        name: aligned_parts[:, index]
        for index, (name, _) in enumerate(MOTION_BODY_JOINTS)
    }
    motion_metrics = _fixed_base_motion_metrics(
        camera_to_world=camera_to_world,
        left_hands=aligned_by_name["leftHand"],
        right_hands=aligned_by_name["rightHand"],
    )
    adapter_hdf5 = adapter_root / f"{clip_motion.clip.stem}.hdf5"
    write_adapter_hdf5(
        adapter_hdf5,
        clip_motion=clip_motion,
        camera=clip_camera,
        camera_to_world=camera_to_world,
        aligned_parts=aligned_parts,
        participant_height_m=participant_height_m,
    )
    motion_times = (
        clip_motion.body_times_seconds - clip_motion.metadata.actual_start_time
    )
    return PreparedClip(
        clip=clip_motion.clip,
        adapter_hdf5=adapter_hdf5,
        frame_count=clip_motion.video.frame_count,
        body_time_start=float(clip_motion.body_times_seconds[0]),
        body_time_end=float(clip_motion.body_times_seconds[-1]),
        motion_time_start=float(motion_times[0]),
        motion_time_end=float(motion_times[-1]),
        maximum_slam_time_error_us=float(np.max(time_errors)),
        minimum_slam_quality=float(np.min(qualities)),
        low_quality_slam_pose_count=int(np.count_nonzero(qualities < 0.5)),
        **motion_metrics,
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _short_name(stem: str) -> str:
    suffix = stem.rsplit("_", 1)[-1]
    return suffix if re.fullmatch(r"c\d+", suffix) else stem


def _select_clips(
    clips: Sequence[NymeriaClip],
    requested: Sequence[str],
) -> list[NymeriaClip]:
    if not requested:
        return list(clips)
    names = set(requested)
    selected = [
        clip
        for clip in clips
        if clip.stem in names or _short_name(clip.stem) in names
    ]
    found = {
        name
        for name in names
        if any(
            name in (clip.stem, _short_name(clip.stem))
            for clip in selected
        )
    }
    missing = sorted(names - found, key=natural_key)
    if missing:
        raise ValueError(f"Requested clips not found: {missing}")
    return selected


def _prepared_report(clip: PreparedClip) -> dict[str, Any]:
    return {
        "clip": clip.clip.stem,
        "source_video": str(clip.clip.video_path),
        "source_motion_208": str(clip.clip.motion_208_path),
        "source_smpl_metadata": str(clip.clip.smpl_path),
        "adapter_hdf5": str(clip.adapter_hdf5),
        "frame_count": clip.frame_count,
        "body_time_seconds": [
            clip.body_time_start,
            clip.body_time_end,
        ],
        "motion_time_seconds": [
            clip.motion_time_start,
            clip.motion_time_end,
        ],
        "maximum_slam_pose_time_error_us": (
            clip.maximum_slam_time_error_us
        ),
        "slam_quality": {
            "minimum": clip.minimum_slam_quality,
            "below_0_5_frame_count": clip.low_quality_slam_pose_count,
            "rejection_threshold": None,
        },
        "raw_camera_motion": {
            "maximum_camera_translation_m": (
                clip.maximum_camera_translation_m
            ),
            "maximum_camera_rotation_degrees": (
                clip.maximum_camera_rotation_degrees
            ),
            "rotation_rejection_threshold_degrees": None,
        },
        "maximum_registered_hand_displacement_m": (
            clip.maximum_hand_displacement_m
        ),
    }


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    episode_root = Path(args.episode_root).expanduser().resolve()
    if not episode_root.is_dir():
        raise NotADirectoryError(f"Nymeria episode not found: {episode_root}")

    discovered = [
        clip
        for clip in discover_nymeria_clips(episode_root.parent)
        if clip.session == episode_root.name
    ]
    if not discovered:
        raise FileNotFoundError(
            f"No complete clip bundles found for session {episode_root.name}"
        )
    selected = _select_clips(discovered, args.clip)
    if args.max_clips is not None:
        if args.max_clips <= 0:
            raise ValueError("--max-clips must be positive")
        selected = selected[: args.max_clips]

    clip_motions: list[ClipMotion] = []
    calibration_rejected: list[dict[str, str]] = []
    with tqdm(
        discovered,
        desc="Nymeria session clips",
        unit="clip",
        dynamic_ncols=True,
    ) as progress:
        for clip in progress:
            try:
                clip_motions.append(load_clip_motion(clip))
            except Exception as error:
                calibration_rejected.append(
                    {
                        "clip": clip.stem,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    by_stem = {motion.clip.stem: motion for motion in clip_motions}
    missing_selected = [
        clip.stem for clip in selected if clip.stem not in by_stem
    ]
    if missing_selected:
        raise ValueError(
            "Selected clips failed temporal/body preparation: "
            f"{missing_selected}"
        )

    slam = load_slam_trajectory(
        episode_root / CAMERA_TRAJECTORY_RELATIVE_PATH
    )
    hand_eye, calibration_stems = estimate_session_hand_eye(
        clips=clip_motions,
        slam=slam,
        sample_fps=HAND_EYE_SAMPLE_FPS,
    )
    tqdm.write(
        "Session hand-eye calibrated from "
        f"{hand_eye.clip_count} clips / {hand_eye.sample_count} motion pairs: "
        f"median={hand_eye.median_rotation_error_degrees:.3f} deg / "
        f"{1000.0 * hand_eye.median_translation_error_m:.2f} mm"
    )

    camera = load_recording_camera(
        episode_root / CAMERA_CALIBRATION_RELATIVE_PATH,
        discovered[0].camera_intrinsics_path,
    )
    participant_height_m = load_participant_height_m(discovered[0])
    adapter_base = (
        Path(args.adapter_output).expanduser().resolve()
        if args.adapter_output is not None
        else Path(args.output).expanduser().resolve() / "camera_alignment"
    )
    adapter_root = adapter_base / episode_root.name
    report_root = Path(args.output).expanduser().resolve() / episode_root.name
    report_path = report_root / "visual_alignment_report.json"

    prepared: list[PreparedClip] = []
    for clip in selected:
        prepared.append(
            prepare_clip(
                clip_motion=by_stem[clip.stem],
                adapter_root=adapter_root,
                camera=camera,
                slam=slam,
                hand_eye=hand_eye,
                participant_height_m=participant_height_m,
            )
        )

    report = {
        "schema_version": 8,
        "status": "prepared_only",
        "episode": episode_root.name,
        "participant": {
            "metadata": str(discovered[0].session_metadata_path),
            "height_m": participant_height_m,
        },
        "timeline_policy": {
            "aria_time": (
                "csv_original_start_time + video_display_timestamp"
            ),
            "motion_time": (
                "csv_start_time - actual_start_time + "
                "video_display_timestamp"
            ),
            "video_timestamps": "ffprobe displayed frames",
        },
        "intrinsics": {
            "path": str(discovered[0].camera_intrinsics_path),
            "distortion_coefficients": [],
        },
        "recording_head": {
            "calibration": str(
                episode_root / CAMERA_CALIBRATION_RELATIVE_PATH
            ),
            "rgb_calibration_record_count": camera.source_record_count,
            "rgb_calibration_is_constant": True,
            "source_camera_model": camera.source_model,
            "source_projection_parameters": (
                camera.source_parameters.tolist()
            ),
            "T_Device_Camera": (
                camera.source_device_from_camera.tolist()
            ),
            "trajectory": str(
                episode_root / CAMERA_TRAJECTORY_RELATIVE_PATH
            ),
        },
        "video_camera": {
            "orientation": "upright",
            "source_image_rotation_clockwise_degrees": (
                VIDEO_ROTATION_CLOCKWISE_DEGREES
            ),
            "T_CameraRGB_CameraUpright": (
                CAMERA_RGB_FROM_VIDEO_CAMERA.tolist()
            ),
            "T_Device_CameraUpright": (
                camera.device_from_video_camera.tolist()
            ),
        },
        "camera_trajectory": {
            "source": "Project Aria closed-loop device poses",
            "uses_208d_pose": False,
            "rotation_rejection_threshold_degrees": None,
            "quality_policy": (
                "finite in-range SLAM poses with timestamp distance <= 2 ms; "
                "tracking quality is audited but does not reject a clip"
            ),
        },
        "hand_eye": hand_eye.as_dict(),
        "hand_eye_sample_fps": HAND_EYE_SAMPLE_FPS,
        "hand_eye_calibration_clips": list(calibration_stems),
        "hand_eye_rejected_clips": calibration_rejected,
        "requested_clips": [clip.stem for clip in selected],
        "adapter_contract": {
            "camera_transform": (
                "transforms/camera = T_SourceWorld_CameraOpenCV"
            ),
            "body_transforms": (
                "208-D body registered by session hand-eye calibration; "
                "spine3 rotation uses one frame-0 MPS-to-motion anchor"
            ),
            "body_transform_datasets": [
                f"transforms/{name}" for name, _ in MOTION_BODY_JOINTS
            ],
            "spine3_transform": (
                "frame-0 MPS-to-208-D rotation anchor with per-frame "
                "MPS head position and 208-D spine3-to-head offset"
            ),
            "hand_targets": (
                "rigid leftHand/rightHand wrist poses; no synthesized "
                "fingertips; validity is independent of RGB visibility"
            ),
            "frame_mapping": "one adapter frame per displayed MP4 frame",
        },
        "prepared": [_prepared_report(clip) for clip in prepared],
        "skipped": [],
        "failed": [],
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_write_json(report_path, report)
    tqdm.write(
        f"Wrote {len(prepared)} selected camera/body HDF5 file(s); "
        f"session calibration used {len(calibration_stems)} clips; "
        f"elapsed={_format_duration(time.perf_counter() - started)}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Register Nymeria 208-D body poses into raw recording-level "
            "Aria camera trajectories."
        )
    )
    parser.add_argument(
        "--episode-root",
        type=Path,
        required=True,
        help="Nymeria session root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Root for <session>/visual_alignment_report.json",
    )
    parser.add_argument(
        "--adapter-output",
        type=Path,
        help=(
            "Root for selected <session>/<clip>.hdf5 alignments; default "
            "<output>/camera_alignment"
        ),
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        help="Only write this full stem or cNNN suffix; repeat as needed",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        help="Write only the first N selected/discovered clips",
    )
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
