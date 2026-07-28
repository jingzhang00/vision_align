"""Run human-to-robot visual alignment on pre-cut Nymeria clips.

The activity narration CSV supplies clip timing and action text, while the
original XSens segment tracks provide the head and wrist rigid transforms
needed by the existing G1 retargeter.  A recording-level hand-eye solve
registers the XSens head to the Project Aria head device.  The generated
adapter HDF5 files are temporary inputs to ``human_to_robot.pipeline``.

Nymeria's body tracks contain rigid hand/palm transforms but no articulated
finger pose.  The adapter therefore declares rigid-palm targets explicitly;
the shared retargeter keeps the robot hands open and uses its regular base
search and 6D IK.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import av
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from human_to_robot.trajectory import (
    HAND_TARGET_MODE_ATTRIBUTE,
    RIGID_PALM_TARGET,
)
from human_to_robot.video_io import atomic_video_writer


CAMERA_CALIBRATION_RELATIVE_PATH = Path(
    "recording_head/mps/slam/online_calibration.jsonl"
)
CAMERA_TRAJECTORY_RELATIVE_PATH = Path(
    "recording_head/mps/slam/closed_loop_trajectory.csv"
)
BODY_RELATIVE_PATH = Path("body/xdata.npz")
INTRINSICS_RELATIVE_PATH = Path("undistorted/camera_intrinsics.json")
ACTIVITY_NARRATION_RELATIVE_PATH = Path(
    "narration/activity_summarization.csv"
)
ACTIVITY_VIDEO_RELATIVE_PATH = Path(
    "undistorted_384_activity_summarization_clips"
)
FULL_UNDISTORTED_VIDEO_NAME = "video_384_undistord.mp4"
MAX_CLIP_BOUNDARY_DRIFT_SECONDS = 0.25
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
PART_NAMES = (
    "Pelvis",
    "L5",
    "L3",
    "T12",
    "T8",
    "Neck",
    "Head",
    "R_Shoulder",
    "R_UpperArm",
    "R_Forearm",
    "R_Hand",
    "L_Shoulder",
    "L_UpperArm",
    "L_Forearm",
    "L_Hand",
    "R_UpperLeg",
    "R_LowerLeg",
    "R_Foot",
    "R_Toe",
    "L_UpperLeg",
    "L_LowerLeg",
    "L_Foot",
    "L_Toe",
)
PART_INDEX = {name: index for index, name in enumerate(PART_NAMES)}


@dataclass(frozen=True)
class ActivityAnnotation:
    row_index: int
    original_start_time: float
    original_end_time: float
    body_start_time: float
    body_end_time: float
    action_text: str


@dataclass(frozen=True)
class ActivityNarration:
    path: Path
    timecode_offset_seconds: float
    annotations: tuple[ActivityAnnotation, ...]


@dataclass(frozen=True)
class ClipPair:
    stem: str
    video: Path
    annotation: ActivityAnnotation

    @property
    def short_name(self) -> str:
        suffix = self.stem.rsplit("_", 1)[-1]
        return suffix if suffix.startswith("c") else self.stem


@dataclass(frozen=True)
class CameraCalibration:
    intrinsic: np.ndarray
    resolution: tuple[int, int]
    source_device_from_camera: np.ndarray
    device_from_video_camera: np.ndarray
    source_model: str
    source_parameters: np.ndarray
    source_record_count: int


@dataclass(frozen=True)
class HandEyeResult:
    device_from_xsens_head: np.ndarray
    sample_count: int
    median_rotation_error_degrees: float
    p95_rotation_error_degrees: float
    median_translation_error_m: float
    p95_translation_error_m: float
    maximum_time_error_us: float
    timecode_to_device_offset_us: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": (
                "AX=XB SO(3)xR3 hand-eye solve using Project Aria "
                "closed-loop device poses and XSens head poses"
            ),
            "sample_count": self.sample_count,
            "T_Device_XSensHead": self.device_from_xsens_head.tolist(),
            "median_rotation_error_degrees": (
                self.median_rotation_error_degrees
            ),
            "p95_rotation_error_degrees": self.p95_rotation_error_degrees,
            "median_translation_error_m": self.median_translation_error_m,
            "p95_translation_error_m": self.p95_translation_error_m,
            "maximum_time_error_us": self.maximum_time_error_us,
            "timecode_to_device_offset_us": (
                self.timecode_to_device_offset_us
            ),
        }


@dataclass(frozen=True)
class BodyMotion:
    timestamps_us: np.ndarray
    translations: np.ndarray
    quaternions_wxyz: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_us.shape[0])


@dataclass(frozen=True)
class SlamTrajectory:
    timestamps_us: np.ndarray
    translations: np.ndarray
    quaternions_wxyz: np.ndarray
    quality: np.ndarray


@dataclass(frozen=True)
class PreparedClip:
    pair: ClipPair
    normalized_video: Path
    adapter_hdf5: Path
    frame_count: int
    source_start_frame: int
    body_time_start: float
    body_time_end: float
    annotation_start_offset_seconds: float
    annotation_end_offset_seconds: float
    left_visible_fraction: float
    right_visible_fraction: float
    maximum_slam_time_error_us: float
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


def _natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def load_activity_narration(path: str | Path) -> ActivityNarration:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Activity narration CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"start_time", "end_time", "Describe my activity"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing narration columns: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Activity narration CSV is empty: {path}")

    try:
        starts = np.asarray(
            [float(row["start_time"]) for row in rows],
            dtype=np.float64,
        )
        ends = np.asarray(
            [float(row["end_time"]) for row in rows],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} contains an invalid clip timestamp") from error
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError(f"{path} contains non-finite clip timestamps")
    if np.any(ends <= starts):
        invalid = int(np.flatnonzero(ends <= starts)[0])
        raise ValueError(
            f"{path} row {invalid} has end_time <= start_time"
        )

    timecode_offset = float(starts[0])
    annotations = tuple(
        ActivityAnnotation(
            row_index=index,
            original_start_time=float(start),
            original_end_time=float(end),
            body_start_time=float(start - timecode_offset),
            body_end_time=float(end - timecode_offset),
            action_text=str(row["Describe my activity"]).strip(),
        )
        for index, (row, start, end) in enumerate(zip(rows, starts, ends))
    )
    return ActivityNarration(
        path=path,
        timecode_offset_seconds=timecode_offset,
        annotations=annotations,
    )


def discover_clip_pairs(
    video_directory: str | Path,
    narration: ActivityNarration,
    *,
    selected: Sequence[str] = (),
) -> list[ClipPair]:
    video_directory = Path(video_directory).expanduser().resolve()
    if not video_directory.is_dir():
        raise NotADirectoryError(f"Video clip directory not found: {video_directory}")

    videos = {path.stem: path for path in video_directory.glob("*.mp4")}
    by_index: dict[int, Path] = {}
    for stem, path in videos.items():
        match = re.search(r"_c(\d+)$", stem)
        if match is None:
            raise ValueError(
                f"Activity clip stem must end in _cNNN: {path.name}"
            )
        index = int(match.group(1))
        if index in by_index:
            raise ValueError(
                f"Multiple activity videos map to narration row {index}"
            )
        by_index[index] = path

    expected = set(range(len(narration.annotations)))
    actual = set(by_index)
    missing_videos = sorted(expected - actual)
    unexpected_videos = sorted(actual - expected)
    if missing_videos or unexpected_videos:
        raise ValueError(
            "Narration/video clips are not one-to-one by cNNN index; "
            f"missing videos={missing_videos}, "
            f"unexpected videos={unexpected_videos}"
        )
    if not videos:
        raise FileNotFoundError(
            f"No activity MP4 clips found in {video_directory}"
        )

    pairs = [
        ClipPair(
            stem=by_index[index].stem,
            video=by_index[index],
            annotation=narration.annotations[index],
        )
        for index in range(len(narration.annotations))
    ]
    if selected:
        requested = set(selected)

        def matches(pair: ClipPair) -> bool:
            return pair.stem in requested or pair.short_name in requested

        filtered = [pair for pair in pairs if matches(pair)]
        found = {
            name
            for name in requested
            if any(name in (pair.stem, pair.short_name) for pair in filtered)
        }
        missing = sorted(requested - found, key=_natural_key)
        if missing:
            raise ValueError(f"Requested clips not found: {missing}")
        pairs = filtered
    return pairs


def correct_xsens_timestamps(
    timestamps_us: np.ndarray,
    *,
    frame_rate: float = 240.0,
    tolerance_us: float = 1000.0,
) -> np.ndarray:
    """Repair isolated XSens timestamp glitches as the official loader does."""

    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("XSens timestamps must be a one-dimensional sequence")
    nominal = 1.0e6 / frame_rate
    differences = np.diff(timestamps)
    invalid = np.abs(differences - nominal) > tolerance_us
    corrected_differences = differences.copy()
    corrected_differences[invalid] = int(nominal)
    corrected = timestamps[0] + np.concatenate(
        [
            np.zeros(1, dtype=np.int64),
            np.cumsum(corrected_differences, dtype=np.int64),
        ]
    )
    if np.any(np.diff(corrected) <= 0):
        raise ValueError("Corrected XSens timestamps are not strictly increasing")
    if abs(int(corrected[-1]) - int(timestamps[-1])) > 10_000:
        raise ValueError(
            "Correcting XSens timestamp glitches changed the final timestamp "
            "by more than 10 ms"
        )
    return corrected


def load_body_motion(path: str | Path) -> BodyMotion:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Nymeria body motion not found: {path}")
    with np.load(path) as archive:
        required = ("timestamps_us", "segment_tXYZ", "segment_qWXYZ")
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        timestamps = correct_xsens_timestamps(archive["timestamps_us"])
        translations = np.asarray(
            archive["segment_tXYZ"], dtype=np.float64
        ).reshape(-1, len(PART_NAMES), 3)
        quaternions = np.asarray(
            archive["segment_qWXYZ"], dtype=np.float64
        ).reshape(-1, len(PART_NAMES), 4)
    if (
        translations.shape[0] != len(timestamps)
        or quaternions.shape[0] != len(timestamps)
    ):
        raise ValueError(f"Body array frame counts disagree in {path}")
    norms = np.linalg.norm(quaternions, axis=2)
    if np.any(norms < 0.1) or not np.isfinite(norms).all():
        raise ValueError(f"Body motion contains invalid segment quaternions: {path}")
    quaternions /= norms[..., None]
    return BodyMotion(
        timestamps_us=timestamps,
        translations=translations,
        quaternions_wxyz=quaternions,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON {path}: {error}") from error


def load_intrinsic(path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    path = Path(path)
    payload = _read_json(path)
    pinhole = payload.get("undistorted", payload)
    try:
        intrinsic = np.asarray(pinhole["K"], dtype=np.float64)
        resolution = tuple(int(value) for value in pinhole["resolution"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing pinhole K/resolution in {path}") from error
    if intrinsic.shape != (3, 3) or resolution.__len__() != 2:
        raise ValueError(f"Invalid pinhole calibration in {path}")
    if not np.isfinite(intrinsic).all() or min(resolution) <= 0:
        raise ValueError(f"Non-finite pinhole calibration in {path}")
    expected_bottom = np.array([0.0, 0.0, 1.0])
    if not np.allclose(intrinsic[2], expected_bottom, atol=1e-12, rtol=0):
        raise ValueError(f"K has an invalid homogeneous row in {path}")
    return intrinsic, (resolution[0], resolution[1])


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
                    f"Invalid JSON at {calibration_path}:{line_number}: {error}"
                ) from error
            cameras = record.get("CameraCalibrations", [])
            selected = next(
                (camera for camera in cameras if camera.get("Label") == label),
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
                    f"Invalid RGB projection at {calibration_path}:{line_number}"
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

    intrinsic, resolution = load_intrinsic(intrinsics_path)
    device_from_video_camera = (
        reference_transform @ CAMERA_RGB_FROM_VIDEO_CAMERA
    )
    return CameraCalibration(
        intrinsic=intrinsic,
        resolution=resolution,
        source_device_from_camera=reference_transform,
        device_from_video_camera=device_from_video_camera,
        source_model=reference_model,
        source_parameters=reference_parameters,
        source_record_count=record_count,
    )


def _relative_transforms(
    rotations: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    relative_rotations = np.einsum(
        "nji,njk->nik", rotations[:-1], rotations[1:]
    )
    relative_translations = np.einsum(
        "nji,nj->ni",
        rotations[:-1],
        translations[1:] - translations[:-1],
    )
    return relative_rotations, relative_translations


def solve_hand_eye(
    device_rotations: np.ndarray,
    device_translations: np.ndarray,
    xsens_rotations: np.ndarray,
    xsens_translations: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve ``A X = X B`` using Nymeria's SO(3)xR3 formulation."""

    arrays = (
        device_rotations,
        device_translations,
        xsens_rotations,
        xsens_translations,
    )
    frame_count = len(device_rotations)
    if frame_count < 20 or any(len(array) != frame_count for array in arrays):
        raise ValueError("Hand-eye solve requires at least 20 paired poses")
    device_relative = _relative_transforms(
        device_rotations, device_translations
    )
    xsens_relative = _relative_transforms(xsens_rotations, xsens_translations)
    device_r, device_t = device_relative
    xsens_r, xsens_t = xsens_relative

    device_logs = Rotation.from_matrix(device_r).as_rotvec().T
    xsens_logs = Rotation.from_matrix(xsens_r).as_rotvec().T
    matrix_u, singular_values, matrix_vh = np.linalg.svd(
        xsens_logs @ device_logs.T
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
        np.einsum("ij,nj->ni", rotation, xsens_t) - device_t
    ).reshape(-1)
    translation = np.linalg.lstsq(jacobian, residual, rcond=None)[0]

    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for rotation_a, translation_a, rotation_b, translation_b in zip(
        device_r,
        device_t,
        xsens_r,
        xsens_t,
        strict=True,
    ):
        left_rotation = rotation_a @ rotation
        right_rotation = rotation @ rotation_b
        rotation_errors.append(
            Rotation.from_matrix(
                left_rotation.T @ right_rotation
            ).magnitude()
        )
        left_translation = rotation_a @ translation + translation_a
        right_translation = rotation @ translation_b + translation
        translation_errors.append(
            float(np.linalg.norm(left_translation - right_translation))
        )
    rotation_degrees = np.degrees(rotation_errors)
    translation_errors_array = np.asarray(translation_errors)
    metrics = {
        "median_rotation_error_degrees": float(
            np.median(rotation_degrees)
        ),
        "p95_rotation_error_degrees": float(
            np.quantile(rotation_degrees, 0.95)
        ),
        "median_translation_error_m": float(
            np.median(translation_errors_array)
        ),
        "p95_translation_error_m": float(
            np.quantile(translation_errors_array, 0.95)
        ),
    }
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform, metrics


def load_slam_trajectory(path: str | Path) -> SlamTrajectory:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Aria closed-loop trajectory not found: {path}")
    # Loading only the nine required numeric columns is substantially faster
    # and smaller than materializing the 28-column CSV as Python dictionaries.
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
        raise ValueError(f"Aria trajectory timestamps are not increasing: {path}")
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
    choose_left = (
        np.abs(timestamps[left] - targets)
        <= np.abs(timestamps[right] - targets)
    )
    selected = np.where(choose_left, left, right)
    return (
        timestamps[selected],
        trajectory.translations[selected],
        trajectory.quaternions_wxyz[selected],
        trajectory.quality[selected],
    )


def estimate_hand_eye(
    *,
    body: BodyMotion,
    slam: SlamTrajectory,
    timecode_offset_seconds: float,
    sample_fps: float,
) -> HandEyeResult:
    timecode_to_device_offset = (
        float(body.timestamps_us[0])
        - float(timecode_offset_seconds) * 1.0e6
    )
    stride = max(1, int(round(240.0 / sample_fps)))
    sample_indices = np.arange(
        240,
        body.frame_count - 240,
        stride,
        dtype=np.int64,
    )
    target_device_times = (
        body.timestamps_us[sample_indices] - timecode_to_device_offset
    )
    (
        selected_times,
        device_translations,
        device_quaternions,
        qualities,
    ) = nearest_slam_poses(slam, target_device_times)
    time_errors = np.abs(selected_times - target_device_times)
    valid = qualities >= 0.5
    if np.count_nonzero(valid) < 20:
        raise ValueError(
            f"Only {np.count_nonzero(valid)} reliable Aria poses are available"
        )
    sample_indices = sample_indices[valid]
    device_translations = device_translations[valid]
    device_rotations = Rotation.from_quat(
        device_quaternions[valid][:, [1, 2, 3, 0]]
    ).as_matrix()
    head_index = PART_INDEX["Head"]
    xsens_translations = body.translations[sample_indices, head_index]
    xsens_rotations = Rotation.from_quat(
        body.quaternions_wxyz[
            sample_indices, head_index
        ][:, [1, 2, 3, 0]]
    ).as_matrix()
    transform, metrics = solve_hand_eye(
        device_rotations,
        device_translations,
        xsens_rotations,
        xsens_translations,
    )
    result = HandEyeResult(
        device_from_xsens_head=transform,
        sample_count=int(np.count_nonzero(valid)),
        maximum_time_error_us=float(np.max(time_errors[valid])),
        timecode_to_device_offset_us=timecode_to_device_offset,
        **metrics,
    )
    if (
        result.maximum_time_error_us > 2_000.0
        or result.median_rotation_error_degrees > 2.0
        or result.p95_rotation_error_degrees > 15.0
        or result.median_translation_error_m > 0.02
        or result.p95_translation_error_m > 0.10
    ):
        raise ValueError(
            "Nymeria body/head hand-eye calibration failed residual checks: "
            f"{result.as_dict()}"
        )
    return result


def _frame_digest(frame_bgr: np.ndarray) -> bytes:
    return hashlib.blake2b(
        np.ascontiguousarray(frame_bgr).data,
        digest_size=16,
    ).digest()


def _decoded_frame_digests(
    path: Path,
) -> list[bytes]:
    digests: list[bytes] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            digests.append(_frame_digest(frame.to_ndarray(format="bgr24")))
    if not digests:
        raise ValueError(f"Video has no visible decoded frames: {path}")
    return digests


def _match_subsequence(
    full_digests: Sequence[bytes],
    clip_digests: Sequence[bytes],
) -> int:
    if not clip_digests:
        raise ValueError("Cannot match an empty clip digest sequence")
    candidates = np.asarray(
        [
            index
            for index, digest in enumerate(full_digests)
            if digest == clip_digests[0]
        ],
        dtype=np.int64,
    )
    if candidates.size == 0:
        raise ValueError("Clip first frame is absent from the full video")
    probe_sizes = sorted(
        {
            min(len(clip_digests), size)
            for size in (5, 15, 60, len(clip_digests))
        }
    )
    for probe_size in probe_sizes:
        candidates = np.asarray(
            [
                int(index)
                for index in candidates
                if list(
                    full_digests[
                        int(index) : int(index) + probe_size
                    ]
                )
                == list(clip_digests[:probe_size])
            ],
            dtype=np.int64,
        )
        if candidates.size == 1:
            return int(candidates[0])
        if candidates.size == 0:
            raise ValueError(
                f"Clip frames do not form a source subsequence "
                f"(probe={probe_size})"
            )
    raise ValueError(
        "Clip is ambiguous in the full video even after matching all "
        f"{len(clip_digests)} decoded frames; matches={candidates.tolist()}"
    )


def load_capture_timestamps(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        description = container.metadata.get("description")
        stream_frames = int(container.streams.video[0].frames)
    if not description:
        raise ValueError(f"Raw video has no capture timestamp description: {path}")
    try:
        timestamps = np.asarray(json.loads(description), dtype=np.int64)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid video capture timestamps in {path}") from error
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError(f"Video capture timestamp list is empty: {path}")
    if stream_frames > 0 and len(timestamps) != stream_frames:
        raise ValueError(
            f"Video timestamp/frame mismatch in {path}: "
            f"{len(timestamps)} != {stream_frames}"
        )
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Video capture timestamps are not increasing: {path}")
    return timestamps


def _normalize_video(
    source: Path,
    destination: Path,
    *,
    hardware_encoding: bool,
) -> int:
    frame_count = 0
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        output_size = (
            int(stream.codec_context.width),
            int(stream.codec_context.height),
        )
        with atomic_video_writer(
            destination,
            source_video=source,
            output_size=output_size,
            hardware_encoding=hardware_encoding,
        ) as writer:
            for frame in container.decode(stream):
                writer.write(frame.to_ndarray(format="bgr24"))
                frame_count += 1
    if frame_count == 0:
        raise ValueError(f"Video contains no visible frames: {source}")
    return frame_count


def _slerp_wxyz(
    quaternion_left: np.ndarray,
    quaternion_right: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    left = np.asarray(quaternion_left, dtype=np.float64)
    right = np.asarray(quaternion_right, dtype=np.float64).copy()
    alpha = np.asarray(alpha, dtype=np.float64)
    dot = np.sum(left * right, axis=-1)
    flip = dot < 0.0
    right[flip] *= -1.0
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    nearly_linear = sin_angle < 1e-6
    left_weight = np.empty_like(alpha)
    right_weight = np.empty_like(alpha)
    left_weight[nearly_linear] = 1.0 - alpha[nearly_linear]
    right_weight[nearly_linear] = alpha[nearly_linear]
    regular = ~nearly_linear
    left_weight[regular] = np.sin(
        (1.0 - alpha[regular]) * angle[regular]
    ) / sin_angle[regular]
    right_weight[regular] = np.sin(
        alpha[regular] * angle[regular]
    ) / sin_angle[regular]
    result = (
        left_weight[..., None] * left
        + right_weight[..., None] * right
    )
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return result


def interpolate_parts(
    body: BodyMotion,
    target_timestamps_us: np.ndarray,
    part_indices: Sequence[int],
) -> np.ndarray:
    targets = np.asarray(target_timestamps_us, dtype=np.float64)
    if (
        targets.ndim != 1
        or len(targets) == 0
        or targets[0] < body.timestamps_us[0]
        or targets[-1] > body.timestamps_us[-1]
    ):
        raise ValueError("Requested body timestamps fall outside XSens motion")
    right = np.searchsorted(body.timestamps_us, targets, side="left")
    right = np.clip(right, 1, body.frame_count - 1)
    left = right - 1
    left_time = body.timestamps_us[left].astype(np.float64)
    right_time = body.timestamps_us[right].astype(np.float64)
    alpha = (targets - left_time) / (right_time - left_time)

    parts = np.asarray(part_indices, dtype=np.int64)
    translations_left = body.translations[left[:, None], parts[None, :]]
    translations_right = body.translations[right[:, None], parts[None, :]]
    translations = (
        (1.0 - alpha[:, None, None]) * translations_left
        + alpha[:, None, None] * translations_right
    )
    quaternions = _slerp_wxyz(
        body.quaternions_wxyz[left[:, None], parts[None, :]],
        body.quaternions_wxyz[right[:, None], parts[None, :]],
        np.repeat(alpha[:, None], len(parts), axis=1),
    )
    transforms = np.repeat(
        np.eye(4, dtype=np.float64)[None, None],
        len(targets),
        axis=0,
    )
    transforms = np.repeat(transforms, len(parts), axis=1)
    transforms[:, :, :3, :3] = Rotation.from_quat(
        quaternions[..., [1, 2, 3, 0]].reshape(-1, 4)
    ).as_matrix().reshape(len(targets), len(parts), 3, 3)
    transforms[:, :, :3, 3] = translations
    return transforms


def camera_space_hands(
    xsens_parts: np.ndarray,
    *,
    camera: CameraCalibration,
    hand_eye: HandEyeResult,
) -> tuple[np.ndarray, np.ndarray]:
    if xsens_parts.ndim != 4 or xsens_parts.shape[1:] != (3, 4, 4):
        raise ValueError(
            "Expected [frame, head/left/right, 4, 4] XSens transforms"
        )
    xsens_head_from_world = np.linalg.inv(xsens_parts[:, 0])
    device_from_camera_inverse = np.linalg.inv(
        camera.device_from_video_camera
    )
    camera_from_xsens_head = (
        device_from_camera_inverse
        @ hand_eye.device_from_xsens_head
    )
    camera_from_xsens_head = np.repeat(
        camera_from_xsens_head[None],
        len(xsens_parts),
        axis=0,
    )
    camera_from_world = np.einsum(
        "nij,njk->nik",
        camera_from_xsens_head,
        xsens_head_from_world,
    )
    left = np.einsum("nij,njk->nik", camera_from_world, xsens_parts[:, 1])
    right = np.einsum("nij,njk->nik", camera_from_world, xsens_parts[:, 2])
    return left, right


def world_space_hands(
    xsens_parts: np.ndarray,
    world_from_device: np.ndarray,
    *,
    camera: CameraCalibration,
    hand_eye: HandEyeResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Register XSens hands and the RGB camera in one Aria world frame."""

    if xsens_parts.ndim != 4 or xsens_parts.shape[1:] != (3, 4, 4):
        raise ValueError(
            "Expected [frame, head/left/right, 4, 4] XSens transforms"
        )
    if world_from_device.shape != (len(xsens_parts), 4, 4):
        raise ValueError(
            "Expected one [4, 4] Aria device pose for every XSens frame"
        )
    xsens_head_from_world = np.linalg.inv(xsens_parts[:, 0])
    device_from_xsens_head = np.repeat(
        hand_eye.device_from_xsens_head[None],
        len(xsens_parts),
        axis=0,
    )
    world_from_xsens_world = np.einsum(
        "nij,njk,nkl->nil",
        world_from_device,
        device_from_xsens_head,
        xsens_head_from_world,
    )
    left = np.einsum(
        "nij,njk->nik",
        world_from_xsens_world,
        xsens_parts[:, 1],
    )
    right = np.einsum(
        "nij,njk->nik",
        world_from_xsens_world,
        xsens_parts[:, 2],
    )
    camera_to_world = np.einsum(
        "nij,jk->nik",
        world_from_device,
        camera.device_from_video_camera,
    )
    return left, right, camera_to_world


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
    camera_translation = np.linalg.norm(
        camera_motion[:, :3, 3],
        axis=1,
    )
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


def _project(
    points_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    z = points_camera[:, 2]
    pixels = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    valid = z > 1e-6
    pixels[valid, 0] = (
        intrinsic[0, 0] * points_camera[valid, 0] / z[valid]
        + intrinsic[0, 2]
    )
    pixels[valid, 1] = (
        intrinsic[1, 1] * points_camera[valid, 1] / z[valid]
        + intrinsic[1, 2]
    )
    return pixels


def _visibility(
    transforms: np.ndarray,
    *,
    intrinsic: np.ndarray,
    resolution: tuple[int, int],
) -> np.ndarray:
    pixels = _project(transforms[:, :3, 3], intrinsic)
    width, height = resolution
    return (
        np.isfinite(pixels).all(axis=1)
        & (transforms[:, 2, 3] > 0.03)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )


def write_adapter_hdf5(
    path: Path,
    *,
    pair: ClipPair,
    narration_path: Path,
    camera: CameraCalibration,
    camera_to_world: np.ndarray,
    left_hands: np.ndarray,
    right_hands: np.ndarray,
    left_confidence: np.ndarray,
    right_confidence: np.ndarray,
    body_times: np.ndarray,
    source_frame_indices: np.ndarray,
) -> None:
    frame_count = len(left_hands)
    if (
        right_hands.shape != left_hands.shape
        or left_hands.shape != (frame_count, 4, 4)
        or camera_to_world.shape != (frame_count, 4, 4)
        or left_confidence.shape != (frame_count,)
        or right_confidence.shape != (frame_count,)
    ):
        raise ValueError("Adapter hand arrays have inconsistent shapes")
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
            handle.create_dataset("camera/intrinsic", data=camera.intrinsic)
            handle.create_dataset(
                "transforms/camera",
                data=camera_to_world,
            )
            for side, hands, confidence in (
                ("left", left_hands, left_confidence),
                ("right", right_hands, right_confidence),
            ):
                handle.create_dataset(f"transforms/{side}Hand", data=hands)
                handle.create_dataset(
                    f"confidences/{side}Hand", data=confidence
                )
            metadata = handle.create_group("nymeria")
            metadata.attrs["source_video"] = str(pair.video)
            metadata.attrs["source_narration"] = str(narration_path)
            metadata.attrs["narration_row_index"] = pair.annotation.row_index
            metadata.attrs["hand_pose"] = (
                "XSens rigid hand/palm pose; articulated finger pose "
                "unavailable"
            )
            metadata.attrs["camera_frame"] = (
                "Aria MPS world; transforms/camera is "
                "T_World_CameraUpright(t)"
            )
            metadata.attrs["action_text"] = pair.annotation.action_text
            metadata.attrs["annotation_original_start_time"] = (
                pair.annotation.original_start_time
            )
            metadata.attrs["annotation_original_end_time"] = (
                pair.annotation.original_end_time
            )
            metadata.create_dataset("body_time_seconds", data=body_times)
            metadata.create_dataset(
                "source_full_video_frame", data=source_frame_indices
            )
            metadata.create_dataset(
                "annotation_relative_time_seconds",
                data=body_times - pair.annotation.body_start_time,
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _prepare_clip(
    *,
    pair: ClipPair,
    adapter_root: Path,
    camera: CameraCalibration,
    body: BodyMotion,
    slam: SlamTrajectory,
    hand_eye: HandEyeResult,
    narration_path: Path,
    timecode_offset_seconds: float,
    capture_timestamps: np.ndarray,
    full_digests: Sequence[bytes],
    hardware_encoding: bool,
    require_visible_hands: bool,
    motion_filter: bool,
    maximum_camera_translation_m: float,
    maximum_camera_rotation_degrees: float,
    maximum_hand_displacement_m: float,
) -> tuple[PreparedClip | None, dict[str, Any] | None]:
    normalized_video = adapter_root / f"{pair.stem}.mp4"
    adapter_hdf5 = adapter_root / f"{pair.stem}.hdf5"
    try:
        with av.open(str(pair.video)) as container:
            stream = container.streams.video[0]
            source_resolution = (
                int(stream.codec_context.width),
                int(stream.codec_context.height),
            )
        if source_resolution != camera.resolution:
            raise ValueError(
                f"{pair.video} resolution {source_resolution} does not match "
                f"intrinsics resolution {camera.resolution}"
            )
        clip_digests = _decoded_frame_digests(pair.video)
        source_start = _match_subsequence(full_digests, clip_digests)
        frame_count = len(clip_digests)
        if source_start + frame_count > len(full_digests):
            raise ValueError(
                f"{pair.video} extends past the full undistorted video"
            )
        source_frames = source_start + np.arange(
            frame_count,
            dtype=np.int64,
        )
        raw_frames = 2 * source_frames
        if raw_frames[-1] >= len(capture_timestamps):
            raise ValueError(
                f"{pair.video} extends past the raw capture timestamps"
            )
        tracking_seconds = (
            capture_timestamps[raw_frames].astype(np.float64) / 1.0e9
        )
        (
            selected_slam_times,
            device_translations,
            device_quaternions,
            device_quality,
        ) = nearest_slam_poses(
            slam,
            capture_timestamps[raw_frames].astype(np.float64) / 1.0e3,
        )
        slam_time_errors = np.abs(
            selected_slam_times
            - capture_timestamps[raw_frames].astype(np.float64) / 1.0e3
        )
        if np.max(slam_time_errors) > 2_000.0:
            raise ValueError(
                f"{pair.video} has an Aria pose timestamp gap of "
                f"{np.max(slam_time_errors):.1f} us"
            )
        if np.any(device_quality < 0.5):
            raise ValueError(
                f"{pair.video} contains unreliable Aria closed-loop poses"
            )
        world_from_device = np.repeat(
            np.eye(4, dtype=np.float64)[None],
            frame_count,
            axis=0,
        )
        world_from_device[:, :3, :3] = Rotation.from_quat(
            device_quaternions[:, [1, 2, 3, 0]]
        ).as_matrix()
        world_from_device[:, :3, 3] = device_translations
        body_times = tracking_seconds - float(timecode_offset_seconds)
        annotation_start = pair.annotation.body_start_time
        annotation_end = pair.annotation.body_end_time
        start_offset = float(body_times[0] - annotation_start)
        end_offset = float(body_times[-1] - annotation_end)
        if (
            start_offset < -MAX_CLIP_BOUNDARY_DRIFT_SECONDS
            or end_offset > MAX_CLIP_BOUNDARY_DRIFT_SECONDS
            or body_times[-1] < annotation_start
            or body_times[0] > annotation_end
        ):
            raise ValueError(
                f"Video/body timing disagrees with narration row "
                f"{pair.annotation.row_index} in {narration_path}: "
                f"video body range={body_times[0]:.6f}.."
                f"{body_times[-1]:.6f}, annotation range="
                f"{annotation_start:.6f}..{annotation_end:.6f}, "
                "allowed boundary drift="
                f"{MAX_CLIP_BOUNDARY_DRIFT_SECONDS:.3f}s"
            )

        body_timestamps = body.timestamps_us[0] + body_times * 1.0e6
        parts = interpolate_parts(
            body,
            body_timestamps,
            [
                PART_INDEX["Head"],
                PART_INDEX["L_Hand"],
                PART_INDEX["R_Hand"],
            ],
        )
        left_hands, right_hands, camera_to_world = world_space_hands(
            parts,
            world_from_device,
            camera=camera,
            hand_eye=hand_eye,
        )
        world_to_camera = np.linalg.inv(camera_to_world)
        left_observation = np.einsum(
            "nij,njk->nik",
            world_to_camera,
            left_hands,
        )
        right_observation = np.einsum(
            "nij,njk->nik",
            world_to_camera,
            right_hands,
        )
        left_visible = _visibility(
            left_observation,
            intrinsic=camera.intrinsic,
            resolution=camera.resolution,
        )
        right_visible = _visibility(
            right_observation,
            intrinsic=camera.intrinsic,
            resolution=camera.resolution,
        )
        motion = _fixed_base_motion_metrics(
            camera_to_world=camera_to_world,
            left_hands=left_hands,
            right_hands=right_hands,
        )
        if motion_filter and (
            motion["maximum_camera_translation_m"]
            > maximum_camera_translation_m
            or motion["maximum_camera_rotation_degrees"]
            > maximum_camera_rotation_degrees
            or motion["maximum_hand_displacement_m"]
            > maximum_hand_displacement_m
        ):
            return None, {
                "clip": pair.stem,
                "reason": (
                    "global wearer motion is incompatible with the current "
                    "fixed-base G1 retargeter"
                ),
                **motion,
            }
        if require_visible_hands and (
            np.count_nonzero(left_visible) < 2
            or np.count_nonzero(right_visible) < 2
        ):
            normalized_video.unlink(missing_ok=True)
            return None, {
                "clip": pair.stem,
                "reason": (
                    "fewer than two frames contain each projected wearer wrist"
                ),
                "left_visible_frames": int(np.count_nonzero(left_visible)),
                "right_visible_frames": int(np.count_nonzero(right_visible)),
            }

        normalized_count = _normalize_video(
            pair.video,
            normalized_video,
            hardware_encoding=hardware_encoding,
        )
        if normalized_count != frame_count:
            raise ValueError(
                f"Visible frame count changed while normalizing {pair.video}: "
                f"{frame_count} -> {normalized_count}"
            )
        if require_visible_hands:
            left_confidence = left_visible.astype(np.float32)
            right_confidence = right_visible.astype(np.float32)
        else:
            left_confidence = np.ones(frame_count, dtype=np.float32)
            right_confidence = np.ones(frame_count, dtype=np.float32)
        write_adapter_hdf5(
            adapter_hdf5,
            pair=pair,
            narration_path=narration_path,
            camera=camera,
            camera_to_world=camera_to_world,
            left_hands=left_hands,
            right_hands=right_hands,
            left_confidence=left_confidence,
            right_confidence=right_confidence,
            body_times=body_times,
            source_frame_indices=source_frames,
        )
        return (
            PreparedClip(
                pair=pair,
                normalized_video=normalized_video,
                adapter_hdf5=adapter_hdf5,
                frame_count=frame_count,
                source_start_frame=source_start,
                body_time_start=float(body_times[0]),
                body_time_end=float(body_times[-1]),
                annotation_start_offset_seconds=start_offset,
                annotation_end_offset_seconds=end_offset,
                left_visible_fraction=float(np.mean(left_visible)),
                right_visible_fraction=float(np.mean(right_visible)),
                maximum_slam_time_error_us=float(np.max(slam_time_errors)),
                maximum_camera_translation_m=motion[
                    "maximum_camera_translation_m"
                ],
                maximum_camera_rotation_degrees=motion[
                    "maximum_camera_rotation_degrees"
                ],
                maximum_hand_displacement_m=motion[
                    "maximum_hand_displacement_m"
                ],
            ),
            None,
        )
    except BaseException:
        normalized_video.unlink(missing_ok=True)
        adapter_hdf5.unlink(missing_ok=True)
        raise


def prepare_clips(
    *,
    pairs: Sequence[ClipPair],
    episode_root: Path,
    adapter_root: Path,
    camera: CameraCalibration,
    body: BodyMotion,
    slam: SlamTrajectory,
    hand_eye: HandEyeResult,
    narration_path: Path,
    timecode_offset_seconds: float,
    hardware_encoding: bool,
    require_visible_hands: bool,
    motion_filter: bool,
    maximum_camera_translation_m: float,
    maximum_camera_rotation_degrees: float,
    maximum_hand_displacement_m: float,
) -> tuple[
    list[PreparedClip],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw_video = episode_root / "video.mp4"
    full_undistorted = episode_root / FULL_UNDISTORTED_VIDEO_NAME
    if not raw_video.is_file():
        raise FileNotFoundError(f"Raw Nymeria video not found: {raw_video}")
    if not full_undistorted.is_file():
        raise FileNotFoundError(
            f"Full 15 fps undistorted video not found: {full_undistorted}"
        )
    capture_timestamps = load_capture_timestamps(raw_video)
    full_digests = _decoded_frame_digests(full_undistorted)
    if abs(len(capture_timestamps) - 2 * len(full_digests)) > 1:
        raise ValueError(
            "The full 15 fps undistorted video is not a one-in-two "
            "downsample of the raw 30 fps video"
        )

    prepared: list[PreparedClip] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    started = time.perf_counter()
    with tqdm(
        pairs,
        desc="Nymeria alignment prepare",
        unit="clip",
        dynamic_ncols=True,
    ) as progress:
        for pair in progress:
            clip_started = time.perf_counter()
            try:
                prepared_clip, skipped_clip = _prepare_clip(
                    pair=pair,
                    adapter_root=adapter_root,
                    camera=camera,
                    body=body,
                    slam=slam,
                    hand_eye=hand_eye,
                    narration_path=narration_path,
                    timecode_offset_seconds=timecode_offset_seconds,
                    capture_timestamps=capture_timestamps,
                    full_digests=full_digests,
                    hardware_encoding=hardware_encoding,
                    require_visible_hands=require_visible_hands,
                    motion_filter=motion_filter,
                    maximum_camera_translation_m=(
                        maximum_camera_translation_m
                    ),
                    maximum_camera_rotation_degrees=(
                        maximum_camera_rotation_degrees
                    ),
                    maximum_hand_displacement_m=maximum_hand_displacement_m,
                )
            except KeyboardInterrupt:
                tqdm.write(
                    f"Interrupted {pair.short_name} after "
                    f"{_format_duration(time.perf_counter() - clip_started)}"
                )
                raise
            except Exception as error:
                failed.append(
                    {
                        "clip": pair.stem,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                tqdm.write(
                    f"Failed {pair.short_name} after "
                    f"{_format_duration(time.perf_counter() - clip_started)}: "
                    f"{type(error).__name__}: {error}"
                )
                progress.set_postfix_str(
                    f"last={pair.short_name} "
                    f"time={_format_duration(time.perf_counter() - clip_started)}",
                    refresh=False,
                )
                continue
            if skipped_clip is not None:
                skipped.append(skipped_clip)
                tqdm.write(
                    f"Skipped {pair.short_name}: {skipped_clip['reason']}"
                )
                progress.set_postfix_str(
                    f"last={pair.short_name} "
                    f"time={_format_duration(time.perf_counter() - clip_started)}",
                    refresh=False,
                )
                continue
            if prepared_clip is None:
                raise RuntimeError(
                    f"Internal error: {pair.short_name} was neither prepared "
                    "nor skipped"
                )
            prepared.append(prepared_clip)
            tqdm.write(
                f"Prepared {pair.short_name}: {prepared_clip.frame_count} "
                "frames in "
                f"{_format_duration(time.perf_counter() - clip_started)}"
            )
            progress.set_postfix_str(
                f"last={pair.short_name} "
                f"time={_format_duration(time.perf_counter() - clip_started)} "
                f"total={_format_duration(time.perf_counter() - started)}",
                refresh=False,
            )
    return prepared, skipped, failed


def _pipeline_command(
    *,
    adapter_root: Path,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "human_to_robot.pipeline",
        "--folder",
        str(adapter_root),
        "--output",
        str(output),
        "--arm-prompt",
        args.arm_prompt,
        "--sam-threshold",
        str(args.sam_threshold),
        (
            "--hardware-encoding"
            if args.hardware_encoding
            else "--no-hardware-encoding"
        ),
    ]
    if args.isaacsim_root is not None:
        command.extend(["--isaacsim-root", str(args.isaacsim_root)])
    if args.debug:
        command.append("--debug")
    if not args.reuse:
        command.append("--no-reuse")
    for argument in args.pipeline_arg:
        command.append(argument)
    return command


def _report_prepared_clip(clip: PreparedClip) -> dict[str, Any]:
    return {
        "clip": clip.pair.stem,
        "source_video": str(clip.pair.video),
        "narration_row_index": clip.pair.annotation.row_index,
        "frame_count": clip.frame_count,
        "source_full_15fps_start_frame": clip.source_start_frame,
        "body_time_seconds": [
            clip.body_time_start,
            clip.body_time_end,
        ],
        "visible_video_boundary_offset_from_annotation_seconds": {
            "start": clip.annotation_start_offset_seconds,
            "end": clip.annotation_end_offset_seconds,
        },
        "left_visible_fraction": clip.left_visible_fraction,
        "right_visible_fraction": clip.right_visible_fraction,
        "maximum_slam_pose_time_error_us": (
            clip.maximum_slam_time_error_us
        ),
        "fixed_base_motion": {
            "maximum_camera_translation_m": (
                clip.maximum_camera_translation_m
            ),
            "maximum_camera_rotation_degrees": (
                clip.maximum_camera_rotation_degrees
            ),
            "maximum_hand_displacement_m": (
                clip.maximum_hand_displacement_m
            ),
        },
        "hand_action": (
            "rigid XSens hand/palm pose; robot fingers fixed open because "
            "source body tracks have no articulated finger pose"
        ),
    }


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    if not 0.0 < args.hand_eye_sample_fps <= 240.0:
        raise ValueError("--hand-eye-sample-fps must be in (0, 240]")
    for name in (
        "maximum_camera_translation_m",
        "maximum_camera_rotation_degrees",
        "maximum_hand_displacement_m",
    ):
        if getattr(args, name) <= 0.0:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must be positive")
    episode_root = Path(args.episode_root).expanduser().resolve()
    if not episode_root.is_dir():
        raise NotADirectoryError(f"Nymeria episode not found: {episode_root}")
    video_directory = (
        Path(args.video_dir).expanduser().resolve()
        if args.video_dir is not None
        else episode_root / ACTIVITY_VIDEO_RELATIVE_PATH
    )
    narration = load_activity_narration(
        episode_root / ACTIVITY_NARRATION_RELATIVE_PATH
    )
    pairs = discover_clip_pairs(
        video_directory,
        narration,
        selected=args.clip,
    )
    if args.max_clips is not None:
        if args.max_clips <= 0:
            raise ValueError("--max-clips must be positive")
        pairs = pairs[: args.max_clips]
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    intrinsics_path = (
        Path(args.intrinsics).expanduser().resolve()
        if args.intrinsics is not None
        else episode_root / INTRINSICS_RELATIVE_PATH
    )
    camera = load_recording_camera(
        episode_root / CAMERA_CALIBRATION_RELATIVE_PATH,
        intrinsics_path,
    )
    body = load_body_motion(episode_root / BODY_RELATIVE_PATH)
    slam = load_slam_trajectory(
        episode_root / CAMERA_TRAJECTORY_RELATIVE_PATH
    )
    tqdm.write(
        f"Discovered {len(pairs)} Nymeria clip pair(s); "
        f"episode={episode_root.name}, resolution="
        f"{camera.resolution[0]}x{camera.resolution[1]}"
    )
    hand_eye = estimate_hand_eye(
        body=body,
        slam=slam,
        timecode_offset_seconds=narration.timecode_offset_seconds,
        sample_fps=args.hand_eye_sample_fps,
    )
    tqdm.write(
        "Hand-eye calibrated: median residual "
        f"{hand_eye.median_rotation_error_degrees:.3f} deg / "
        f"{1000.0 * hand_eye.median_translation_error_m:.2f} mm"
    )

    kept_adapter = (
        Path(args.adapter_output).expanduser().resolve()
        if args.adapter_output is not None
        else None
    )
    if kept_adapter is not None and (
        kept_adapter == output
        or kept_adapter.is_relative_to(output)
        or output.is_relative_to(kept_adapter)
    ):
        raise ValueError(
            "--adapter-output and --output must be disjoint directories"
        )
    temporary_context: tempfile.TemporaryDirectory[str] | None = None
    if kept_adapter is None:
        temporary_context = tempfile.TemporaryDirectory(
            prefix=".nymeria-visual-alignment-",
            dir=output.parent,
        )
        adapter_base = Path(temporary_context.name)
    else:
        adapter_base = kept_adapter
        adapter_base.mkdir(parents=True, exist_ok=True)
    adapter_root = adapter_base / episode_root.name
    if kept_adapter is not None and adapter_root.exists() and any(
        adapter_root.iterdir()
    ):
        raise FileExistsError(
            f"--adapter-output episode directory must be empty: {adapter_root}"
        )

    published_root = output / episode_root.name
    report_path = published_root / "visual_alignment_report.json"
    keep_report = args.debug or args.prepare_only
    if not keep_report:
        removed: list[Path] = []
        if report_path.is_file() or report_path.is_symlink():
            report_path.unlink()
            removed.append(report_path)
        previous_adapter = published_root / "adapter"
        if previous_adapter.is_symlink() or previous_adapter.is_file():
            previous_adapter.unlink()
            removed.append(previous_adapter)
        elif previous_adapter.is_dir():
            shutil.rmtree(previous_adapter)
            removed.append(previous_adapter)
        if removed:
            tqdm.write(
                "Removed previous Nymeria debug/prepare outputs: "
                + ", ".join(str(path) for path in removed)
            )

    try:
        prepared, skipped, failed = prepare_clips(
            pairs=pairs,
            episode_root=episode_root,
            adapter_root=adapter_root,
            camera=camera,
            body=body,
            slam=slam,
            hand_eye=hand_eye,
            narration_path=narration.path,
            timecode_offset_seconds=narration.timecode_offset_seconds,
            hardware_encoding=args.hardware_encoding,
            require_visible_hands=args.require_visible_hands,
            motion_filter=args.motion_filter,
            maximum_camera_translation_m=(
                args.maximum_camera_translation_m
            ),
            maximum_camera_rotation_degrees=(
                args.maximum_camera_rotation_degrees
            ),
            maximum_hand_displacement_m=args.maximum_hand_displacement_m,
        )
        report: dict[str, Any] = {
            "schema_version": 3,
            "status": "prepared",
            "episode": episode_root.name,
            "video_directory": str(video_directory),
            "activity_narration": {
                "path": str(narration.path),
                "timecode_offset_seconds": (
                    narration.timecode_offset_seconds
                ),
            },
            "intrinsics": {
                "path": str(intrinsics_path),
                "resolution": list(camera.resolution),
                "K": camera.intrinsic.tolist(),
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
            "camera_parameter_policy": (
                "K describes the upright video. Raw T_Device_Camera comes "
                "from recording_head; T_Device_CameraUpright composes the "
                "Project Aria clockwise-90-degree video orientation."
            ),
            "fixed_base_motion_filter": {
                "enabled": args.motion_filter,
                "maximum_camera_translation_m": (
                    args.maximum_camera_translation_m
                ),
                "maximum_camera_rotation_degrees": (
                    args.maximum_camera_rotation_degrees
                ),
                "maximum_hand_displacement_m": (
                    args.maximum_hand_displacement_m
                ),
            },
            "hand_eye": hand_eye.as_dict(),
            "prepared": [_report_prepared_clip(clip) for clip in prepared],
            "skipped": skipped,
            "failed": failed,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if keep_report:
            _atomic_write_json(report_path, report)
        if not prepared:
            reason = (
                f"all {len(failed)} selected clip(s) failed"
                if failed
                else "all selected clips were rejected by eligibility filters"
            )
            tqdm.write(f"No clips prepared ({reason}); nothing to synthesize")
            return 1
        if args.prepare_only:
            if kept_adapter is None:
                destination = output / episode_root.name / "adapter"
                if destination.exists():
                    raise FileExistsError(
                        f"Prepare-only adapter already exists: {destination}"
                    )
                shutil.copytree(adapter_root, destination)
            report["status"] = "prepared_only"
            if failed:
                report["status"] = "partial_failure"
            report["elapsed_seconds"] = time.perf_counter() - started
            _atomic_write_json(report_path, report)
            tqdm.write(
                f"Prepared {len(prepared)} clip(s) in "
                f"{_format_duration(time.perf_counter() - started)}; "
                f"skipped={len(skipped)}, failed={len(failed)}"
            )
            return 1 if failed else 0

        command = _pipeline_command(
            adapter_root=adapter_base,
            output=output,
            args=args,
        )
        tqdm.write("Starting SAM3 / ProPainter / Pinocchio / Isaac pipeline")
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            report["status"] = "failed"
            report["pipeline_return_code"] = completed.returncode
            report["elapsed_seconds"] = time.perf_counter() - started
            if keep_report:
                _atomic_write_json(report_path, report)
            return int(completed.returncode)

        missing_outputs: list[str] = []
        for clip in prepared:
            published_video = published_root / f"{clip.pair.stem}.mp4"
            published_hdf5 = published_root / f"{clip.pair.stem}.hdf5"
            for path in (published_video, published_hdf5):
                if not path.is_file():
                    missing_outputs.append(str(path))
        if missing_outputs:
            report["status"] = "failed"
            report["missing_outputs"] = missing_outputs
            if keep_report:
                _atomic_write_json(report_path, report)
            tqdm.write(
                "Visual alignment pipeline returned success but final outputs "
                f"are missing: {missing_outputs}"
            )
            return 1

        report["status"] = "partial_failure" if failed else "completed"
        report["elapsed_seconds"] = time.perf_counter() - started
        if keep_report:
            _atomic_write_json(report_path, report)
        tqdm.write(
            f"Nymeria visual alignment completed: {len(prepared)} clip(s), "
            f"skipped={len(skipped)}, failed={len(failed)}, "
            f"elapsed={_format_duration(time.perf_counter() - started)}"
        )
        return 1 if failed else 0
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pair Nymeria activity narration/video clips, align XSens wrists "
            "to the Aria RGB camera, and run the existing human-to-G1 visual "
            "synthesis pipeline without activity SMPL files."
        )
    )
    parser.add_argument(
        "--episode-root",
        type=Path,
        required=True,
        help="Nymeria episode root",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help=(
            "Directory containing cNNN activity MP4 clips; default "
            "<episode>/undistorted_384_activity_summarization_clips"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--intrinsics",
        type=Path,
        help=(
            "Virtual pinhole JSON; default "
            "<episode>/undistorted/camera_intrinsics.json"
        ),
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        help="Only process this full stem or cNNN suffix; repeat as needed",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        help="Process only the first N selected/discovered clips",
    )
    parser.add_argument(
        "--isaacsim-root",
        type=Path,
        help="Isaac Sim standalone directory containing python.sh",
    )
    parser.add_argument(
        "--arm-prompt",
        default="first-person human arm and hand",
        help="SAM3 text prompt for the camera wearer's arm",
    )
    parser.add_argument("--sam-threshold", type=float, default=0.4)
    parser.add_argument(
        "--hand-eye-sample-fps",
        type=float,
        default=10.0,
        help="Recording-level hand-eye solve sample rate (default: %(default)s)",
    )
    parser.add_argument(
        "--require-visible-hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip clips unless each projected wearer wrist is visible in at "
            "least two frames (default: true)"
        ),
    )
    parser.add_argument(
        "--motion-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip global-motion clips that cannot be represented by the "
            "current fixed-base G1 retargeter (default: true)"
        ),
    )
    parser.add_argument(
        "--maximum-camera-translation-m",
        type=float,
        default=0.25,
        help="Fixed-base clip limit in meters (default: %(default)s)",
    )
    parser.add_argument(
        "--maximum-camera-rotation-degrees",
        type=float,
        default=45.0,
        help="Fixed-base clip rotation limit (default: %(default)s)",
    )
    parser.add_argument(
        "--maximum-hand-displacement-m",
        type=float,
        default=0.75,
        help="Fixed-base hand-target motion limit (default: %(default)s)",
    )
    parser.add_argument(
        "--hardware-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVENC when available (default: true)",
    )
    parser.add_argument(
        "--reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate and validate normalized MP4/HDF5 adapters only",
    )
    parser.add_argument(
        "--adapter-output",
        type=Path,
        help=(
            "Keep adapter inputs here; otherwise they are temporary and "
            "automatically removed"
        ),
    )
    parser.add_argument(
        "--pipeline-arg",
        action="append",
        default=[],
        help=(
            "Append one advanced argument token to human_to_robot.pipeline; "
            "repeat for multiple tokens"
        ),
    )
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
