#!/usr/bin/env python3
"""Source-driven Nymeria clip discovery and temporal adaptation.

This module deliberately contains no robot or CUDA imports.  It is the
testable boundary between Nymeria's 208-D motion clips and downstream
retargeting:

* video geometry, cadence, and *displayed* frame timestamps come from
  ``ffprobe``;
* motion cadence comes from the same-stem SMPL metadata;
* same-stem MP4/208-D/text/SMPL assets are required;
* the motion-to-video mapping uses timestamps rather than fixed frame ratios.

In particular, the displayed frame list returned by ``ffprobe -show_frames``
honours MP4 edit lists.  Encoded GOP preroll packets therefore never become
phantom output samples.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from repaint.vda_pipeline_common import natural_key


N_JOINTS = 22
MOTION_DIM = 208


@dataclass(frozen=True)
class DisplayVideoInfo:
    width: int
    height: int
    fps: float
    display_timestamps: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(len(self.display_timestamps))


@dataclass(frozen=True)
class NymeriaClip:
    session: str
    stem: str
    video_path: Path
    motion_208_path: Path
    text_path: Path
    smpl_path: Path
    session_metadata_path: Path
    camera_intrinsics_path: Path


@dataclass(frozen=True)
class MotionMetadata:
    fps: float
    action_text: str
    csv_start_time: float
    csv_original_start_time: float
    actual_start_time: float
    sample_count: int


@dataclass(frozen=True)
class ResampledMotion:
    timestamps: np.ndarray
    root_position_y_up: np.ndarray
    root_rotation: Rotation
    relative_joint_positions: np.ndarray
    relative_joint_rotations: tuple[Rotation, ...]
    source_sample_positions: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(len(self.timestamps))


def _parse_positive_rate(value: str, *, label: str) -> float:
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return rate


def _optional_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    result = int(str(value))
    return result if result >= 0 else None


def probe_display_video(path: Path) -> DisplayVideoInfo:
    """Probe the canonical displayed timeline, not encoded packet count."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-show_frames",
        "-show_entries",
        (
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,"
            "nb_read_frames:frame=best_effort_timestamp_time"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required but was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffprobe failed for {path}:\n{exc.stderr.strip()}"
        ) from exc

    payload = json.loads(result.stdout)
    streams = [
        stream
        for stream in payload.get("streams", [])
        if stream.get("codec_type") == "video"
    ]
    if len(streams) != 1:
        raise ValueError(
            f"{path} must expose exactly one selected video stream, "
            f"found {len(streams)}"
        )
    stream = streams[0]
    fps: float | None = None
    for candidate in (
        stream.get("avg_frame_rate"),
        stream.get("r_frame_rate"),
    ):
        if candidate in (None, "", "N/A"):
            continue
        try:
            candidate_fps = _parse_positive_rate(
                str(candidate),
                label="video frame rate",
            )
        except ValueError:
            continue
        fps = candidate_fps
        break
    if fps is None:
        raise ValueError(f"No valid video frame rate reported for {path}")

    raw_timestamps: list[float] = []
    for frame in payload.get("frames", []):
        timestamp_text = frame.get("best_effort_timestamp_time")
        if timestamp_text in (None, "", "N/A"):
            raise ValueError(f"Displayed frame in {path} has no timestamp")
        raw_timestamps.append(float(timestamp_text))
    if not raw_timestamps:
        raise ValueError(f"ffprobe decoded no displayed frames from {path}")

    timestamps = np.asarray(raw_timestamps, dtype=np.float64)
    if not np.isfinite(timestamps).all():
        raise ValueError(f"Non-finite displayed timestamp in {path}")
    timestamps -= timestamps[0]
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Displayed timestamps are not strictly increasing: {path}")

    decoded_count = _optional_int(stream.get("nb_read_frames"))
    if decoded_count is not None and decoded_count != len(timestamps):
        raise ValueError(
            f"ffprobe frame accounting disagrees for {path}: "
            f"nb_read_frames={decoded_count}, displayed={len(timestamps)}"
        )
    return DisplayVideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        display_timestamps=timestamps,
    )


def discover_nymeria_clips(src_root: Path) -> list[NymeriaClip]:
    """Discover complete same-stem clip bundles under each Nymeria session."""

    src_root = src_root.resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Nymeria source root does not exist: {src_root}")

    clips: list[NymeriaClip] = []
    incomplete: list[str] = []
    session_dirs = sorted(
        (path for path in src_root.iterdir() if path.is_dir()),
        key=lambda path: natural_key(path.name),
    )
    for session_dir in session_dirs:
        video_dirs = sorted(
            (
                path
                for path in session_dir.glob(
                    "undistorted_*_activity_summarization_clips"
                )
                if path.is_dir()
            ),
            key=lambda path: natural_key(path.name),
        )
        if not video_dirs:
            continue
        if len(video_dirs) != 1:
            raise ValueError(
                f"{session_dir} contains multiple candidate clip directories: "
                + ", ".join(path.name for path in video_dirs)
            )
        video_dir = video_dirs[0]
        session_metadata = session_dir / "metadata.json"
        intrinsics = session_dir / "undistorted/camera_intrinsics.json"
        for video_path in sorted(
            video_dir.glob("*.mp4"),
            key=lambda path: natural_key(path.name),
        ):
            stem = video_path.stem
            bundle = NymeriaClip(
                session=session_dir.name,
                stem=stem,
                video_path=video_path.resolve(),
                motion_208_path=(
                    session_dir / "act_sum_208_clips" / f"{stem}.npy"
                ).resolve(),
                text_path=(
                    session_dir / "act_sum_text" / f"{stem}.txt"
                ).resolve(),
                smpl_path=(
                    session_dir / "act_sum_smpl" / f"{stem}.npz"
                ).resolve(),
                session_metadata_path=session_metadata.resolve(),
                camera_intrinsics_path=intrinsics.resolve(),
            )
            missing = [
                path
                for path in (
                    bundle.motion_208_path,
                    bundle.text_path,
                    bundle.smpl_path,
                    bundle.session_metadata_path,
                    bundle.camera_intrinsics_path,
                )
                if not path.is_file()
            ]
            if missing:
                incomplete.append(
                    f"{bundle.stem}: " + ", ".join(str(path) for path in missing)
                )
            else:
                clips.append(bundle)
    if incomplete:
        raise FileNotFoundError(
            "Every Nymeria MP4 must have same-stem 208-D motion, text, SMPL "
            "metadata, plus session metadata/intrinsics. Missing:\n  - "
            + "\n  - ".join(incomplete)
        )
    if not clips:
        raise FileNotFoundError(
            f"No complete Nymeria clip bundles found under {src_root}"
        )
    return clips


def load_motion_metadata(clip: NymeriaClip) -> MotionMetadata:
    with np.load(clip.smpl_path, allow_pickle=True) as smpl:
        required = (
            "mocap_frame_rate",
            "csv_start_time",
            "csv_original_start_time",
            "actual_start_time",
            "actual_end_time",
        )
        missing = [key for key in required if key not in smpl]
        if missing:
            raise KeyError(
                f"{clip.smpl_path} is missing metadata: {', '.join(missing)}"
            )
        fps = float(np.asarray(smpl["mocap_frame_rate"]).item())
        csv_start = float(np.asarray(smpl["csv_start_time"]).item())
        csv_original_start = float(
            np.asarray(smpl["csv_original_start_time"]).item()
        )
        start = float(np.asarray(smpl["actual_start_time"]).item())
        end = float(np.asarray(smpl["actual_end_time"]).item())
        smpl_count = int(len(smpl["trans"])) if "trans" in smpl else -1
        embedded_text = (
            str(np.asarray(smpl["action_text"]).item()).strip()
            if "action_text" in smpl
            else ""
        )
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid motion FPS {fps} in {clip.smpl_path}")
    if (
        not np.isfinite(
            [csv_start, csv_original_start, start, end]
        ).all()
        or end < start
    ):
        raise ValueError(f"Invalid motion time bounds in {clip.smpl_path}")
    motion = np.load(clip.motion_208_path, mmap_mode="r")
    if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
        raise ValueError(
            f"{clip.motion_208_path} must have shape [N,{MOTION_DIM}], "
            f"got {motion.shape}"
        )
    sample_count = int(motion.shape[0])
    if smpl_count >= 0 and smpl_count != sample_count:
        raise ValueError(
            f"Same-stem motion length mismatch for {clip.stem}: "
            f"208-D={sample_count}, SMPL={smpl_count}"
        )
    expected_end = start + (sample_count - 1) / fps
    if not np.isclose(end, expected_end, atol=0.5 / fps + 1e-6):
        raise ValueError(
            f"SMPL time bounds do not match source motion cadence for "
            f"{clip.stem}: metadata_end={end:.9f}, "
            f"sample_end={expected_end:.9f}"
        )
    text = clip.text_path.read_text(encoding="utf-8").strip()
    if embedded_text and text and embedded_text != text:
        raise ValueError(
            f"Action text differs between {clip.text_path} and {clip.smpl_path}"
        )
    action_text = text or embedded_text
    if not action_text:
        raise ValueError(f"Empty action text for {clip.stem}")
    return MotionMetadata(
        fps=fps,
        action_text=action_text,
        csv_start_time=csv_start,
        csv_original_start_time=csv_original_start,
        actual_start_time=start,
        sample_count=sample_count,
    )


def load_participant_height_m(clip: NymeriaClip) -> float:
    payload = json.loads(clip.session_metadata_path.read_text(encoding="utf-8"))
    height_cm = float(payload["participant_height_cm"])
    height_m = height_cm / 100.0
    if not np.isfinite(height_m) or not 1.0 <= height_m <= 2.5:
        raise ValueError(
            f"Implausible participant height in {clip.session_metadata_path}: "
            f"{height_cm} cm"
        )
    return height_m


def load_scaled_intrinsic(
    clip: NymeriaClip,
    *,
    video_width: int,
    video_height: int,
) -> np.ndarray:
    """Load the calibrated pinhole K and scale it to the probed clip size."""

    payload = json.loads(
        clip.camera_intrinsics_path.read_text(encoding="utf-8")
    )
    undistorted = payload["undistorted"]
    intrinsic = np.asarray(undistorted["K"], dtype=np.float64)
    calibration_width, calibration_height = map(
        int,
        undistorted["resolution"],
    )
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(
            f"Invalid K matrix in {clip.camera_intrinsics_path}: "
            f"{intrinsic.shape}"
        )
    if min(calibration_width, calibration_height, video_width, video_height) <= 0:
        raise ValueError("Camera/video dimensions must be positive")
    scale_x = video_width / calibration_width
    scale_y = video_height / calibration_height
    scaled = intrinsic.copy()
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] = scale_x * (intrinsic[0, 2] + 0.5) - 0.5
    scaled[1, 2] = scale_y * (intrinsic[1, 2] + 0.5) - 0.5
    scaled[0, 1] *= scale_x
    return scaled


def sixd_to_rotation(values: np.ndarray) -> Rotation:
    """Convert Nymeria's row-vector 6-D rotation representation."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != 6:
        raise ValueError(f"6-D rotations must end in dimension 6: {values.shape}")
    first = values[..., :3]
    second = values[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    first_unit = first / np.clip(first_norm, 1e-8, None)
    second_orthogonal = (
        second
        - np.sum(first_unit * second, axis=-1, keepdims=True) * first_unit
    )
    second_norm = np.linalg.norm(
        second_orthogonal,
        axis=-1,
        keepdims=True,
    )
    second_unit = second_orthogonal / np.clip(second_norm, 1e-8, None)
    third_unit = np.cross(first_unit, second_unit, axis=-1)
    matrices = np.stack(
        [first_unit, second_unit, third_unit],
        axis=-2,
    )
    if not np.isfinite(matrices).all():
        raise ValueError("Non-finite rotation reconstructed from 6-D data")
    return Rotation.from_matrix(matrices.reshape(-1, 3, 3))


def integrate_root_displacements(root_displacements: np.ndarray) -> np.ndarray:
    """Integrate per-sample root displacement without the old one-frame lag.

    Nymeria's first root delta is the clip reset value (normally zero), while
    sample ``i`` stores the displacement that reaches sample ``i``.  The old
    implementation accumulated ``[:-1]`` and therefore applied the reset value
    as frame 1, delaying every displacement by one motion frame.
    """

    displacements = np.asarray(root_displacements, dtype=np.float64)
    if displacements.ndim != 2 or displacements.shape[1] != 3:
        raise ValueError(
            f"Root displacements must have shape [N,3], got {displacements.shape}"
        )
    if not len(displacements):
        raise ValueError("Root displacement sequence is empty")
    if not np.isfinite(displacements).all():
        raise ValueError("Root displacement sequence contains non-finite values")
    positions = np.zeros_like(displacements, dtype=np.float64)
    if len(displacements) > 1:
        positions[1:] = np.cumsum(displacements[1:], axis=0)
    return positions


def motion_sample_positions(
    display_timestamps: np.ndarray,
    *,
    motion_fps: float,
    motion_sample_count: int,
    maximum_boundary_clamp_seconds: float = 0.0,
) -> np.ndarray:
    """Map video-aligned timestamps to fractional source motion indices.

    By default the target timeline must be fully covered by the motion.  A
    caller with an independently calibrated clip boundary may explicitly
    permit a small boundary clamp and must record that clamp in its provenance.
    """

    timestamps = np.asarray(display_timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps):
        raise ValueError("Display timestamps must be a non-empty vector")
    if not np.isfinite(timestamps).all():
        raise ValueError("Display timestamps must be finite")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("Display timestamps must be strictly increasing")
    if not np.isfinite(motion_fps) or motion_fps <= 0:
        raise ValueError("Motion FPS must be positive")
    if motion_sample_count < 1:
        raise ValueError("Motion sample count must be positive")
    if (
        not np.isfinite(maximum_boundary_clamp_seconds)
        or maximum_boundary_clamp_seconds < 0.0
    ):
        raise ValueError("maximum_boundary_clamp_seconds must be non-negative")

    positions = timestamps * float(motion_fps)
    maximum = float(motion_sample_count - 1)
    upper_tolerance = (
        0.5
        +
        float(maximum_boundary_clamp_seconds) * float(motion_fps)
        + 1e-6
    )
    lower_tolerance = (
        1e-9
        if maximum_boundary_clamp_seconds == 0.0
        else float(maximum_boundary_clamp_seconds) * float(motion_fps) + 1e-6
    )
    if (
        positions[0] < -lower_tolerance
        or positions[-1] > maximum + upper_tolerance
    ):
        raise ValueError(
            "Displayed video extends beyond the allowed same-stem motion "
            "support: "
            f"first motion index={positions[0]:.6f}, "
            f"last motion index={positions[-1]:.6f}, maximum={maximum:.6f}, "
            f"allowed clamp={maximum_boundary_clamp_seconds:.6f}s"
        )
    return np.clip(positions, 0.0, maximum)


def _linear_resample(
    values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    result = np.empty((len(target_times), flat.shape[1]), dtype=np.float64)
    for index in range(flat.shape[1]):
        result[:, index] = np.interp(
            target_times,
            source_times,
            flat[:, index],
        )
    return result.reshape((len(target_times),) + values.shape[1:])


def resample_motion_to_video(
    motion: np.ndarray,
    *,
    motion_fps: float,
    display_timestamps: np.ndarray,
    motion_timestamps: np.ndarray | None = None,
    maximum_boundary_clamp_seconds: float = 0.0,
) -> ResampledMotion:
    """Timestamp-resample the 208-D body motion onto displayed video frames."""

    motion = np.asarray(motion, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
        raise ValueError(
            f"Motion must have shape [N,{MOTION_DIM}], got {motion.shape}"
        )
    if not np.isfinite(motion).all():
        raise ValueError("Motion contains non-finite values")
    video_timestamps = np.asarray(display_timestamps, dtype=np.float64)
    target_motion_timestamps = (
        video_timestamps
        if motion_timestamps is None
        else np.asarray(motion_timestamps, dtype=np.float64)
    )
    if target_motion_timestamps.shape != video_timestamps.shape:
        raise ValueError(
            "motion_timestamps must have the same shape as "
            "display_timestamps"
        )
    positions = motion_sample_positions(
        target_motion_timestamps,
        motion_fps=motion_fps,
        motion_sample_count=len(motion),
        maximum_boundary_clamp_seconds=maximum_boundary_clamp_seconds,
    )
    source_times = np.arange(len(motion), dtype=np.float64) / motion_fps
    target_times = positions / motion_fps

    root_positions = integrate_root_displacements(motion[:, :3])
    root_positions[:, 1] += float(motion[0, 9])
    root_rotations = sixd_to_rotation(motion[:, 3:9])
    root_slerp = Slerp(source_times, root_rotations)

    relative_positions = motion[
        :,
        10 : 10 + 3 * N_JOINTS,
    ].reshape(len(motion), N_JOINTS, 3)
    relative_rotation_6d = motion[
        :,
        10 + 3 * N_JOINTS :,
    ].reshape(len(motion), N_JOINTS, 6)
    relative_rotations: list[Rotation] = []
    for joint_index in range(N_JOINTS):
        rotations = sixd_to_rotation(
            relative_rotation_6d[:, joint_index]
        )
        relative_rotations.append(Slerp(source_times, rotations)(target_times))

    return ResampledMotion(
        timestamps=video_timestamps.copy(),
        root_position_y_up=_linear_resample(
            root_positions,
            source_times,
            target_times,
        ),
        root_rotation=root_slerp(target_times),
        relative_joint_positions=_linear_resample(
            relative_positions,
            source_times,
            target_times,
        ),
        relative_joint_rotations=tuple(relative_rotations),
        source_sample_positions=positions,
    )
