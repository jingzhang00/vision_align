"""Batch-rectify Nymeria RGB fisheye videos and save matching intrinsics.

Nymeria's Project Aria RGB calibration uses the 15-parameter
FisheyeRadTanThinPrism (Fisheye624) model.  OpenCV's four-parameter fisheye
model is not equivalent, so the destination-to-source map is implemented
directly from the Project Aria projection definition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import av
import cv2
import numpy as np
from tqdm import tqdm

from human_to_robot.video_io import (
    VideoEncodingProfile,
    atomic_video_writer,
    video_metadata,
)


PARAMETER_NAMES = (
    "f",
    "cx",
    "cy",
    "k0",
    "k1",
    "k2",
    "k3",
    "k4",
    "k5",
    "p0",
    "p1",
    "s0",
    "s1",
    "s2",
    "s3",
)
EXPECTED_MODEL = "FisheyeRadTanThinPrism"
DEFAULT_CALIBRATION_RELATIVE_PATH = Path(
    "recording_head/mps/slam/online_calibration.jsonl"
)


@dataclass(frozen=True)
class CameraCalibration:
    label: str
    model: str
    description: str
    parameters: tuple[float, ...]
    serial_number: str
    transform_device_camera: dict[str, Any]
    calibrated: bool
    utc_timestamp_ns: int | None
    tracking_timestamp_us: int | None


@dataclass(frozen=True)
class PinholeCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> list[list[float]]:
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]


@dataclass(frozen=True)
class Remap:
    map_x: np.ndarray
    map_y: np.ndarray
    valid_mask: np.ndarray
    source_parameters: tuple[float, ...]
    source_scale: float


@dataclass(frozen=True)
class Episode:
    name: str
    directory: Path
    relative_directory: Path
    source_video: Path
    calibration_file: Path
    output_directory: Path


@dataclass(frozen=True)
class EpisodeResult:
    episode: Episode
    status: str
    elapsed_seconds: float
    frame_count: int = 0
    message: str = ""


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def load_rgb_calibration(
    path: str | Path,
    *,
    camera_label: str = "camera-rgb",
) -> CameraCalibration:
    """Load the requested camera calibration from the first JSONL record."""

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            first_line = next((line for line in handle if line.strip()), None)
    except FileNotFoundError:
        raise FileNotFoundError(f"Calibration file not found: {path}") from None
    if first_line is None:
        raise ValueError(f"Calibration file is empty: {path}")

    record = json.loads(first_line)
    cameras = record.get("CameraCalibrations")
    if not isinstance(cameras, list):
        raise ValueError(f"CameraCalibrations is missing from {path}")
    selected = next(
        (camera for camera in cameras if camera.get("Label") == camera_label),
        None,
    )
    if selected is None:
        labels = sorted(
            str(camera.get("Label"))
            for camera in cameras
            if camera.get("Label") is not None
        )
        raise ValueError(
            f"Camera {camera_label!r} not found in {path}; available: {labels}"
        )

    projection = selected.get("Projection", {})
    model = str(projection.get("Name", ""))
    parameters = tuple(float(value) for value in projection.get("Params", ()))
    if model != EXPECTED_MODEL:
        raise ValueError(
            f"Unsupported camera model {model!r} in {path}; "
            f"expected {EXPECTED_MODEL!r}"
        )
    if len(parameters) != len(PARAMETER_NAMES):
        raise ValueError(
            f"{model} in {path} has {len(parameters)} parameters; "
            f"expected {len(PARAMETER_NAMES)}"
        )
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError(f"{model} contains non-finite parameters in {path}")
    if parameters[0] <= 0:
        raise ValueError(f"{model} focal length must be positive in {path}")
    calibrated = bool(selected.get("Calibrated", False))
    if not calibrated:
        raise ValueError(f"Camera {camera_label!r} is not calibrated in {path}")

    return CameraCalibration(
        label=camera_label,
        model=model,
        description=str(projection.get("Description", "")),
        parameters=parameters,
        serial_number=str(selected.get("SerialNumber", "")),
        transform_device_camera=dict(selected.get("T_Device_Camera", {})),
        calibrated=calibrated,
        utc_timestamp_ns=_optional_int(record.get("utc_timestamp_ns")),
        tracking_timestamp_us=_optional_int(record.get("tracking_timestamp_us")),
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def scale_fisheye_parameters(
    parameters: Sequence[float],
    scale: float,
) -> tuple[float, ...]:
    """Scale Fisheye624 intrinsics using Project Aria's pixel convention."""

    if len(parameters) != len(PARAMETER_NAMES):
        raise ValueError(
            f"Expected {len(PARAMETER_NAMES)} parameters, got {len(parameters)}"
        )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Scale must be positive and finite, got {scale}")
    scaled = [float(value) for value in parameters]
    scaled[0] *= scale
    scaled[1] = scale * (scaled[1] + 0.5) - 0.5
    scaled[2] = scale * (scaled[2] + 0.5) - 0.5
    return tuple(scaled)


def build_remap(
    calibration: CameraCalibration,
    *,
    source_size: tuple[int, int],
    calibration_size: tuple[int, int],
    output: PinholeCalibration,
) -> Remap:
    """Build a pinhole-output to fisheye-source lookup map."""

    source_width, source_height = source_size
    calibration_width, calibration_height = calibration_size
    if min(
        source_width,
        source_height,
        calibration_width,
        calibration_height,
        output.width,
        output.height,
    ) <= 0:
        raise ValueError("Image dimensions must be positive")

    scale_x = source_width / calibration_width
    scale_y = source_height / calibration_height
    if not math.isclose(scale_x, scale_y, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(
            "Fisheye624 calibration rescaling must be isotropic, but "
            f"{calibration_width}x{calibration_height} -> "
            f"{source_width}x{source_height} gives "
            f"scale_x={scale_x:.9g}, scale_y={scale_y:.9g}"
        )

    params = scale_fisheye_parameters(calibration.parameters, scale_x)
    f, source_cx, source_cy = params[:3]
    radial_coefficients = np.asarray(params[3:9], dtype=np.float64)
    p0, p1 = params[9:11]
    s0, s1, s2, s3 = params[11:15]

    pixel_y, pixel_x = np.indices(
        (output.height, output.width),
        dtype=np.float64,
    )
    a = (pixel_x - output.cx) / output.fx
    b = (pixel_y - output.cy) / output.fy
    radius = np.hypot(a, b)
    theta = np.arctan(radius)
    theta_squared = theta * theta

    radial = np.ones_like(theta)
    theta_power = theta_squared.copy()
    for coefficient in radial_coefficients:
        radial += coefficient * theta_power
        theta_power *= theta_squared

    theta_over_radius = np.divide(
        theta,
        radius,
        out=np.ones_like(theta),
        where=radius > np.finfo(np.float64).eps,
    )
    radial_scale = radial * theta_over_radius
    x_radial = radial_scale * a
    y_radial = radial_scale * b
    distorted_radius_squared = x_radial**2 + y_radial**2
    distorted_radius_fourth = distorted_radius_squared**2

    tangential_common = 2.0 * (x_radial * p0 + y_radial * p1)
    x_distorted = (
        x_radial
        + tangential_common * x_radial
        + distorted_radius_squared * p0
        + s0 * distorted_radius_squared
        + s1 * distorted_radius_fourth
    )
    y_distorted = (
        y_radial
        + tangential_common * y_radial
        + distorted_radius_squared * p1
        + s2 * distorted_radius_squared
        + s3 * distorted_radius_fourth
    )

    map_x = f * x_distorted + source_cx
    map_y = f * y_distorted + source_cy
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= -0.5)
        & (map_x <= source_width - 0.5)
        & (map_y >= -0.5)
        & (map_y <= source_height - 0.5)
    )
    map_x = np.where(valid, map_x, -1.0).astype(np.float32)
    map_y = np.where(valid, map_y, -1.0).astype(np.float32)
    return Remap(
        map_x=map_x,
        map_y=map_y,
        valid_mask=valid,
        source_parameters=params,
        source_scale=scale_x,
    )


def discover_episodes(
    dataset_root: str | Path,
    *,
    output_root: str | Path | None = None,
    calibration_relative_path: Path = DEFAULT_CALIBRATION_RELATIVE_PATH,
) -> list[Episode]:
    """Find recording roots containing both video.mp4 and RGB calibration."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root not found: {dataset_root}")
    external_output_root = (
        None
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )

    candidate_videos: Iterable[Path]
    if (dataset_root / "video.mp4").is_file():
        candidate_videos = (dataset_root / "video.mp4",)
    else:
        candidate_videos = dataset_root.rglob("video.mp4")

    episodes: list[Episode] = []
    for source_video in sorted(candidate_videos):
        directory = source_video.parent
        calibration_file = directory / calibration_relative_path
        if not calibration_file.is_file():
            continue
        if directory == dataset_root:
            relative_directory = Path(directory.name)
        else:
            relative_directory = directory.relative_to(dataset_root)
        if external_output_root is None:
            output_directory = directory / "undistorted"
        else:
            output_directory = (
                external_output_root / relative_directory / "undistorted"
            )
        episodes.append(
            Episode(
                name=relative_directory.as_posix(),
                directory=directory,
                relative_directory=relative_directory,
                source_video=source_video,
                calibration_file=calibration_file,
                output_directory=output_directory,
            )
        )
    return episodes


def _valid_roi(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    if not len(columns):
        return [0, 0, 0, 0]
    left = int(columns.min())
    top = int(rows.min())
    return [
        left,
        top,
        int(columns.max()) - left + 1,
        int(rows.max()) - top + 1,
    ]


def _interpolation_flag(name: str) -> int:
    return {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }[name]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative_path(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start=start)).as_posix()


def _metadata_payload(
    *,
    episode: Episode,
    calibration: CameraCalibration,
    calibration_size: tuple[int, int],
    source_size: tuple[int, int],
    source_frame_count: int,
    source_fps: float,
    output: PinholeCalibration,
    remap: Remap,
    encoding: VideoEncodingProfile,
    processed_frames: int,
    max_frames: int | None,
    interpolation: str,
    elapsed_seconds: float,
    output_video: Path,
    output_json: Path,
) -> dict[str, Any]:
    horizontal_fov = math.degrees(
        2.0 * math.atan(output.width / (2.0 * output.fx))
    )
    vertical_fov = math.degrees(
        2.0 * math.atan(output.height / (2.0 * output.fy))
    )
    return {
        "schema_version": 1,
        "episode": episode.name,
        "source": {
            "video": _relative_path(episode.source_video, output_json.parent),
            "resolution": list(source_size),
            "fps": source_fps,
            "frame_count": source_frame_count,
            "calibration_file": _relative_path(
                episode.calibration_file,
                output_json.parent,
            ),
            "calibration_reference_resolution": list(calibration_size),
            "calibration_reference_resolution_source": (
                "CLI-configured Project Aria RGB sensor resolution; "
                "online_calibration.jsonl does not store image dimensions"
            ),
            "camera_label": calibration.label,
            "camera_serial_number": calibration.serial_number,
            "calibrated": calibration.calibrated,
            "selected_record_timestamp": {
                "utc_timestamp_ns": calibration.utc_timestamp_ns,
                "tracking_timestamp_us": calibration.tracking_timestamp_us,
            },
            "camera_model": calibration.model,
            "camera_model_description": calibration.description,
            "parameter_order": list(PARAMETER_NAMES),
            "projection_parameters_at_reference_resolution": list(
                calibration.parameters
            ),
            "projection_parameters_at_source_resolution": list(
                remap.source_parameters
            ),
            "reference_to_source_scale": remap.source_scale,
            "T_Device_Camera": calibration.transform_device_camera,
        },
        "undistorted": {
            "video": output_video.name,
            "camera_model": "pinhole",
            "project_aria_model_name": "Linear",
            "distortion_model": "none",
            "distortion_coefficients": [],
            "resolution": [output.width, output.height],
            "fps": float(encoding.frame_rate),
            "frame_rate": {
                "numerator": encoding.frame_rate.numerator,
                "denominator": encoding.frame_rate.denominator,
            },
            "frame_count": processed_frames,
            "K": output.matrix,
            "fx": output.fx,
            "fy": output.fy,
            "cx": output.cx,
            "cy": output.cy,
            "horizontal_fov_degrees": horizontal_fov,
            "vertical_fov_degrees": vertical_fov,
            "valid_pixel_fraction": float(remap.valid_mask.mean()),
            "valid_roi_xywh": _valid_roi(remap.valid_mask),
            "T_Device_Camera": calibration.transform_device_camera,
        },
        "processing": {
            "method": (
                "destination pinhole ray projected through Project Aria "
                "FisheyeRadTanThinPrism/Fisheye624"
            ),
            "pixel_scaling": "c_scaled = scale * (c + 0.5) - 0.5",
            "interpolation": interpolation,
            "border_mode": "constant_black",
            "source_format_preserved": {
                "frame_rate": True,
                "codec_family": True,
                "color_metadata": True,
            },
            "encoding": encoding.to_dict(),
            "max_frames": max_frames,
            "truncated": processed_frames < source_frame_count,
            "elapsed_seconds": elapsed_seconds,
        },
    }


def process_episode(
    episode: Episode,
    *,
    output: PinholeCalibration,
    calibration_size: tuple[int, int],
    camera_label: str,
    interpolation: str,
    hardware_encoding: bool,
    overwrite: bool,
    max_frames: int | None,
) -> EpisodeResult:
    started = time.perf_counter()
    output_video = episode.output_directory / "video.mp4"
    output_json = episode.output_directory / "camera_intrinsics.json"
    success_marker = episode.output_directory / "_SUCCESS"
    expected_outputs = (output_video, output_json, success_marker)
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        if len(existing) == len(expected_outputs):
            try:
                existing_metadata = json.loads(
                    output_json.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Existing metadata is invalid for {episode.name}: {error}"
                ) from error
            existing_truncated = bool(
                existing_metadata.get("processing", {}).get("truncated", True)
            )
            if max_frames is None and existing_truncated:
                raise FileExistsError(
                    f"Existing output for {episode.name} is truncated; "
                    "run again with --overwrite"
                )
            return EpisodeResult(
                episode=episode,
                status="skipped",
                elapsed_seconds=time.perf_counter() - started,
                message="outputs already exist",
            )
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Partial output exists for {episode.name} ({names}); "
            "use --overwrite after checking it"
        )

    calibration = load_rgb_calibration(
        episode.calibration_file,
        camera_label=camera_label,
    )
    source_width, source_height, source_frame_count, source_fps = (
        video_metadata(episode.source_video)
    )
    if source_frame_count <= 0:
        raise ValueError(f"Video contains no frames: {episode.source_video}")
    remap = build_remap(
        calibration,
        source_size=(source_width, source_height),
        calibration_size=calibration_size,
        output=output,
    )
    nearest_interpolation = interpolation == "nearest"
    fixed_map, fractional_map = cv2.convertMaps(
        remap.map_x,
        remap.map_y,
        cv2.CV_16SC2,
        nninterpolation=nearest_interpolation,
    )
    expected_frames = (
        source_frame_count
        if max_frames is None
        else min(source_frame_count, max_frames)
    )
    episode.output_directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{episode.output_directory.name}-",
        dir=episode.output_directory.parent,
    ) as temporary_directory:
        staging = Path(temporary_directory)
        staging_video = staging / output_video.name
        staging_json = staging / output_json.name
        staging_success = staging / success_marker.name
        processed_frames = 0
        encoding: VideoEncodingProfile | None = None
        with (
            av.open(str(episode.source_video)) as source_container,
            atomic_video_writer(
                staging_video,
                source_video=episode.source_video,
                output_size=(output.width, output.height),
                hardware_encoding=hardware_encoding,
            ) as writer,
            tqdm(
                total=expected_frames,
                desc=episode.name,
                unit="frame",
                dynamic_ncols=True,
                leave=False,
            ) as progress,
        ):
            encoding = writer.profile
            video_stream = source_container.streams.video[0]
            for frame in source_container.decode(video_stream):
                if max_frames is not None and processed_frames >= max_frames:
                    break
                source_frame = frame.to_ndarray(format="bgr24")
                rectified = cv2.remap(
                    source_frame,
                    fixed_map,
                    fractional_map,
                    interpolation=_interpolation_flag(interpolation),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                writer.write(rectified)
                processed_frames += 1
                progress.update()

        if processed_frames == 0 or encoding is None:
            raise ValueError(
                f"Video contains no decodable frames: {episode.source_video}"
            )
        if processed_frames != expected_frames:
            raise RuntimeError(
                f"Decoded {processed_frames} frames from {episode.source_video}, "
                f"expected {expected_frames}; output was not published"
            )
        elapsed = time.perf_counter() - started
        payload = _metadata_payload(
            episode=episode,
            calibration=calibration,
            calibration_size=calibration_size,
            source_size=(source_width, source_height),
            source_frame_count=source_frame_count,
            source_fps=source_fps,
            output=output,
            remap=remap,
            encoding=encoding,
            processed_frames=processed_frames,
            max_frames=max_frames,
            interpolation=interpolation,
            elapsed_seconds=elapsed,
            output_video=output_video,
            output_json=output_json,
        )
        _write_json(staging_json, payload)
        _write_json(
            staging_success,
            {
                "schema_version": 1,
                "artifacts": [output_video.name, output_json.name],
            },
        )
        episode.output_directory.mkdir(parents=True, exist_ok=True)
        success_marker.unlink(missing_ok=True)
        os.replace(staging_video, output_video)
        os.replace(staging_json, output_json)
        os.replace(staging_success, success_marker)

    return EpisodeResult(
        episode=episode,
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        frame_count=processed_frames,
    )


def run(args: argparse.Namespace) -> int:
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.max_frames is not None and args.output_root is None:
        raise ValueError(
            "--max-frames requires --output-root so a validation clip cannot "
            "be mistaken for the in-place full output"
        )
    focal_length = (
        260.0 * args.width / 384.0
        if args.focal_length is None
        else args.focal_length
    )
    output = PinholeCalibration(
        width=args.width,
        height=args.height,
        fx=focal_length,
        fy=focal_length,
        cx=(
            (args.width - 1) / 2.0
            if args.principal_x is None
            else args.principal_x
        ),
        cy=(
            (args.height - 1) / 2.0
            if args.principal_y is None
            else args.principal_y
        ),
    )
    if min(output.width, output.height) <= 0:
        raise ValueError("--width and --height must be positive")
    if not math.isfinite(output.fx) or output.fx <= 0:
        raise ValueError("--focal-length must be positive and finite")
    if not math.isfinite(output.cx) or not math.isfinite(output.cy):
        raise ValueError("Principal point must be finite")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    episodes = discover_episodes(
        dataset_root,
        output_root=args.output_root,
    )
    if args.episode:
        requested = set(args.episode)
        episodes = [
            episode
            for episode in episodes
            if episode.name in requested
            or episode.directory.name in requested
        ]
        found = {
            requested_name
            for requested_name in requested
            if any(
                episode.name == requested_name
                or episode.directory.name == requested_name
                for episode in episodes
            )
        }
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested episodes not found: {missing}")
    if not episodes:
        raise ValueError(
            f"No Nymeria episodes with video.mp4 and "
            f"{DEFAULT_CALIBRATION_RELATIVE_PATH} found under {dataset_root}"
        )

    batch_started = time.perf_counter()
    tqdm.write(
        f"Discovered {len(episodes)} Nymeria episode(s); "
        f"output={output.width}x{output.height}, "
        f"focal={output.fx:g}, fps=source"
    )
    results: list[EpisodeResult] = []
    with tqdm(
        episodes,
        desc="Nymeria fisheye",
        unit="episode",
        dynamic_ncols=True,
    ) as batch_progress:
        for episode in batch_progress:
            episode_started = time.perf_counter()
            try:
                result = process_episode(
                    episode,
                    output=output,
                    calibration_size=(
                        args.calibration_width,
                        args.calibration_height,
                    ),
                    camera_label=args.camera_label,
                    interpolation=args.interpolation,
                    hardware_encoding=args.hardware_encoding,
                    overwrite=args.overwrite,
                    max_frames=args.max_frames,
                )
            except (Exception, KeyboardInterrupt) as error:
                elapsed = time.perf_counter() - episode_started
                if isinstance(error, KeyboardInterrupt):
                    tqdm.write(
                        f"Interrupted {episode.name} after "
                        f"{_format_duration(elapsed)}"
                    )
                    raise
                result = EpisodeResult(
                    episode=episode,
                    status="failed",
                    elapsed_seconds=elapsed,
                    message=f"{type(error).__name__}: {error}",
                )
            results.append(result)
            if result.status == "completed":
                tqdm.write(
                    f"Completed {episode.name}: {result.frame_count} frames in "
                    f"{_format_duration(result.elapsed_seconds)}"
                )
            elif result.status == "skipped":
                tqdm.write(f"Skipped {episode.name}: {result.message}")
            else:
                tqdm.write(
                    f"Failed {episode.name} after "
                    f"{_format_duration(result.elapsed_seconds)}: "
                    f"{result.message}"
                )
            batch_progress.set_postfix_str(
                f"last={episode.name} time="
                f"{_format_duration(result.elapsed_seconds)}",
                refresh=False,
            )

    elapsed = time.perf_counter() - batch_started
    completed = sum(result.status == "completed" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)
    tqdm.write(
        f"Finished Nymeria fisheye batch: completed={completed}, "
        f"skipped={skipped}, failed={failed}, elapsed={_format_duration(elapsed)}"
    )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-undistort Nymeria Project Aria RGB videos and save the "
            "matching pinhole camera intrinsics."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/nymeria"),
        help="Nymeria dataset or single episode directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional external output root. By default each episode writes to "
            "<episode>/undistorted."
        ),
    )
    parser.add_argument(
        "--episode",
        action="append",
        help="Only process this episode name; may be supplied more than once",
    )
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument(
        "--focal-length",
        type=float,
        help=(
            "Output pinhole fx=fy in pixels "
            "(default: 260 scaled from width 384)"
        ),
    )
    parser.add_argument(
        "--principal-x",
        type=float,
        help="Output cx (default: (width - 1) / 2)",
    )
    parser.add_argument(
        "--principal-y",
        type=float,
        help="Output cy (default: (height - 1) / 2)",
    )
    parser.add_argument(
        "--calibration-width",
        type=int,
        default=2880,
        help="RGB calibration reference width (default: %(default)s)",
    )
    parser.add_argument(
        "--calibration-height",
        type=int,
        default=2880,
        help="RGB calibration reference height (default: %(default)s)",
    )
    parser.add_argument("--camera-label", default="camera-rgb")
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "linear", "cubic", "lanczos"),
        default="linear",
    )
    parser.add_argument(
        "--hardware-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVENC when available (default: enabled)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a complete or partial existing output",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process only the first N frames (for validation)",
    )
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
