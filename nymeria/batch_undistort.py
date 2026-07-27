"""Write the virtual pinhole intrinsics used by Nymeria RGB clips.

The Nymeria videos consumed by this project are already rectified.  This
module therefore does not decode, remap, or encode video; it only records the
matching pinhole camera matrix for downstream rendering.
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
from typing import Any, Iterable

from tqdm import tqdm


DEFAULT_CALIBRATION_RELATIVE_PATH = Path(
    "recording_head/mps/slam/online_calibration.jsonl"
)


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
class Episode:
    name: str
    directory: Path
    relative_directory: Path
    calibration_file: Path
    output_directory: Path


@dataclass(frozen=True)
class EpisodeResult:
    episode: Episode
    status: str
    elapsed_seconds: float
    message: str = ""


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def discover_episodes(
    dataset_root: str | Path,
    *,
    output_root: str | Path | None = None,
    calibration_relative_path: Path = DEFAULT_CALIBRATION_RELATIVE_PATH,
) -> list[Episode]:
    """Find Nymeria recording roots without recursing into generated media."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root not found: {dataset_root}")
    external_output_root = (
        None if output_root is None else Path(output_root).expanduser().resolve()
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
        relative_directory = (
            Path(directory.name)
            if directory == dataset_root
            else directory.relative_to(dataset_root)
        )
        output_directory = (
            directory / "undistorted"
            if external_output_root is None
            else external_output_root / relative_directory / "undistorted"
        )
        episodes.append(
            Episode(
                name=relative_directory.as_posix(),
                directory=directory,
                relative_directory=relative_directory,
                calibration_file=calibration_file,
                output_directory=output_directory,
            )
        )
    return episodes


def build_pinhole_calibration(args: argparse.Namespace) -> PinholeCalibration:
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
    return output


def _payload(output: PinholeCalibration) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "camera_model": "pinhole",
        "distortion_model": "none",
        "distortion_coefficients": [],
        "resolution": [output.width, output.height],
        "K": output.matrix,
        "fx": output.fx,
        "fy": output.fy,
        "cx": output.cx,
        "cy": output.cy,
    }


def _read_existing_pinhole(path: Path) -> tuple[list[int], list[list[float]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid existing intrinsics JSON {path}: {error}") from error
    # Accept the former rich schema so rerunning the metadata-only command is
    # idempotent on datasets processed by the original rectifier.
    pinhole = payload.get("undistorted", payload)
    try:
        resolution = [int(value) for value in pinhole["resolution"]]
        matrix = [
            [float(value) for value in row]
            for row in pinhole["K"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Existing intrinsics are incomplete: {path}") from error
    return resolution, matrix


def _same_intrinsics(
    existing: tuple[list[int], list[list[float]]],
    requested: PinholeCalibration,
) -> bool:
    resolution, matrix = existing
    if resolution != [requested.width, requested.height]:
        return False
    expected = requested.matrix
    return len(matrix) == 3 and all(
        len(row) == 3
        and all(
            math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-9)
            for actual, target in zip(row, expected_row)
        )
        for row, expected_row in zip(matrix, expected)
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


def process_episode(
    episode: Episode,
    *,
    output: PinholeCalibration,
    overwrite: bool,
) -> EpisodeResult:
    started = time.perf_counter()
    output_json = episode.output_directory / "camera_intrinsics.json"
    if output_json.exists() and not overwrite:
        if _same_intrinsics(_read_existing_pinhole(output_json), output):
            return EpisodeResult(
                episode=episode,
                status="skipped",
                elapsed_seconds=time.perf_counter() - started,
                message="matching intrinsics already exist",
            )
        raise FileExistsError(
            f"Different intrinsics already exist for {episode.name}: "
            f"{output_json}; use --overwrite to replace only this JSON"
        )
    _atomic_write_json(output_json, _payload(output))
    return EpisodeResult(
        episode=episode,
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
    )


def run(args: argparse.Namespace) -> int:
    output = build_pinhole_calibration(args)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    episodes = discover_episodes(dataset_root, output_root=args.output_root)
    if args.episode:
        requested = set(args.episode)
        episodes = [
            episode
            for episode in episodes
            if episode.name in requested or episode.directory.name in requested
        ]
        found = {
            name
            for name in requested
            if any(
                episode.name == name or episode.directory.name == name
                for episode in episodes
            )
        }
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested episodes not found: {missing}")
    if not episodes:
        raise ValueError(
            "No Nymeria episodes containing root video.mp4 and "
            f"{DEFAULT_CALIBRATION_RELATIVE_PATH} found under {dataset_root}"
        )

    batch_started = time.perf_counter()
    tqdm.write(
        f"Discovered {len(episodes)} Nymeria episode(s); "
        f"K={output.fx:g},{output.fy:g},{output.cx:g},{output.cy:g}; "
        "video processing=disabled"
    )
    results: list[EpisodeResult] = []
    with tqdm(
        episodes,
        desc="Nymeria intrinsics",
        unit="episode",
        dynamic_ncols=True,
    ) as progress:
        for episode in progress:
            started = time.perf_counter()
            try:
                result = process_episode(
                    episode,
                    output=output,
                    overwrite=args.overwrite,
                )
            except (Exception, KeyboardInterrupt) as error:
                if isinstance(error, KeyboardInterrupt):
                    tqdm.write(
                        f"Interrupted {episode.name} after "
                        f"{_format_duration(time.perf_counter() - started)}"
                    )
                    raise
                result = EpisodeResult(
                    episode=episode,
                    status="failed",
                    elapsed_seconds=time.perf_counter() - started,
                    message=f"{type(error).__name__}: {error}",
                )
            results.append(result)
            if result.status == "completed":
                tqdm.write(
                    f"Completed {episode.name} in "
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
            progress.set_postfix_str(
                f"last={episode.name} "
                f"time={_format_duration(result.elapsed_seconds)}",
                refresh=False,
            )

    elapsed = time.perf_counter() - batch_started
    completed = sum(result.status == "completed" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)
    tqdm.write(
        f"Finished Nymeria intrinsics: completed={completed}, "
        f"skipped={skipped}, failed={failed}, "
        f"elapsed={_format_duration(elapsed)}"
    )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write the virtual pinhole K used by already-undistorted Nymeria "
            "clips. No video is decoded or generated."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/nymeria"),
        help="Nymeria dataset or one episode directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional external root. By default write "
            "<episode>/undistorted/camera_intrinsics.json."
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
        help="Virtual pinhole fx=fy (default: 260 scaled from width 384)",
    )
    parser.add_argument(
        "--principal-x",
        type=float,
        help="Virtual pinhole cx (default: (width - 1) / 2)",
    )
    parser.add_argument(
        "--principal-y",
        type=float,
        help="Virtual pinhole cy (default: (height - 1) / 2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only camera_intrinsics.json when it differs",
    )
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
