"""Minimal LeRobot v3 trajectory loading for the fast FK replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from repaint.vda_pipeline_common import probe_video, read_json as _read_json


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = WORKSPACE_ROOT / "egodex/basic_pick_place/lerobot"
DEFAULT_G1_URDF = (
    WORKSPACE_ROOT
    / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
)
VIDEO_FEATURE = "observation.images.ego"
SUPPORTED_FIELDS = ("observation.state", "action")


@dataclass(frozen=True)
class Trajectory:
    dataset_root: Path
    episode_index: int
    fps: float
    source_video: Path
    source_size: tuple[int, int]
    field: str
    joint_names: tuple[str, ...]
    frame_indices: np.ndarray
    timestamps: np.ndarray
    root_pose_xyzw: np.ndarray
    joint_positions: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_indices))


def _episode_metadata_rows(dataset_root: Path) -> list[dict[str, Any]]:
    paths = sorted(
        (dataset_root / "meta/episodes").glob("chunk-*/*.parquet")
    )
    if not paths:
        raise FileNotFoundError(
            f"No LeRobot episode metadata under {dataset_root / 'meta/episodes'}"
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def find_dataset_roots(folder: Path) -> tuple[Path, ...]:
    """Find LeRobot dataset roots recursively under a folder."""

    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    roots = tuple(
        sorted(info.parent.parent for info in folder.rglob("meta/info.json"))
    )
    if not roots:
        raise FileNotFoundError(f"No LeRobot datasets under {folder}")
    return roots


def _episode_metadata_row(
    dataset_root: Path,
    episode_index: int,
) -> dict[str, Any]:
    matching = [
        row
        for row in _episode_metadata_rows(dataset_root)
        if int(row["episode_index"]) == episode_index
    ]
    if len(matching) != 1:
        available = sorted(
            int(row["episode_index"])
            for row in _episode_metadata_rows(dataset_root)
        )
        raise ValueError(
            f"Episode {episode_index} was not found exactly once; "
            f"available episodes: {available}"
        )
    return matching[0]


def _source_video_metadata(path: Path) -> tuple[int, int, int, float]:
    metadata = probe_video(path)
    return (
        metadata.width,
        metadata.height,
        metadata.frame_count,
        metadata.fps,
    )


def _episode_source_video(
    *,
    dataset_root: Path,
    info: dict[str, Any],
    episode_row: dict[str, Any],
) -> Path:
    template = info.get("video_path")
    if not isinstance(template, str) or not template:
        raise ValueError("LeRobot metadata does not declare video_path")
    chunk_key = f"videos/{VIDEO_FEATURE}/chunk_index"
    file_key = f"videos/{VIDEO_FEATURE}/file_index"
    if chunk_key not in episode_row or file_key not in episode_row:
        raise KeyError(
            f"Episode metadata does not locate {VIDEO_FEATURE!r}"
        )
    relative = template.format(
        video_key=VIDEO_FEATURE,
        chunk_index=int(episode_row[chunk_key]),
        file_index=int(episode_row[file_key]),
    )
    path = dataset_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing source video: {path}")
    return path


def list_episode_indices(dataset_root: Path) -> tuple[int, ...]:
    """Return every episode index declared by a LeRobot dataset."""

    indices = sorted(
        int(row["episode_index"])
        for row in _episode_metadata_rows(dataset_root.resolve())
    )
    if len(indices) != len(set(indices)):
        raise ValueError(
            f"Duplicate episode metadata under {dataset_root.resolve()}"
        )
    return tuple(indices)


def load_trajectory(
    dataset_root: Path,
    episode_index: int,
    field: str,
    start_frame: int,
    end_frame: int | None,
) -> Trajectory:
    """Load one trajectory and derive playback metadata from its source MP4."""

    dataset_root = dataset_root.resolve()
    info = _read_json(dataset_root / "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{dataset_root} is not a LeRobot v3.0 dataset")
    if info.get("robot_type") not in {
        "unitree_g1",
        "unitree_g1_with_hands",
    }:
        raise ValueError(
            f"Unsupported robot_type: {info.get('robot_type')!r}"
        )
    feature = info.get("features", {}).get(field)
    if field not in SUPPORTED_FIELDS or not isinstance(feature, dict):
        raise KeyError(f"Dataset does not contain supported field {field!r}")
    joint_names = tuple(feature.get("names", ()))
    if not joint_names:
        raise ValueError(f"{field} does not declare any G1 joint names")

    episode_row = _episode_metadata_row(dataset_root, episode_index)
    length = int(episode_row["length"])
    source_video = _episode_source_video(
        dataset_root=dataset_root,
        info=info,
        episode_row=episode_row,
    )
    width, height, source_frame_count, source_fps = (
        _source_video_metadata(source_video)
    )
    if source_frame_count != length:
        raise ValueError(
            "Source video and trajectory lengths disagree: "
            f"{source_frame_count} != {length}"
        )

    data_path = dataset_root / str(info["data_path"]).format(
        chunk_index=int(episode_row["data/chunk_index"]),
        file_index=int(episode_row["data/file_index"]),
    )
    table = pq.read_table(
        data_path,
        columns=[
            field,
            "observation.root_pose",
            "episode_index",
            "frame_index",
            "timestamp",
        ],
    )
    mask = (
        np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        == episode_index
    )
    frame_indices = np.asarray(
        table["frame_index"].to_pylist(),
        dtype=np.int64,
    )[mask]
    order = np.argsort(frame_indices)
    frame_indices = frame_indices[order]
    timestamps = np.asarray(
        table["timestamp"].to_pylist(),
        dtype=np.float64,
    )[mask][order]
    root_pose = np.asarray(
        table["observation.root_pose"].to_pylist(),
        dtype=np.float64,
    )[mask][order]
    joint_positions = np.asarray(
        table[field].to_pylist(),
        dtype=np.float64,
    )[mask][order]

    if not np.array_equal(frame_indices, np.arange(length)):
        raise ValueError(
            f"Episode {episode_index} frame indices are not contiguous"
        )
    if root_pose.shape != (length, 7):
        raise ValueError(f"Invalid root pose shape: {root_pose.shape}")
    if joint_positions.shape != (length, len(joint_names)):
        raise ValueError(
            f"Invalid joint trajectory shape: {joint_positions.shape}"
        )
    if not (
        np.isfinite(root_pose).all()
        and np.isfinite(joint_positions).all()
        and np.isfinite(timestamps).all()
    ):
        raise ValueError(f"Episode {episode_index} contains non-finite values")
    if not np.allclose(
        np.linalg.norm(root_pose[:, 3:7], axis=1),
        1.0,
        atol=2e-4,
    ):
        raise ValueError(
            f"Episode {episode_index} has non-unit root quaternions"
        )

    stop = length if end_frame is None else end_frame
    if not 0 <= start_frame < stop <= length:
        raise ValueError(
            f"Expected 0 <= start-frame < end-frame <= {length}, "
            f"got {start_frame} and {stop}"
        )
    selection = slice(start_frame, stop)
    return Trajectory(
        dataset_root=dataset_root,
        episode_index=episode_index,
        fps=source_fps,
        source_video=source_video,
        source_size=(width, height),
        field=field,
        joint_names=joint_names,
        frame_indices=frame_indices[selection],
        timestamps=timestamps[selection],
        root_pose_xyzw=root_pose[selection],
        joint_positions=joint_positions[selection],
    )
