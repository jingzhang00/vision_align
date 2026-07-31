"""LeRobot v3 trajectory loading shared by the replay tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = WORKSPACE_ROOT / "output_new/basic_pick_place"
DEFAULT_G1_URDF = (
    WORKSPACE_ROOT
    / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
)
SUPPORTED_FIELDS = ("observation.state", "action")


@dataclass(frozen=True)
class Trajectory:
    dataset_root: Path
    episode_index: int
    fps: float
    field: str
    joint_names: tuple[str, ...]
    frame_indices: np.ndarray
    timestamps: np.ndarray
    root_pose_xyzw: np.ndarray
    joint_positions: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_indices))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


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

    episode_rows = _episode_metadata_rows(dataset_root)
    matching = [
        row
        for row in episode_rows
        if int(row["episode_index"]) == episode_index
    ]
    if len(matching) != 1:
        available = sorted(
            int(row["episode_index"])
            for row in episode_rows
        )
        raise ValueError(
            f"Episode {episode_index} was not found exactly once; "
            f"available episodes: {available}"
        )
    episode_row = matching[0]
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

    length = int(episode_row["length"])
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
        fps=float(info["fps"]),
        field=field,
        joint_names=joint_names,
        frame_indices=frame_indices[selection],
        timestamps=timestamps[selection],
        root_pose_xyzw=root_pose[selection],
        joint_positions=joint_positions[selection],
    )
