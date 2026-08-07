"""Load one authoritative EgoDex or Nymeria G1 RGB-D episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from repaint.vda_pipeline_common import probe_video, read_json


VIDEO_FEATURE = "observation.images.ego"
TRAJECTORY_FIELD = "observation.state"


@dataclass(frozen=True)
class RgbdEpisode:
    dataset_root: Path
    source_dataset_kind: str
    episode_index: int
    source_stem: str
    source_video: Path
    source_hdf5: Path
    removed_video: Path
    fps: float
    source_size: tuple[int, int]
    full_frame_count: int
    frame_indices: np.ndarray
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    root_pose_xyzw: np.ndarray
    camera_pose_cv_xyzw: np.ndarray
    intrinsic: np.ndarray
    scene_scale: float

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_indices))


def _one(rows: Any, episode_index: int, label: str) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and int(row.get("episode_index", -1)) == episode_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{label} episode {episode_index} was found {len(matches)} times"
        )
    return matches[0]


def _episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    paths = sorted((dataset_root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(dataset_root / "meta/episodes")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _source_path(value: object, source_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else source_root / path).resolve()


def _load_episode(
    *,
    dataset_root: Path,
    source_dataset_kind: str,
    episode_index: int,
    source_stem: str,
    source_video: Path,
    source_hdf5: Path,
    removed_video: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
    camera_pose_cv_xyzw: np.ndarray,
    intrinsic: np.ndarray,
    frame_indices: np.ndarray,
    scene_scale: float,
    start_frame: int,
    end_frame: int | None,
) -> RgbdEpisode:
    length = len(frame_indices)
    if (
        not joint_names
        or joint_positions.shape != (length, len(joint_names))
        or root_pose_xyzw.shape != (length, 7)
        or camera_pose_cv_xyzw.shape != (length, 7)
        or intrinsic.shape != (3, 3)
        or not np.array_equal(frame_indices, np.arange(length))
        or not all(
            np.isfinite(value).all()
            for value in (
                joint_positions,
                root_pose_xyzw,
                camera_pose_cv_xyzw,
                intrinsic,
            )
        )
        or not np.isfinite(scene_scale)
        or scene_scale <= 0.0
    ):
        raise ValueError(f"Invalid {source_dataset_kind} trajectory episode")
    stop = length if end_frame is None else end_frame
    if not 0 <= start_frame < stop <= length:
        raise ValueError(f"Expected 0 <= start < end <= {length}")
    for path in (source_video, source_hdf5, removed_video):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = probe_video(source_video)
    removed = probe_video(removed_video)
    if not source.compatible_with(removed) or source.frame_count != length:
        raise ValueError("Source, handless video, and trajectory timing disagree")
    selection = slice(start_frame, stop)
    return RgbdEpisode(
        dataset_root=dataset_root,
        source_dataset_kind=source_dataset_kind,
        episode_index=episode_index,
        source_stem=source_stem,
        source_video=source_video,
        source_hdf5=source_hdf5,
        removed_video=removed_video,
        fps=source.fps,
        source_size=(source.width, source.height),
        full_frame_count=length,
        frame_indices=frame_indices[selection],
        joint_names=joint_names,
        joint_positions=joint_positions[selection],
        root_pose_xyzw=root_pose_xyzw[selection],
        camera_pose_cv_xyzw=camera_pose_cv_xyzw[selection],
        intrinsic=intrinsic,
        scene_scale=float(scene_scale),
    )


def _load_nymeria(
    dataset_root: Path,
    source_root: Path,
    removed_root: Path,
    episode_index: int,
    start_frame: int,
    end_frame: int | None,
) -> RgbdEpisode:
    manifest = read_json(dataset_root / "manifest.json")
    if manifest.get("format") != "nymeria-g1-trajectory-v1":
        raise ValueError("Unsupported native G1 trajectory format")
    episode = _one(manifest.get("episodes", ()), episode_index, "Nymeria")
    alignment = episode.get("camera_alignment")
    if not isinstance(alignment, dict):
        raise ValueError("Nymeria episode has no camera alignment")
    trajectory_path = (dataset_root / str(episode["trajectory_path"])).resolve()
    if not trajectory_path.is_relative_to(dataset_root) or not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    with np.load(trajectory_path) as trajectory:
        return _load_episode(
            dataset_root=dataset_root,
            source_dataset_kind="nymeria",
            episode_index=episode_index,
            source_stem=str(episode["source_stem"]),
            source_video=_source_path(episode["source_video"], source_root),
            source_hdf5=_source_path(alignment["adapter_hdf5"], source_root),
            removed_video=(removed_root / str(episode["video_path"])).resolve(),
            joint_names=tuple(str(name) for name in manifest["joint_names"]),
            joint_positions=np.asarray(trajectory["joint_positions"], dtype=float),
            root_pose_xyzw=np.asarray(trajectory["root_pose_xyzw"], dtype=float),
            camera_pose_cv_xyzw=np.asarray(
                trajectory["camera_pose_cv_xyzw"], dtype=float
            ),
            intrinsic=np.asarray(trajectory["camera_intrinsic"], dtype=float),
            frame_indices=np.asarray(trajectory["frame_indices"], dtype=np.int64),
            scene_scale=float(alignment["scene_scale"]),
            start_frame=start_frame,
            end_frame=end_frame,
        )


def _load_egodex(
    dataset_root: Path,
    source_root: Path,
    removed_root: Path,
    episode_index: int,
    start_frame: int,
    end_frame: int | None,
) -> RgbdEpisode:
    info = read_json(dataset_root / "meta/info.json")
    manifest = read_json(dataset_root / "meta/egodex.json")
    if int(manifest.get("schema_version", -1)) != 10:
        raise ValueError("EgoDex trajectory must use the position-anchored schema")
    episode_row = _one(_episode_rows(dataset_root), episode_index, "LeRobot")
    source_episode = _one(manifest.get("episodes", ()), episode_index, "EgoDex")
    feature = info.get("features", {}).get(TRAJECTORY_FIELD, {})
    joint_names = tuple(str(name) for name in feature.get("names", ()))
    data_path = dataset_root / str(info["data_path"]).format(
        chunk_index=int(episode_row["data/chunk_index"]),
        file_index=int(episode_row["data/file_index"]),
    )
    table = pq.read_table(
        data_path,
        columns=[
            TRAJECTORY_FIELD,
            "observation.root_pose",
            "observation.camera_pose",
            "observation.camera_intrinsics",
            "episode_index",
            "frame_index",
        ],
    )
    rows = np.flatnonzero(
        np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        == episode_index
    )
    order = np.argsort(
        np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)[rows]
    )
    rows = rows[order]
    frame_indices = np.asarray(
        table["frame_index"].to_pylist(), dtype=np.int64
    )[rows]
    intrinsics = np.asarray(
        table["observation.camera_intrinsics"].to_pylist(), dtype=float
    )[rows].reshape(len(rows), 3, 3)
    if not np.allclose(intrinsics, intrinsics[:1], atol=1e-6):
        raise ValueError("EgoDex camera intrinsics changed within the episode")
    video_relative = str(info["video_path"]).format(
        video_key=VIDEO_FEATURE,
        chunk_index=int(episode_row[f"videos/{VIDEO_FEATURE}/chunk_index"]),
        file_index=int(episode_row[f"videos/{VIDEO_FEATURE}/file_index"]),
    )
    return _load_episode(
        dataset_root=dataset_root,
        source_dataset_kind="egodex",
        episode_index=episode_index,
        source_stem=str(source_episode["source_stem"]),
        source_video=_source_path(source_episode["source_video"], source_root),
        source_hdf5=_source_path(source_episode["source_hdf5"], source_root),
        removed_video=(removed_root / video_relative).resolve(),
        joint_names=joint_names,
        joint_positions=np.asarray(table[TRAJECTORY_FIELD].to_pylist(), dtype=float)[rows],
        root_pose_xyzw=np.asarray(
            table["observation.root_pose"].to_pylist(), dtype=float
        )[rows],
        camera_pose_cv_xyzw=np.asarray(
            table["observation.camera_pose"].to_pylist(), dtype=float
        )[rows],
        intrinsic=intrinsics[0],
        frame_indices=frame_indices,
        scene_scale=float(source_episode["scene_scale"]),
        start_frame=start_frame,
        end_frame=end_frame,
    )


def load_rgbd_episode(
    *,
    dataset_root: Path,
    source_root: Path,
    removed_root: Path,
    episode_index: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> RgbdEpisode:
    dataset_root = dataset_root.expanduser().resolve()
    loader = _load_nymeria if (dataset_root / "manifest.json").is_file() else _load_egodex
    return loader(
        dataset_root,
        source_root.expanduser().resolve(),
        removed_root.expanduser().resolve(),
        episode_index,
        start_frame,
        end_frame,
    )
