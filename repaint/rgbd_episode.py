"""Load one LeRobot G1 episode plus its original EgoDex camera data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


VIDEO_FEATURE = "observation.images.ego"
SUPPORTED_FIELDS = ("observation.state", "action")


@dataclass(frozen=True)
class RgbdEpisode:
    dataset_root: Path
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
    source_camera_pose_cv_xyzw: np.ndarray
    recorded_camera_pose_cv_xyzw: np.ndarray
    intrinsic: np.ndarray
    scene_scale: float
    recorded_scene_scale: float
    source_to_robot_rotation: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_indices))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def _episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    paths = sorted(
        (dataset_root / "meta/episodes").glob("chunk-*/*.parquet")
    )
    if not paths:
        raise FileNotFoundError(
            f"No episode metadata under {dataset_root / 'meta/episodes'}"
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _one_episode_row(
    dataset_root: Path,
    episode_index: int,
) -> dict[str, Any]:
    matching = [
        row
        for row in _episode_rows(dataset_root)
        if int(row["episode_index"]) == episode_index
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Episode {episode_index} was found {len(matching)} times"
        )
    return matching[0]


def _one_egodex_episode(
    dataset_root: Path,
    episode_index: int,
) -> dict[str, Any]:
    metadata = _read_json(dataset_root / "meta/egodex.json")
    matching = [
        episode
        for episode in metadata.get("episodes", [])
        if int(episode["episode_index"]) == episode_index
    ]
    if len(matching) != 1:
        raise ValueError(
            f"EgoDex episode {episode_index} was found {len(matching)} times"
        )
    return matching[0]


def _video_metadata(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(width, height, frame_count) <= 0 or fps <= 0.0:
        raise ValueError(f"Invalid video metadata: {path}")
    return width, height, frame_count, fps


def _mapped_camera_poses(
    *,
    root_positions: np.ndarray,
    source_camera_to_world: np.ndarray,
    source_hip_to_world: np.ndarray,
    source_to_robot_rotation: np.ndarray,
    scene_scale: float,
) -> np.ndarray:
    relative = (
        source_camera_to_world[:, :3, 3]
        - source_hip_to_world[:, :3, 3]
    )
    mapped_position = (
        root_positions
        + scene_scale
        * (source_to_robot_rotation @ relative.T).T
    )
    mapped_rotation = np.einsum(
        "ij,njk->nik",
        source_to_robot_rotation,
        source_camera_to_world[:, :3, :3],
    )
    quaternions = Rotation.from_matrix(mapped_rotation).as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    return np.concatenate([mapped_position, quaternions], axis=1)


def load_rgbd_episode(
    *,
    dataset_root: Path,
    source_root: Path,
    episode_index: int,
    field: str = "observation.state",
    start_frame: int = 0,
    end_frame: int | None = None,
    scene_scale: float = 1.0,
) -> RgbdEpisode:
    """Load aligned robot/camera data without invoking the legacy pipeline."""

    dataset_root = dataset_root.resolve()
    source_root = source_root.resolve()
    if field not in SUPPORTED_FIELDS:
        raise ValueError(f"Unsupported trajectory field: {field}")
    if not np.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError("scene_scale must be positive and finite")

    info = _read_json(dataset_root / "meta/info.json")
    feature = info.get("features", {}).get(field)
    if not isinstance(feature, dict):
        raise KeyError(f"Dataset has no {field!r} feature")
    joint_names = tuple(str(name) for name in feature.get("names", ()))
    if not joint_names:
        raise ValueError(f"{field!r} has no joint names")

    episode_row = _one_episode_row(dataset_root, episode_index)
    egodex_episode = _one_egodex_episode(dataset_root, episode_index)
    length = int(episode_row["length"])
    stop = length if end_frame is None else end_frame
    if not 0 <= start_frame < stop <= length:
        raise ValueError(
            f"Expected 0 <= start < end <= {length}, got "
            f"{start_frame} and {stop}"
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
            "observation.camera_pose",
            "observation.camera_intrinsics",
            "episode_index",
            "frame_index",
        ],
    )
    episode_column = np.asarray(
        table["episode_index"].to_pylist(),
        dtype=np.int64,
    )
    selected_rows = np.flatnonzero(episode_column == episode_index)
    frame_indices = np.asarray(
        table["frame_index"].to_pylist(),
        dtype=np.int64,
    )[selected_rows]
    order = np.argsort(frame_indices)
    selected_rows = selected_rows[order]
    frame_indices = frame_indices[order]
    if not np.array_equal(frame_indices, np.arange(length)):
        raise ValueError(f"Episode {episode_index} frames are not contiguous")

    joint_positions = np.asarray(
        table[field].to_pylist(),
        dtype=np.float64,
    )[selected_rows]
    root_pose = np.asarray(
        table["observation.root_pose"].to_pylist(),
        dtype=np.float64,
    )[selected_rows]
    recorded_camera_pose = np.asarray(
        table["observation.camera_pose"].to_pylist(),
        dtype=np.float64,
    )[selected_rows]
    intrinsics = np.asarray(
        table["observation.camera_intrinsics"].to_pylist(),
        dtype=np.float64,
    )[selected_rows].reshape(length, 3, 3)
    if not np.allclose(intrinsics, intrinsics[:1], atol=1e-6):
        raise ValueError("Per-frame camera intrinsics are not constant")

    source_stem = str(egodex_episode["source_stem"])
    source_video = source_root / f"{source_stem}.mp4"
    source_hdf5 = source_root / f"{source_stem}.hdf5"
    video_path_template = str(info["video_path"])
    video_relative_path = Path(video_path_template.format(
        video_key=VIDEO_FEATURE,
        chunk_index=int(
            episode_row[f"videos/{VIDEO_FEATURE}/chunk_index"]
        ),
        file_index=int(
            episode_row[f"videos/{VIDEO_FEATURE}/file_index"]
        ),
    ))
    removed_video = dataset_root / "repaint" / video_relative_path
    for path in (source_video, source_hdf5, removed_video, data_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_meta = _video_metadata(source_video)
    removed_meta = _video_metadata(removed_video)
    if source_meta[:3] != removed_meta[:3]:
        raise ValueError(
            "Original and removed videos disagree in size/frame count: "
            f"{source_meta[:3]} != {removed_meta[:3]}"
        )
    if source_meta[2] != length:
        raise ValueError(
            f"Video and trajectory lengths disagree: {source_meta[2]} != {length}"
        )
    if abs(source_meta[3] - removed_meta[3]) > 0.01:
        raise ValueError("Original and removed video FPS values disagree")

    with h5py.File(source_hdf5, "r") as handle:
        source_intrinsic = np.asarray(
            handle["camera/intrinsic"],
            dtype=np.float64,
        )
        source_camera_to_world = np.asarray(
            handle["transforms/camera"],
            dtype=np.float64,
        )
        source_hip_to_world = np.asarray(
            handle["transforms/hip"],
            dtype=np.float64,
        )
    if not np.allclose(source_intrinsic, intrinsics[0], atol=1e-5):
        raise ValueError("Source and LeRobot camera intrinsics disagree")
    expected_transform_shape = (length, 4, 4)
    if (
        source_camera_to_world.shape != expected_transform_shape
        or source_hip_to_world.shape != expected_transform_shape
    ):
        raise ValueError("Source camera/hip transform shapes are invalid")

    source_to_robot_rotation = np.asarray(
        egodex_episode["source_to_robot_rotation"],
        dtype=np.float64,
    )
    if source_to_robot_rotation.shape != (3, 3):
        raise ValueError("source_to_robot_rotation must be 3x3")
    mapped_camera_pose = _mapped_camera_poses(
        root_positions=root_pose[:, :3],
        source_camera_to_world=source_camera_to_world,
        source_hip_to_world=source_hip_to_world,
        source_to_robot_rotation=source_to_robot_rotation,
        scene_scale=scene_scale,
    )
    source_camera_quaternion = Rotation.from_matrix(
        source_camera_to_world[:, :3, :3]
    ).as_quat()
    source_camera_pose = np.concatenate(
        [
            source_camera_to_world[:, :3, 3],
            source_camera_quaternion,
        ],
        axis=1,
    )

    selection = slice(start_frame, stop)
    return RgbdEpisode(
        dataset_root=dataset_root,
        episode_index=episode_index,
        source_stem=source_stem,
        source_video=source_video,
        source_hdf5=source_hdf5,
        removed_video=removed_video,
        fps=float(source_meta[3]),
        source_size=(source_meta[0], source_meta[1]),
        full_frame_count=length,
        frame_indices=frame_indices[selection],
        joint_names=joint_names,
        joint_positions=joint_positions[selection],
        root_pose_xyzw=root_pose[selection],
        camera_pose_cv_xyzw=mapped_camera_pose[selection],
        source_camera_pose_cv_xyzw=source_camera_pose[selection],
        recorded_camera_pose_cv_xyzw=recorded_camera_pose[selection],
        intrinsic=source_intrinsic,
        scene_scale=float(scene_scale),
        recorded_scene_scale=float(
            egodex_episode[
                "source_to_robot_camera_translation_scale"
            ]
        ),
        source_to_robot_rotation=source_to_robot_rotation,
    )
