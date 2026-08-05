"""Load one calibrated EgoDex or Nymeria G1 episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from repaint.robot_camera import (
    HEAD_CAMERA_PARENT_FRAME,
    TORSO_ANCHORED_CAMERA_MAPPING,
    robot_frame_poses_world,
    torso_anchored_camera_poses_cv,
)
from repaint.vda_pipeline_common import probe_video, read_json as _read_json


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
    source_camera_pose_in_spine3_xyzw: np.ndarray
    intrinsic: np.ndarray
    scene_scale: float
    source_spine3_to_robot_torso_rotation: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_indices))


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


def _one_egodex_source_episode(
    dataset_root: Path,
    episode_index: int,
) -> dict[str, Any]:
    manifest_path = dataset_root / "meta/egodex.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    metadata = _read_json(manifest_path)
    if int(metadata.get("schema_version", -1)) != 9:
        raise ValueError("EgoDex manifest predates torso-anchored cameras")
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
    metadata = probe_video(path)
    return (
        metadata.width,
        metadata.height,
        metadata.frame_count,
        metadata.fps,
    )


def _manifest_path(
    value: object,
    *,
    source_root: Path,
) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = source_root / path
    return path.resolve()


def _load_nymeria_robot_episode(
    *,
    dataset_root: Path,
    source_root: Path,
    removed_root: Path,
    episode_index: int,
    start_frame: int,
    end_frame: int | None,
) -> RgbdEpisode:
    manifest = _read_json(dataset_root / "manifest.json")
    if (
        manifest.get("format") != "nymeria-g1-trajectory-v1"
        or int(manifest.get("schema_version", -1)) != 3
    ):
        raise ValueError("Unsupported Nymeria robot trajectory format")
    matching = [
        item
        for item in manifest.get("episodes", ())
        if int(item.get("episode_index", -1)) == episode_index
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Nymeria episode {episode_index} was found "
            f"{len(matching)} times"
        )
    episode = matching[0]
    alignment = episode.get("camera_alignment")
    if not isinstance(alignment, dict) or alignment.get("status") != "validated":
        raise ValueError(f"Nymeria episode {episode_index} lacks camera alignment")
    scene_scale = float(alignment["scene_scale"])
    mapping_mode = str(alignment.get("mapping_mode", ""))
    if mapping_mode != TORSO_ANCHORED_CAMERA_MAPPING:
        raise ValueError(f"Unsupported Nymeria camera mapping: {mapping_mode}")
    source_spine3_to_robot_torso_rotation = np.asarray(
        alignment["source_spine3_to_robot_torso_rotation"],
        dtype=np.float64,
    )
    if source_spine3_to_robot_torso_rotation.shape != (3, 3):
        raise ValueError("Nymeria source-spine3-to-G1-torso basis is invalid")
    trajectory_path = (dataset_root / str(episode["trajectory_path"])).resolve()
    if (
        not trajectory_path.is_relative_to(dataset_root)
        or not trajectory_path.is_file()
    ):
        raise FileNotFoundError(trajectory_path)
    with np.load(trajectory_path) as trajectory:
        joint_positions = np.asarray(
            trajectory["joint_positions"], dtype=np.float64
        )
        root_pose = np.asarray(trajectory["root_pose_xyzw"], dtype=np.float64)
        recorded_camera_pose = np.asarray(
            trajectory["camera_pose_cv_xyzw"], dtype=np.float64
        )
        source_camera_pose_in_spine3 = np.asarray(
            trajectory["source_camera_pose_in_spine3_xyzw"],
            dtype=np.float64,
        )
        intrinsic = np.asarray(trajectory["camera_intrinsic"], dtype=np.float64)
        frame_indices = np.asarray(trajectory["frame_indices"], dtype=np.int64)
    joint_names = tuple(str(name) for name in manifest.get("joint_names", ()))
    length = int(episode["frame_count"])
    if (
        not joint_names
        or joint_positions.shape != (length, len(joint_names))
        or root_pose.shape != (length, 7)
        or recorded_camera_pose.shape != (length, 7)
        or source_camera_pose_in_spine3.shape != (length, 7)
        or intrinsic.shape != (3, 3)
        or not np.array_equal(frame_indices, np.arange(length))
        or not all(
            np.isfinite(values).all()
            for values in (
                joint_positions,
                root_pose,
                recorded_camera_pose,
                source_camera_pose_in_spine3,
                intrinsic,
            )
        )
    ):
        raise ValueError(f"Nymeria episode {episode_index} trajectory is invalid")
    stop = length if end_frame is None else end_frame
    if not 0 <= start_frame < stop <= length:
        raise ValueError(
            f"Expected 0 <= start < end <= {length}, got {start_frame} and {stop}"
        )

    source_video = _manifest_path(episode["source_video"], source_root=source_root)
    source_hdf5 = _manifest_path(alignment["adapter_hdf5"], source_root=source_root)
    removed_video = (removed_root / str(episode["video_path"])).resolve()
    for path in (source_video, source_hdf5, removed_video):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_meta = _video_metadata(source_video)
    removed_meta = _video_metadata(removed_video)
    if source_meta[:3] != removed_meta[:3] or source_meta[2] != length:
        raise ValueError("Nymeria source/removed video geometry is inconsistent")
    if abs(source_meta[3] - removed_meta[3]) > 0.01:
        raise ValueError("Nymeria source/removed video FPS values disagree")

    with h5py.File(source_hdf5, "r") as handle:
        source_intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
    if not np.allclose(source_intrinsic, intrinsic, atol=1e-5):
        raise ValueError("Nymeria camera adapter and trajectory disagree")
    robot_urdf = _manifest_path(
        manifest["robot_urdf"],
        source_root=dataset_root,
    )
    if not robot_urdf.is_file():
        raise FileNotFoundError(robot_urdf)
    robot_torso_poses = robot_frame_poses_world(
        urdf_path=robot_urdf,
        joint_names=joint_names,
        joint_positions=joint_positions,
        root_pose_xyzw=root_pose,
        frame_name=HEAD_CAMERA_PARENT_FRAME,
    )
    mapped_camera_pose = torso_anchored_camera_poses_cv(
        source_camera_pose_in_spine3_xyzw=(
            source_camera_pose_in_spine3
        ),
        robot_torso_pose_world_xyzw=robot_torso_poses,
        source_spine3_to_robot_torso_rotation=(
            source_spine3_to_robot_torso_rotation
        ),
        scene_scale=scene_scale,
    )
    rotation_error = (
        Rotation.from_quat(mapped_camera_pose[:, 3:7]).inv()
        * Rotation.from_quat(recorded_camera_pose[:, 3:7])
    ).magnitude()
    if float(np.max(rotation_error)) > 1e-5:
        raise ValueError("Torso-anchored and recorded Nymeria rotations disagree")
    position_error = np.linalg.norm(
        mapped_camera_pose[:, :3] - recorded_camera_pose[:, :3],
        axis=1,
    )
    if float(np.max(position_error)) > 1e-5:
        raise ValueError("Torso-anchored and recorded Nymeria positions disagree")
    selection = slice(start_frame, stop)
    return RgbdEpisode(
        dataset_root=dataset_root,
        source_dataset_kind="nymeria",
        episode_index=episode_index,
        source_stem=str(episode["source_stem"]),
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
        camera_pose_cv_xyzw=recorded_camera_pose[selection],
        source_camera_pose_in_spine3_xyzw=(
            source_camera_pose_in_spine3[selection]
        ),
        intrinsic=source_intrinsic,
        scene_scale=scene_scale,
        source_spine3_to_robot_torso_rotation=(
            source_spine3_to_robot_torso_rotation
        ),
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
    """Load aligned robot/camera data without invoking the legacy pipeline."""

    dataset_root = dataset_root.resolve()
    source_root = source_root.resolve()
    removed_root = removed_root.resolve()
    if (dataset_root / "manifest.json").is_file():
        return _load_nymeria_robot_episode(
            dataset_root=dataset_root,
            source_root=source_root,
            removed_root=removed_root,
            episode_index=episode_index,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    info = _read_json(dataset_root / "meta/info.json")
    feature = info.get("features", {}).get(TRAJECTORY_FIELD)
    if not isinstance(feature, dict):
        raise KeyError(f"Dataset has no {TRAJECTORY_FIELD!r} feature")
    joint_names = tuple(str(name) for name in feature.get("names", ()))
    if not joint_names:
        raise ValueError(f"{TRAJECTORY_FIELD!r} has no joint names")

    episode_row = _one_episode_row(dataset_root, episode_index)
    source_episode = _one_egodex_source_episode(
        dataset_root,
        episode_index,
    )
    mapping_mode = str(source_episode.get("mapping_mode", ""))
    if mapping_mode != TORSO_ANCHORED_CAMERA_MAPPING:
        raise ValueError(f"Unsupported EgoDex camera mapping: {mapping_mode}")
    scene_scale = float(source_episode["scene_scale"])
    source_spine3_to_robot_torso_rotation = np.asarray(
        source_episode["source_spine3_to_robot_torso_rotation"],
        dtype=np.float64,
    )
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
            TRAJECTORY_FIELD,
            "observation.root_pose",
            "observation.camera_pose",
            "observation.camera_intrinsics",
            "observation.source_camera_pose_in_spine3_xyzw",
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
        table[TRAJECTORY_FIELD].to_pylist(),
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
    source_camera_pose_in_spine3 = np.asarray(
        table["observation.source_camera_pose_in_spine3_xyzw"].to_pylist(),
        dtype=np.float64,
    )[selected_rows]
    intrinsics = np.asarray(
        table["observation.camera_intrinsics"].to_pylist(),
        dtype=np.float64,
    )[selected_rows].reshape(length, 3, 3)
    if not np.allclose(intrinsics, intrinsics[:1], atol=1e-6):
        raise ValueError("Per-frame camera intrinsics are not constant")

    source_stem = str(source_episode["source_stem"])
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
    removed_video = removed_root / video_relative_path
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
    if not np.allclose(source_intrinsic, intrinsics[0], atol=1e-5):
        raise ValueError("Source and stored camera intrinsics disagree")
    if source_spine3_to_robot_torso_rotation.shape != (3, 3):
        raise ValueError("EgoDex source-spine3-to-G1-torso basis is invalid")
    if source_camera_pose_in_spine3.shape != (length, 7) or not np.isfinite(
        source_camera_pose_in_spine3
    ).all():
        raise ValueError("EgoDex source camera-in-spine3 poses are invalid")
    egodex_manifest = _read_json(dataset_root / "meta/egodex.json")
    gmr_model = egodex_manifest.get("gmr_robot_model")
    if not isinstance(gmr_model, dict):
        raise ValueError("EgoDex manifest has no G1 model provenance")
    robot_urdf = _manifest_path(
        gmr_model["source_urdf"],
        source_root=dataset_root,
    )
    robot_torso_poses = robot_frame_poses_world(
        urdf_path=robot_urdf,
        joint_names=joint_names,
        joint_positions=joint_positions,
        root_pose_xyzw=root_pose,
        frame_name=HEAD_CAMERA_PARENT_FRAME,
    )
    mapped_camera_pose = torso_anchored_camera_poses_cv(
        source_camera_pose_in_spine3_xyzw=(
            source_camera_pose_in_spine3
        ),
        robot_torso_pose_world_xyzw=robot_torso_poses,
        source_spine3_to_robot_torso_rotation=(
            source_spine3_to_robot_torso_rotation
        ),
        scene_scale=scene_scale,
    )
    rotation_error = (
        Rotation.from_quat(mapped_camera_pose[:, 3:7]).inv()
        * Rotation.from_quat(recorded_camera_pose[:, 3:7])
    ).magnitude()
    if float(np.max(rotation_error)) > 1e-5:
        raise ValueError(
            "Recomputed and recorded camera rotations disagree: "
            f"max={float(np.max(rotation_error)):.6g} rad"
        )
    position_error = np.linalg.norm(
        mapped_camera_pose[:, :3] - recorded_camera_pose[:, :3],
        axis=1,
    )
    if float(np.max(position_error)) > 1e-5:
        raise ValueError(
            "Recomputed and recorded camera positions disagree: "
            f"max={float(np.max(position_error)):.6g} m"
        )
    selection = slice(start_frame, stop)
    return RgbdEpisode(
        dataset_root=dataset_root,
        source_dataset_kind="egodex",
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
        camera_pose_cv_xyzw=recorded_camera_pose[selection],
        source_camera_pose_in_spine3_xyzw=(
            source_camera_pose_in_spine3[selection]
        ),
        intrinsic=source_intrinsic,
        scene_scale=scene_scale,
        source_spine3_to_robot_torso_rotation=(
            source_spine3_to_robot_torso_rotation
        ),
    )
