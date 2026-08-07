"""Load EgoDex human trajectories for trajectory comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from repaint.vda_pipeline_common import probe_video


BODY_CHAINS = (
    (
        "hip",
        "spine1",
        "spine2",
        "spine3",
        "spine4",
        "spine5",
        "spine6",
        "spine7",
        "neck1",
        "neck2",
        "neck3",
        "neck4",
    ),
    ("spine7", "leftShoulder", "leftArm", "leftForearm", "leftHand"),
    ("spine7", "rightShoulder", "rightArm", "rightForearm", "rightHand"),
)
FINGER_CHAINS = (
    (
        "Hand",
        "ThumbKnuckle",
        "ThumbIntermediateBase",
        "ThumbIntermediateTip",
        "ThumbTip",
    ),
    (
        "Hand",
        "IndexFingerMetacarpal",
        "IndexFingerKnuckle",
        "IndexFingerIntermediateBase",
        "IndexFingerIntermediateTip",
        "IndexFingerTip",
    ),
    (
        "Hand",
        "MiddleFingerMetacarpal",
        "MiddleFingerKnuckle",
        "MiddleFingerIntermediateBase",
        "MiddleFingerIntermediateTip",
        "MiddleFingerTip",
    ),
    (
        "Hand",
        "RingFingerMetacarpal",
        "RingFingerKnuckle",
        "RingFingerIntermediateBase",
        "RingFingerIntermediateTip",
        "RingFingerTip",
    ),
    (
        "Hand",
        "LittleFingerMetacarpal",
        "LittleFingerKnuckle",
        "LittleFingerIntermediateBase",
        "LittleFingerIntermediateTip",
        "LittleFingerTip",
    ),
)


def _all_chains() -> tuple[tuple[str, ...], ...]:
    hand_chains = tuple(
        tuple(f"{side}{name}" for name in chain)
        for side in ("left", "right")
        for chain in FINGER_CHAINS
    )
    return BODY_CHAINS + hand_chains


SKELETON_CHAINS = _all_chains()
CANONICAL_NAMES = tuple(
    dict.fromkeys(name for chain in SKELETON_CHAINS for name in chain)
)


@dataclass(frozen=True)
class HumanTrajectory:
    dataset_root: Path
    stem: str
    hdf5_path: Path
    video_path: Path
    description: str
    fps: float
    source_size: tuple[int, int]
    full_frame_count: int
    frame_indices: np.ndarray
    names: tuple[str, ...]
    positions_world: np.ndarray
    confidence: np.ndarray
    camera_positions_world: np.ndarray
    intrinsic: np.ndarray
    edges: tuple[tuple[int, int], ...]

    @property
    def frame_count(self) -> int:
        return int(self.positions_world.shape[0])


def _video_metadata(path: Path) -> tuple[int, int, int, float]:
    metadata = probe_video(path)
    return metadata.width, metadata.height, metadata.frame_count, metadata.fps


def _attribute_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return ""
    return str(value)


def _skeleton_edges(names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    name_to_index = {name: index for index, name in enumerate(names)}
    edges: list[tuple[int, int]] = []
    for chain in SKELETON_CHAINS:
        for parent, child in zip(chain, chain[1:]):
            if parent in name_to_index and child in name_to_index:
                edge = (name_to_index[parent], name_to_index[child])
                if edge not in edges:
                    edges.append(edge)
    if not edges:
        raise ValueError("No recognized human skeleton edges were found")
    return tuple(edges)


def load_human_trajectory(
    hdf5_path: Path,
    start_frame: int,
    end_frame: int | None,
) -> HumanTrajectory:
    hdf5_path = hdf5_path.expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    video_path = hdf5_path.with_suffix(".mp4")
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Expected a same-name source video next to the HDF5: {video_path}"
        )

    width, height, video_frame_count, fps = _video_metadata(video_path)
    with h5py.File(hdf5_path, "r") as handle:
        if "transforms/camera" not in handle:
            raise KeyError(f"{hdf5_path} has no transforms/camera dataset")
        camera_dataset = handle["transforms/camera"]
        if camera_dataset.ndim != 3 or camera_dataset.shape[1:] != (4, 4):
            raise ValueError(
                "transforms/camera must have shape (frames, 4, 4), got "
                f"{camera_dataset.shape}"
            )
        full_frame_count = int(camera_dataset.shape[0])
        if video_frame_count != full_frame_count:
            raise ValueError(
                "Source video and HDF5 lengths disagree: "
                f"{video_frame_count} != {full_frame_count}"
            )
        stop = full_frame_count if end_frame is None else end_frame
        if not 0 <= start_frame < stop <= full_frame_count:
            raise ValueError(
                f"Expected 0 <= start < end <= {full_frame_count}, got "
                f"{start_frame} and {stop}"
            )
        selection = slice(start_frame, stop)

        if "camera/intrinsic" not in handle:
            raise KeyError(f"{hdf5_path} has no camera/intrinsic dataset")
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("camera/intrinsic must be a finite 3x3 matrix")

        transform_group = handle.get("transforms")
        if transform_group is None:
            raise KeyError(f"{hdf5_path} has no transforms group")
        names = tuple(name for name in CANONICAL_NAMES if name in transform_group)
        if not names:
            raise ValueError(f"{hdf5_path} has no recognized human joints")

        position_values: list[np.ndarray] = []
        confidence_values: list[np.ndarray] = []
        for name in names:
            transform = handle[f"transforms/{name}"]
            if transform.shape != (full_frame_count, 4, 4):
                raise ValueError(
                    f"transforms/{name} has invalid shape {transform.shape}"
                )
            position_values.append(
                np.asarray(transform[selection, :3, 3], dtype=np.float64)
            )
            confidence_key = f"confidences/{name}"
            if confidence_key in handle:
                confidence_dataset = handle[confidence_key]
                if confidence_dataset.shape != (full_frame_count,):
                    raise ValueError(
                        f"{confidence_key} has invalid shape "
                        f"{confidence_dataset.shape}"
                    )
                confidence_values.append(
                    np.asarray(confidence_dataset[selection], dtype=np.float64)
                )
            else:
                confidence_values.append(
                    np.ones(stop - start_frame, dtype=np.float64)
                )

        positions_world = np.stack(position_values, axis=1)
        confidence = np.clip(np.stack(confidence_values, axis=1), 0.0, 1.0)
        camera_positions_world = np.asarray(
            camera_dataset[selection, :3, 3],
            dtype=np.float64,
        )
        description = _attribute_text(
            handle.attrs.get(
                "description",
                handle.attrs.get("llm_description", ""),
            )
        )

    if not np.isfinite(positions_world).all():
        raise ValueError(f"{hdf5_path} contains non-finite joint positions")
    if not np.isfinite(confidence).all():
        raise ValueError(f"{hdf5_path} contains non-finite confidence values")
    if not np.isfinite(camera_positions_world).all():
        raise ValueError(f"{hdf5_path} contains non-finite camera positions")

    return HumanTrajectory(
        dataset_root=hdf5_path.parent,
        stem=hdf5_path.stem,
        hdf5_path=hdf5_path,
        video_path=video_path,
        description=description,
        fps=fps,
        source_size=(width, height),
        full_frame_count=full_frame_count,
        frame_indices=np.arange(start_frame, stop, dtype=np.int64),
        names=names,
        positions_world=positions_world,
        confidence=confidence,
        camera_positions_world=camera_positions_world,
        intrinsic=intrinsic,
        edges=_skeleton_edges(names),
    )


def display_coordinates(points_world: np.ndarray) -> np.ndarray:
    """Map EgoDex's Y-up world coordinates to a Z-up plot."""

    result = np.asarray(points_world, dtype=np.float64)[..., (0, 2, 1)].copy()
    result[..., 1] *= -1.0
    return result
