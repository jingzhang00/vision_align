"""One constant EgoDex scene translation from initial bilateral palms."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from repaint.g1_hand import (
    PALM_OFFSET_FROM_WRIST,
    TRACKED_HAND_TO_G1_PALM_ROTATIONS,
)
from repaint.robot_camera import robot_frame_points_in_camera_cv


INITIAL_PALM_FRAME_COUNT = 5
MAX_SCENE_TRANSLATION_M = 0.3
SOURCE_HAND_TRANSFORMS = ("leftHand", "rightHand")
ROBOT_WRIST_FRAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _load_source_palms_camera_cv(
    source_hdf5: Path,
    frame_indices: np.ndarray,
    wrist_to_palm_offsets_local_m: np.ndarray,
) -> np.ndarray:
    """Return tracked source palm points in the original CV camera frame."""

    indices = np.asarray(frame_indices, dtype=np.int64)
    offsets = np.asarray(wrist_to_palm_offsets_local_m, dtype=np.float64)
    if (
        indices.ndim != 1
        or len(indices) == 0
        or np.any(indices < 0)
        or offsets.shape != (2, 3)
        or not np.isfinite(offsets).all()
    ):
        raise ValueError("Source palm inputs are invalid")
    with h5py.File(source_hdf5, "r") as handle:
        camera_dataset = handle["transforms/camera"]
        if (
            camera_dataset.ndim != 3
            or camera_dataset.shape[1:] != (4, 4)
            or int(indices[-1]) >= camera_dataset.shape[0]
        ):
            raise ValueError("Source camera transforms are invalid")
        camera = np.asarray(camera_dataset[indices], dtype=np.float64)
        palms_world = []
        for side, name in enumerate(SOURCE_HAND_TRANSFORMS):
            dataset = handle[f"transforms/{name}"]
            if dataset.shape != camera_dataset.shape:
                raise ValueError(f"Source {name} transforms are invalid")
            transform = np.asarray(dataset[indices], dtype=np.float64)
            palms_world.append(
                transform[:, :3, 3]
                + np.einsum(
                    "nij,j->ni",
                    transform[:, :3, :3],
                    offsets[side],
                )
            )
    return np.einsum(
        "nji,nsj->nsi",
        camera[:, :3, :3],
        np.stack(palms_world, axis=1) - camera[:, None, :3, 3],
    )


def initial_palm_scene_translation(
    *,
    source_hdf5: Path,
    frame_indices: np.ndarray,
    scene_scale: float,
    scene_camera_pose_cv_xyzw: np.ndarray,
    urdf_path: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
) -> np.ndarray:
    """Fit one world XYZ translation from the first five bilateral palms."""

    indices = np.asarray(frame_indices, dtype=np.int64)
    camera = np.asarray(scene_camera_pose_cv_xyzw, dtype=np.float64)
    count = min(INITIAL_PALM_FRAME_COUNT, len(indices))
    if (
        count < 3
        or camera.shape != (len(indices), 7)
        or not np.isfinite(camera).all()
        or not np.isfinite(scene_scale)
        or scene_scale <= 0.0
    ):
        raise ValueError("Initial palm alignment inputs are invalid")
    camera = camera[:count]
    palm_offsets = np.repeat(PALM_OFFSET_FROM_WRIST[None], 2, axis=0)
    source_palms_camera = _load_source_palms_camera_cv(
        source_hdf5,
        indices[:count],
        np.einsum(
            "sij,sj->si",
            TRACKED_HAND_TO_G1_PALM_ROTATIONS,
            palm_offsets,
        )
        / scene_scale,
    )
    robot_palms_camera = robot_frame_points_in_camera_cv(
        urdf_path=urdf_path,
        joint_names=joint_names,
        joint_positions=np.asarray(joint_positions)[:count],
        root_pose_xyzw=np.asarray(root_pose_xyzw)[:count],
        camera_pose_cv_xyzw=camera,
        frame_names=ROBOT_WRIST_FRAMES,
        frame_local_offsets_m=palm_offsets,
    )
    valid = (
        np.isfinite(source_palms_camera).all(axis=(1, 2))
        & np.isfinite(robot_palms_camera).all(axis=(1, 2))
        & np.all(source_palms_camera[:, :, 2] > 0.0, axis=1)
    )
    if int(valid.sum()) < 3:
        raise ValueError("Fewer than three valid initial bilateral palm frames")
    camera_rotation = Rotation.from_quat(camera[:, 3:7]).as_matrix()
    residual_world = np.einsum(
        "nij,nj->ni",
        camera_rotation,
        np.mean(
            robot_palms_camera - scene_scale * source_palms_camera,
            axis=1,
        ),
    )[valid]
    translation = np.median(residual_world, axis=0)
    if float(np.linalg.norm(translation)) > MAX_SCENE_TRANSLATION_M:
        raise ValueError(
            "Initial palm scene translation exceeds "
            f"{MAX_SCENE_TRANSLATION_M:.1f} m"
        )
    return translation
