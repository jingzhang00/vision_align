"""Calibrate the mapped source camera against robot hand trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from repaint.rgbd_episode import RgbdEpisode


@dataclass(frozen=True)
class HandTranslationCalibration:
    camera_pose_cv_xyzw: np.ndarray
    translation_camera_xyz: np.ndarray
    correspondence_count: int
    confidence_threshold: float
    rmse_before_m: float
    rmse_after_m: float
    median_error_before_m: float
    median_error_after_m: float


def _source_hand_camera_points(
    source_hdf5: Path,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(source_hdf5, "r") as handle:
        camera = np.asarray(handle["transforms/camera"], dtype=np.float64)[
            frame_indices
        ]
        hand = np.stack(
            [
                np.asarray(
                    handle["transforms/leftHand"],
                    dtype=np.float64,
                )[frame_indices],
                np.asarray(
                    handle["transforms/rightHand"],
                    dtype=np.float64,
                )[frame_indices],
            ],
            axis=1,
        )
        confidence = np.stack(
            [
                np.asarray(
                    handle["confidences/leftHand"],
                    dtype=np.float64,
                )[frame_indices],
                np.asarray(
                    handle["confidences/rightHand"],
                    dtype=np.float64,
                )[frame_indices],
            ],
            axis=1,
        )
    camera_rotation = camera[:, :3, :3]
    relative_world = hand[:, :, :3, 3] - camera[:, None, :3, 3]
    points_camera = np.einsum(
        "nji,nsj->nsi",
        camera_rotation,
        relative_world,
    )
    return points_camera, confidence


def _robot_hand_camera_points(
    episode: RgbdEpisode,
    urdf_path: Path,
) -> np.ndarray:
    model = pin.buildModelFromUrdf(str(urdf_path.resolve()))
    if tuple(model.names[1:]) != episode.joint_names:
        raise ValueError(
            "Episode joints do not match the calibration G1 URDF"
        )
    hand_frame_ids = [
        model.getFrameId("left_hand_base_link"),
        model.getFrameId("right_hand_base_link"),
    ]
    if any(frame_id >= model.nframes for frame_id in hand_frame_ids):
        raise KeyError("G1 URDF is missing a hand base frame")

    data = model.createData()
    points_camera = np.empty(
        (episode.frame_count, 2, 3),
        dtype=np.float64,
    )
    for frame_id in range(episode.frame_count):
        pin.forwardKinematics(
            model,
            data,
            episode.joint_positions[frame_id],
        )
        pin.updateFramePlacements(model, data)
        root = episode.root_pose_xyzw[frame_id]
        root_rotation = Rotation.from_quat(root[3:7]).as_matrix()
        camera = episode.camera_pose_cv_xyzw[frame_id]
        camera_rotation = Rotation.from_quat(
            camera[3:7]
        ).as_matrix()
        for side, hand_frame_id in enumerate(hand_frame_ids):
            point_world = (
                root_rotation
                @ np.asarray(
                    data.oMf[hand_frame_id].translation,
                    dtype=np.float64,
                )
                + root[:3]
            )
            points_camera[frame_id, side] = (
                camera_rotation.T @ (point_world - camera[:3])
            )
    return points_camera


def calibrate_camera_translation_to_hands(
    episode: RgbdEpisode,
    urdf_path: Path,
    *,
    confidence_threshold: float = 0.5,
) -> HandTranslationCalibration:
    """Apply one robust episode-level camera translation from hand anchors.

    The source camera rotation and intrinsic remain untouched.  The correction
    compensates for the static camera/body offset left by imperfect human-to-G1
    morphology and IK, while the scene and camera still share the source
    extrinsic motion.
    """

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in [0, 1]")
    source_hand, confidence = _source_hand_camera_points(
        episode.source_hdf5,
        episode.frame_indices,
    )
    robot_hand = _robot_hand_camera_points(episode, urdf_path)
    target_hand = source_hand * episode.scene_scale
    valid = (
        (confidence >= confidence_threshold)
        & np.isfinite(source_hand).all(axis=-1)
        & np.isfinite(robot_hand).all(axis=-1)
    )
    if int(valid.sum()) < 4:
        raise ValueError(
            "Too few confident hand correspondences for camera calibration"
        )

    residual_before = robot_hand[valid] - target_hand[valid]
    translation_camera = np.median(residual_before, axis=0)
    residual_after = residual_before - translation_camera
    camera_pose = episode.camera_pose_cv_xyzw.astype(
        np.float64,
        copy=True,
    )
    camera_rotation = Rotation.from_quat(
        camera_pose[:, 3:7]
    )
    camera_pose[:, :3] += camera_rotation.apply(translation_camera)

    before_norm = np.linalg.norm(residual_before, axis=1)
    after_norm = np.linalg.norm(residual_after, axis=1)
    return HandTranslationCalibration(
        camera_pose_cv_xyzw=camera_pose,
        translation_camera_xyz=translation_camera,
        correspondence_count=int(valid.sum()),
        confidence_threshold=float(confidence_threshold),
        rmse_before_m=float(np.sqrt(np.mean(np.square(before_norm)))),
        rmse_after_m=float(np.sqrt(np.mean(np.square(after_norm)))),
        median_error_before_m=float(np.median(before_norm)),
        median_error_after_m=float(np.median(after_norm)),
    )
