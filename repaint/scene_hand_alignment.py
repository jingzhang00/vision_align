"""One-shot scene placement from initial source/G1 hand anchors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from repaint.robot_camera import (
    load_source_hand_points_camera_cv,
    robot_frame_points_in_camera_cv,
)


@dataclass(frozen=True)
class SceneHandAlignment:
    """A constant world translation applied to scene camera poses once."""

    camera_pose_cv_xyzw: np.ndarray
    translation_world_xyz_m: np.ndarray
    report: dict[str, object]


def _norm_distribution(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or len(finite) == 0 or not np.isfinite(finite).all():
        raise ValueError("Expected a non-empty finite residual vector")
    return {
        "p50": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
        "rmse": float(np.sqrt(np.mean(np.square(finite)))),
    }


def estimate_initial_hand_midpoint_translation(
    *,
    source_hand_points_camera_cv: np.ndarray,
    source_hand_confidence: np.ndarray,
    robot_hand_points_camera_cv: np.ndarray,
    scene_camera_pose_cv_xyzw: np.ndarray,
    scene_scale: float,
    confidence_threshold: float = 0.5,
    max_translation_m: float = 0.3,
    max_inlier_p95_m: float = 0.03,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one world translation from per-frame bilateral hand midpoints.

    Using the left/right midpoint prevents the human/G1 arm-span difference
    from being mistaken for scene translation. No rotation or scale is fitted.
    """

    source = np.asarray(source_hand_points_camera_cv, dtype=np.float64)
    confidence = np.asarray(source_hand_confidence, dtype=np.float64)
    robot = np.asarray(robot_hand_points_camera_cv, dtype=np.float64)
    camera_pose = np.asarray(
        scene_camera_pose_cv_xyzw,
        dtype=np.float64,
    )
    frame_count = len(camera_pose)
    if (
        source.shape != (frame_count, 2, 3)
        or robot.shape != source.shape
        or confidence.shape != (frame_count, 2)
        or camera_pose.shape != (frame_count, 7)
    ):
        raise ValueError(
            "Hand alignment inputs must have shapes [N,2,3], [N,2], "
            "and [N,7]"
        )
    if frame_count < 3:
        raise ValueError("Hand alignment needs at least three initial frames")
    if (
        not np.isfinite(scene_scale)
        or scene_scale <= 0.0
        or not 0.0 <= confidence_threshold <= 1.0
        or not np.isfinite(max_translation_m)
        or max_translation_m <= 0.0
        or not np.isfinite(max_inlier_p95_m)
        or max_inlier_p95_m <= 0.0
    ):
        raise ValueError("Hand alignment thresholds and scale are invalid")
    quaternion_norm = np.linalg.norm(camera_pose[:, 3:7], axis=1)
    if np.any(quaternion_norm <= 1e-8):
        raise ValueError("Scene camera poses contain a zero quaternion")

    paired_valid = (
        np.isfinite(source).all(axis=(1, 2))
        & np.isfinite(robot).all(axis=(1, 2))
        & np.isfinite(confidence).all(axis=1)
        & np.all(confidence >= confidence_threshold, axis=1)
        & np.all(source[:, :, 2] > 0.0, axis=1)
    )
    valid_count = int(np.sum(paired_valid))
    minimum_count = min(frame_count, max(3, min(8, int(np.ceil(frame_count / 2)))))
    if valid_count < minimum_count:
        raise ValueError(
            "Too few paired initial hand frames for scene alignment: "
            f"{valid_count} < {minimum_count}"
        )

    camera_rotation = Rotation.from_quat(
        camera_pose[:, 3:7] / quaternion_norm[:, None]
    ).as_matrix()
    midpoint_difference_camera = (
        np.mean(robot, axis=1)
        - scene_scale * np.mean(source, axis=1)
    )
    residual_world = np.einsum(
        "nij,nj->ni",
        camera_rotation,
        midpoint_difference_camera,
    )[paired_valid]

    preliminary = np.median(residual_world, axis=0)
    preliminary_deviation = np.linalg.norm(
        residual_world - preliminary,
        axis=1,
    )
    deviation_median = float(np.median(preliminary_deviation))
    deviation_mad = float(
        np.median(np.abs(preliminary_deviation - deviation_median))
    )
    inlier_threshold = max(
        0.02,
        deviation_median + 3.0 * 1.4826 * deviation_mad,
    )
    inlier = preliminary_deviation <= inlier_threshold
    inlier_count = int(np.sum(inlier))
    minimum_inlier_count = max(
        minimum_count,
        int(np.ceil(0.8 * valid_count)),
    )
    if inlier_count < minimum_inlier_count:
        raise ValueError(
            "Initial hand midpoint translation is not temporally stable: "
            f"{inlier_count} robust inliers < {minimum_inlier_count}"
        )

    translation_world = np.median(residual_world[inlier], axis=0)
    translation_norm = float(np.linalg.norm(translation_world))
    before_norm = np.linalg.norm(residual_world, axis=1)
    after_norm = np.linalg.norm(
        residual_world - translation_world,
        axis=1,
    )
    inlier_after = after_norm[inlier]
    inlier_p95 = float(np.percentile(inlier_after, 95.0))
    if translation_norm > max_translation_m:
        raise ValueError(
            "Initial hand alignment translation is implausibly large: "
            f"{translation_norm:.4f} m > {max_translation_m:.4f} m"
        )
    if inlier_p95 > max_inlier_p95_m:
        raise ValueError(
            "Initial hand midpoint is not stable enough for one rigid scene "
            f"translation: p95={inlier_p95:.4f} m > "
            f"{max_inlier_p95_m:.4f} m"
        )

    individual_difference_camera = (
        robot - scene_scale * source
    )[paired_valid]
    individual_world = np.einsum(
        "nij,nsj->nsi",
        camera_rotation[paired_valid],
        individual_difference_camera,
    )
    individual_after_norm = np.linalg.norm(
        individual_world - translation_world[None, None, :],
        axis=2,
    ).reshape(-1)
    report: dict[str, object] = {
        "schema_version": 1,
        "method": (
            "initial bilateral hand midpoint, robust constant world "
            "translation"
        ),
        "estimator": (
            "per-frame left/right midpoint residual, temporal coordinate "
            "median with MAD outlier rejection"
        ),
        "formula": (
            "delta_world = median(G1_hand_midpoint_world - "
            "source_hand_midpoint_scene_world)"
        ),
        "world_translation_xyz_m": translation_world.tolist(),
        "world_translation_norm_m": translation_norm,
        "scene_scale": float(scene_scale),
        "confidence_threshold": float(confidence_threshold),
        "window_frame_count": frame_count,
        "paired_frame_count": valid_count,
        "minimum_paired_frame_count": minimum_count,
        "robust_inlier_count": inlier_count,
        "minimum_robust_inlier_count": minimum_inlier_count,
        "robust_inlier_fraction": float(inlier_count / valid_count),
        "outlier_threshold_m": float(inlier_threshold),
        "midpoint_residual_before_m": _norm_distribution(before_norm),
        "midpoint_residual_after_m": _norm_distribution(after_norm),
        "midpoint_inlier_residual_after_m": _norm_distribution(inlier_after),
        "individual_hand_residual_after_m": _norm_distribution(
            individual_after_norm
        ),
        "individual_hand_residual_is_diagnostic_only": True,
        "fits_rotation": False,
        "fits_scale": False,
        "application_count": 1,
        "applied_to": "scene mesh camera-pose translations only",
    }
    return translation_world, report


def align_scene_camera_poses_to_initial_hands(
    *,
    source_hdf5: Path,
    frame_indices: np.ndarray,
    scene_scale: float,
    scene_camera_pose_cv_xyzw: np.ndarray,
    urdf_path: Path,
    joint_names: tuple[str, ...],
    joint_positions: np.ndarray,
    root_pose_xyzw: np.ndarray,
    window_frames: int,
    confidence_threshold: float = 0.5,
) -> SceneHandAlignment:
    """Load initial anchors/FK and translate the complete scene pose sequence."""

    if window_frames <= 0:
        raise ValueError("--scene-alignment-window-frames must be positive")
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    camera_pose = np.asarray(scene_camera_pose_cv_xyzw, dtype=np.float64)
    selected_count = min(int(window_frames), len(frame_indices))
    if selected_count < 3:
        raise ValueError("Prepared sequence has fewer than three frames")
    initial_indices = frame_indices[:selected_count]
    initial_camera_pose = camera_pose[:selected_count]
    source_hand, confidence = load_source_hand_points_camera_cv(
        source_hdf5,
        initial_indices,
    )
    robot_hand = robot_frame_points_in_camera_cv(
        urdf_path=urdf_path,
        joint_names=joint_names,
        joint_positions=np.asarray(joint_positions)[:selected_count],
        root_pose_xyzw=np.asarray(root_pose_xyzw)[:selected_count],
        camera_pose_cv_xyzw=initial_camera_pose,
        frame_names=(
            "left_hand_base_link",
            "right_hand_base_link",
        ),
    )
    translation_world, report = estimate_initial_hand_midpoint_translation(
        source_hand_points_camera_cv=source_hand,
        source_hand_confidence=confidence,
        robot_hand_points_camera_cv=robot_hand,
        scene_camera_pose_cv_xyzw=initial_camera_pose,
        scene_scale=scene_scale,
        confidence_threshold=confidence_threshold,
    )

    with h5py.File(source_hdf5, "r") as handle:
        confidence_keys = (
            "confidences/leftHand",
            "confidences/rightHand",
        )
        confidence_source = (
            "source HDF5"
            if all(key in handle for key in confidence_keys)
            else "implicit ones because source HDF5 confidence is absent"
        )
    report.update(
        {
            "source_hdf5": str(source_hdf5.resolve()),
            "g1_urdf": str(urdf_path.resolve()),
            "window_local_frame_start": 0,
            "window_local_frame_end_exclusive": selected_count,
            "window_source_frame_indices": initial_indices.tolist(),
            "confidence_source": confidence_source,
            "hand_frame_mapping": {
                "source": ["leftHand", "rightHand"],
                "robot": [
                    "left_hand_base_link",
                    "right_hand_base_link",
                ],
            },
        }
    )
    aligned_pose = camera_pose.copy()
    aligned_pose[:, :3] += translation_world
    return SceneHandAlignment(
        camera_pose_cv_xyzw=aligned_pose,
        translation_world_xyz_m=translation_world,
        report=report,
    )
