"""One-shot scene placement from bilateral source/G1 hand anchors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from repaint.g1_hand import (
    TRACKED_HAND_TO_G1_PALM_ROTATIONS,
)
from repaint.hand_trajectory import (
    FINGERTIP_GRASP_TARGET,
    RIGID_PALM_TARGET,
    hdf5_hand_target_mode,
)
from repaint.robot_camera import (
    ROBOT_WRIST_FRAMES,
    load_source_palm_points_camera_cv,
    robot_frame_points_in_camera_cv,
    robot_palm_offsets_local_m,
)


PALM_ALIGNMENT_METHOD = (
    "initial bilateral tracked-Hand palm midpoint, "
    "coordinate-median constant world translation"
)
PALM_ALIGNMENT_STAGE = (
    "tracked-hand-palm-constant-xyz-translation"
)


@dataclass(frozen=True)
class SceneHandAlignment:
    """Scene camera poses plus any downstream constant world translation."""

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


def _anchor_points_to_world(
    *,
    source_anchor_camera_cv: np.ndarray,
    robot_anchor_camera_cv: np.ndarray,
    camera_pose_cv_xyzw: np.ndarray,
    scene_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    quaternion_norm = np.linalg.norm(
        camera_pose_cv_xyzw[:, 3:7],
        axis=1,
    )
    if np.any(quaternion_norm <= 1e-8):
        raise ValueError("Scene camera poses contain a zero quaternion")
    camera_rotation = Rotation.from_quat(
        camera_pose_cv_xyzw[:, 3:7] / quaternion_norm[:, None]
    ).as_matrix()
    source_world = (
        camera_pose_cv_xyzw[:, None, :3]
        + np.einsum(
            "nij,nsj->nsi",
            camera_rotation,
            scene_scale * source_anchor_camera_cv,
        )
    )
    robot_world = (
        camera_pose_cv_xyzw[:, None, :3]
        + np.einsum(
            "nij,nsj->nsi",
            camera_rotation,
            robot_anchor_camera_cv,
        )
    )
    return source_world, robot_world


def _estimate_initial_bilateral_midpoint_translation(
    *,
    source_anchor_camera_cv: np.ndarray,
    robot_anchor_camera_cv: np.ndarray,
    scene_camera_pose_cv_xyzw: np.ndarray,
    scene_scale: float,
    frame_validity: np.ndarray,
    anchor_name: str,
    require_positive_source_depth: bool,
    max_translation_m: float = 0.3,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    """Fit one median XYZ translation from valid bilateral anchors."""

    source = np.asarray(source_anchor_camera_cv, dtype=np.float64)
    robot = np.asarray(robot_anchor_camera_cv, dtype=np.float64)
    camera_pose = np.asarray(
        scene_camera_pose_cv_xyzw,
        dtype=np.float64,
    )
    validity = np.asarray(frame_validity, dtype=bool)
    frame_count = len(source)
    if (
        source.shape != (frame_count, 2, 3)
        or robot.shape != source.shape
        or camera_pose.shape != (frame_count, 7)
        or validity.shape != (frame_count,)
        or frame_count < 3
    ):
        raise ValueError(
            f"{anchor_name} alignment inputs must have shapes [N,2,3], "
            "[N,7], and [N] with at least three frames"
        )
    if (
        not np.isfinite(camera_pose).all()
        or not np.isfinite(scene_scale)
        or scene_scale <= 0.0
        or not np.isfinite(max_translation_m)
        or max_translation_m <= 0.0
    ):
        raise ValueError(
            f"{anchor_name} alignment transforms and thresholds are invalid"
        )
    paired_valid = (
        np.isfinite(source).all(axis=(1, 2))
        & np.isfinite(robot).all(axis=(1, 2))
        & validity
    )
    if require_positive_source_depth:
        paired_valid &= np.all(source[:, :, 2] > 0.0, axis=1)
    valid_count = int(np.sum(paired_valid))
    minimum_count = min(
        frame_count,
        max(3, min(8, int(np.ceil(frame_count / 2)))),
    )
    if valid_count < minimum_count:
        raise ValueError(
            f"Too few paired initial {anchor_name} frames for scene alignment: "
            f"{valid_count} < {minimum_count}"
        )

    source_world, robot_world = _anchor_points_to_world(
        source_anchor_camera_cv=source,
        robot_anchor_camera_cv=robot,
        camera_pose_cv_xyzw=camera_pose,
        scene_scale=scene_scale,
    )
    residual_world = (
        np.mean(robot_world, axis=1)
        - np.mean(source_world, axis=1)
    )[paired_valid]
    translation_world = np.median(residual_world, axis=0)
    translation_norm = float(np.linalg.norm(translation_world))
    before_norm = np.linalg.norm(residual_world, axis=1)
    after_norm = np.linalg.norm(
        residual_world - translation_world,
        axis=1,
    )
    if translation_norm > max_translation_m:
        raise ValueError(
            f"Initial {anchor_name} alignment translation is implausibly "
            "large: "
            f"{translation_norm:.4f} m > {max_translation_m:.4f} m"
        )

    individual_after = np.linalg.norm(
        (
            robot_world
            - source_world
            - translation_world[None, None, :]
        )[paired_valid],
        axis=2,
    ).reshape(-1)
    report: dict[str, object] = {
        "schema_version": 2,
        "world_translation_xyz_m": translation_world.tolist(),
        "world_translation_norm_m": translation_norm,
        "world_translation_z_m": float(translation_world[2]),
        "scene_scale": float(scene_scale),
        "window_frame_count": frame_count,
        "paired_frame_count": valid_count,
        "minimum_paired_frame_count": minimum_count,
        "temporal_stability_required": False,
        "maximum_translation_m": float(max_translation_m),
        "midpoint_residual_before_m": _norm_distribution(before_norm),
        "midpoint_residual_after_m": _norm_distribution(after_norm),
        "individual_anchor_residual_after_m": _norm_distribution(
            individual_after
        ),
        "fits_rotation": False,
        "fits_scale": False,
        "application_count": 1,
        "applied_to": "one downstream scene-world translation",
    }
    return translation_world, report, paired_valid


def estimate_initial_palm_midpoint_translation(
    *,
    source_palm_camera_cv: np.ndarray,
    robot_palm_camera_cv: np.ndarray,
    source_wrist_to_palm_offsets_local_m: np.ndarray,
    robot_wrist_to_palm_offsets_local_m: np.ndarray,
    scene_camera_pose_cv_xyzw: np.ndarray,
    scene_scale: float,
    source_hand_target_mode: str = RIGID_PALM_TARGET,
    max_translation_m: float = 0.3,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one XYZ translation between tracked-Hand and G1 palm anchors."""

    source = np.asarray(source_palm_camera_cv, dtype=np.float64)
    source_offsets = np.asarray(
        source_wrist_to_palm_offsets_local_m,
        dtype=np.float64,
    )
    robot_offsets = np.asarray(
        robot_wrist_to_palm_offsets_local_m,
        dtype=np.float64,
    )
    if source_hand_target_mode not in {
        FINGERTIP_GRASP_TARGET,
        RIGID_PALM_TARGET,
    }:
        raise ValueError(
            f"Unsupported source hand target mode: {source_hand_target_mode!r}"
        )
    if (
        source_offsets.shape != (2, 3)
        or robot_offsets.shape != (2, 3)
        or not np.isfinite(source_offsets).all()
        or not np.isfinite(robot_offsets).all()
    ):
        raise ValueError("Palm offsets must be finite [2,3] arrays")
    translation_world, report, _ = (
        _estimate_initial_bilateral_midpoint_translation(
            source_anchor_camera_cv=source,
            robot_anchor_camera_cv=robot_palm_camera_cv,
            scene_camera_pose_cv_xyzw=scene_camera_pose_cv_xyzw,
            scene_scale=scene_scale,
            frame_validity=np.ones(len(source), dtype=bool),
            anchor_name="palm",
            require_positive_source_depth=True,
            max_translation_m=max_translation_m,
        )
    )
    report.update(
        {
            "method": PALM_ALIGNMENT_METHOD,
            "estimator": (
                "per-frame equal-left/right tracked-palm midpoint residual, "
                "coordinate median without temporal stability rejection"
            ),
            "formula": (
                "delta_world = median(G1_palm_midpoint_world - "
                "source_palm_midpoint_scene_world)"
            ),
            "anchor_semantics": {
                "source_transform_origins": [
                    "transforms/leftHand",
                    "transforms/rightHand",
                ],
                "source_action_mode": source_hand_target_mode,
                "source_transform_semantics": "tracked Hand wrist-root poses",
                "source_wrist_to_palm_offsets_local_m": (
                    source_offsets.tolist()
                ),
                "source_palm_offset_policy": (
                    "fixed side-specific Hand-to-G1-palm axis map applied to "
                    "canonical wrist-to-palm offsets, then divided by "
                    "scene_scale"
                ),
                "robot_parent_frame_origins": list(ROBOT_WRIST_FRAMES),
                "robot_palm_local_offsets_m": (
                    robot_offsets.tolist()
                ),
                "robot_action_frames": ["left_palm", "right_palm"],
                "bilateral_aggregation": (
                    "equal-weight midpoint of left and right palms"
                ),
                "matches_retarget_tracked_hand_palm_targets": True,
                "uses_source_wrist_origins": True,
                "uses_fingertips": False,
                "uses_hand_base_origins": False,
            },
            "height_policy": (
                "world Z is estimated from the same 3-D palm residual; "
                "no fixed scene or table height is used"
            ),
            "validity_gate": (
                "finite bilateral source/G1 3-D palm origins with positive "
                "source camera depth"
            ),
            "source_visibility_confidence_used": False,
            "source_visibility_confidence_semantics": (
                "Nymeria hand confidence is projection visibility, not "
                "3-D rigid-palm tracking confidence"
            ),
            "individual_palm_residual_after_m": report.pop(
                "individual_anchor_residual_after_m"
            ),
            "individual_palm_residual_is_diagnostic_only": True,
        }
    )
    return translation_world, report


def align_scene_camera_poses_to_initial_palms(
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
    max_translation_m: float = 0.3,
) -> SceneHandAlignment:
    """Place a scene from matching tracked-Hand/G1 palm anchors."""

    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    camera_pose = np.asarray(scene_camera_pose_cv_xyzw, dtype=np.float64)
    selected_count = min(int(window_frames), len(frame_indices))
    if selected_count < 3:
        raise ValueError("Prepared sequence has fewer than three frames")
    source_hdf5 = source_hdf5.resolve()
    target_mode = hdf5_hand_target_mode(source_hdf5)
    if not np.isfinite(scene_scale) or scene_scale <= 0.0:
        raise ValueError("scene_scale must be positive and finite")

    initial_indices = frame_indices[:selected_count]
    initial_camera_pose = camera_pose[:selected_count]
    robot_palm_offsets = robot_palm_offsets_local_m(urdf_path)
    source_palm_offsets = np.einsum(
        "sij,sj->si",
        TRACKED_HAND_TO_G1_PALM_ROTATIONS,
        robot_palm_offsets,
    ) / scene_scale
    source_palm = load_source_palm_points_camera_cv(
        source_hdf5,
        initial_indices,
        wrist_to_palm_offsets_local_m=source_palm_offsets,
    )
    robot_palm = robot_frame_points_in_camera_cv(
        urdf_path=urdf_path,
        joint_names=joint_names,
        joint_positions=np.asarray(joint_positions)[:selected_count],
        root_pose_xyzw=np.asarray(root_pose_xyzw)[:selected_count],
        camera_pose_cv_xyzw=initial_camera_pose,
        frame_names=ROBOT_WRIST_FRAMES,
        frame_local_offsets_m=robot_palm_offsets,
    )
    translation_world, report = estimate_initial_palm_midpoint_translation(
        source_palm_camera_cv=source_palm,
        robot_palm_camera_cv=robot_palm,
        source_wrist_to_palm_offsets_local_m=source_palm_offsets,
        robot_wrist_to_palm_offsets_local_m=robot_palm_offsets,
        scene_camera_pose_cv_xyzw=initial_camera_pose,
        scene_scale=scene_scale,
        source_hand_target_mode=target_mode,
        max_translation_m=max_translation_m,
    )
    report.update(
        {
            "source_hdf5": str(source_hdf5),
            "source_hand_target_mode": target_mode,
            "g1_urdf": str(urdf_path.resolve()),
            "window_local_frame_start": 0,
            "window_local_frame_end_exclusive": selected_count,
            "window_source_frame_indices": initial_indices.tolist(),
            "hand_frame_mapping": {
                "source": (
                    "tracked leftHand/rightHand poses plus the fixed "
                    "side-specific GMR palm-axis map"
                ),
                "robot": (
                    "left/right wrist_yaw_link plus the canonical retarget "
                    "palm offset"
                ),
                "aggregation": (
                    "equal bilateral tracked-palm midpoint"
                ),
            },
            "camera_pose_output_semantics": "translation-only scene placement",
        }
    )
    aligned_pose = camera_pose.copy()
    aligned_pose[:, :3] += translation_world
    return SceneHandAlignment(
        camera_pose_cv_xyzw=aligned_pose,
        translation_world_xyz_m=translation_world,
        report=report,
    )
