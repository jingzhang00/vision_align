from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from .trajectory import HandTrajectory, load_hand_trajectory
from .video_io import video_metadata


CV_TO_G1_WORLD = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

LEG_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

WRIST_JOINTS = (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

HAND_JOINT_SUFFIXES = (
    "index_mcp_pitch",
    "index_dip",
    "middle_mcp_pitch",
    "middle_dip",
    "pinky_mcp_pitch",
    "pinky_dip",
    "ring_mcp_pitch",
    "ring_dip",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "thumb_ip",
)

HAND_JOINTS = tuple(
    f"{side}_{suffix}"
    for side in ("left", "right")
    for suffix in HAND_JOINT_SUFFIXES
)

HAND_TIP_LOCAL_OFFSETS = {
    "thumb": np.array([-0.00679, 0.0, 0.04790], dtype=np.float64),
    "index": np.array([0.01458, 0.0, 0.03707], dtype=np.float64),
    "middle": np.array([0.01458, 0.0, 0.03707], dtype=np.float64),
}
PALM_FRAMES = ("left_palm", "right_palm")
PALM_OFFSET_FROM_WRIST = np.array([0.08, 0.0, 0.0], dtype=np.float64)
TRAJECTORY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PinocchioRetargetResult:
    qpos: np.ndarray
    left_targets: np.ndarray
    right_targets: np.ndarray
    left_target_rotations: np.ndarray
    right_target_rotations: np.ndarray
    left_errors: np.ndarray
    right_errors: np.ndarray
    left_orientation_errors: np.ndarray
    right_orientation_errors: np.ndarray
    left_visible: np.ndarray
    right_visible: np.ndarray
    hand_closures: np.ndarray
    target_hand_widths: np.ndarray
    achieved_hand_widths: np.ndarray
    hand_aperture_profiles: list[dict[str, float]]
    base_position: np.ndarray
    scale: float
    search_results: list[dict[str, float | list[float]]]


def build_pinocchio_g1(urdf_path: str | Path) -> pin.Model:
    model = pin.buildModelFromUrdf(str(Path(urdf_path).resolve()))
    for side in ("left", "right"):
        wrist_frame_id = model.getFrameId(f"{side}_wrist_yaw_link")
        if wrist_frame_id >= model.nframes:
            raise ValueError(f"Missing {side} wrist frame in G1 URDF")
        wrist_frame = model.frames[wrist_frame_id]
        palm = pin.Frame(
            f"{side}_palm",
            wrist_frame.parent,
            wrist_frame.previousFrame,
            wrist_frame.placement
            * pin.SE3(np.eye(3), PALM_OFFSET_FROM_WRIST),
            pin.FrameType.OP_FRAME,
        )
        model.addFrame(palm)
        hand_frame_id = model.getFrameId(f"{side}_hand_base_link")
        if hand_frame_id >= model.nframes:
            raise ValueError(f"Missing {side} hand frame in G1 URDF")
    missing_joints = [
        name
        for name in HAND_JOINTS
        if model.getJointId(name) >= model.njoints
    ]
    if missing_joints:
        raise ValueError(
            "G1 hand URDF is missing joints: " + ", ".join(missing_joints)
        )
    return model


def _closed_hand_joint_values(
    model: pin.Model,
    side: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for suffix in HAND_JOINT_SUFFIXES:
        joint_name = f"{side}_{suffix}"
        joint_id = model.getJointId(joint_name)
        q_index = model.idx_qs[joint_id]
        values[suffix] = float(model.upperPositionLimit[q_index])

    mimic_scale = {
        "index_dip": 0.89,
        "middle_dip": 0.89,
        "pinky_dip": 0.89,
        "ring_dip": 0.89,
    }
    for suffix, scale in mimic_scale.items():
        master = suffix.replace("_dip", "_mcp_pitch")
        values[suffix] = min(values[suffix], scale * values[master])
    thumb_scale = 2.29 if side == "left" else 1.86
    values["thumb_cmc_pitch"] = min(
        values["thumb_cmc_pitch"],
        values["thumb_ip"] / thumb_scale,
    )
    values["thumb_ip"] = min(
        values["thumb_ip"],
        thumb_scale * values["thumb_cmc_pitch"],
    )
    return values


def _set_continuous_hand_configuration(
    model: pin.Model,
    q: np.ndarray,
    closures: np.ndarray,
) -> None:
    """Map continuous ``[left, right]`` closure values into LinkerHand joints."""

    action = np.asarray(closures, dtype=np.float64)
    if (
        action.shape != (2,)
        or not np.isfinite(action).all()
        or np.any(action < 0.0)
        or np.any(action > 1.0)
    ):
        raise ValueError(
            "Continuous hand closure must be finite [left, right] values "
            f"within [0, 1], got {action}"
        )
    for side_index, side in enumerate(("left", "right")):
        closure = float(action[side_index])
        closed_values = _closed_hand_joint_values(model, side)
        for suffix in HAND_JOINT_SUFFIXES:
            joint_name = f"{side}_{suffix}"
            joint_id = model.getJointId(joint_name)
            q_index = model.idx_qs[joint_id]
            lower = float(model.lowerPositionLimit[q_index])
            q[q_index] = lower + closure * (
                closed_values[suffix] - lower
            )


def _current_hand_jaw_width(
    model: pin.Model,
    data: pin.Data,
    *,
    side: str,
) -> float:
    tips: dict[str, np.ndarray] = {}
    for digit, offset in HAND_TIP_LOCAL_OFFSETS.items():
        placement = data.oMf[model.getFrameId(f"{side}_{digit}_distal")]
        tips[digit] = placement.translation + placement.rotation @ offset
    virtual_finger = 0.7 * tips["index"] + 0.3 * tips["middle"]
    return float(np.linalg.norm(tips["thumb"] - virtual_finger))


def _hand_aperture_curve(
    model: pin.Model,
    *,
    side: str,
    sample_count: int = 257,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the monotonic closing jaw-width branch for one hand."""

    closures = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    widths = np.empty(sample_count, dtype=np.float64)
    data = model.createData()
    side_index = 0 if side == "left" else 1
    for sample_id, closure in enumerate(closures):
        q = pin.neutral(model)
        action = np.zeros(2, dtype=np.float64)
        action[side_index] = closure
        _set_continuous_hand_configuration(model, q, action)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        widths[sample_id] = _current_hand_jaw_width(
            model,
            data,
            side=side,
        )

    minimum_index = int(np.argmin(widths))
    closures = closures[: minimum_index + 1]
    widths = widths[: minimum_index + 1]
    if len(widths) < 2 or np.any(np.diff(widths) >= 0.0):
        raise RuntimeError(
            f"{side} hand aperture is not strictly decreasing before contact"
        )
    return closures, widths


def _retarget_hand_widths(
    model: pin.Model,
    human_widths: np.ndarray,
    *,
    scale: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, float]],
]:
    """Map the paper's continuous ``w_t`` to calibrated hand closure."""

    widths = np.asarray(human_widths, dtype=np.float64)
    if widths.ndim != 2 or widths.shape[1] != 2:
        raise ValueError(
            f"Human hand widths must have shape [N, 2], got {widths.shape}"
        )
    if not np.isfinite(widths).all() or np.any(widths < 0.0):
        raise ValueError("Human hand widths must be finite and non-negative")
    target_widths = scale * widths
    closures = np.empty_like(target_widths)
    profiles: list[dict[str, float]] = []
    for side_index, side in enumerate(("left", "right")):
        curve_closures, curve_widths = _hand_aperture_curve(
            model,
            side=side,
        )
        clamped_targets = np.clip(
            target_widths[:, side_index],
            curve_widths[-1],
            curve_widths[0],
        )
        closures[:, side_index] = np.interp(
            clamped_targets,
            curve_widths[::-1],
            curve_closures[::-1],
        )
        profiles.append(
            {
                "open_width_m": float(curve_widths[0]),
                "minimum_width_m": float(curve_widths[-1]),
                "maximum_calibrated_closure": float(curve_closures[-1]),
            }
        )

    achieved_widths = np.empty_like(target_widths)
    data = model.createData()
    for frame_id, frame_closures in enumerate(closures):
        q = pin.neutral(model)
        _set_continuous_hand_configuration(model, q, frame_closures)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for side_index, side in enumerate(("left", "right")):
            achieved_widths[frame_id, side_index] = (
                _current_hand_jaw_width(model, data, side=side)
            )
    return closures, target_widths, achieved_widths, profiles


def _visible(
    pixels: np.ndarray,
    confidence: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    return (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
        & (confidence >= 0.5)
    )


def _tracked(points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=1) & (confidence >= 0.5)


def _map_targets(
    trajectory: HandTrajectory,
    scale: float,
    camera_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = camera_origin + (
        CV_TO_G1_WORLD @ (scale * trajectory.left_reference_camera).T
    ).T
    right = camera_origin + (
        CV_TO_G1_WORLD @ (scale * trajectory.right_reference_camera).T
    ).T
    return left, right


def _map_target_rotations(
    trajectory: HandTrajectory,
    left_calibration: np.ndarray,
    right_calibration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    for source, calibration in (
        (trajectory.left_reference_rotations, left_calibration),
        (trajectory.right_reference_rotations, right_calibration),
    ):
        mapped = np.einsum("ij,njk->nik", CV_TO_G1_WORLD, source)
        hand_to_palm = mapped[0].T @ calibration
        targets.append(np.einsum("nij,jk->nik", mapped, hand_to_palm))
    return targets[0], targets[1]


def _camera_poses(
    trajectory: HandTrajectory,
    scale: float,
    camera_origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions: list[np.ndarray] = []
    cv_rotations: list[np.ndarray] = []
    gl_quaternions: list[np.ndarray] = []
    cv_to_gl = np.diag([1.0, -1.0, -1.0])
    for motion in trajectory.camera_motion:
        position = camera_origin + CV_TO_G1_WORLD @ (
            scale * motion[:3, 3]
        )
        world_from_cv = CV_TO_G1_WORLD @ motion[:3, :3]
        world_from_gl = world_from_cv @ cv_to_gl
        xyzw = Rotation.from_matrix(world_from_gl).as_quat()
        positions.append(position)
        cv_rotations.append(world_from_cv)
        gl_quaternions.append(np.roll(xyzw, 1))
    return (
        np.asarray(positions),
        np.asarray(cv_rotations),
        np.asarray(gl_quaternions),
    )


def _base_candidates() -> Iterable[tuple[np.ndarray, float]]:
    scale = 0.84
    for x in (-0.12, 0.0, 0.12):
        for y in (-0.10, 0.0, 0.10):
            for z in (0.70, 0.793, 0.88):
                yield np.array([x, y, z], dtype=np.float64), scale


def _active_velocity_indices(
    model: pin.Model,
    *,
    position_only: bool,
) -> np.ndarray:
    frozen = set(LEG_JOINTS) | set(HAND_JOINTS)
    if position_only:
        frozen.update(WRIST_JOINTS)
    indices: list[int] = []
    for joint_name in model.names[1:]:
        if joint_name in frozen:
            continue
        joint_id = model.getJointId(joint_name)
        indices.extend(
            range(
                model.idx_vs[joint_id],
                model.idx_vs[joint_id] + model.nvs[joint_id],
            )
        )
    return np.asarray(indices, dtype=np.int64)


def _posture_weights(
    model: pin.Model,
    active_indices: np.ndarray,
) -> np.ndarray:
    weights = np.full(len(active_indices), 1e-4, dtype=np.float64)
    active_lookup = {
        int(velocity_index): local_index
        for local_index, velocity_index in enumerate(active_indices)
    }
    for joint_name in (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ):
        velocity_index = model.idx_vs[model.getJointId(joint_name)]
        if velocity_index in active_lookup:
            weights[active_lookup[velocity_index]] = 1e-3
    for joint_name in WRIST_JOINTS:
        velocity_index = model.idx_vs[model.getJointId(joint_name)]
        if velocity_index in active_lookup:
            weights[active_lookup[velocity_index]] = 5e-5
    return weights


def _solve_frame(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    *,
    base_position: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    left_rotation: np.ndarray | None,
    right_rotation: np.ndarray | None,
    left_active: bool,
    right_active: bool,
    active_indices: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    frame_ids = tuple(model.getFrameId(name) for name in PALM_FRAMES)
    posture_weights = _posture_weights(model, active_indices)
    orientation_weight = 0.15
    frame_start_q = q.copy()
    for _ in range(iterations):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        jacobians: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        for frame_id, target, target_rotation, active in (
            (frame_ids[0], left_target, left_rotation, left_active),
            (frame_ids[1], right_target, right_rotation, right_active),
        ):
            if not active:
                continue
            placement = data.oMf[frame_id]
            jacobian = pin.computeFrameJacobian(
                model,
                data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:, active_indices]
            position_error = (
                target - base_position - placement.translation
            )
            if target_rotation is None:
                jacobians.append(jacobian[:3])
                errors.append(position_error)
            else:
                orientation_error = Rotation.from_matrix(
                    target_rotation @ placement.rotation.T
                ).as_rotvec()
                jacobians.append(
                    np.vstack(
                        [
                            jacobian[:3],
                            orientation_weight * jacobian[3:],
                        ]
                    )
                )
                errors.append(
                    np.concatenate(
                        [
                            position_error,
                            orientation_weight * orientation_error,
                        ]
                    )
                )
        regularization = np.sqrt(posture_weights)
        posture_jacobian = np.diag(regularization)
        posture_error = regularization * (
            frame_start_q[active_indices] - q[active_indices]
        )
        jacobians.append(posture_jacobian)
        errors.append(posture_error)
        jacobian = np.vstack(jacobians)
        error = np.concatenate(errors)
        hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ error
        delta = np.linalg.solve(
            hessian + 1e-6 * np.eye(len(active_indices)),
            gradient,
        )
        maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
        if maximum > 0.20:
            delta *= 0.20 / maximum
        q[active_indices] += 0.8 * delta
        q[:] = np.clip(
            q,
            model.lowerPositionLimit + 0.02,
            model.upperPositionLimit - 0.02,
        )
        if np.linalg.norm(error) < 1e-4:
            break

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    metrics: list[float] = []
    for frame_id, target, target_rotation, active in (
        (frame_ids[0], left_target, left_rotation, left_active),
        (frame_ids[1], right_target, right_rotation, right_active),
    ):
        placement = data.oMf[frame_id]
        metrics.append(
            float(
                np.linalg.norm(
                    placement.translation - (target - base_position)
                )
            )
            if active
            else float("nan")
        )
        metrics.append(
            float(
                Rotation.from_matrix(
                    placement.rotation.T @ target_rotation
                ).magnitude()
            )
            if active and target_rotation is not None
            else float("nan")
        )
    return q, (metrics[0], metrics[2], metrics[1], metrics[3])


def _evaluate_candidate(
    model: pin.Model,
    trajectory: HandTrajectory,
    base_position: np.ndarray,
    scale: float,
    camera_origin: np.ndarray,
    left_tracked: np.ndarray,
    right_tracked: np.ndarray,
) -> dict[str, float | list[float]]:
    left_targets, right_targets = _map_targets(
        trajectory,
        scale,
        camera_origin,
    )
    q = pin.neutral(model)
    data = model.createData()
    active_indices = _active_velocity_indices(model, position_only=True)
    sample_ids = np.unique(
        np.concatenate(
            [
                np.arange(0, trajectory.frame_count, 12),
                np.flatnonzero(
                    np.diff(
                        right_tracked.astype(np.int8),
                        prepend=0,
                    )
                    != 0
                ),
            ]
        )
    )
    sample_ids = sample_ids[
        (sample_ids >= 0) & (sample_ids < trajectory.frame_count)
    ]
    errors: list[float] = []
    feasible = 0
    target_count = 0
    for frame_id in sample_ids:
        q, metrics = _solve_frame(
            model,
            data,
            q,
            base_position=base_position,
            left_target=left_targets[frame_id],
            right_target=right_targets[frame_id],
            left_rotation=None,
            right_rotation=None,
            left_active=bool(left_tracked[frame_id]),
            right_active=bool(right_tracked[frame_id]),
            active_indices=active_indices,
            iterations=45,
        )
        for active, error in (
            (left_tracked[frame_id], metrics[0]),
            (right_tracked[frame_id], metrics[1]),
        ):
            if active:
                target_count += 1
                errors.append(error)
                feasible += int(error < 0.035)
    mean_error = float(np.mean(errors)) if errors else float("inf")
    feasible_fraction = feasible / max(target_count, 1)
    base_prior = float(
        np.linalg.norm(
            (base_position - np.array([0.0, 0.0, 0.793])) / 0.2
        )
    )
    score = 10.0 * feasible_fraction - mean_error - 0.005 * base_prior
    return {
        "base_position": base_position.tolist(),
        "scale": float(scale),
        "feasible_fraction": float(feasible_fraction),
        "mean_error_m": mean_error,
        "score": float(score),
    }


def retarget_with_pinocchio(
    model: pin.Model,
    trajectory: HandTrajectory,
    *,
    source_width: int,
    source_height: int,
    camera_origin: np.ndarray | None = None,
) -> PinocchioRetargetResult:
    if camera_origin is None:
        camera_origin = np.array([0.10, 0.0, 1.30], dtype=np.float64)
    left_visible = _visible(
        trajectory.left_pixels,
        trajectory.left_confidence,
        source_width,
        source_height,
    )
    right_visible = _visible(
        trajectory.right_pixels,
        trajectory.right_confidence,
        source_width,
        source_height,
    )
    left_tracked = _tracked(
        trajectory.left_reference_camera,
        trajectory.left_confidence,
    )
    right_tracked = _tracked(
        trajectory.right_reference_camera,
        trajectory.right_confidence,
    )
    search_results = [
        _evaluate_candidate(
            model,
            trajectory,
            base_position,
            scale,
            camera_origin,
            left_tracked,
            right_tracked,
        )
        for base_position, scale in _base_candidates()
    ]
    search_results.sort(key=lambda item: float(item["score"]), reverse=True)
    best = search_results[0]
    base_position = np.asarray(best["base_position"], dtype=np.float64)
    scale = float(best["scale"])
    left_targets, right_targets = _map_targets(
        trajectory,
        scale,
        camera_origin,
    )
    (
        hand_closures,
        target_hand_widths,
        achieved_hand_widths,
        hand_aperture_profiles,
    ) = _retarget_hand_widths(
        model,
        np.stack(
            [trajectory.left_widths, trajectory.right_widths],
            axis=1,
        ),
        scale=scale,
    )

    q = pin.neutral(model)
    data = model.createData()
    position_indices = _active_velocity_indices(model, position_only=True)
    q, _ = _solve_frame(
        model,
        data,
        q,
        base_position=base_position,
        left_target=left_targets[0],
        right_target=right_targets[0],
        left_rotation=None,
        right_rotation=None,
        left_active=bool(left_tracked[0]),
        right_active=bool(right_tracked[0]),
        active_indices=position_indices,
        iterations=100,
    )
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    left_calibration = data.oMf[model.getFrameId("left_palm")].rotation.copy()
    right_calibration = data.oMf[model.getFrameId("right_palm")].rotation.copy()
    left_rotations, right_rotations = _map_target_rotations(
        trajectory,
        left_calibration,
        right_calibration,
    )

    active_indices = _active_velocity_indices(model, position_only=False)
    qpos = np.zeros((trajectory.frame_count, model.nq), dtype=np.float64)
    left_errors = np.full(trajectory.frame_count, np.nan)
    right_errors = np.full(trajectory.frame_count, np.nan)
    left_orientation_errors = np.full(trajectory.frame_count, np.nan)
    right_orientation_errors = np.full(trajectory.frame_count, np.nan)
    for frame_id in range(trajectory.frame_count):
        q, metrics = _solve_frame(
            model,
            data,
            q,
            base_position=base_position,
            left_target=left_targets[frame_id],
            right_target=right_targets[frame_id],
            left_rotation=left_rotations[frame_id],
            right_rotation=right_rotations[frame_id],
            left_active=bool(left_tracked[frame_id]),
            right_active=bool(right_tracked[frame_id]),
            active_indices=active_indices,
            iterations=60 if frame_id == 0 else 20,
        )
        finite_position_errors = np.asarray(metrics[:2], dtype=np.float64)
        if (
            np.isfinite(finite_position_errors).any()
            and np.nanmax(finite_position_errors) > 0.005
        ):
            q, _ = _solve_frame(
                model,
                data,
                q,
                base_position=base_position,
                left_target=left_targets[frame_id],
                right_target=right_targets[frame_id],
                left_rotation=None,
                right_rotation=None,
                left_active=bool(left_tracked[frame_id]),
                right_active=bool(right_tracked[frame_id]),
                active_indices=position_indices,
                iterations=20,
            )
            q, metrics = _solve_frame(
                model,
                data,
                q,
                base_position=base_position,
                left_target=left_targets[frame_id],
                right_target=right_targets[frame_id],
                left_rotation=left_rotations[frame_id],
                right_rotation=right_rotations[frame_id],
                left_active=bool(left_tracked[frame_id]),
                right_active=bool(right_tracked[frame_id]),
                active_indices=active_indices,
                iterations=0,
            )
        _set_continuous_hand_configuration(
            model,
            q,
            hand_closures[frame_id],
        )
        qpos[frame_id] = q
        (
            left_errors[frame_id],
            right_errors[frame_id],
            left_orientation_errors[frame_id],
            right_orientation_errors[frame_id],
        ) = metrics
    return PinocchioRetargetResult(
        qpos=qpos,
        left_targets=left_targets,
        right_targets=right_targets,
        left_target_rotations=left_rotations,
        right_target_rotations=right_rotations,
        left_errors=left_errors,
        right_errors=right_errors,
        left_orientation_errors=left_orientation_errors,
        right_orientation_errors=right_orientation_errors,
        left_visible=left_visible,
        right_visible=right_visible,
        hand_closures=hand_closures,
        target_hand_widths=target_hand_widths,
        achieved_hand_widths=achieved_hand_widths,
        hand_aperture_profiles=hand_aperture_profiles,
        base_position=base_position,
        scale=scale,
        search_results=search_results,
    )


def save_pinocchio_trajectory(
    *,
    model: pin.Model,
    trajectory: HandTrajectory,
    result: PinocchioRetargetResult,
    output_root: Path,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    link_frame_ids = [
        frame_id
        for frame_id, frame in enumerate(model.frames)
        if frame.type == pin.FrameType.BODY
    ]
    link_names = [model.frames[frame_id].name for frame_id in link_frame_ids]
    link_positions = np.empty(
        (trajectory.frame_count, len(link_frame_ids), 3),
        dtype=np.float32,
    )
    link_quaternions = np.empty(
        (trajectory.frame_count, len(link_frame_ids), 4),
        dtype=np.float32,
    )
    data = model.createData()
    for trajectory_frame, qpos in enumerate(result.qpos):
        pin.forwardKinematics(model, data, qpos)
        pin.updateFramePlacements(model, data)
        for link_index, frame_id in enumerate(link_frame_ids):
            placement = data.oMf[frame_id]
            link_positions[trajectory_frame, link_index] = (
                placement.translation
            )
            link_quaternions[trajectory_frame, link_index] = np.roll(
                Rotation.from_matrix(placement.rotation).as_quat(),
                1,
            )
    camera_positions, camera_cv_rotations, camera_gl_quaternions = (
        _camera_poses(
            trajectory,
            result.scale,
            np.array([0.10, 0.0, 1.30], dtype=np.float64),
        )
    )
    joint_steps = np.abs(np.diff(result.qpos, axis=0))
    hand_q_indices = np.asarray(
        [
            q_index
            for joint_name in HAND_JOINTS
            for q_index in range(
                model.idx_qs[model.getJointId(joint_name)],
                model.idx_qs[model.getJointId(joint_name)]
                + model.nqs[model.getJointId(joint_name)],
            )
        ],
        dtype=np.int64,
    )
    arm_q_mask = np.ones(model.nq, dtype=bool)
    arm_q_mask[hand_q_indices] = False
    arm_joint_steps = joint_steps[:, arm_q_mask]
    hand_joint_steps = joint_steps[:, hand_q_indices]
    wrist_motion: dict[str, dict[str, float]] = {}
    for joint_name in WRIST_JOINTS:
        joint_id = model.getJointId(joint_name)
        q_index = model.idx_qs[joint_id]
        values = result.qpos[:, q_index]
        steps = np.abs(np.diff(values))
        wrist_motion[joint_name] = {
            "range_rad": float(np.ptp(values)),
            "max_step_rad": float(np.max(steps)) if steps.size else 0.0,
        }
    report: dict[str, object] = {
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "ik_solver": f"Pinocchio {pin.__version__}",
        "ik_target_source": "HDF5 leftHand/rightHand rigid-body tracks",
        "hand_action_source": "continuous HDF5 fingertip width",
        "frame_count": trajectory.frame_count,
        "joint_count": model.nq,
        "joint_names": list(model.names[1:]),
        "link_count": len(link_names),
        "base_position": result.base_position.tolist(),
        "scale": result.scale,
        "left_mean_error_m": float(np.nanmean(result.left_errors)),
        "left_max_error_m": float(np.nanmax(result.left_errors)),
        "right_mean_error_m": float(np.nanmean(result.right_errors)),
        "right_max_error_m": float(np.nanmax(result.right_errors)),
        "left_mean_orientation_error_deg": float(
            np.degrees(np.nanmean(result.left_orientation_errors))
        ),
        "right_mean_orientation_error_deg": float(
            np.degrees(np.nanmean(result.right_orientation_errors))
        ),
        "max_joint_step_rad": (
            float(np.max(joint_steps)) if joint_steps.size else 0.0
        ),
        "max_arm_joint_step_rad": (
            float(np.max(arm_joint_steps))
            if arm_joint_steps.size
            else 0.0
        ),
        "max_hand_joint_step_rad": (
            float(np.max(hand_joint_steps))
            if hand_joint_steps.size
            else 0.0
        ),
        "wrist_motion": wrist_motion,
        "hand_action": {
            "type": "continuous_width",
            "shape": [trajectory.frame_count, 2],
            "order": ["left", "right"],
            "formula": (
                "w=||thumb-(0.7*index+0.3*middle)||_2"
            ),
            "closure_semantics": (
                "0=open; increasing values close the hand along the "
                "calibrated monotonic aperture branch"
            ),
            "source": list(trajectory.hand_sources),
            "human_width_range_m": {
                side: [
                    float(np.min(widths)),
                    float(np.max(widths)),
                ]
                for side, widths in (
                    ("left", trajectory.left_widths),
                    ("right", trajectory.right_widths),
                )
            },
            "target_width_range_m": {
                side: [
                    float(np.min(result.target_hand_widths[:, side_index])),
                    float(np.max(result.target_hand_widths[:, side_index])),
                ]
                for side_index, side in enumerate(("left", "right"))
            },
            "achieved_width_range_m": {
                side: [
                    float(np.min(result.achieved_hand_widths[:, side_index])),
                    float(np.max(result.achieved_hand_widths[:, side_index])),
                ]
                for side_index, side in enumerate(("left", "right"))
            },
            "closure_range": {
                side: [
                    float(np.min(result.hand_closures[:, side_index])),
                    float(np.max(result.hand_closures[:, side_index])),
                ]
                for side_index, side in enumerate(("left", "right"))
            },
            "aperture_profiles": {
                side: result.hand_aperture_profiles[side_index]
                for side_index, side in enumerate(("left", "right"))
            },
        },
        "top_base_search_results": result.search_results[:10],
    }
    (output_root / "ik_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_root / "g1_trajectory.npz",
        trajectory_schema_version=np.asarray(
            TRAJECTORY_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        kinematics=np.asarray("pinocchio"),
        ik_target_source=np.asarray(
            "HDF5 leftHand/rightHand rigid-body tracks"
        ),
        hand_action_source=np.asarray("continuous HDF5 fingertip width"),
        joint_names=np.asarray(model.names[1:]),
        qpos=result.qpos,
        link_names=np.asarray(link_names),
        link_positions=link_positions,
        link_quaternions=link_quaternions,
        left_targets=result.left_targets,
        right_targets=result.right_targets,
        left_target_rotations=result.left_target_rotations,
        right_target_rotations=result.right_target_rotations,
        left_errors=result.left_errors,
        right_errors=result.right_errors,
        left_orientation_errors=result.left_orientation_errors,
        right_orientation_errors=result.right_orientation_errors,
        left_visible=result.left_visible,
        right_visible=result.right_visible,
        hand_closures=result.hand_closures,
        hand_closure_order=np.asarray(["left", "right"]),
        hand_closure_semantics=np.asarray(
            "0=open; increasing closes along the calibrated monotonic "
            "aperture branch"
        ),
        hand_target_widths=result.target_hand_widths,
        hand_achieved_widths=result.achieved_hand_widths,
        hand_sources=np.asarray(trajectory.hand_sources),
        hand_widths=np.stack(
            [trajectory.left_widths, trajectory.right_widths], axis=1
        ),
        hand_normalized_widths=np.stack(
            [
                trajectory.left_normalized_widths,
                trajectory.right_normalized_widths,
            ],
            axis=1,
        ),
        left_observation_camera=trajectory.left_observation_camera,
        right_observation_camera=trajectory.right_observation_camera,
        base_position=result.base_position,
        scale=result.scale,
        camera_positions=camera_positions,
        camera_cv_rotations=camera_cv_rotations,
        camera_gl_quaternions=camera_gl_quaternions,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retarget HDF5 hands to G1 with Pinocchio for Isaac Sim"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_width, source_height, frame_count, _ = video_metadata(args.video)
    trajectory = load_hand_trajectory(args.hdf5)
    if trajectory.frame_count != frame_count:
        raise ValueError(
            f"HDF5 has {trajectory.frame_count} frames; video has "
            f"{frame_count}"
        )
    model = build_pinocchio_g1(args.urdf)
    result = retarget_with_pinocchio(
        model,
        trajectory,
        source_width=source_width,
        source_height=source_height,
    )
    report = save_pinocchio_trajectory(
        model=model,
        trajectory=trajectory,
        result=result,
        output_root=args.output.resolve(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
