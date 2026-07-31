from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pinocchio as pin
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .trajectory import (
    FINGERTIP_GRASP_TARGET,
    HAND_LANDMARK_NAMES,
    RIGID_PALM_TARGET,
    ArticulatedHandTrajectory,
    HandTrajectory,
    load_hand_trajectory,
)
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
O6_MOTOR_SUFFIXES = (
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
O6_DIGITS = ("thumb", "index", "middle", "ring", "pinky")
O6_FLEX_DIGITS = ("index", "middle", "ring", "pinky")
O6_MOTOR_COUNT = len(O6_MOTOR_SUFFIXES)
ARTICULATED_DIGIT_CHAINS = {
    "thumb": (
        "ThumbKnuckle",
        "ThumbIntermediateBase",
        "ThumbIntermediateTip",
        "ThumbTip",
    ),
    "index": (
        "IndexFingerKnuckle",
        "IndexFingerIntermediateBase",
        "IndexFingerIntermediateTip",
        "IndexFingerTip",
    ),
    "middle": (
        "MiddleFingerKnuckle",
        "MiddleFingerIntermediateBase",
        "MiddleFingerIntermediateTip",
        "MiddleFingerTip",
    ),
    "ring": (
        "RingFingerKnuckle",
        "RingFingerIntermediateBase",
        "RingFingerIntermediateTip",
        "RingFingerTip",
    ),
    "pinky": (
        "LittleFingerKnuckle",
        "LittleFingerIntermediateBase",
        "LittleFingerIntermediateTip",
        "LittleFingerTip",
    ),
}

HAND_JOINTS = tuple(
    f"{side}_{suffix}"
    for side in ("left", "right")
    for suffix in HAND_JOINT_SUFFIXES
)

HAND_TIP_LOCAL_OFFSETS = {
    "thumb": np.array([-0.00679, 0.0, 0.04790], dtype=np.float64),
    "index": np.array([0.01458, 0.0, 0.03707], dtype=np.float64),
    "middle": np.array([0.01458, 0.0, 0.03707], dtype=np.float64),
    "ring": np.array([0.01458, 0.0, 0.03707], dtype=np.float64),
    "pinky": np.array([0.01458, 0.0, 0.03707], dtype=np.float64),
}
PALM_FRAMES = ("left_palm", "right_palm")
G1_BASE_FRAME = "pelvis"
G1_SUPPORT_FRAMES = ("LL_FOOT", "LR_FOOT")
G1_SHOULDER_JOINTS = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
)
PALM_OFFSET_FROM_WRIST = np.array([0.08, 0.0, 0.0], dtype=np.float64)
BASE_GRID_POINTS_PER_AXIS = 5
BASE_YAW_OFFSETS_RAD = np.deg2rad(
    np.asarray(
        [0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0, -60.0, 60.0],
        dtype=np.float64,
    )
)
IK_FEASIBLE_POSITION_ERROR_M = 0.02
IK_FEASIBLE_ORIENTATION_ERROR_RAD = np.deg2rad(15.0)
IK_ORIENTATION_WEIGHT = 0.05
TRAJECTORY_SCHEMA_VERSION = 13
EPISODE_SKIP_REPORT = "episode_skip.json"


class NoCommonBasePlacementError(RuntimeError):
    """The representative end-effector targets share no reachable base pose."""


@dataclass(frozen=True)
class PinocchioRetargetResult:
    qpos: np.ndarray
    left_targets: np.ndarray
    right_targets: np.ndarray
    left_target_rotations: np.ndarray
    right_target_rotations: np.ndarray
    left_palm_targets: np.ndarray
    right_palm_targets: np.ndarray
    left_palm_target_rotations: np.ndarray
    right_palm_target_rotations: np.ndarray
    left_errors: np.ndarray
    right_errors: np.ndarray
    left_orientation_errors: np.ndarray
    right_orientation_errors: np.ndarray
    left_visible: np.ndarray
    right_visible: np.ndarray
    hand_closures: np.ndarray
    hand_motor_commands: np.ndarray
    target_hand_widths: np.ndarray
    achieved_hand_widths: np.ndarray
    hand_aperture_profiles: list[dict[str, object]]
    hand_action_mode: str
    hand_action_diagnostics: dict[str, dict[str, object]]
    hand_tcp_translations: np.ndarray
    hand_tcp_rotations: np.ndarray
    base_frame: str
    nominal_pelvis_height: float
    base_position: np.ndarray
    base_rotation: np.ndarray
    base_search_keyframes: np.ndarray
    base_search_centroid: np.ndarray
    maximum_reach: float
    scale: float
    scale_source: str
    source_arm_lengths: np.ndarray
    robot_arm_lengths: np.ndarray
    pelvis_anchor_reference_camera: np.ndarray
    shoulder_anchor_reference_camera: np.ndarray
    robot_shoulder_center_from_pelvis: np.ndarray
    mapped_shoulder_center: np.ndarray
    camera_origin: np.ndarray
    world_from_reference_camera: np.ndarray
    search_results: list[dict[str, object]]


@dataclass(frozen=True)
class _SpatialAlignment:
    scale: float
    scale_source: str
    source_arm_lengths: np.ndarray
    robot_arm_lengths: np.ndarray
    pelvis_anchor_reference_camera: np.ndarray
    shoulder_anchor_reference_camera: np.ndarray
    robot_shoulder_center_from_pelvis: np.ndarray
    camera_origin: np.ndarray
    world_from_reference_camera: np.ndarray


@dataclass(frozen=True)
class O6HandRetargetTrajectory:
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    action_mode: str
    source_action_transforms_reference_camera: np.ndarray
    wrist_from_action_transforms: np.ndarray
    diagnostics: dict[str, dict[str, object]]


def build_pinocchio_g1(urdf_path: str | Path) -> pin.Model:
    resolved_urdf = Path(urdf_path).resolve()
    model = pin.buildModelFromUrdf(str(resolved_urdf))
    model.vision_align_mimic_joints = _load_urdf_mimic_joints(
        resolved_urdf
    )
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


def _load_urdf_mimic_joints(
    urdf_path: Path,
) -> dict[str, tuple[str, float, float]]:
    root = ET.parse(urdf_path).getroot()
    result: dict[str, tuple[str, float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        mimic = joint.find("mimic")
        if name is None or mimic is None or mimic.get("joint") is None:
            continue
        result[name] = (
            str(mimic.get("joint")),
            float(mimic.get("multiplier", "1")),
            float(mimic.get("offset", "0")),
        )
    return result


def _mimic_joints(
    model: pin.Model,
) -> dict[str, tuple[str, float, float]]:
    loaded = getattr(model, "vision_align_mimic_joints", None)
    if isinstance(loaded, dict):
        return loaded
    fallback: dict[str, tuple[str, float, float]] = {}
    for side in ("left", "right"):
        for digit in O6_FLEX_DIGITS:
            fallback[f"{side}_{digit}_dip"] = (
                f"{side}_{digit}_mcp_pitch",
                0.89,
                0.0,
            )
        fallback[f"{side}_thumb_ip"] = (
            f"{side}_thumb_cmc_pitch",
            2.29 if side == "left" else 1.86,
            0.0,
        )
    return fallback


def _o6_motor_limits(
    model: pin.Model,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown hand side: {side}")
    lower = np.empty(O6_MOTOR_COUNT, dtype=np.float64)
    upper = np.empty(O6_MOTOR_COUNT, dtype=np.float64)
    motor_index: dict[str, int] = {}
    for index, suffix in enumerate(O6_MOTOR_SUFFIXES):
        name = f"{side}_{suffix}"
        joint_id = model.getJointId(name)
        q_index = model.idx_qs[joint_id]
        lower[index] = float(model.lowerPositionLimit[q_index])
        upper[index] = float(model.upperPositionLimit[q_index])
        motor_index[name] = index

    for child, (master, multiplier, offset) in _mimic_joints(model).items():
        if not child.startswith(f"{side}_") or master not in motor_index:
            continue
        if abs(multiplier) < 1e-12:
            continue
        child_id = model.getJointId(child)
        child_q = model.idx_qs[child_id]
        child_lower = float(model.lowerPositionLimit[child_q])
        child_upper = float(model.upperPositionLimit[child_q])
        bounds = sorted(
            (
                (child_lower - offset) / multiplier,
                (child_upper - offset) / multiplier,
            )
        )
        index = motor_index[master]
        lower[index] = max(lower[index], bounds[0])
        upper[index] = min(upper[index], bounds[1])
    if np.any(upper <= lower):
        raise RuntimeError(f"Invalid O6 motor limits for {side} hand")
    return lower, upper


def _set_hand_motor_configuration(
    model: pin.Model,
    q: np.ndarray,
    motor_commands: np.ndarray,
) -> None:
    commands = np.asarray(motor_commands, dtype=np.float64)
    expected = (2, O6_MOTOR_COUNT)
    if commands.shape != expected or not np.isfinite(commands).all():
        raise ValueError(
            f"O6 motor commands must be finite with shape {expected}, "
            f"got {commands.shape}"
        )

    command_by_joint: dict[str, float] = {}
    for side_index, side in enumerate(("left", "right")):
        lower, upper = _o6_motor_limits(model, side)
        if np.any(commands[side_index] < lower - 1e-9) or np.any(
            commands[side_index] > upper + 1e-9
        ):
            raise ValueError(
                f"{side} O6 motor command exceeds limits: "
                f"{commands[side_index]}"
            )
        for motor_index, suffix in enumerate(O6_MOTOR_SUFFIXES):
            command_by_joint[f"{side}_{suffix}"] = float(
                np.clip(
                    commands[side_index, motor_index],
                    lower[motor_index],
                    upper[motor_index],
                )
            )

    mimic = _mimic_joints(model)
    for joint_name in HAND_JOINTS:
        if joint_name in command_by_joint:
            value = command_by_joint[joint_name]
        elif joint_name in mimic:
            master, multiplier, offset = mimic[joint_name]
            value = multiplier * command_by_joint[master] + offset
        else:
            raise RuntimeError(
                f"Hand joint {joint_name} is neither an O6 motor nor mimic"
            )
        joint_id = model.getJointId(joint_name)
        q_index = model.idx_qs[joint_id]
        q[q_index] = np.clip(
            value,
            model.lowerPositionLimit[q_index],
            model.upperPositionLimit[q_index],
        )


def _closures_to_motor_commands(
    model: pin.Model,
    closures: np.ndarray,
) -> np.ndarray:
    action = np.asarray(closures, dtype=np.float64)
    squeeze = action.ndim == 1
    if squeeze:
        action = action[None]
    if (
        action.ndim != 2
        or action.shape[1] != 2
        or not np.isfinite(action).all()
        or np.any(action < 0.0)
        or np.any(action > 1.0)
    ):
        raise ValueError(
            "Continuous hand closure must have shape [N, 2], be finite, "
            f"and remain within [0, 1], got {action}"
        )
    result = np.empty(
        (len(action), 2, O6_MOTOR_COUNT),
        dtype=np.float64,
    )
    for side_index, side in enumerate(("left", "right")):
        lower, upper = _o6_motor_limits(model, side)
        result[:, side_index] = lower + action[:, side_index, None] * (
            upper - lower
        )
    return result[0] if squeeze else result


def _closed_hand_joint_values(
    model: pin.Model,
    side: str,
) -> dict[str, float]:
    q = pin.neutral(model)
    commands = _closures_to_motor_commands(
        model,
        np.ones(2, dtype=np.float64),
    )
    _set_hand_motor_configuration(model, q, commands)
    return {
        suffix: float(
            q[model.idx_qs[model.getJointId(f"{side}_{suffix}")]]
        )
        for suffix in HAND_JOINT_SUFFIXES
    }


def _set_continuous_hand_configuration(
    model: pin.Model,
    q: np.ndarray,
    closures: np.ndarray,
) -> None:
    """Map continuous ``[left, right]`` closure values into LinkerHand joints."""

    _set_hand_motor_configuration(
        model,
        q,
        _closures_to_motor_commands(model, closures),
    )


def _current_hand_jaw_width(
    model: pin.Model,
    data: pin.Data,
    *,
    side: str,
) -> float:
    tips = _current_hand_tip_positions(model, data, side=side)
    virtual_finger = 0.7 * tips["index"] + 0.3 * tips["middle"]
    return float(np.linalg.norm(tips["thumb"] - virtual_finger))


def _current_hand_tip_positions(
    model: pin.Model,
    data: pin.Data,
    *,
    side: str,
) -> dict[str, np.ndarray]:
    tips: dict[str, np.ndarray] = {}
    for digit, offset in HAND_TIP_LOCAL_OFFSETS.items():
        placement = data.oMf[model.getFrameId(f"{side}_{digit}_distal")]
        tips[digit] = placement.translation + placement.rotation @ offset
    return tips


def _canonical_tcp_rotations(model: pin.Model) -> np.ndarray:
    """Return open-hand semantic gripper bases in each palm frame.

    Only the TCP origin follows the fingers while they close.  Recomputing the
    full basis from curled fingertips makes the jaw axis flip near a fist and
    feeds an artificial 80--170 degree rotation into arm IK.
    """

    q = pin.neutral(model)
    _set_continuous_hand_configuration(
        model,
        q,
        np.zeros(2, dtype=np.float64),
    )
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    rotations: list[np.ndarray] = []
    for side in ("left", "right"):
        tips = _current_hand_tip_positions(model, data, side=side)
        virtual_finger = 0.7 * tips["index"] + 0.3 * tips["middle"]
        jaw = tips["thumb"] - virtual_finger
        jaw_width = float(np.linalg.norm(jaw))
        sign = 1.0 if side == "right" else -1.0
        z_axis = sign * jaw / max(jaw_width, 1e-9)
        wrist = data.oMf[
            model.getFrameId(f"{side}_wrist_yaw_link")
        ].translation
        direction = virtual_finger - wrist
        y_axis = np.cross(z_axis, direction)
        y_axis /= max(float(np.linalg.norm(y_axis)), 1e-9)
        x_axis = np.cross(y_axis, z_axis)
        world_from_tcp = np.column_stack([x_axis, y_axis, z_axis])
        world_from_palm = data.oMf[
            model.getFrameId(f"{side}_palm")
        ].rotation
        rotations.append(world_from_palm.T @ world_from_tcp)
    return np.asarray(rotations, dtype=np.float64)


def _hand_tcp_transforms(
    model: pin.Model,
    hand_commands: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute state-dependent jaw centers with fixed semantic TCP bases."""

    commands = np.asarray(hand_commands, dtype=np.float64)
    if commands.ndim == 2 and commands.shape[1] == 2:
        motor_commands = _closures_to_motor_commands(model, commands)
    elif commands.ndim == 3 and commands.shape[1:] == (
        2,
        O6_MOTOR_COUNT,
    ):
        motor_commands = commands
    else:
        raise ValueError(
            "Hand commands must have shape [N, 2] or [N, 2, 6], "
            f"got {commands.shape}"
        )
    canonical_rotations = _canonical_tcp_rotations(model)
    translations = np.empty((len(motor_commands), 2, 3), dtype=np.float64)
    data = model.createData()
    for frame_id, command in enumerate(motor_commands):
        q = pin.neutral(model)
        _set_hand_motor_configuration(model, q, command)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for side_index, side in enumerate(("left", "right")):
            tips = _current_hand_tip_positions(model, data, side=side)
            virtual_finger = 0.7 * tips["index"] + 0.3 * tips["middle"]
            tcp_position = 0.5 * (tips["thumb"] + virtual_finger)
            palm = data.oMf[model.getFrameId(f"{side}_palm")]
            translations[frame_id, side_index] = (
                palm.rotation.T @ (tcp_position - palm.translation)
            )
    return translations, canonical_rotations


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
    """Map the continuous ``w_t`` to calibrated hand closure."""

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


def _open_hand_motor_commands(model: pin.Model) -> np.ndarray:
    return np.stack(
        [_o6_motor_limits(model, side)[0] for side in ("left", "right")],
        axis=0,
    )


def _hand_anatomical_basis(
    palm: np.ndarray,
    index_base: np.ndarray,
    middle_base: np.ndarray,
    pinky_base: np.ndarray,
) -> np.ndarray | None:
    forward = np.asarray(middle_base, dtype=np.float64) - palm
    forward_norm = float(np.linalg.norm(forward))
    if not np.isfinite(forward_norm) or forward_norm < 1e-9:
        return None
    forward /= forward_norm
    lateral = np.asarray(index_base, dtype=np.float64) - pinky_base
    lateral -= forward * float(np.dot(forward, lateral))
    lateral_norm = float(np.linalg.norm(lateral))
    if not np.isfinite(lateral_norm) or lateral_norm < 1e-9:
        return None
    lateral /= lateral_norm
    normal = np.cross(forward, lateral)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm < 1e-9:
        return None
    normal /= normal_norm
    return np.column_stack([forward, lateral, normal])


def _robot_digit_feature(
    model: pin.Model,
    data: pin.Data,
    *,
    side: str,
    digit: str,
) -> np.ndarray:
    palm = data.oMf[model.getFrameId(f"{side}_palm")].translation
    bases = {
        name: data.oMf[
            model.getFrameId(
                f"{side}_{name}_"
                + (
                    "metacarpals_base2"
                    if name == "thumb"
                    else "proximal"
                )
            )
        ].translation
        for name in O6_DIGITS
    }
    basis = _hand_anatomical_basis(
        palm,
        bases["index"],
        bases["middle"],
        bases["pinky"],
    )
    if basis is None:
        raise RuntimeError(f"Degenerate {side} robot palm geometry")

    distal = data.oMf[model.getFrameId(f"{side}_{digit}_distal")]
    tip = (
        distal.translation
        + distal.rotation @ HAND_TIP_LOCAL_OFFSETS[digit]
    )
    if digit == "thumb":
        nodes = (
            bases[digit],
            data.oMf[
                model.getFrameId(f"{side}_thumb_metacarpals")
            ].translation,
            distal.translation,
            tip,
        )
    else:
        nodes = (bases[digit], distal.translation, tip)
    chain_length = sum(
        float(np.linalg.norm(following - previous))
        for previous, following in zip(nodes[:-1], nodes[1:], strict=True)
    )
    vector = basis.T @ (tip - bases[digit])
    terminal = basis.T @ (tip - distal.translation)
    chord = float(np.linalg.norm(vector))
    terminal_length = float(np.linalg.norm(terminal))
    if (
        chain_length < 1e-9
        or chord < 1e-9
        or terminal_length < 1e-9
    ):
        raise RuntimeError(f"Degenerate {side} {digit} robot geometry")
    return np.concatenate(
        [
            vector / chord,
            terminal / terminal_length,
            np.asarray([chord / chain_length]),
        ]
    )


def _articulated_source_features(
    track: ArticulatedHandTrajectory,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if track.names != HAND_LANDMARK_NAMES:
        raise ValueError("Articulated hand landmarks use an unknown order")
    points = np.asarray(track.points_palm, dtype=np.float64)
    confidence = np.asarray(track.confidence, dtype=np.float64)
    if points.shape != (track.frame_count, len(track.names), 3):
        raise ValueError(
            f"Invalid articulated hand point shape: {points.shape}"
        )
    if confidence.shape != points.shape[:2]:
        raise ValueError(
            f"Invalid articulated hand confidence shape: {confidence.shape}"
        )
    landmark_index = {
        name: index for index, name in enumerate(track.names)
    }
    basis_names = (
        "Hand",
        "IndexFingerKnuckle",
        "MiddleFingerKnuckle",
        "LittleFingerKnuckle",
    )
    basis_indices = [landmark_index[name] for name in basis_names]
    features = {
        digit: np.full((track.frame_count, 7), np.nan, dtype=np.float64)
        for digit in O6_DIGITS
    }
    valid = {
        digit: np.zeros(track.frame_count, dtype=bool)
        for digit in O6_DIGITS
    }
    for frame_id in range(track.frame_count):
        basis_points = points[frame_id, basis_indices]
        basis_confidence = confidence[frame_id, basis_indices]
        if (
            not np.isfinite(basis_points).all()
            or not np.isfinite(basis_confidence).all()
            or np.any(basis_confidence <= 0.0)
        ):
            continue
        basis = _hand_anatomical_basis(*basis_points)
        if basis is None:
            continue
        for digit, names in ARTICULATED_DIGIT_CHAINS.items():
            indices = [landmark_index[name] for name in names]
            chain = points[frame_id, indices]
            chain_confidence = confidence[frame_id, indices]
            if (
                not np.isfinite(chain).all()
                or not np.isfinite(chain_confidence).all()
                or np.any(chain_confidence <= 0.0)
            ):
                continue
            segment_lengths = np.linalg.norm(
                np.diff(chain, axis=0),
                axis=1,
            )
            chain_length = float(np.sum(segment_lengths))
            vector = basis.T @ (chain[-1] - chain[0])
            terminal = basis.T @ (chain[-1] - chain[-2])
            chord = float(np.linalg.norm(vector))
            terminal_length = float(np.linalg.norm(terminal))
            if (
                chain_length < 1e-9
                or chord < 1e-9
                or terminal_length < 1e-9
            ):
                continue
            features[digit][frame_id] = np.concatenate(
                [
                    vector / chord,
                    terminal / terminal_length,
                    np.asarray([chord / chain_length]),
                ]
            )
            valid[digit][frame_id] = True
    return features, valid


def _robot_feature_grid(
    model: pin.Model,
    *,
    side: str,
    digit: str,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if digit not in O6_FLEX_DIGITS:
        raise ValueError(f"Expected a flex digit, got {digit}")
    motor_index = O6_MOTOR_SUFFIXES.index(f"{digit}_mcp_pitch")
    lower, upper = _o6_motor_limits(model, side)
    values = np.linspace(
        lower[motor_index],
        upper[motor_index],
        samples,
        dtype=np.float64,
    )
    features = np.empty((samples, 7), dtype=np.float64)
    commands = _open_hand_motor_commands(model)
    data = model.createData()
    side_index = 0 if side == "left" else 1
    for sample_id, value in enumerate(values):
        sample_commands = commands.copy()
        sample_commands[side_index, motor_index] = value
        q = pin.neutral(model)
        _set_hand_motor_configuration(model, q, sample_commands)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        features[sample_id] = _robot_digit_feature(
            model,
            data,
            side=side,
            digit=digit,
        )
    return values[:, None], features


def _robot_thumb_feature_grid(
    model: pin.Model,
    *,
    side: str,
    samples_per_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _o6_motor_limits(model, side)
    yaw = np.linspace(
        lower[0],
        upper[0],
        samples_per_axis,
        dtype=np.float64,
    )
    pitch = np.linspace(
        lower[1],
        upper[1],
        samples_per_axis,
        dtype=np.float64,
    )
    grid_commands = np.asarray(
        [(yaw_value, pitch_value) for yaw_value in yaw for pitch_value in pitch],
        dtype=np.float64,
    )
    features = np.empty((len(grid_commands), 7), dtype=np.float64)
    commands = _open_hand_motor_commands(model)
    data = model.createData()
    side_index = 0 if side == "left" else 1
    for sample_id, values in enumerate(grid_commands):
        sample_commands = commands.copy()
        sample_commands[side_index, :2] = values
        q = pin.neutral(model)
        _set_hand_motor_configuration(model, q, sample_commands)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        features[sample_id] = _robot_digit_feature(
            model,
            data,
            side=side,
            digit="thumb",
        )
    return grid_commands, features


def _fill_and_smooth_motor_fit(
    fitted: np.ndarray,
    valid: np.ndarray,
    fallback: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    result = np.asarray(fitted, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    fallback = np.asarray(fallback, dtype=np.float64)
    if result.shape != fallback.shape:
        raise ValueError("Fitted and fallback motor arrays differ in shape")
    frame_ids = np.arange(len(result), dtype=np.float64)
    for column in range(result.shape[1]):
        finite = valid & np.isfinite(result[:, column])
        if np.count_nonzero(finite) >= 2:
            result[:, column] = np.interp(
                frame_ids,
                frame_ids[finite],
                result[finite, column],
            )
        elif np.count_nonzero(finite) == 1:
            result[:, column] = result[finite, column][0]
        else:
            result[:, column] = fallback[:, column]
        if len(result) >= 5:
            window = min(7, len(result) if len(result) % 2 else len(result) - 1)
            if window >= 5:
                result[:, column] = savgol_filter(
                    result[:, column],
                    window_length=window,
                    polyorder=2,
                    mode="interp",
                )
    return np.clip(result, lower, upper)


def _fit_feature_grid(
    source: np.ndarray,
    valid: np.ndarray,
    grid_commands: np.ndarray,
    grid_features: np.ndarray,
    fallback: np.ndarray,
    *,
    feature_weights: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fitted = np.full_like(fallback, np.nan, dtype=np.float64)
    errors = np.full(len(source), np.nan, dtype=np.float64)
    previous: np.ndarray | None = None
    command_range = np.maximum(upper - lower, 1e-9)
    for frame_id in np.flatnonzero(valid):
        residual = (grid_features - source[frame_id]) * feature_weights
        costs = np.sum(np.square(residual), axis=1)
        if previous is not None:
            costs += 0.005 * np.sum(
                np.square((grid_commands - previous) / command_range),
                axis=1,
            )
        best = int(np.argmin(costs))
        fitted[frame_id] = grid_commands[best]
        errors[frame_id] = float(
            np.linalg.norm(
                (grid_features[best] - source[frame_id])
                * feature_weights
            )
        )
        previous = grid_commands[best]
    return (
        _fill_and_smooth_motor_fit(
            fitted,
            valid,
            fallback,
            lower=lower,
            upper=upper,
        ),
        errors,
    )


def _achieved_hand_widths_from_motors(
    model: pin.Model,
    motor_commands: np.ndarray,
) -> np.ndarray:
    commands = np.asarray(motor_commands, dtype=np.float64)
    if commands.ndim != 3 or commands.shape[1:] != (
        2,
        O6_MOTOR_COUNT,
    ):
        raise ValueError(
            f"Expected O6 motor commands [N, 2, 6], got {commands.shape}"
        )
    achieved = np.empty((len(commands), 2), dtype=np.float64)
    data = model.createData()
    for frame_id, command in enumerate(commands):
        q = pin.neutral(model)
        _set_hand_motor_configuration(model, q, command)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for side_index, side in enumerate(("left", "right")):
            achieved[frame_id, side_index] = _current_hand_jaw_width(
                model,
                data,
                side=side,
            )
    return achieved


def _motor_closure_summary(
    model: pin.Model,
    motor_commands: np.ndarray,
) -> np.ndarray:
    result = np.empty((len(motor_commands), 2), dtype=np.float64)
    for side_index, side in enumerate(("left", "right")):
        lower, upper = _o6_motor_limits(model, side)
        normalized = (
            motor_commands[:, side_index] - lower
        ) / np.maximum(upper - lower, 1e-9)
        result[:, side_index] = np.mean(normalized, axis=1)
    return np.clip(result, 0.0, 1.0)


def _retarget_articulated_hands(
    model: pin.Model,
    trajectory: HandTrajectory,
    human_widths: np.ndarray,
    *,
    scale: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
    dict[str, dict[str, object]],
]:
    (
        fallback_closures,
        target_widths,
        _,
        fallback_profiles,
    ) = _retarget_hand_widths(model, human_widths, scale=scale)
    motor_commands = _closures_to_motor_commands(
        model,
        fallback_closures,
    )
    tracks = (
        trajectory.left_articulated_hand,
        trajectory.right_articulated_hand,
    )
    profiles: list[dict[str, object]] = []
    diagnostics: dict[str, dict[str, object]] = {}
    for side_index, side in enumerate(("left", "right")):
        track = tracks[side_index]
        side_profile: dict[str, object] = {
            "motor_order": list(O6_MOTOR_SUFFIXES),
            "fallback_aperture_profile": fallback_profiles[side_index],
        }
        digit_diagnostics: dict[str, object] = {}
        if track is None:
            side_profile["source"] = "continuous_width_fallback"
            side_profile["fallback_reason"] = (
                "complete 25-landmark hand track unavailable"
            )
            profiles.append(side_profile)
            diagnostics[side] = {
                "source": "continuous_width_fallback",
                "digits": digit_diagnostics,
            }
            continue

        source_features, source_valid = _articulated_source_features(track)
        lower, upper = _o6_motor_limits(model, side)
        thumb_grid, thumb_features = _robot_thumb_feature_grid(
            model,
            side=side,
            samples_per_axis=65,
        )
        fitted_thumb, thumb_errors = _fit_feature_grid(
            source_features["thumb"],
            source_valid["thumb"],
            thumb_grid,
            thumb_features,
            motor_commands[:, side_index, :2],
            feature_weights=np.asarray(
                [
                    np.sqrt(0.5),
                    np.sqrt(0.5),
                    np.sqrt(0.5),
                    np.sqrt(0.4),
                    np.sqrt(0.4),
                    np.sqrt(0.4),
                    np.sqrt(0.1),
                ]
            ),
            lower=lower[:2],
            upper=upper[:2],
        )
        motor_commands[:, side_index, :2] = fitted_thumb
        finite_thumb_error = thumb_errors[np.isfinite(thumb_errors)]
        digit_diagnostics["thumb"] = {
            "valid_fraction": float(np.mean(source_valid["thumb"])),
            "mean_feature_error": (
                float(np.mean(finite_thumb_error))
                if finite_thumb_error.size
                else None
            ),
            "max_feature_error": (
                float(np.max(finite_thumb_error))
                if finite_thumb_error.size
                else None
            ),
        }

        for digit in O6_FLEX_DIGITS:
            motor_index = O6_MOTOR_SUFFIXES.index(f"{digit}_mcp_pitch")
            grid, grid_features = _robot_feature_grid(
                model,
                side=side,
                digit=digit,
                samples=257,
            )
            fitted, errors = _fit_feature_grid(
                source_features[digit],
                source_valid[digit],
                grid,
                grid_features,
                motor_commands[:, side_index, motor_index : motor_index + 1],
                feature_weights=np.asarray(
                    [
                        np.sqrt(0.5),
                        0.0,
                        np.sqrt(0.5),
                        np.sqrt(0.4),
                        0.0,
                        np.sqrt(0.4),
                        np.sqrt(0.1),
                    ]
                ),
                lower=lower[motor_index : motor_index + 1],
                upper=upper[motor_index : motor_index + 1],
            )
            motor_commands[:, side_index, motor_index] = fitted[:, 0]
            finite_error = errors[np.isfinite(errors)]
            digit_diagnostics[digit] = {
                "valid_fraction": float(np.mean(source_valid[digit])),
                "mean_feature_error": (
                    float(np.mean(finite_error))
                    if finite_error.size
                    else None
                ),
                "max_feature_error": (
                    float(np.max(finite_error))
                    if finite_error.size
                    else None
                ),
            }
        side_profile.update(
            {
                "source": "25-landmark palm-local hand geometry",
                "motor_limits_rad": {
                    suffix: [float(lower[index]), float(upper[index])]
                    for index, suffix in enumerate(O6_MOTOR_SUFFIXES)
                },
                "fit": digit_diagnostics,
            }
        )
        profiles.append(side_profile)
        diagnostics[side] = {
            "source": "articulated_hand_geometry",
            "digits": digit_diagnostics,
        }

    achieved_widths = _achieved_hand_widths_from_motors(
        model,
        motor_commands,
    )
    return (
        _motor_closure_summary(model, motor_commands),
        motor_commands,
        target_widths,
        achieved_widths,
        profiles,
        diagnostics,
    )


def retarget_o6_hands(
    model: pin.Model,
    hdf5_path: str | Path,
    *,
    scale: float,
) -> O6HandRetargetTrajectory:
    """Retarget EgoDex hand articulation and preserve its action-frame pose.

    The returned transforms make the hand mapping explicit:

    - ``source_action_transforms_reference_camera`` is the source action frame
      in the first camera frame. It is the semantic fingertip TCP for normal
      EgoDex input and the tracked palm for explicit ``rigid_palm`` input.
    - ``wrist_from_action_transforms`` is the corresponding robot action frame
      expressed in the G1 wrist-yaw frame for the retargeted O6 state.

    A body retargeter can therefore place the complete hand with
    ``world_from_wrist = world_from_action @ inv(wrist_from_action)`` instead
    of incorrectly treating the source Hand origin as the robot wrist.
    """

    trajectory = load_hand_trajectory(hdf5_path)
    rigid_palm_mode = trajectory.target_mode == RIGID_PALM_TARGET
    if rigid_palm_mode:
        motor_commands = np.repeat(
            _open_hand_motor_commands(model)[None],
            trajectory.frame_count,
            axis=0,
        )
        diagnostics = {
            side: {
                "source": "rigid_palm_fixed_open",
                "reason": (
                    "rigid palm input has no articulated finger action"
                ),
            }
            for side in ("left", "right")
        }
    else:
        if (
            trajectory.left_articulated_hand is None
            or trajectory.right_articulated_hand is None
        ):
            raise ValueError(
                f"{hdf5_path} must contain complete 25-landmark tracks "
                "for both hands"
            )
        human_widths = np.stack(
            [trajectory.left_widths, trajectory.right_widths],
            axis=1,
        )
        (
            _,
            motor_commands,
            _,
            _,
            _,
            diagnostics,
        ) = _retarget_articulated_hands(
            model,
            trajectory,
            human_widths,
            scale=scale,
        )

    joint_positions = np.empty(
        (trajectory.frame_count, len(HAND_JOINTS)),
        dtype=np.float64,
    )
    for frame_index, commands in enumerate(motor_commands):
        qpos = pin.neutral(model)
        _set_hand_motor_configuration(model, qpos, commands)
        for joint_index, joint_name in enumerate(HAND_JOINTS):
            joint_id = model.getJointId(joint_name)
            joint_positions[frame_index, joint_index] = qpos[
                model.idx_qs[joint_id]
            ]

    action_positions = np.stack(
        [
            trajectory.left_action_reference_camera,
            trajectory.right_action_reference_camera,
        ],
        axis=1,
    )
    action_rotations = np.stack(
        [
            trajectory.left_action_reference_rotations,
            trajectory.right_action_reference_rotations,
        ],
        axis=1,
    )
    source_action_transforms = np.repeat(
        np.eye(4, dtype=np.float64)[None, None],
        trajectory.frame_count,
        axis=0,
    )
    source_action_transforms = np.repeat(
        source_action_transforms,
        2,
        axis=1,
    )
    source_action_transforms[:, :, :3, :3] = action_rotations
    source_action_transforms[:, :, :3, 3] = action_positions

    wrist_from_action = np.repeat(
        np.eye(4, dtype=np.float64)[None, None],
        trajectory.frame_count,
        axis=0,
    )
    wrist_from_action = np.repeat(wrist_from_action, 2, axis=1)
    if rigid_palm_mode:
        wrist_from_action[:, :, :3, 3] = (
            PALM_OFFSET_FROM_WRIST[None, None]
        )
    else:
        tcp_translations, tcp_rotations = _hand_tcp_transforms(
            model,
            motor_commands,
        )
        wrist_from_action[:, :, :3, :3] = tcp_rotations[None]
        wrist_from_action[:, :, :3, 3] = (
            PALM_OFFSET_FROM_WRIST[None, None] + tcp_translations
        )

    if (
        source_action_transforms.shape
        != (trajectory.frame_count, 2, 4, 4)
        or wrist_from_action.shape
        != (trajectory.frame_count, 2, 4, 4)
        or not np.isfinite(source_action_transforms).all()
        or not np.isfinite(wrist_from_action).all()
    ):
        raise ValueError("Retargeted O6 action transforms are invalid")

    return O6HandRetargetTrajectory(
        joint_names=HAND_JOINTS,
        joint_positions=joint_positions,
        action_mode=trajectory.target_mode,
        source_action_transforms_reference_camera=(
            source_action_transforms
        ),
        wrist_from_action_transforms=wrist_from_action,
        diagnostics=diagnostics,
    )


def _fixed_open_hand_action(
    model: pin.Model,
    frame_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
]:
    """Return an explicit fixed-open action when fingers are unobserved."""

    commands = np.zeros((frame_count, 2), dtype=np.float64)
    target_widths = np.full((frame_count, 2), np.nan, dtype=np.float64)
    open_q = pin.neutral(model)
    _set_continuous_hand_configuration(model, open_q, commands[0])
    data = model.createData()
    pin.forwardKinematics(model, data, open_q)
    pin.updateFramePlacements(model, data)
    open_widths = np.asarray(
        [
            _current_hand_jaw_width(model, data, side=side)
            for side in ("left", "right")
        ],
        dtype=np.float64,
    )
    achieved_widths = np.repeat(open_widths[None], frame_count, axis=0)
    profiles = [
        {
            "open_width_m": float(open_widths[side_index]),
            "fixed_command": 0.0,
            "source": "finger articulation unavailable",
        }
        for side_index in range(2)
    ]
    return commands, target_widths, achieved_widths, profiles


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


def _map_action_targets(
    trajectory: HandTrajectory,
    scale: float,
    camera_origin: np.ndarray,
    tcp_translations: np.ndarray,
    tcp_rotations: np.ndarray,
    world_from_reference_camera: np.ndarray = CV_TO_G1_WORLD,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Map the declared action frame to the shared palm IK targets."""

    if tcp_translations.shape != (trajectory.frame_count, 2, 3):
        raise ValueError(
            "TCP translations must have shape "
            f"{(trajectory.frame_count, 2, 3)}, got {tcp_translations.shape}"
        )
    if tcp_rotations.shape != (2, 3, 3):
        raise ValueError(
            f"TCP rotations must have shape (2, 3, 3), got {tcp_rotations.shape}"
        )
    if world_from_reference_camera.shape != (3, 3):
        raise ValueError(
            "world_from_reference_camera must have shape (3, 3)"
        )

    target_mode = getattr(
        trajectory,
        "target_mode",
        FINGERTIP_GRASP_TARGET,
    )
    if target_mode not in {FINGERTIP_GRASP_TARGET, RIGID_PALM_TARGET}:
        raise ValueError(f"Unsupported hand target mode: {target_mode!r}")
    grasp_positions: list[np.ndarray] = []
    grasp_rotations: list[np.ndarray] = []
    palm_positions: list[np.ndarray] = []
    palm_rotations: list[np.ndarray] = []
    for side_index, (positions, rotations) in enumerate(
        (
            (
                trajectory.left_action_reference_camera,
                trajectory.left_action_reference_rotations,
            ),
            (
                trajectory.right_action_reference_camera,
                trajectory.right_action_reference_rotations,
            ),
        )
    ):
        action_position = camera_origin + (
            world_from_reference_camera @ (scale * positions).T
        ).T
        action_rotation = np.einsum(
            "ij,njk->nik",
            world_from_reference_camera,
            rotations,
        )
        if target_mode == RIGID_PALM_TARGET:
            palm_position = action_position
            palm_rotation = action_rotation
            offset_world = np.einsum(
                "nij,nj->ni",
                palm_rotation,
                tcp_translations[:, side_index],
            )
            grasp_position = palm_position + offset_world
            grasp_rotation = np.einsum(
                "nij,jk->nik",
                palm_rotation,
                tcp_rotations[side_index],
            )
        else:
            grasp_position = action_position
            grasp_rotation = action_rotation
            palm_rotation = np.einsum(
                "nij,jk->nik",
                grasp_rotation,
                tcp_rotations[side_index].T,
            )
            offset_world = np.einsum(
                "nij,nj->ni",
                palm_rotation,
                tcp_translations[:, side_index],
            )
            palm_position = grasp_position - offset_world
        grasp_positions.append(grasp_position)
        grasp_rotations.append(grasp_rotation)
        palm_positions.append(palm_position)
        palm_rotations.append(palm_rotation)
    return (
        grasp_positions[0],
        grasp_positions[1],
        grasp_rotations[0],
        grasp_rotations[1],
        palm_positions[0],
        palm_positions[1],
        palm_rotations[0],
        palm_rotations[1],
    )


def _camera_poses(
    trajectory: HandTrajectory,
    scale: float,
    camera_origin: np.ndarray,
    world_from_reference_camera: np.ndarray = CV_TO_G1_WORLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions: list[np.ndarray] = []
    cv_rotations: list[np.ndarray] = []
    gl_quaternions: list[np.ndarray] = []
    cv_to_gl = np.diag([1.0, -1.0, -1.0])
    for motion in trajectory.camera_motion:
        position = camera_origin + world_from_reference_camera @ (
            scale * motion[:3, 3]
        )
        world_from_cv = world_from_reference_camera @ motion[:3, :3]
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


def _representative_keyframes(
    left_targets: np.ndarray,
    right_targets: np.ndarray,
    left_tracked: np.ndarray,
    right_tracked: np.ndarray,
) -> np.ndarray:
    """Select the frame set K from per-hand spatial trajectory extremes."""

    frame_ids: set[int] = set()
    for targets, tracked in (
        (left_targets, left_tracked),
        (right_targets, right_tracked),
    ):
        valid_ids = np.flatnonzero(
            tracked & np.isfinite(targets).all(axis=1)
        )
        if valid_ids.size == 0:
            continue
        valid_targets = targets[valid_ids]
        for axis in range(3):
            frame_ids.add(
                int(valid_ids[int(np.argmin(valid_targets[:, axis]))])
            )
            frame_ids.add(
                int(valid_ids[int(np.argmax(valid_targets[:, axis]))])
            )
    if not frame_ids:
        raise ValueError("No tracked end-effector poses for base search")
    return np.asarray(sorted(frame_ids), dtype=np.int64)


def _keyframe_target_positions(
    keyframes: np.ndarray,
    left_targets: np.ndarray,
    right_targets: np.ndarray,
    left_tracked: np.ndarray,
    right_tracked: np.ndarray,
) -> np.ndarray:
    positions = [
        targets[frame_id]
        for frame_id in keyframes
        for targets, tracked in (
            (left_targets, left_tracked),
            (right_targets, right_tracked),
        )
        if tracked[frame_id]
    ]
    if not positions:
        raise ValueError("Representative keyframes contain no active targets")
    return np.asarray(positions, dtype=np.float64)


def _trajectory_centroid(
    left_targets: np.ndarray,
    right_targets: np.ndarray,
    left_tracked: np.ndarray,
    right_tracked: np.ndarray,
) -> np.ndarray:
    """Average per-frame active-EEF centers over the complete trajectory."""

    centers: list[np.ndarray] = []
    for frame_id in range(left_targets.shape[0]):
        active = [
            targets[frame_id]
            for targets, tracked in (
                (left_targets, left_tracked),
                (right_targets, right_tracked),
            )
            if tracked[frame_id]
            and np.isfinite(targets[frame_id]).all()
        ]
        if active:
            centers.append(np.mean(active, axis=0))
    if not centers:
        raise ValueError("No tracked end-effector poses for trajectory centroid")
    return np.mean(centers, axis=0)


def _robot_maximum_reach(model: pin.Model) -> float:
    """Return the morphology-specific base-to-palm kinematic reach bound."""

    reaches: list[float] = []
    for frame_name in PALM_FRAMES:
        frame = model.frames[model.getFrameId(frame_name)]
        joint_id = frame.parent
        reach = float(np.linalg.norm(frame.placement.translation))
        while joint_id > 0:
            reach += float(
                np.linalg.norm(
                    model.jointPlacements[joint_id].translation
                )
            )
            joint_id = model.parents[joint_id]
        reaches.append(reach)
    if not reaches or not np.isfinite(reaches).all():
        raise ValueError("Could not derive a finite robot kinematic reach")
    return float(max(reaches))


def _robot_nominal_pelvis_height(model: pin.Model) -> float:
    """Return the upright pelvis height that puts both feet on z=0."""

    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    support_heights: list[float] = []
    for frame_name in G1_SUPPORT_FRAMES:
        frame_id = model.getFrameId(frame_name)
        if frame_id >= model.nframes:
            raise ValueError(
                f"G1 URDF is missing support frame {frame_name!r}"
            )
        support_heights.append(float(data.oMf[frame_id].translation[2]))
    pelvis_height = -min(support_heights)
    if not np.isfinite(pelvis_height) or pelvis_height <= 0.0:
        raise ValueError(
            "Could not derive a positive neutral pelvis height from "
            f"{G1_SUPPORT_FRAMES}"
        )
    return pelvis_height


def _robot_arm_lengths(model: pin.Model) -> np.ndarray:
    """Return shoulder-to-palm kinematic chain lengths for both G1 arms."""

    lengths: list[float] = []
    for side, shoulder_name in zip(
        ("left", "right"),
        G1_SHOULDER_JOINTS,
        strict=True,
    ):
        shoulder_id = model.getJointId(shoulder_name)
        if shoulder_id >= model.njoints:
            raise ValueError(f"G1 URDF is missing joint {shoulder_name!r}")
        palm_id = model.getFrameId(f"{side}_palm")
        if palm_id >= model.nframes:
            raise ValueError(f"G1 URDF is missing {side!r} palm frame")
        palm = model.frames[palm_id]
        length = float(np.linalg.norm(palm.placement.translation))
        joint_id = palm.parent
        while joint_id != shoulder_id and joint_id > 0:
            length += float(
                np.linalg.norm(model.jointPlacements[joint_id].translation)
            )
            joint_id = model.parents[joint_id]
        if joint_id != shoulder_id or not np.isfinite(length) or length <= 0.0:
            raise ValueError(
                f"Could not derive the {side} shoulder-to-palm chain length"
            )
        lengths.append(length)
    return np.asarray(lengths, dtype=np.float64)


def _robot_shoulder_center(model: pin.Model) -> np.ndarray:
    """Return the neutral shoulder midpoint in the pelvis frame."""

    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    centers: list[np.ndarray] = []
    for joint_name in G1_SHOULDER_JOINTS:
        joint_id = model.getJointId(joint_name)
        if joint_id >= model.njoints:
            raise ValueError(f"G1 URDF is missing joint {joint_name!r}")
        centers.append(
            np.asarray(data.oMi[joint_id].translation, dtype=np.float64)
        )
    shoulder_center = np.mean(centers, axis=0)
    if shoulder_center.shape != (3,) or not np.isfinite(
        shoulder_center
    ).all():
        raise ValueError("G1 neutral shoulder center is invalid")
    return shoulder_center


def _episode_reference_camera_rotation(
    trajectory: HandTrajectory,
) -> np.ndarray:
    """Level the source world while retaining the episode camera attitude."""

    body = trajectory.body_alignment
    if body is None:
        raise ValueError("Episode body alignment is unavailable")
    source_from_reference_camera = Rotation.from_matrix(
        np.asarray(
            trajectory.camera_to_world[0, :3, :3],
            dtype=np.float64,
        )
    ).as_matrix()
    source_up = np.asarray(body.world_up, dtype=np.float64)
    if (
        source_from_reference_camera.shape != (3, 3)
        or source_up.shape != (3,)
        or not np.isfinite(source_from_reference_camera).all()
        or not np.isfinite(source_up).all()
        or not np.isclose(np.linalg.norm(source_up), 1.0, atol=1e-8)
    ):
        raise ValueError("Episode camera rotation or world-up axis is invalid")

    source_forward = source_from_reference_camera[:, 2]
    level_forward = source_forward - source_up * float(
        np.dot(source_up, source_forward)
    )
    if np.linalg.norm(level_forward) <= 1e-6:
        source_right = source_from_reference_camera[:, 0]
        level_right = source_right - source_up * float(
            np.dot(source_up, source_right)
        )
        if np.linalg.norm(level_right) <= 1e-6:
            raise ValueError(
                "Episode camera axes are degenerate relative to world up"
            )
        level_right /= np.linalg.norm(level_right)
        level_forward = np.cross(level_right, source_up)
    level_forward /= np.linalg.norm(level_forward)
    source_left = np.cross(source_up, level_forward)
    source_left /= np.linalg.norm(source_left)
    level_forward = np.cross(source_left, source_up)
    source_basis = np.column_stack(
        [level_forward, source_left, source_up]
    )
    aligned_from_source = source_basis.T
    world_from_reference_camera = (
        aligned_from_source @ source_from_reference_camera
    )
    if not np.allclose(
        world_from_reference_camera.T @ world_from_reference_camera,
        np.eye(3),
        rtol=0.0,
        atol=1e-8,
    ) or not np.isclose(
        np.linalg.det(world_from_reference_camera),
        1.0,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("Derived episode camera rotation is not in SO(3)")
    return world_from_reference_camera


def _derive_spatial_alignment(
    model: pin.Model,
    trajectory: HandTrajectory,
    *,
    nominal_pelvis_height: float,
    scale: float | None,
    camera_origin: np.ndarray | None,
) -> _SpatialAlignment:
    """Align an episode body rig to G1 before retargeting its hands."""

    body = trajectory.body_alignment
    if body is None and (scale is None or camera_origin is None):
        raise ValueError(
            "HDF5 lacks the upper-body anchors required for spatial "
            "alignment (hip, left/right Arm, Forearm, and Hand). Provide "
            "an explicit scale and camera origin only for a pre-calibrated "
            "adapter."
        )
    robot_arm_lengths = _robot_arm_lengths(model)
    source_arm_lengths = (
        np.asarray(body.arm_lengths, dtype=np.float64)
        if body is not None
        else np.full(2, np.nan, dtype=np.float64)
    )
    if scale is None:
        if (
            source_arm_lengths.shape != (2,)
            or not np.isfinite(source_arm_lengths).all()
            or np.any(source_arm_lengths <= 0.0)
        ):
            raise ValueError("Episode source arm lengths are invalid")
        scale = float(np.median(robot_arm_lengths / source_arm_lengths))
        scale_source = (
            "median G1 shoulder-to-palm chain length divided by episode "
            f"shoulder-to-hand chain length ({body.source})"
        )
    else:
        scale = float(scale)
        scale_source = "explicit caller override"
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Spatial scale must be positive and finite")
    robot_shoulder_center = _robot_shoulder_center(model)

    if body is None:
        world_from_reference_camera = CV_TO_G1_WORLD.copy()
        pelvis_anchor = np.full(3, np.nan, dtype=np.float64)
        shoulder_anchor = np.full(3, np.nan, dtype=np.float64)
    else:
        world_from_reference_camera = _episode_reference_camera_rotation(
            trajectory
        )
        pelvis_samples = np.asarray(
            body.pelvis_reference_camera,
            dtype=np.float64,
        )
        valid = np.isfinite(pelvis_samples).all(axis=1)
        if np.count_nonzero(valid) < 2:
            raise ValueError("Episode pelvis anchor has fewer than two samples")
        pelvis_anchor = np.median(pelvis_samples[valid], axis=0)
        shoulder_samples = np.asarray(
            body.shoulder_center_reference_camera,
            dtype=np.float64,
        )
        if (
            shoulder_samples.ndim != 2
            or shoulder_samples.shape[1] != 3
        ):
            raise ValueError(
                "Episode shoulder-center anchor must have shape [N, 3]"
            )
        valid = np.isfinite(shoulder_samples).all(axis=1)
        if np.count_nonzero(valid) < 2:
            raise ValueError(
                "Episode shoulder-center anchor has fewer than two samples"
            )
        shoulder_anchor = np.median(shoulder_samples[valid], axis=0)

    if camera_origin is None:
        target_pelvis = np.asarray(
            [0.0, 0.0, nominal_pelvis_height],
            dtype=np.float64,
        )
        camera_origin = target_pelvis - world_from_reference_camera @ (
            scale * pelvis_anchor
        )
    else:
        camera_origin = np.asarray(camera_origin, dtype=np.float64)
    if camera_origin.shape != (3,) or not np.isfinite(camera_origin).all():
        raise ValueError("Camera origin must be a finite 3-vector")

    return _SpatialAlignment(
        scale=scale,
        scale_source=scale_source,
        source_arm_lengths=source_arm_lengths,
        robot_arm_lengths=robot_arm_lengths,
        pelvis_anchor_reference_camera=pelvis_anchor,
        shoulder_anchor_reference_camera=shoulder_anchor,
        robot_shoulder_center_from_pelvis=robot_shoulder_center,
        camera_origin=camera_origin,
        world_from_reference_camera=world_from_reference_camera,
    )


def _yaw_rotation(yaw: float) -> np.ndarray:
    return Rotation.from_euler("z", yaw).as_matrix()


def _shoulder_aligned_base_candidates(
    mapped_shoulder_center: np.ndarray,
    robot_shoulder_center_from_pelvis: np.ndarray,
    *,
    trajectory_centroid: np.ndarray,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Derive pelvis translations that align the two shoulder centers.

    For every candidate yaw ``R``, the pelvis translation is solved exactly:

    ``p_pelvis = p_source_shoulder - R @ p_robot_shoulder_from_pelvis``.

    Translation is therefore not an independent search variable.  This avoids
    transferring the source hip-to-shoulder length to a robot with a different
    torso morphology.
    """

    for name, value in (
        ("mapped_shoulder_center", mapped_shoulder_center),
        (
            "robot_shoulder_center_from_pelvis",
            robot_shoulder_center_from_pelvis,
        ),
        ("trajectory_centroid", trajectory_centroid),
    ):
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite 3-vector")

    direction = trajectory_centroid[:2] - mapped_shoulder_center[:2]
    facing_yaw = (
        float(np.arctan2(direction[1], direction[0]))
        if np.linalg.norm(direction) > 1e-9
        else 0.0
    )
    for yaw_offset in BASE_YAW_OFFSETS_RAD:
        rotation = _yaw_rotation(facing_yaw + float(yaw_offset))
        position = (
            mapped_shoulder_center
            - rotation @ robot_shoulder_center_from_pelvis
        )
        yield position, rotation


def _base_candidates(
    keyframe_target_positions: np.ndarray,
    *,
    trajectory_centroid: np.ndarray,
    r_max: float,
    base_height: float | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Grid upright pelvis poses around the trajectory centroid within r_max.

    ``base_height`` fixes the humanoid root height so frozen neutral legs stay
    grounded. Leaving it unset retains the generic three-axis helper used by
    unit tests and non-grounded morphology experiments.
    """

    if (
        keyframe_target_positions.ndim != 2
        or keyframe_target_positions.shape[1] != 3
    ):
        raise ValueError("keyframe_target_positions must have shape [N, 3]")
    if not np.isfinite(keyframe_target_positions).all():
        raise ValueError("keyframe_target_positions must be finite")
    if trajectory_centroid.shape != (3,) or not np.isfinite(
        trajectory_centroid
    ).all():
        raise ValueError("trajectory_centroid must be a finite 3-vector")
    if not np.isfinite(r_max) or r_max <= 0.0:
        raise ValueError("r_max must be positive and finite")
    if base_height is not None and (
        not np.isfinite(base_height) or base_height <= 0.0
    ):
        raise ValueError("base_height must be positive and finite")

    centroid = trajectory_centroid
    lower = np.max(keyframe_target_positions - r_max, axis=0)
    upper = np.min(keyframe_target_positions + r_max, axis=0)
    grounded_height_in_reach = (
        base_height is None
        or lower[2] - 1e-9 <= base_height <= upper[2] + 1e-9
    )
    missing_overlap = np.any(lower[:2] > upper[:2]) or (
        base_height is None and lower[2] > upper[2]
    )
    if missing_overlap or not grounded_height_in_reach:
        raise NoCommonBasePlacementError(
            "No grounded pelvis placement can cover all representative "
            f"trajectory extremes within r_max={r_max:.4f} m"
        )

    axes: list[np.ndarray] = []
    for axis in range(3):
        if axis == 2 and base_height is not None:
            axes.append(np.asarray([base_height], dtype=np.float64))
            continue
        values = np.linspace(
            lower[axis],
            upper[axis],
            BASE_GRID_POINTS_PER_AXIS,
            dtype=np.float64,
        )
        values = np.unique(
            np.append(values, np.clip(centroid[axis], lower[axis], upper[axis]))
        )
        axes.append(
            values[
                np.argsort(
                    np.abs(values - centroid[axis]),
                    kind="stable",
                )
            ]
        )

    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                position = np.asarray([x, y, z], dtype=np.float64)
                if np.any(
                    np.linalg.norm(
                        keyframe_target_positions - position,
                        axis=1,
                    )
                    > r_max + 1e-9
                ):
                    continue
                direction = centroid[:2] - position[:2]
                facing_yaw = (
                    float(np.arctan2(direction[1], direction[0]))
                    if np.linalg.norm(direction) > 1e-9
                    else 0.0
                )
                for yaw_offset in BASE_YAW_OFFSETS_RAD:
                    yield position.copy(), _yaw_rotation(
                        facing_yaw + float(yaw_offset)
                    )


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
    base_rotation: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    left_rotation: np.ndarray | None,
    right_rotation: np.ndarray | None,
    left_active: bool,
    right_active: bool,
    active_indices: np.ndarray,
    iterations: int,
    orientation_weight: float = IK_ORIENTATION_WEIGHT,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    frame_ids = tuple(model.getFrameId(name) for name in PALM_FRAMES)
    posture_weights = _posture_weights(model, active_indices)
    if not np.isfinite(orientation_weight) or orientation_weight < 0.0:
        raise ValueError("orientation_weight must be finite and non-negative")
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
            target_in_base = base_rotation.T @ (
                target - base_position
            )
            jacobian = pin.computeFrameJacobian(
                model,
                data,
                q,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:, active_indices]
            position_error = target_in_base - placement.translation
            if target_rotation is None:
                jacobians.append(jacobian[:3])
                errors.append(position_error)
            else:
                rotation_in_base = base_rotation.T @ target_rotation
                orientation_error = Rotation.from_matrix(
                    rotation_in_base @ placement.rotation.T
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
        target_in_base = base_rotation.T @ (target - base_position)
        metrics.append(
            float(
                np.linalg.norm(
                    placement.translation - target_in_base
                )
            )
            if active
            else float("nan")
        )
        metrics.append(
            float(
                Rotation.from_matrix(
                    placement.rotation.T
                    @ (base_rotation.T @ target_rotation)
                ).magnitude()
            )
            if active and target_rotation is not None
            else float("nan")
        )
    return q, (metrics[0], metrics[2], metrics[1], metrics[3])


def _evaluate_candidate(
    model: pin.Model,
    keyframes: np.ndarray,
    base_position: np.ndarray,
    base_rotation: np.ndarray,
    left_targets: np.ndarray,
    right_targets: np.ndarray,
    left_rotations: np.ndarray,
    right_rotations: np.ndarray,
    left_tracked: np.ndarray,
    right_tracked: np.ndarray,
) -> dict[str, object]:
    data = model.createData()
    position_indices = _active_velocity_indices(model, position_only=True)
    pose_indices = _active_velocity_indices(model, position_only=False)
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    feasibility_violations: list[float] = []
    feasible_keyframes = 0
    evaluated_keyframes = 0
    feasible_targets = 0
    evaluated_targets = 0
    for frame_id in keyframes:
        left_active = bool(left_tracked[frame_id])
        right_active = bool(right_tracked[frame_id])
        if not left_active and not right_active:
            continue
        evaluated_keyframes += 1
        q, _ = _solve_frame(
            model,
            data,
            pin.neutral(model),
            base_position=base_position,
            base_rotation=base_rotation,
            left_target=left_targets[frame_id],
            right_target=right_targets[frame_id],
            left_rotation=None,
            right_rotation=None,
            left_active=left_active,
            right_active=right_active,
            active_indices=position_indices,
            iterations=60,
        )
        q, metrics = _solve_frame(
            model,
            data,
            q,
            base_position=base_position,
            base_rotation=base_rotation,
            left_target=left_targets[frame_id],
            right_target=right_targets[frame_id],
            left_rotation=left_rotations[frame_id],
            right_rotation=right_rotations[frame_id],
            left_active=left_active,
            right_active=right_active,
            active_indices=pose_indices,
            iterations=100,
        )
        frame_feasible = True
        for active, position_error, orientation_error in (
            (left_active, metrics[0], metrics[2]),
            (right_active, metrics[1], metrics[3]),
        ):
            if active:
                position_errors.append(position_error)
                orientation_errors.append(orientation_error)
                target_feasible = (
                    position_error <= IK_FEASIBLE_POSITION_ERROR_M
                    and orientation_error
                    <= IK_FEASIBLE_ORIENTATION_ERROR_RAD
                )
                frame_feasible = frame_feasible and target_feasible
                feasible_targets += int(target_feasible)
                evaluated_targets += 1
                feasibility_violations.append(
                    max(
                        position_error / IK_FEASIBLE_POSITION_ERROR_M - 1.0,
                        0.0,
                    )
                    + max(
                        orientation_error
                        / IK_FEASIBLE_ORIENTATION_ERROR_RAD
                        - 1.0,
                        0.0,
                    )
                )
        feasible_keyframes += int(frame_feasible)
    if evaluated_keyframes == 0:
        raise ValueError("No active representative keyframes to evaluate")
    feasible_fraction = feasible_keyframes / evaluated_keyframes
    feasible_target_fraction = (
        feasible_targets / evaluated_targets
        if evaluated_targets
        else 0.0
    )
    yaw = float(
        np.arctan2(base_rotation[1, 0], base_rotation[0, 0])
    )
    mean_position_error = (
        float(np.mean(position_errors))
        if position_errors
        else float("inf")
    )
    mean_orientation_error = (
        float(np.mean(orientation_errors))
        if orientation_errors
        else float("inf")
    )
    return {
        "base_position": base_position.tolist(),
        "base_rotation": base_rotation.tolist(),
        "base_yaw_rad": yaw,
        "feasible_keyframes": feasible_keyframes,
        "keyframe_count": evaluated_keyframes,
        "feasible_fraction": float(feasible_fraction),
        "feasible_targets": feasible_targets,
        "target_count": evaluated_targets,
        "feasible_target_fraction": float(feasible_target_fraction),
        "mean_feasibility_violation": float(
            np.mean(feasibility_violations)
        ),
        "mean_position_error_m": mean_position_error,
        "mean_orientation_error_rad": mean_orientation_error,
        # Eq. (3) optimizes only the keyframe feasibility average. Independent
        # target feasibility and normalized violations make ties deterministic
        # without letting sub-millimetre position differences select a
        # physically nonsensical pelvis pose.
        "score": float(feasible_fraction),
    }


def retarget_with_pinocchio(
    model: pin.Model,
    trajectory: HandTrajectory,
    *,
    source_width: int,
    source_height: int,
    camera_origin: np.ndarray | None = None,
    scale: float | None = None,
) -> PinocchioRetargetResult:
    nominal_pelvis_height = _robot_nominal_pelvis_height(model)
    spatial_alignment = _derive_spatial_alignment(
        model,
        trajectory,
        nominal_pelvis_height=nominal_pelvis_height,
        scale=scale,
        camera_origin=camera_origin,
    )
    scale = spatial_alignment.scale
    camera_origin = spatial_alignment.camera_origin
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
    human_widths = np.stack(
        [trajectory.left_widths, trajectory.right_widths],
        axis=1,
    )

    rigid_palm_mode = trajectory.target_mode == RIGID_PALM_TARGET
    if rigid_palm_mode:
        (
            hand_closures,
            target_hand_widths,
            achieved_hand_widths,
            hand_aperture_profiles,
        ) = _fixed_open_hand_action(model, trajectory.frame_count)
        hand_motor_commands = _closures_to_motor_commands(
            model,
            hand_closures,
        )
        hand_action_mode = "rigid_palm_fixed_open"
        hand_action_diagnostics = {}
    elif (
        trajectory.left_articulated_hand is not None
        or trajectory.right_articulated_hand is not None
    ):
        (
            hand_closures,
            hand_motor_commands,
            target_hand_widths,
            achieved_hand_widths,
            hand_aperture_profiles,
            hand_action_diagnostics,
        ) = _retarget_articulated_hands(
            model,
            trajectory,
            human_widths,
            scale=scale,
        )
        hand_action_mode = "articulated_o6"
    else:
        (
            hand_closures,
            target_hand_widths,
            achieved_hand_widths,
            hand_aperture_profiles,
        ) = _retarget_hand_widths(
            model,
            human_widths,
            scale=scale,
        )
        hand_motor_commands = _closures_to_motor_commands(
            model,
            hand_closures,
        )
        hand_action_mode = "continuous_width_fallback"
        hand_action_diagnostics = {
            side: {
                "source": "continuous_width_fallback",
                "reason": "complete 25-landmark hand track unavailable",
            }
            for side in ("left", "right")
        }

    # All input modes produce the same palm target arrays consumed by the same
    # pelvis-pose search and 6D IK. Fingertip actions are converted through the
    # robot TCP; explicit rigid-palm inputs map directly to the palm frame.
    hand_tcp_translations, hand_tcp_rotations = _hand_tcp_transforms(
        model,
        hand_motor_commands,
    )
    (
        left_targets,
        right_targets,
        left_rotations,
        right_rotations,
        left_palm_targets,
        right_palm_targets,
        left_palm_rotations,
        right_palm_rotations,
    ) = _map_action_targets(
        trajectory,
        scale,
        camera_origin,
        hand_tcp_translations,
        hand_tcp_rotations,
        world_from_reference_camera=(
            spatial_alignment.world_from_reference_camera
        ),
    )

    keyframes = _representative_keyframes(
        left_targets,
        right_targets,
        left_tracked,
        right_tracked,
    )
    centroid = _trajectory_centroid(
        left_targets,
        right_targets,
        left_tracked,
        right_tracked,
    )
    r_max = _robot_maximum_reach(model)
    if not rigid_palm_mode:
        r_max += float(
            np.max(
                np.linalg.norm(
                    hand_tcp_translations,
                    axis=2,
                )
            )
        )
    shoulder_anchor = (
        spatial_alignment.shoulder_anchor_reference_camera
    )
    if np.isfinite(shoulder_anchor).all():
        mapped_shoulder_center = camera_origin + (
            spatial_alignment.world_from_reference_camera
            @ (scale * shoulder_anchor)
        )
        candidates = list(
            _shoulder_aligned_base_candidates(
                mapped_shoulder_center,
                spatial_alignment.robot_shoulder_center_from_pelvis,
                trajectory_centroid=centroid,
            )
        )
    else:
        mapped_shoulder_center = np.full(3, np.nan, dtype=np.float64)
        keyframe_positions = _keyframe_target_positions(
            keyframes,
            left_targets,
            right_targets,
            left_tracked,
            right_tracked,
        )
        candidates = list(
            _base_candidates(
                keyframe_positions,
                trajectory_centroid=centroid,
                r_max=r_max,
                base_height=nominal_pelvis_height,
            )
        )
    if not candidates:
        raise RuntimeError(
            "Pelvis-pose grid contains no candidate within the G1 reach"
        )
    search_results = [
        _evaluate_candidate(
            model,
            keyframes,
            base_position,
            base_rotation,
            left_palm_targets,
            right_palm_targets,
            left_palm_rotations,
            right_palm_rotations,
            left_tracked,
            right_tracked,
        )
        for base_position, base_rotation in candidates
    ]
    search_results.sort(
        key=lambda item: (
            -float(item["feasible_fraction"]),
            -float(item["feasible_target_fraction"]),
            float(item["mean_feasibility_violation"]),
            float(item["mean_position_error_m"]),
            float(item["mean_orientation_error_rad"]),
        )
    )
    best = search_results[0]
    if float(best["feasible_fraction"]) == 0.0:
        print(
            "Warning: no pelvis candidate makes a complete representative "
            "keyframe feasible; selecting the candidate with the best "
            "independent end-effector feasibility and normalized residual",
            file=sys.stderr,
        )
    base_position = np.asarray(best["base_position"], dtype=np.float64)
    base_rotation = np.asarray(best["base_rotation"], dtype=np.float64)

    q = pin.neutral(model)
    data = model.createData()
    position_indices = _active_velocity_indices(model, position_only=True)
    q, _ = _solve_frame(
        model,
        data,
        q,
        base_position=base_position,
        base_rotation=base_rotation,
        left_target=left_palm_targets[0],
        right_target=right_palm_targets[0],
        left_rotation=None,
        right_rotation=None,
        left_active=bool(left_tracked[0]),
        right_active=bool(right_tracked[0]),
        active_indices=position_indices,
        iterations=100,
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
            base_rotation=base_rotation,
            left_target=left_palm_targets[frame_id],
            right_target=right_palm_targets[frame_id],
            left_rotation=left_palm_rotations[frame_id],
            right_rotation=right_palm_rotations[frame_id],
            left_active=bool(left_tracked[frame_id]),
            right_active=bool(right_tracked[frame_id]),
            active_indices=active_indices,
            iterations=60 if frame_id == 0 else 20,
        )
        _set_hand_motor_configuration(
            model,
            q,
            hand_motor_commands[frame_id],
        )
        qpos[frame_id] = q
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        tcp_metrics: list[float] = []
        for side_index, (
            side,
            target,
            target_rotation,
            active,
        ) in enumerate(
            (
                (
                    "left",
                    left_targets[frame_id],
                    left_rotations[frame_id],
                    left_tracked[frame_id],
                ),
                (
                    "right",
                    right_targets[frame_id],
                    right_rotations[frame_id],
                    right_tracked[frame_id],
                ),
            )
        ):
            palm = data.oMf[model.getFrameId(f"{side}_palm")]
            if rigid_palm_mode:
                actual_position = (
                    base_position
                    + base_rotation @ palm.translation
                )
                actual_rotation = base_rotation @ palm.rotation
                metric_target = (
                    left_palm_targets[frame_id]
                    if side == "left"
                    else right_palm_targets[frame_id]
                )
                metric_rotation = (
                    left_palm_rotations[frame_id]
                    if side == "left"
                    else right_palm_rotations[frame_id]
                )
            else:
                actual_position = (
                    base_position
                    + base_rotation
                    @ (
                        palm.translation
                        + palm.rotation
                        @ hand_tcp_translations[frame_id, side_index]
                    )
                )
                actual_rotation = (
                    base_rotation
                    @ palm.rotation
                    @ hand_tcp_rotations[side_index]
                )
                metric_target = target
                metric_rotation = target_rotation
            tcp_metrics.append(
                float(np.linalg.norm(actual_position - metric_target))
                if active
                else float("nan")
            )
            tcp_metrics.append(
                float(
                    Rotation.from_matrix(
                        actual_rotation.T @ metric_rotation
                    ).magnitude()
                )
                if active
                else float("nan")
            )
        metrics = (
            tcp_metrics[0],
            tcp_metrics[2],
            tcp_metrics[1],
            tcp_metrics[3],
        )
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
        left_palm_targets=left_palm_targets,
        right_palm_targets=right_palm_targets,
        left_palm_target_rotations=left_palm_rotations,
        right_palm_target_rotations=right_palm_rotations,
        left_errors=left_errors,
        right_errors=right_errors,
        left_orientation_errors=left_orientation_errors,
        right_orientation_errors=right_orientation_errors,
        left_visible=left_visible,
        right_visible=right_visible,
        hand_closures=hand_closures,
        hand_motor_commands=hand_motor_commands,
        target_hand_widths=target_hand_widths,
        achieved_hand_widths=achieved_hand_widths,
        hand_aperture_profiles=hand_aperture_profiles,
        hand_action_mode=hand_action_mode,
        hand_action_diagnostics=hand_action_diagnostics,
        hand_tcp_translations=hand_tcp_translations,
        hand_tcp_rotations=hand_tcp_rotations,
        base_frame=G1_BASE_FRAME,
        nominal_pelvis_height=nominal_pelvis_height,
        base_position=base_position,
        base_rotation=base_rotation,
        base_search_keyframes=keyframes,
        base_search_centroid=centroid,
        maximum_reach=r_max,
        scale=scale,
        scale_source=spatial_alignment.scale_source,
        source_arm_lengths=spatial_alignment.source_arm_lengths,
        robot_arm_lengths=spatial_alignment.robot_arm_lengths,
        pelvis_anchor_reference_camera=(
            spatial_alignment.pelvis_anchor_reference_camera
        ),
        shoulder_anchor_reference_camera=(
            spatial_alignment.shoulder_anchor_reference_camera
        ),
        robot_shoulder_center_from_pelvis=(
            spatial_alignment.robot_shoulder_center_from_pelvis
        ),
        mapped_shoulder_center=mapped_shoulder_center,
        camera_origin=spatial_alignment.camera_origin,
        world_from_reference_camera=(
            spatial_alignment.world_from_reference_camera
        ),
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
            result.camera_origin,
            result.world_from_reference_camera,
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
    rigid_palm_mode = result.hand_action_mode == "rigid_palm_fixed_open"
    articulated_mode = result.hand_action_mode == "articulated_o6"
    achieved_width_range = {
        side: [
            float(np.min(result.achieved_hand_widths[:, side_index])),
            float(np.max(result.achieved_hand_widths[:, side_index])),
        ]
        for side_index, side in enumerate(("left", "right"))
    }
    command_range = {
        side: [
            float(np.min(result.hand_closures[:, side_index])),
            float(np.max(result.hand_closures[:, side_index])),
        ]
        for side_index, side in enumerate(("left", "right"))
    }
    motor_command_range = {
        side: {
            suffix: [
                float(
                    np.min(
                        result.hand_motor_commands[
                            :, side_index, motor_index
                        ]
                    )
                ),
                float(
                    np.max(
                        result.hand_motor_commands[
                            :, side_index, motor_index
                        ]
                    )
                ),
            ]
            for motor_index, suffix in enumerate(O6_MOTOR_SUFFIXES)
        }
        for side_index, side in enumerate(("left", "right"))
    }
    if rigid_palm_mode:
        ik_target_source = (
            "HDF5 leftHand/rightHand rigid pose mapped directly to G1 palm"
        )
        hand_action_source = (
            "fixed-open robot hands; source finger articulation unavailable"
        )
        hand_action_report: dict[str, object] = {
            "type": result.hand_action_mode,
            "shape": [trajectory.frame_count, 2],
            "order": ["left", "right"],
            "formula": (
                "source rigid Hand position and orientation map directly to "
                "the robot palm target"
            ),
            "closure_semantics": (
                "both LinkerHand O6 commands remain exactly 0=open"
            ),
            "source": list(trajectory.hand_sources),
            "human_width_range_m": None,
            "target_width_range_m": None,
            "achieved_width_range_m": achieved_width_range,
            "command_range": command_range,
            "motor_command_range_rad": motor_command_range,
            "motor_order": list(O6_MOTOR_SUFFIXES),
            "aperture_profiles": {
                side: result.hand_aperture_profiles[side_index]
                for side_index, side in enumerate(("left", "right"))
            },
            "tcp_mapping": (
                "not used to construct IK targets; the open-hand TCP pose is "
                "derived from the solved palm only for trajectory metadata"
            ),
        }
    else:
        ik_target_source = (
            "fingertip p,R with state-dependent LinkerHand O6 jaw/TCP"
        )
        hand_action_source = (
            "25-landmark palm-local geometry retargeted to independent "
            "LinkerHand O6 motors"
            if articulated_mode
            else "continuous fingertip p,R,w width fallback"
        )
        hand_action_report = {
            "type": result.hand_action_mode,
            "shape": (
                [trajectory.frame_count, 2, O6_MOTOR_COUNT]
                if articulated_mode
                else [trajectory.frame_count, 2]
            ),
            "order": ["left", "right"],
            "motor_order": list(O6_MOTOR_SUFFIXES),
            "formula": (
                "p,R retain the fingertip-derived grasp TCP; each hand's "
                "25 landmarks are converted to a palm-anatomical frame; "
                "thumb fits yaw+pitch and index/middle/ring/pinky fit "
                "independent flexion through O6 forward-kinematics LUTs"
                if articulated_mode
                else
                "p=0.5*(thumb+0.7*index+0.3*middle); "
                "R=[x y z] from the jaw/wrist axes; "
                "w=||thumb-(0.7*index+0.3*middle)||_2"
            ),
            "closure_semantics": (
                "six independent motor angles per hand in radians; "
                "URDF mimic joints are expanded after fitting"
                if articulated_mode
                else
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
            "achieved_width_range_m": achieved_width_range,
            "command_range": command_range,
            "motor_command_range_rad": motor_command_range,
            "aperture_profiles": {
                side: result.hand_aperture_profiles[side_index]
                for side_index, side in enumerate(("left", "right"))
            },
            "tcp_mapping": (
                "p,R define the robot grasp TCP target; the TCP origin "
                "follows the current O6 thumb/virtual-finger midpoint, and "
                "its orientation uses the fixed open-hand canonical gripper "
                "basis"
            ),
        }
        hand_action_report["target_width_range_m"] = {
            side: [
                float(
                    np.min(result.target_hand_widths[:, side_index])
                ),
                float(
                    np.max(result.target_hand_widths[:, side_index])
                ),
            ]
            for side_index, side in enumerate(("left", "right"))
        }
        hand_action_report["width_role"] = (
            "diagnostic only; articulated O6 motors come from full "
            "hand geometry"
            if articulated_mode
            else "continuous scalar actuator target"
        )
        hand_action_report["diagnostics"] = result.hand_action_diagnostics
    shoulder_aligned = np.isfinite(result.mapped_shoulder_center).all()
    resolved_shoulder_center = (
        result.base_position
        + result.base_rotation
        @ result.robot_shoulder_center_from_pelvis
    )
    shoulder_alignment_error = (
        float(
            np.linalg.norm(
                resolved_shoulder_center - result.mapped_shoulder_center
            )
        )
        if shoulder_aligned
        else float("nan")
    )
    base_placement_mode = (
        "shoulder_center_aligned_yaw_search"
        if shoulder_aligned
        else "grounded_xyz_yaw_search"
    )
    report: dict[str, object] = {
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "ik_solver": f"Pinocchio {pin.__version__}",
        "ik_target_source": ik_target_source,
        "hand_target_mode": trajectory.target_mode,
        "hand_action_source": hand_action_source,
        "frame_count": trajectory.frame_count,
        "joint_count": model.nq,
        "joint_names": list(model.names[1:]),
        "link_count": len(link_names),
        "base_position": result.base_position.tolist(),
        "base_rotation": result.base_rotation.tolist(),
        "base_yaw_rad": float(
            np.arctan2(
                result.base_rotation[1, 0],
                result.base_rotation[0, 0],
            )
        ),
        "base_placement": {
            "mode": base_placement_mode,
            "base_frame": result.base_frame,
            "transform_semantics": "world_from_pelvis",
            "candidate_translation_axes": (
                []
                if shoulder_aligned
                else ["x", "y"]
            ),
            "pelvis_position_m": result.base_position.tolist(),
            "nominal_grounded_pelvis_height_m": (
                result.nominal_pelvis_height
            ),
            "pelvis_height_offset_from_grounded_m": float(
                result.base_position[2] - result.nominal_pelvis_height
            ),
            "height_source": (
                "source/G1 shoulder-center alignment"
                if shoulder_aligned
                else "neutral URDF pelvis-to-lowest-foot-support distance"
            ),
            "candidate_rotation_axes": ["yaw"],
            "translation_policy": (
                "p_pelvis = p_source_shoulder_center_world - "
                "R_world_from_pelvis @ "
                "p_g1_shoulder_center_from_pelvis"
                if shoulder_aligned
                else "grounded trajectory-centered grid"
            ),
            "shoulder_alignment_error_m": shoulder_alignment_error,
            "waist_policy": (
                "waist joints participate in IK with posture regularization"
            ),
            "leg_policy": (
                "leg joints remain neutral and frozen; ground contact is not "
                "enforced when shoulder alignment changes pelvis height"
                if shoulder_aligned
                else "leg joints remain neutral and frozen"
            ),
        },
        "spatial_alignment": {
            "mode": (
                "episode_shoulder_center_to_g1_morphology"
                if shoulder_aligned
                else "episode_upper_body_to_g1_morphology"
            ),
            "scale": result.scale,
            "scale_source": result.scale_source,
            "source_arm_lengths_m": result.source_arm_lengths.tolist(),
            "g1_arm_lengths_m": result.robot_arm_lengths.tolist(),
            "source_pelvis_in_reference_camera_m": (
                result.pelvis_anchor_reference_camera.tolist()
            ),
            "source_shoulder_center_in_reference_camera_m": (
                result.shoulder_anchor_reference_camera.tolist()
            ),
            "mapped_source_shoulder_center_world_m": (
                result.mapped_shoulder_center.tolist()
            ),
            "g1_neutral_shoulder_center_in_pelvis_m": (
                result.robot_shoulder_center_from_pelvis.tolist()
            ),
            "camera_origin_m": result.camera_origin.tolist(),
            "world_from_reference_camera_rotation": (
                result.world_from_reference_camera.tolist()
            ),
            "camera_policy": (
                "episode first-camera attitude leveled by the tracked body "
                "up axis; the hip anchor defines the shared reference world, "
                "then the G1 pelvis translation is solved from shoulder-center "
                "alignment"
                if shoulder_aligned
                else (
                    "episode first-camera attitude leveled by the tracked body "
                    "up axis; translation maps the episode hip anchor to the "
                    "grounded neutral G1 pelvis"
                )
            ),
        },
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
        "hand_action": hand_action_report,
        "base_search": {
            "objective": (
                "Eq. (3): mean representative-keyframe indicator that all "
                "active 6D end-effector IK targets are feasible"
            ),
            "representative_keyframes": (
                result.base_search_keyframes.tolist()
            ),
            "trajectory_centroid": result.base_search_centroid.tolist(),
            "maximum_reach_m": result.maximum_reach,
            "candidate_count": len(result.search_results),
            "selection_status": (
                "eq3_feasible"
                if float(result.search_results[0]["feasible_fraction"]) > 0.0
                else "partial_end_effector_feasibility_fallback"
            ),
            "best_feasible_keyframe_fraction": float(
                result.search_results[0]["feasible_fraction"]
            ),
            "best_feasible_target_fraction": float(
                result.search_results[0]["feasible_target_fraction"]
            ),
            "position_tolerance_m": IK_FEASIBLE_POSITION_ERROR_M,
            "orientation_tolerance_deg": float(
                np.degrees(IK_FEASIBLE_ORIENTATION_ERROR_RAD)
            ),
            "orientation_residual_weight": IK_ORIENTATION_WEIGHT,
            "translation_grid_points_per_axis": (
                1 if shoulder_aligned else BASE_GRID_POINTS_PER_AXIS
            ),
            "yaw_offsets_deg": np.degrees(
                BASE_YAW_OFFSETS_RAD
            ).tolist(),
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
        ik_target_source=np.asarray(ik_target_source),
        hand_target_mode=np.asarray(trajectory.target_mode),
        hand_action_source=np.asarray(hand_action_source),
        hand_action_mode=np.asarray(result.hand_action_mode),
        joint_names=np.asarray(model.names[1:]),
        qpos=result.qpos,
        link_names=np.asarray(link_names),
        link_positions=link_positions,
        link_quaternions=link_quaternions,
        left_targets=result.left_targets,
        right_targets=result.right_targets,
        left_target_rotations=result.left_target_rotations,
        right_target_rotations=result.right_target_rotations,
        left_palm_targets=result.left_palm_targets,
        right_palm_targets=result.right_palm_targets,
        left_palm_target_rotations=result.left_palm_target_rotations,
        right_palm_target_rotations=result.right_palm_target_rotations,
        left_errors=result.left_errors,
        right_errors=result.right_errors,
        left_orientation_errors=result.left_orientation_errors,
        right_orientation_errors=result.right_orientation_errors,
        left_visible=result.left_visible,
        right_visible=result.right_visible,
        hand_closures=result.hand_closures,
        hand_motor_commands=result.hand_motor_commands,
        hand_motor_order=np.asarray(O6_MOTOR_SUFFIXES),
        hand_closure_order=np.asarray(["left", "right"]),
        hand_closure_semantics=np.asarray(
            (
                "both LinkerHand O6 commands remain exactly 0=open because "
                "source finger articulation is unavailable"
                if rigid_palm_mode
                else (
                    "mean normalized value of the six independent O6 "
                    "motors; inspect hand_motor_commands for actuator "
                    "angles"
                    if articulated_mode
                    else (
                        "0=open; increasing closes along the calibrated "
                        "monotonic aperture branch"
                    )
                )
            )
        ),
        hand_tcp_translations=result.hand_tcp_translations,
        hand_tcp_rotations=result.hand_tcp_rotations,
        hand_target_widths=result.target_hand_widths,
        hand_scaled_human_widths=result.target_hand_widths,
        hand_width_role=np.asarray(
            (
                "unavailable_fixed_open"
                if rigid_palm_mode
                else (
                    "diagnostic_only_not_actuator_target"
                    if articulated_mode
                    else "continuous_actuator_target"
                )
            )
        ),
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
        intrinsic=trajectory.intrinsic,
        left_action_reference_camera=(
            trajectory.left_action_reference_camera
        ),
        right_action_reference_camera=(
            trajectory.right_action_reference_camera
        ),
        left_action_reference_rotations=(
            trajectory.left_action_reference_rotations
        ),
        right_action_reference_rotations=(
            trajectory.right_action_reference_rotations
        ),
        base_position=result.base_position,
        base_rotation=result.base_rotation,
        base_placement_mode=np.asarray(
            base_placement_mode
        ),
        base_frame=np.asarray(result.base_frame),
        nominal_pelvis_height=np.asarray(
            result.nominal_pelvis_height,
            dtype=np.float64,
        ),
        base_search_keyframes=result.base_search_keyframes,
        base_search_centroid=result.base_search_centroid,
        maximum_reach=result.maximum_reach,
        scale=result.scale,
        scale_source=np.asarray(result.scale_source),
        source_arm_lengths=result.source_arm_lengths,
        robot_arm_lengths=result.robot_arm_lengths,
        pelvis_anchor_reference_camera=(
            result.pelvis_anchor_reference_camera
        ),
        shoulder_anchor_reference_camera=(
            result.shoulder_anchor_reference_camera
        ),
        robot_shoulder_center_from_pelvis=(
            result.robot_shoulder_center_from_pelvis
        ),
        mapped_shoulder_center=result.mapped_shoulder_center,
        shoulder_alignment_error=np.asarray(
            shoulder_alignment_error,
            dtype=np.float64,
        ),
        camera_origin=result.camera_origin,
        world_from_reference_camera=result.world_from_reference_camera,
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
    output_root = args.output.resolve()
    skip_report_path = output_root / EPISODE_SKIP_REPORT
    skip_report_path.unlink(missing_ok=True)
    source_width, source_height, frame_count, _ = video_metadata(args.video)
    trajectory = load_hand_trajectory(args.hdf5)
    if trajectory.frame_count != frame_count:
        raise ValueError(
            f"HDF5 has {trajectory.frame_count} frames; video has "
            f"{frame_count}"
    )
    model = build_pinocchio_g1(args.urdf)
    try:
        result = retarget_with_pinocchio(
            model,
            trajectory,
            source_width=source_width,
            source_height=source_height,
        )
    except NoCommonBasePlacementError as error:
        skip_report: dict[str, object] = {
            "status": "skipped",
            "reason_code": "no_common_base_search_region",
            "reason": str(error),
            "episode_path": str(args.hdf5.resolve()),
            "video": str(args.video.resolve()),
            "hdf5": str(args.hdf5.resolve()),
        }
        output_root.mkdir(parents=True, exist_ok=True)
        skip_report_path.write_text(
            json.dumps(skip_report, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(skip_report, indent=2))
        return
    report = save_pinocchio_trajectory(
        model=model,
        trajectory=trajectory,
        result=result,
        output_root=output_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
