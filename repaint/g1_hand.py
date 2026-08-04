"""G1/O6 hand model loading and EgoDex hand retargeting."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.signal import savgol_filter

from repaint.hand_trajectory import (
    HAND_LANDMARK_NAMES,
    RIGID_PALM_TARGET,
    ArticulatedHandTrajectory,
    HandTrajectory,
    load_hand_trajectory,
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
PALM_OFFSET_FROM_WRIST = np.array([0.08, 0.0, 0.0], dtype=np.float64)
TRACKED_HAND_TO_G1_PALM_ROTATIONS = np.asarray(
    [
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class O6HandRetargetTrajectory:
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    action_mode: str
    source_hand_transforms_reference_camera: np.ndarray
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
) -> np.ndarray:
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
    return closures


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
        if not np.isfinite(basis_points).all():
            continue
        basis = _hand_anatomical_basis(*basis_points)
        if basis is None:
            continue
        for digit, names in ARTICULATED_DIGIT_CHAINS.items():
            indices = [landmark_index[name] for name in names]
            chain = points[frame_id, indices]
            if not np.isfinite(chain).all():
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


def _retarget_articulated_hands(
    model: pin.Model,
    trajectory: HandTrajectory,
    human_widths: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    fallback_closures = _retarget_hand_widths(
        model,
        human_widths,
        scale=scale,
    )
    motor_commands = _closures_to_motor_commands(
        model,
        fallback_closures,
    )
    tracks = (
        trajectory.left_articulated_hand,
        trajectory.right_articulated_hand,
    )
    diagnostics: dict[str, dict[str, object]] = {}
    for side_index, side in enumerate(("left", "right")):
        track = tracks[side_index]
        assert track is not None
        digit_diagnostics: dict[str, object] = {}
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
        diagnostics[side] = {
            "source": "articulated_hand_geometry",
            "digits": digit_diagnostics,
        }
    return motor_commands, diagnostics


def retarget_o6_hands(
    model: pin.Model,
    hdf5_path: str | Path,
    *,
    scale: float,
) -> O6HandRetargetTrajectory:
    """Retarget all O6 finger joints and preserve the tracked EgoDex Hand pose."""

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
        motor_commands, diagnostics = _retarget_articulated_hands(
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

    hand_positions = np.stack(
        [
            trajectory.left_action_reference_camera,
            trajectory.right_action_reference_camera,
        ],
        axis=1,
    )
    hand_rotations = np.stack(
        [
            trajectory.left_action_reference_rotations,
            trajectory.right_action_reference_rotations,
        ],
        axis=1,
    )
    source_hand_transforms = np.repeat(
        np.eye(4, dtype=np.float64)[None, None],
        trajectory.frame_count,
        axis=0,
    )
    source_hand_transforms = np.repeat(
        source_hand_transforms,
        2,
        axis=1,
    )
    source_hand_transforms[:, :, :3, :3] = hand_rotations
    source_hand_transforms[:, :, :3, 3] = hand_positions

    if (
        source_hand_transforms.shape
        != (trajectory.frame_count, 2, 4, 4)
        or not np.isfinite(source_hand_transforms).all()
    ):
        raise ValueError("Tracked EgoDex Hand transforms are invalid")

    return O6HandRetargetTrajectory(
        joint_names=HAND_JOINTS,
        joint_positions=joint_positions,
        action_mode=trajectory.target_mode,
        source_hand_transforms_reference_camera=(
            source_hand_transforms
        ),
        diagnostics=diagnostics,
    )
