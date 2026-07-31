"""Independent G1 arm-position repair from EgoDex 3D hand anchors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from repaint.rgbd_episode import RgbdEpisode


@dataclass(frozen=True)
class ArmPositionRepair:
    joint_positions: np.ndarray
    correspondence_count: int
    confidence_threshold: float
    rmse_before_m: float
    rmse_after_m: float
    max_error_before_m: float
    max_error_after_m: float
    max_joint_correction_rad: float
    max_repaired_joint_step_rad: float


def _source_anchors(
    source_hdf5: Path,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(source_hdf5, "r") as handle:
        hip = np.asarray(
            handle["transforms/hip"],
            dtype=np.float64,
        )[frame_indices, :3, 3]
        hand = np.stack(
            [
                np.asarray(
                    handle["transforms/leftHand"],
                    dtype=np.float64,
                )[frame_indices, :3, 3],
                np.asarray(
                    handle["transforms/rightHand"],
                    dtype=np.float64,
                )[frame_indices, :3, 3],
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
    return hip, hand, confidence


def repair_arm_positions_to_source_hands(
    episode: RgbdEpisode,
    urdf_path: Path,
    *,
    confidence_threshold: float = 0.5,
    maximum_position_error_m: float = 0.005,
) -> ArmPositionRepair:
    """Repair shoulder/elbow joints so G1 palms follow source 3D hands."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie in [0, 1]")
    if maximum_position_error_m <= 0.0:
        raise ValueError("maximum_position_error_m must be positive")
    model = pin.buildModelFromUrdf(str(urdf_path.resolve()))
    if tuple(model.names[1:]) != episode.joint_names:
        raise ValueError("Episode joints do not match the arm IK G1 URDF")

    hip, source_hand, confidence = _source_anchors(
        episode.source_hdf5,
        episode.frame_indices,
    )
    repaired = episode.joint_positions.astype(np.float64, copy=True)
    errors_before: list[float] = []
    errors_after: list[float] = []
    corrections: list[float] = []

    for frame_id in range(episode.frame_count):
        root_rotation = Rotation.from_quat(
            episode.root_pose_xyzw[frame_id, 3:7]
        ).as_matrix()
        source_relative = source_hand[frame_id] - hip[frame_id]
        target_root = np.stack(
            [
                root_rotation.T
                @ (
                    episode.scene_scale
                    * (
                        episode.source_to_robot_rotation
                        @ source_relative[side]
                    )
                )
                for side in range(2)
            ]
        )
        for side_index, side in enumerate(("left", "right")):
            if (
                confidence[frame_id, side_index]
                < confidence_threshold
                or not np.isfinite(target_root[side_index]).all()
            ):
                continue
            joint_names = [
                f"{side}_shoulder_pitch_joint",
                f"{side}_shoulder_roll_joint",
                f"{side}_shoulder_yaw_joint",
                f"{side}_elbow_joint",
            ]
            joint_indices = np.asarray(
                [
                    model.joints[model.getJointId(name)].idx_q
                    for name in joint_names
                ],
                dtype=np.int64,
            )
            hand_frame_id = model.getFrameId(f"{side}_hand_base_link")
            if hand_frame_id >= model.nframes:
                raise KeyError(f"G1 URDF is missing the {side} hand base")
            lower = model.lowerPositionLimit[joint_indices]
            upper = model.upperPositionLimit[joint_indices]
            reference = np.clip(
                repaired[frame_id, joint_indices],
                lower,
                upper,
            )
            data = model.createData()

            def hand_position(candidate: np.ndarray) -> np.ndarray:
                configuration = repaired[frame_id].copy()
                configuration[joint_indices] = candidate
                pin.forwardKinematics(model, data, configuration)
                pin.updateFramePlacements(model, data)
                return np.array(
                    data.oMf[hand_frame_id].translation,
                    dtype=np.float64,
                    copy=True,
                )

            before = hand_position(reference)
            target = target_root[side_index]

            def residual(candidate: np.ndarray) -> np.ndarray:
                return np.concatenate(
                    [
                        30.0 * (hand_position(candidate) - target),
                        0.01 * (candidate - reference),
                    ]
                )

            result = least_squares(
                residual,
                reference,
                bounds=(lower, upper),
                max_nfev=200,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            after = hand_position(result.x)
            error_before = float(np.linalg.norm(before - target))
            error_after = float(np.linalg.norm(after - target))
            if not result.success or error_after > maximum_position_error_m:
                raise RuntimeError(
                    f"Arm IK failed at frame {frame_id} {side}: "
                    f"success={result.success}, error={error_after:.6f} m"
                )
            repaired[frame_id, joint_indices] = result.x
            errors_before.append(error_before)
            errors_after.append(error_after)
            corrections.extend(np.abs(result.x - reference).tolist())

    if len(errors_after) < 4:
        raise ValueError("Too few confident source-hand anchors for arm IK")
    repaired_steps = (
        np.abs(np.diff(repaired, axis=0))
        if len(repaired) > 1
        else np.zeros_like(repaired)
    )
    return ArmPositionRepair(
        joint_positions=repaired,
        correspondence_count=len(errors_after),
        confidence_threshold=float(confidence_threshold),
        rmse_before_m=float(
            np.sqrt(np.mean(np.square(errors_before)))
        ),
        rmse_after_m=float(np.sqrt(np.mean(np.square(errors_after)))),
        max_error_before_m=float(np.max(errors_before)),
        max_error_after_m=float(np.max(errors_after)),
        max_joint_correction_rad=float(np.max(corrections)),
        max_repaired_joint_step_rad=float(np.max(repaired_steps)),
    )
