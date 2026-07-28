from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pinocchio as pin

from human_to_robot.retarget_g1_pinocchio import (
    O6_FLEX_DIGITS,
    O6_MOTOR_COUNT,
    O6_MOTOR_SUFFIXES,
    _fit_feature_grid,
    _mimic_joints,
    _o6_motor_limits,
    _open_hand_motor_commands,
    _retarget_articulated_hands,
    _robot_digit_feature,
    _robot_feature_grid,
    _set_hand_motor_configuration,
    build_pinocchio_g1,
)
from human_to_robot.trajectory import load_hand_trajectory


REPOSITORY = Path(__file__).resolve().parents[1]
HAND_URDF = (
    REPOSITORY
    / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
)
EPISODE_9 = REPOSITORY / "dataset/basic_pick_place/9.hdf5"


def _joint_q_value(
    model: pin.Model,
    q: np.ndarray,
    joint_name: str,
) -> float:
    joint_id = model.getJointId(joint_name)
    return float(q[model.idx_qs[joint_id]])


class ArticulatedO6MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_pinocchio_g1(HAND_URDF)

    def test_six_motors_write_independently_with_exact_mimics_and_limits(
        self,
    ) -> None:
        fractions = np.array(
            [
                [0.07, 0.19, 0.31, 0.43, 0.55, 0.67],
                [0.83, 0.71, 0.59, 0.47, 0.35, 0.23],
            ],
            dtype=np.float64,
        )
        commands = np.empty((2, O6_MOTOR_COUNT), dtype=np.float64)
        for side_index, side in enumerate(("left", "right")):
            lower, upper = _o6_motor_limits(self.model, side)
            commands[side_index] = lower + fractions[side_index] * (
                upper - lower
            )

        q = pin.neutral(self.model)
        _set_hand_motor_configuration(self.model, q, commands)

        for side_index, side in enumerate(("left", "right")):
            lower, upper = _o6_motor_limits(self.model, side)
            self.assertTrue(np.all(commands[side_index] >= lower))
            self.assertTrue(np.all(commands[side_index] <= upper))
            self.assertEqual(
                len(np.unique(np.round(commands[side_index], decimals=8))),
                O6_MOTOR_COUNT,
            )
            for motor_index, suffix in enumerate(O6_MOTOR_SUFFIXES):
                np.testing.assert_allclose(
                    _joint_q_value(self.model, q, f"{side}_{suffix}"),
                    commands[side_index, motor_index],
                    atol=1e-12,
                )

        mimic = _mimic_joints(self.model)
        for child, (master, multiplier, offset) in mimic.items():
            child_value = _joint_q_value(self.model, q, child)
            master_value = _joint_q_value(self.model, q, master)
            np.testing.assert_allclose(
                child_value,
                multiplier * master_value + offset,
                atol=1e-12,
            )
            child_id = self.model.getJointId(child)
            child_q = self.model.idx_qs[child_id]
            self.assertGreaterEqual(
                child_value,
                float(self.model.lowerPositionLimit[child_q]) - 1e-12,
            )
            self.assertLessEqual(
                child_value,
                float(self.model.upperPositionLimit[child_q]) + 1e-12,
            )

        # The left thumb IP mimic is tighter than the CMC pitch's own upper
        # bound; the exposed O6 limit must therefore be their intersection.
        left_lower, left_upper = _o6_motor_limits(self.model, "left")
        thumb_pitch_id = self.model.getJointId("left_thumb_cmc_pitch")
        thumb_pitch_q = self.model.idx_qs[thumb_pitch_id]
        thumb_ip_id = self.model.getJointId("left_thumb_ip")
        thumb_ip_q = self.model.idx_qs[thumb_ip_id]
        multiplier = mimic["left_thumb_ip"][1]
        expected_joint_upper = min(
            float(self.model.upperPositionLimit[thumb_pitch_q]),
            float(self.model.upperPositionLimit[thumb_ip_q]) / multiplier,
        )
        self.assertAlmostEqual(left_lower[1], 0.0)
        self.assertAlmostEqual(left_upper[1], expected_joint_upper)

        at_upper = _open_hand_motor_commands(self.model)
        at_upper[0] = left_upper
        q_at_upper = pin.neutral(self.model)
        _set_hand_motor_configuration(self.model, q_at_upper, at_upper)
        self.assertAlmostEqual(
            _joint_q_value(self.model, q_at_upper, "left_thumb_ip"),
            float(self.model.upperPositionLimit[thumb_ip_q]),
        )

    def test_flex_feature_luts_recover_independent_robot_fk_commands(
        self,
    ) -> None:
        side = "right"
        side_index = 1
        sample_count = 65
        selected_sample = {
            "index": 8,
            "middle": 22,
            "ring": 39,
            "pinky": 55,
        }
        recovered_columns: list[np.ndarray] = []
        expected_commands: list[float] = []
        feature_weights = np.asarray(
            [
                np.sqrt(0.5),
                0.0,
                np.sqrt(0.5),
                np.sqrt(0.4),
                0.0,
                np.sqrt(0.4),
                np.sqrt(0.1),
            ],
            dtype=np.float64,
        )
        lower, upper = _o6_motor_limits(self.model, side)

        for digit in O6_FLEX_DIGITS:
            with self.subTest(digit=digit):
                motor_index = O6_MOTOR_SUFFIXES.index(
                    f"{digit}_mcp_pitch"
                )
                grid_commands, grid_features = _robot_feature_grid(
                    self.model,
                    side=side,
                    digit=digit,
                    samples=sample_count,
                )
                expected = float(
                    grid_commands[selected_sample[digit], 0]
                )
                commands = _open_hand_motor_commands(self.model)
                commands[side_index, motor_index] = expected
                q = pin.neutral(self.model)
                _set_hand_motor_configuration(self.model, q, commands)
                data = self.model.createData()
                pin.forwardKinematics(self.model, data, q)
                pin.updateFramePlacements(self.model, data)
                source_feature = _robot_digit_feature(
                    self.model,
                    data,
                    side=side,
                    digit=digit,
                )

                frame_count = 7
                fitted, errors = _fit_feature_grid(
                    np.repeat(
                        source_feature[None],
                        frame_count,
                        axis=0,
                    ),
                    np.ones(frame_count, dtype=bool),
                    grid_commands,
                    grid_features,
                    np.full((frame_count, 1), lower[motor_index]),
                    feature_weights=feature_weights,
                    lower=lower[motor_index : motor_index + 1],
                    upper=upper[motor_index : motor_index + 1],
                )
                np.testing.assert_allclose(fitted[:, 0], expected, atol=1e-12)
                np.testing.assert_allclose(errors, 0.0, atol=1e-12)
                recovered_columns.append(fitted[:, 0])
                expected_commands.append(expected)

        self.assertEqual(
            len(np.unique(np.round(expected_commands, decimals=8))),
            len(O6_FLEX_DIGITS),
        )
        for first in range(len(recovered_columns)):
            for second in range(first + 1, len(recovered_columns)):
                self.assertFalse(
                    np.allclose(
                        recovered_columns[first],
                        recovered_columns[second],
                    )
                )

    @unittest.skipUnless(
        EPISODE_9.exists(),
        "local basic_pick_place/9 HDF5 is not available",
    )
    def test_episode_9_produces_independent_bounded_motor_tracks(self) -> None:
        trajectory = load_hand_trajectory(EPISODE_9)
        self.assertIsNotNone(trajectory.left_articulated_hand)
        self.assertIsNotNone(trajectory.right_articulated_hand)
        human_widths = np.stack(
            [trajectory.left_widths, trajectory.right_widths],
            axis=1,
        )

        (
            closures,
            motor_commands,
            target_widths,
            achieved_widths,
            profiles,
            diagnostics,
        ) = _retarget_articulated_hands(
            self.model,
            trajectory,
            human_widths,
            scale=0.84,
        )

        self.assertEqual(
            motor_commands.shape,
            (trajectory.frame_count, 2, O6_MOTOR_COUNT),
        )
        self.assertEqual(closures.shape, (trajectory.frame_count, 2))
        self.assertEqual(target_widths.shape, (trajectory.frame_count, 2))
        self.assertEqual(achieved_widths.shape, (trajectory.frame_count, 2))
        self.assertTrue(np.isfinite(motor_commands).all())
        self.assertEqual(len(profiles), 2)
        self.assertEqual(set(diagnostics), {"left", "right"})

        for side_index, side in enumerate(("left", "right")):
            with self.subTest(side=side):
                lower, upper = _o6_motor_limits(self.model, side)
                side_commands = motor_commands[:, side_index]
                self.assertTrue(
                    np.all(side_commands >= lower[None] - 1e-12)
                )
                self.assertTrue(
                    np.all(side_commands <= upper[None] + 1e-12)
                )
                flex = side_commands[:, 2:]
                for first in range(flex.shape[1]):
                    for second in range(first + 1, flex.shape[1]):
                        self.assertFalse(
                            np.allclose(
                                flex[:, first],
                                flex[:, second],
                                atol=1e-8,
                            ),
                            msg=(
                                f"{side} finger columns {first} and "
                                f"{second} collapsed to one command"
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
