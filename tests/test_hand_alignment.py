from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from human_to_robot.retarget_g1_pinocchio import (
    HAND_JOINTS,
    _active_velocity_indices,
    _set_binary_hand_configuration,
    build_pinocchio_g1,
)
from human_to_robot.trajectory import (
    _binary_close_action,
    align_hand_action,
)


REPOSITORY = Path(__file__).resolve().parents[1]
HAND_URDF = (
    REPOSITORY
    / "assets/unitree_description/urdf/g1/main_with_hand_new.urdf"
)


def _constant_keypoints(frame_count: int = 15) -> np.ndarray:
    """Return [wrist, thumb, index, middle] with a non-degenerate jaw plane."""

    keypoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return np.repeat(keypoints[None], frame_count, axis=0)


def _joint_q_value(model, q: np.ndarray, joint_name: str) -> float:
    joint_id = model.getJointId(joint_name)
    return float(q[model.idx_qs[joint_id]])


class HandActionAlignmentTests(unittest.TestCase):
    def test_paper_position_width_and_right_handed_orientation(self) -> None:
        points = _constant_keypoints()
        confidence = np.ones(points.shape[:2], dtype=np.float64)
        expected_position = np.array([0.5, 1.0, 0.0])
        expected_width = 1.0
        expected_right = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        expected_left = np.array(
            [
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

        for side, expected_rotation in (
            ("left", expected_left),
            ("right", expected_right),
        ):
            with self.subTest(side=side):
                aligned = align_hand_action(
                    points,
                    confidence,
                    side=side,
                )
                np.testing.assert_allclose(
                    aligned.position,
                    np.repeat(
                        expected_position[None],
                        len(points),
                        axis=0,
                    ),
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    aligned.width,
                    expected_width,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    aligned.rotation,
                    np.repeat(
                        expected_rotation[None],
                        len(points),
                        axis=0,
                    ),
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    np.linalg.det(aligned.rotation),
                    1.0,
                    atol=1e-12,
                )

    def test_binary_action_semantics_and_hysteresis_are_stable(self) -> None:
        # Low jaw width means closed (1), high width means open (0). Once
        # opened, small oscillations around the inferred threshold must not
        # chatter until the lower hysteresis boundary is crossed.
        normalized_width = np.array(
            [
                0.20,
                0.21,
                0.19,
                0.20,
                0.22,
                0.60,
                0.59,
                0.61,
                0.60,
                0.58,
                0.39,
                0.41,
                0.38,
                0.42,
                0.20,
            ],
            dtype=np.float64,
        )
        valid = np.ones(len(normalized_width), dtype=bool)

        left, left_threshold = _binary_close_action(normalized_width, valid)
        right, right_threshold = _binary_close_action(
            normalized_width[::-1],
            valid,
        )
        actions = np.stack([left, right], axis=1)

        self.assertEqual(actions.shape, (len(normalized_width), 2))
        self.assertEqual(actions.dtype, np.uint8)
        self.assertTrue(np.isin(actions, (0, 1)).all())
        np.testing.assert_array_equal(left[:5], 1)
        np.testing.assert_array_equal(left[5:14], 0)
        self.assertEqual(left[-1], 1)
        self.assertGreater(left_threshold, 0.22)
        self.assertLess(left_threshold, 0.58)
        self.assertGreater(right_threshold, 0.22)
        self.assertLess(right_threshold, 0.58)


class LinkerHandUrdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_pinocchio_g1(HAND_URDF)
        cls.urdf_root = ET.parse(HAND_URDF).getroot()

    def test_new_urdf_dimensions_and_hand_joints_frozen_from_ik(self) -> None:
        self.assertEqual(self.model.nq, 51)
        self.assertEqual(self.model.nv, 51)
        self.assertEqual(len(HAND_JOINTS), 22)
        self.assertEqual(len(set(HAND_JOINTS)), 22)

        hand_velocity_indices = {
            velocity_index
            for joint_name in HAND_JOINTS
            for velocity_index in range(
                self.model.idx_vs[self.model.getJointId(joint_name)],
                self.model.idx_vs[self.model.getJointId(joint_name)]
                + self.model.nvs[self.model.getJointId(joint_name)],
            )
        }
        for position_only in (False, True):
            with self.subTest(position_only=position_only):
                active = set(
                    _active_velocity_indices(
                        self.model,
                        position_only=position_only,
                    ).tolist()
                )
                self.assertTrue(hand_velocity_indices.isdisjoint(active))

    def test_binary_open_closed_mapping_respects_limits_and_mimic(self) -> None:
        open_q = np.zeros(self.model.nq, dtype=np.float64)
        closed_q = np.zeros(self.model.nq, dtype=np.float64)
        _set_binary_hand_configuration(
            self.model,
            open_q,
            np.array([0, 0], dtype=np.uint8),
        )
        _set_binary_hand_configuration(
            self.model,
            closed_q,
            np.array([1, 1], dtype=np.uint8),
        )

        for joint_name in HAND_JOINTS:
            with self.subTest(joint=joint_name):
                joint_id = self.model.getJointId(joint_name)
                q_index = self.model.idx_qs[joint_id]
                lower = float(self.model.lowerPositionLimit[q_index])
                upper = float(self.model.upperPositionLimit[q_index])
                open_value = float(open_q[q_index])
                closed_value = float(closed_q[q_index])
                self.assertAlmostEqual(open_value, lower)
                self.assertGreaterEqual(closed_value, lower)
                self.assertLessEqual(closed_value, upper)
                self.assertGreater(closed_value, open_value)

        for joint_element in self.urdf_root.findall("joint"):
            mimic = joint_element.find("mimic")
            if mimic is None:
                continue
            child = joint_element.attrib["name"]
            if child not in HAND_JOINTS:
                continue
            parent = mimic.attrib["joint"]
            multiplier = float(mimic.attrib.get("multiplier", "1"))
            offset = float(mimic.attrib.get("offset", "0"))
            for state_name, q in (("open", open_q), ("closed", closed_q)):
                with self.subTest(
                    mimic_child=child,
                    state=state_name,
                ):
                    expected = (
                        multiplier * _joint_q_value(self.model, q, parent)
                        + offset
                    )
                    self.assertAlmostEqual(
                        _joint_q_value(self.model, q, child),
                        expected,
                        places=7,
                    )

    def test_every_urdf_mesh_path_exists(self) -> None:
        meshes = self.urdf_root.findall(".//mesh")
        self.assertGreater(len(meshes), 0)
        missing: list[str] = []
        for mesh in meshes:
            filename = mesh.attrib["filename"]
            if filename.startswith("package://"):
                missing.append(filename)
                continue
            resolved = (HAND_URDF.parent / filename).resolve()
            if not resolved.is_file():
                missing.append(filename)
        self.assertEqual(missing, [], f"Missing URDF meshes: {missing}")


if __name__ == "__main__":
    unittest.main()
