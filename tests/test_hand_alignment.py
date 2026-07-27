from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
from scipy.signal import savgol_filter

from human_to_robot.retarget_g1_pinocchio import (
    HAND_JOINTS,
    PALM_OFFSET_FROM_WRIST,
    _active_velocity_indices,
    _retarget_hand_widths,
    _set_continuous_hand_configuration,
    build_pinocchio_g1,
)
from human_to_robot.render_isaacsim import (
    _build_arm_hand_urdf,
    _is_render_link,
)
from human_to_robot.trajectory import (
    align_hand_action,
    load_hand_trajectory,
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

    def test_width_remains_continuous_after_savgol_smoothing(self) -> None:
        widths = np.array(
            [
                0.02,
                0.02,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.08,
                0.08,
                0.08,
                0.08,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
                0.02,
                0.02,
                0.02,
            ],
            dtype=np.float64,
        )
        points = _constant_keypoints(len(widths))
        points[:, 1, 0] = widths
        confidence = np.ones(points.shape[:2], dtype=np.float64)
        confidence[8:15, 1:] = 0.1

        aligned = align_hand_action(points, confidence, side="right")

        np.testing.assert_allclose(
            aligned.width,
            savgol_filter(widths, 11, 2, mode="interp"),
            atol=1e-12,
        )
        self.assertGreater(
            len(np.unique(np.round(aligned.width, decimals=8))),
            2,
        )

    def test_width_does_not_depend_on_wrist_or_orientation_validity(
        self,
    ) -> None:
        widths = np.linspace(0.02, 0.08, 21, dtype=np.float64)
        points = _constant_keypoints(len(widths))
        points[:, 1, 0] = widths
        confidence = np.ones(points.shape[:2], dtype=np.float64)
        confidence[7:14, 0] = 0.0
        points[10, 0] = points[10, 2]

        aligned = align_hand_action(points, confidence, side="right")

        np.testing.assert_allclose(
            aligned.width,
            savgol_filter(widths, 11, 2, mode="interp"),
            atol=1e-12,
        )

    def test_savgol_width_overshoot_is_clipped_to_physical_range(self) -> None:
        widths = np.full(21, 0.001, dtype=np.float64)
        widths[10] = 1.0
        points = _constant_keypoints(len(widths))
        points[:, 1, 0] = widths
        confidence = np.ones(points.shape[:2], dtype=np.float64)
        raw_smoothed = savgol_filter(widths, 11, 2, mode="interp")
        self.assertLess(float(np.min(raw_smoothed)), 0.0)

        aligned = align_hand_action(points, confidence, side="right")

        np.testing.assert_allclose(
            aligned.width,
            np.maximum(raw_smoothed, 0.0),
            atol=1e-12,
        )
        self.assertTrue(np.all(aligned.width >= 0.0))

    def test_unannotated_hdf5_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            hdf5_path = Path(temporary) / "episode.hdf5"
            with h5py.File(hdf5_path, "w") as handle:
                handle.create_dataset("camera/intrinsic", data=np.eye(3))
                handle.create_dataset(
                    "transforms/camera",
                    data=np.repeat(np.eye(4)[None], 3, axis=0),
                )

            with self.assertRaisesRegex(
                ValueError,
                "lacks usable annotated keypoints",
            ):
                load_hand_trajectory(hdf5_path)

    def test_loader_keeps_wrist_ik_independent_of_finger_actions(self) -> None:
        frame_count = 15
        wrist_rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        with TemporaryDirectory() as temporary:
            hdf5_path = Path(temporary) / "episode.hdf5"
            with h5py.File(hdf5_path, "w") as handle:
                handle.create_dataset("camera/intrinsic", data=np.eye(3))
                handle.create_dataset(
                    "transforms/camera",
                    data=np.repeat(np.eye(4)[None], frame_count, axis=0),
                )
                for side, x_offset in (("left", -0.2), ("right", 0.2)):
                    wrist_position = np.array(
                        [x_offset, 0.0, 1.0],
                        dtype=np.float64,
                    )
                    widths = np.concatenate(
                        [
                            np.full(7, 0.02, dtype=np.float64),
                            np.full(8, 0.08, dtype=np.float64),
                        ]
                    )
                    positions = {
                        "Hand": np.repeat(
                            wrist_position[None],
                            frame_count,
                            axis=0,
                        ),
                        "ThumbTip": np.stack(
                            [
                                wrist_position
                                + np.array([width, 0.10, 0.0])
                                for width in widths
                            ]
                        ),
                        "IndexFingerTip": np.repeat(
                            (
                                wrist_position
                                + np.array([0.0, 0.10, 0.0])
                            )[None],
                            frame_count,
                            axis=0,
                        ),
                        "MiddleFingerTip": np.repeat(
                            (
                                wrist_position
                                + np.array([0.0, 0.10, 0.0])
                            )[None],
                            frame_count,
                            axis=0,
                        ),
                    }
                    for name, values in positions.items():
                        transforms = np.repeat(
                            np.eye(4)[None],
                            frame_count,
                            axis=0,
                        )
                        transforms[:, :3, 3] = values
                        if name == "Hand":
                            transforms[:, :3, :3] = wrist_rotation
                        handle.create_dataset(
                            f"transforms/{side}{name}",
                            data=transforms,
                        )
                        handle.create_dataset(
                            f"confidences/{side}{name}",
                            data=np.full(
                                frame_count,
                                1.0 if name == "Hand" else 0.1,
                                dtype=np.float64,
                            ),
                        )

            trajectory = load_hand_trajectory(hdf5_path)

        for side, expected_x in (("left", -0.2), ("right", 0.2)):
            with self.subTest(side=side):
                positions = getattr(
                    trajectory,
                    f"{side}_reference_camera",
                )
                rotations = getattr(
                    trajectory,
                    f"{side}_reference_rotations",
                )
                confidence = getattr(trajectory, f"{side}_confidence")
                np.testing.assert_allclose(
                    positions,
                    np.repeat(
                        np.array([[expected_x, 0.0, 1.0]]),
                        frame_count,
                        axis=0,
                    ),
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    rotations,
                    np.repeat(
                        wrist_rotation[None],
                        frame_count,
                        axis=0,
                    ),
                    atol=1e-12,
                )
                np.testing.assert_allclose(confidence, 1.0)
        for widths in (
            trajectory.left_widths,
            trajectory.right_widths,
        ):
            self.assertLess(
                float(np.mean(widths[:3])),
                float(np.mean(widths[-3:])),
            )
            self.assertGreater(len(np.unique(np.round(widths, decimals=8))), 2)


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

    def test_arm_ik_frame_preserves_the_legacy_wrist_offset(self) -> None:
        for side in ("left", "right"):
            with self.subTest(side=side):
                wrist = self.model.frames[
                    self.model.getFrameId(f"{side}_wrist_yaw_link")
                ]
                palm = self.model.frames[
                    self.model.getFrameId(f"{side}_palm")
                ]
                self.assertEqual(palm.parent, wrist.parent)
                relative = wrist.placement.inverse() * palm.placement
                np.testing.assert_allclose(
                    relative.translation,
                    PALM_OFFSET_FROM_WRIST,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    relative.rotation,
                    np.eye(3),
                    atol=1e-12,
                )

    def test_continuous_hand_mapping_respects_limits_and_mimic(self) -> None:
        open_q = np.zeros(self.model.nq, dtype=np.float64)
        half_q = np.zeros(self.model.nq, dtype=np.float64)
        closed_q = np.zeros(self.model.nq, dtype=np.float64)
        _set_continuous_hand_configuration(
            self.model,
            open_q,
            np.array([0.0, 0.0]),
        )
        _set_continuous_hand_configuration(
            self.model,
            half_q,
            np.array([0.5, 0.5]),
        )
        _set_continuous_hand_configuration(
            self.model,
            closed_q,
            np.array([1.0, 1.0]),
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
                self.assertAlmostEqual(
                    _joint_q_value(self.model, half_q, joint_name),
                    0.5 * (open_value + closed_value),
                )

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
            for state_name, q in (
                ("open", open_q),
                ("half", half_q),
                ("closed", closed_q),
            ):
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

    def test_human_width_maps_to_continuous_robot_aperture(self) -> None:
        increasing = np.linspace(0.04, 0.09, 21)
        human_widths = np.stack(
            [increasing, increasing[::-1]],
            axis=1,
        )

        closures, targets, achieved, profiles = _retarget_hand_widths(
            self.model,
            human_widths,
            scale=0.84,
        )

        self.assertEqual(closures.shape, human_widths.shape)
        np.testing.assert_allclose(targets, 0.84 * human_widths)
        np.testing.assert_allclose(achieved, targets, atol=1e-5)
        self.assertTrue(np.all(np.diff(closures[:, 0]) < 0.0))
        self.assertTrue(np.all(np.diff(closures[:, 1]) > 0.0))
        self.assertEqual(len(profiles), 2)
        for profile in profiles:
            self.assertGreater(
                profile["open_width_m"],
                profile["minimum_width_m"],
            )
            self.assertGreater(
                profile["maximum_calibrated_closure"],
                0.0,
            )
            self.assertLess(
                profile["maximum_calibrated_closure"],
                1.0,
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

    def test_isaac_render_urdf_keeps_only_arm_and_hand_visuals(self) -> None:
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "g1_render.urdf"
            visible = _build_arm_hand_urdf(HAND_URDF, destination)
            root = ET.parse(destination).getroot()

            self.assertEqual(len(root.findall("link")), 63)
            self.assertEqual(len(root.findall("joint")), 62)
            self.assertEqual(len(visible), 38)
            for link in root.findall("link"):
                visuals = link.findall("visual")
                if visuals:
                    self.assertTrue(_is_render_link(link.attrib["name"]))
                for mesh in link.findall(".//mesh"):
                    path = (destination.parent / mesh.attrib["filename"]).resolve()
                    self.assertTrue(path.is_file(), path)


if __name__ == "__main__":
    unittest.main()
