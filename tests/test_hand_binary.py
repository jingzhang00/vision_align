from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from human_to_robot.hand_binary import (
    CLOSE_RATIO_THRESHOLD,
    HAND_LANDMARKS,
    infer_binary_grasps_from_hdf5,
    infer_hand_binary,
)


REPOSITORY = Path(__file__).resolve().parents[1]
BASIC_PICK_PLACE = REPOSITORY / "dataset/basic_pick_place"


def _tracks_from_ratios(
    ratios: list[float],
    *,
    wrist_x: list[float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ratio = np.asarray(ratios, dtype=np.float64)
    frame_count = len(ratio)
    x = (
        np.zeros(frame_count, dtype=np.float64)
        if wrist_x is None
        else np.asarray(wrist_x, dtype=np.float64)
    )
    hand = np.stack(
        [x, np.zeros(frame_count), np.ones(frame_count)],
        axis=1,
    )
    virtual_finger = hand + np.array([0.0, 0.1, 0.0])
    positions = {
        "Hand": hand,
        "ThumbTip": (
            virtual_finger
            + np.stack(
                [
                    0.1 * ratio,
                    np.zeros(frame_count),
                    np.zeros(frame_count),
                ],
                axis=1,
            )
        ),
        "IndexFingerTip": virtual_finger.copy(),
        "MiddleFingerTip": virtual_finger.copy(),
    }
    confidences = {
        name: np.ones(frame_count, dtype=np.float64)
        for name in HAND_LANDMARKS
    }
    return positions, confidences


def _write_hdf5(
    path: Path,
    *,
    ratios: list[float],
    operation_hand: str | None = "right",
    right_wrist_x: list[float] | None = None,
    include_confidences: bool = True,
) -> None:
    right_positions, right_confidences = _tracks_from_ratios(
        ratios,
        wrist_x=right_wrist_x,
    )
    left_positions, left_confidences = _tracks_from_ratios(
        [0.1] * len(ratios),
    )
    with h5py.File(path, "w") as handle:
        if operation_hand is not None:
            handle.attrs["environment"] = (
                f"table:metal, task:test, hand:{operation_hand}"
            )
        handle.create_dataset(
            "camera/intrinsic",
            data=np.array(
                [
                    [100.0, 0.0, 50.0],
                    [0.0, 100.0, 50.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        handle.create_dataset(
            "transforms/camera",
            data=np.repeat(
                np.eye(4, dtype=np.float64)[None],
                len(ratios),
                axis=0,
            ),
        )
        for side, positions, confidences in (
            ("left", left_positions, left_confidences),
            ("right", right_positions, right_confidences),
        ):
            for name in HAND_LANDMARKS:
                transform = np.repeat(
                    np.eye(4, dtype=np.float64)[None],
                    len(ratios),
                    axis=0,
                )
                transform[:, :3, 3] = positions[name]
                handle.create_dataset(
                    f"transforms/{side}{name}",
                    data=transform,
                )
                if include_confidences:
                    handle.create_dataset(
                        f"confidences/{side}{name}",
                        data=confidences[name],
                    )


class StatelessHandBinaryTests(unittest.TestCase):
    def test_missing_hdf5_confidences_default_to_one(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.hdf5"
            _write_hdf5(
                path,
                ratios=[0.2, 0.5, 0.2],
                include_confidences=False,
            )

            result = infer_binary_grasps_from_hdf5(
                path,
                image_size=(100, 100),
            )["right"]

        np.testing.assert_array_equal(result.binary, [1, 0, 1])
        np.testing.assert_array_equal(
            result.diagnostics.frame_confidence,
            np.ones(3),
        )

    def test_exact_ratio_threshold_without_temporal_processing(self) -> None:
        ratios = [0.1, 0.5, 0.2, 0.6, CLOSE_RATIO_THRESHOLD]
        positions, confidences = _tracks_from_ratios(ratios)

        result = infer_hand_binary(
            positions,
            confidences,
            side="right",
        )

        np.testing.assert_allclose(
            result.diagnostics.normalized_jaw_ratio,
            ratios,
            atol=1e-12,
        )
        np.testing.assert_array_equal(result.binary, [1, 0, 1, 0, 1])
        self.assertEqual(
            [(run.start, run.end) for run in result.runs],
            [(0, 1), (2, 3), (4, 5)],
        )
        self.assertFalse(hasattr(result, "blend"))

    def test_a_frame_does_not_depend_on_past_or_future_frames(self) -> None:
        first_positions, first_confidences = _tracks_from_ratios(
            [0.2, 0.4, 0.2, 0.7],
        )
        second_positions, second_confidences = _tracks_from_ratios(
            [0.2, 0.4, 10.0, 0.01],
        )

        first = infer_hand_binary(
            first_positions,
            first_confidences,
            side="right",
        )
        second = infer_hand_binary(
            second_positions,
            second_confidences,
            side="right",
        )

        np.testing.assert_array_equal(first.binary[:2], second.binary[:2])
        np.testing.assert_allclose(
            first.diagnostics.normalized_jaw_ratio[:2],
            second.diagnostics.normalized_jaw_ratio[:2],
            atol=1e-12,
        )

    def test_invalid_frame_is_open_and_is_not_interpolated(self) -> None:
        positions, confidences = _tracks_from_ratios([0.2, 0.2, 0.2])
        positions["ThumbTip"][1] = np.nan
        confidences["IndexFingerTip"][2] = 0.0

        result = infer_hand_binary(
            positions,
            confidences,
            side="right",
        )

        np.testing.assert_array_equal(result.binary, [1, 0, 0])
        np.testing.assert_array_equal(
            result.diagnostics.geometry_valid,
            [True, False, False],
        )
        self.assertTrue(
            np.isnan(result.diagnostics.normalized_jaw_ratio[1])
        )

    def test_operation_hand_attribute_masks_inactive_hand(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.hdf5"
            _write_hdf5(
                path,
                ratios=[0.2, 0.2, 0.2],
                operation_hand="right",
            )

            result = infer_binary_grasps_from_hdf5(
                path,
                image_size=(100, 100),
            )

        np.testing.assert_array_equal(result["right"].binary, 1)
        np.testing.assert_array_equal(result["left"].binary, 0)
        self.assertEqual(
            result["right"].diagnostics.operation_hands,
            ("right",),
        )
        self.assertIn(
            "masked_by_hdf5_operation_hand_attribute",
            result["left"].diagnostics.warnings,
        )

    def test_raw_camera_projection_forces_out_of_frame_pose_open(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode.hdf5"
            _write_hdf5(
                path,
                ratios=[0.2, 0.2, 0.2],
                right_wrist_x=[0.0, 1.0, 0.0],
            )

            result = infer_binary_grasps_from_hdf5(
                path,
                image_size=(100, 100),
            )["right"]

        np.testing.assert_array_equal(result.binary, [1, 0, 1])
        np.testing.assert_array_equal(result.diagnostics.visible, [1, 0, 1])
        self.assertEqual(
            result.diagnostics.visibility_source,
            "camera_projection_explicit_image_size:rightHand",
        )

    @unittest.skipUnless(
        (BASIC_PICK_PLACE / "9.hdf5").exists(),
        "local episode 9 is not available",
    )
    def test_episode_9_real_grasp_is_closed_not_initialised_closed(self) -> None:
        result = infer_binary_grasps_from_hdf5(
            BASIC_PICK_PLACE / "9.hdf5",
            image_size=(1920, 1080),
        )

        np.testing.assert_array_equal(result["left"].binary, 0)
        right = result["right"]
        np.testing.assert_array_equal(right.binary[:18], 0)
        np.testing.assert_array_equal(right.binary[31:53], 1)
        np.testing.assert_array_equal(right.binary[54:56], 0)
        self.assertEqual(
            [(run.start, run.end) for run in right.runs],
            [(18, 54), (56, 57)],
        )

    @unittest.skipUnless(
        (BASIC_PICK_PLACE / "10.hdf5").exists(),
        "local episode 10 is not available",
    )
    def test_episode_10_active_right_closes_and_releases(self) -> None:
        result = infer_binary_grasps_from_hdf5(
            BASIC_PICK_PLACE / "10.hdf5",
            image_size=(1920, 1080),
        )

        np.testing.assert_array_equal(result["left"].binary, 0)
        right = result["right"]
        np.testing.assert_array_equal(right.binary[:31], 0)
        np.testing.assert_array_equal(right.binary[31:121], 1)
        np.testing.assert_array_equal(right.binary[121:], 0)
        self.assertEqual(
            [(run.start, run.end) for run in right.runs],
            [(31, 121)],
        )


if __name__ == "__main__":
    unittest.main()
