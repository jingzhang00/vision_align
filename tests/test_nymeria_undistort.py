from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nymeria.batch_undistort import (
    CameraCalibration,
    PinholeCalibration,
    build_remap,
    discover_episodes,
    load_rgb_calibration,
    scale_fisheye_parameters,
)


def _camera(parameters: tuple[float, ...]) -> CameraCalibration:
    return CameraCalibration(
        label="camera-rgb",
        model="FisheyeRadTanThinPrism",
        description="test",
        parameters=parameters,
        serial_number="rgb-test",
        transform_device_camera={},
        calibrated=True,
        utc_timestamp_ns=1,
        tracking_timestamp_us=2,
    )


class NymeriaUndistortTests(unittest.TestCase):
    def test_scale_parameters_uses_pixel_center_rule(self) -> None:
        parameters = tuple(float(value) for value in range(15))

        scaled = scale_fisheye_parameters(parameters, 0.5)

        self.assertEqual(scaled[0], 0.0)
        self.assertEqual(scaled[1], 0.25)
        self.assertEqual(scaled[2], 0.75)
        self.assertEqual(scaled[3:], parameters[3:])

    def test_remap_center_projects_to_fisheye_principal_point(self) -> None:
        parameters = (
            100.0,
            150.0,
            120.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        output = PinholeCalibration(
            width=9,
            height=9,
            fx=20.0,
            fy=20.0,
            cx=4.0,
            cy=4.0,
        )

        remap = build_remap(
            _camera(parameters),
            source_size=(160, 160),
            calibration_size=(320, 320),
            output=output,
        )

        np.testing.assert_allclose(
            [remap.map_x[4, 4], remap.map_y[4, 4]],
            [74.75, 59.75],
            rtol=0,
            atol=1e-6,
        )
        self.assertTrue(remap.valid_mask[4, 4])

    def test_loads_camera_rgb_from_first_jsonl_record(self) -> None:
        record = {
            "utc_timestamp_ns": 11,
            "tracking_timestamp_us": 22,
            "CameraCalibrations": [
                {
                    "Label": "camera-rgb",
                    "Calibrated": True,
                    "Projection": {
                        "Name": "FisheyeRadTanThinPrism",
                        "Description": "test model",
                        "Params": list(range(1, 16)),
                    },
                    "SerialNumber": "serial",
                    "T_Device_Camera": {"Translation": [1, 2, 3]},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "online_calibration.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            calibration = load_rgb_calibration(path)

        self.assertEqual(calibration.serial_number, "serial")
        self.assertEqual(calibration.parameters, tuple(map(float, range(1, 16))))
        self.assertEqual(calibration.utc_timestamp_ns, 11)
        self.assertEqual(calibration.tracking_timestamp_us, 22)

    def test_discovery_ignores_generated_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode = root / "episode-a"
            calibration = (
                episode
                / "recording_head"
                / "mps"
                / "slam"
                / "online_calibration.jsonl"
            )
            calibration.parent.mkdir(parents=True)
            calibration.touch()
            (episode / "video.mp4").touch()
            generated = episode / "undistorted"
            generated.mkdir()
            (generated / "video.mp4").touch()

            episodes = discover_episodes(root)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].name, "episode-a")
        self.assertEqual(
            episodes[0].output_directory,
            episode.resolve() / "undistorted",
        )


if __name__ == "__main__":
    unittest.main()
