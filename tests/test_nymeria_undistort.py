from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from nymeria_scripts.batch_undistort import (
    PinholeCalibration,
    build_pinhole_calibration,
    discover_episodes,
    process_episode,
)


def _arguments(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "width": 384,
        "height": 384,
        "focal_length": None,
        "principal_x": None,
        "principal_y": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _make_episode(root: Path, name: str = "episode-a") -> Path:
    episode = root / name
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
    return episode


class NymeriaIntrinsicsTests(unittest.TestCase):
    def test_default_virtual_pinhole_scales_with_resolution(self) -> None:
        calibration = build_pinhole_calibration(
            _arguments(width=768, height=512)
        )

        self.assertEqual(
            calibration,
            PinholeCalibration(
                width=768,
                height=512,
                fx=520.0,
                fy=520.0,
                cx=383.5,
                cy=255.5,
            ),
        )

    def test_discovery_ignores_generated_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode = _make_episode(root)
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

    def test_process_writes_only_intrinsics_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode_directory = _make_episode(root)
            output_directory = episode_directory / "undistorted"
            output_directory.mkdir()
            sibling_video = output_directory / "video.mp4"
            sibling_video.write_bytes(b"keep-video")
            episode = discover_episodes(root)[0]
            calibration = build_pinhole_calibration(_arguments())

            first = process_episode(
                episode,
                output=calibration,
                overwrite=False,
            )
            second = process_episode(
                episode,
                output=calibration,
                overwrite=False,
            )
            payload = json.loads(
                (output_directory / "camera_intrinsics.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(first.status, "completed")
            self.assertEqual(second.status, "skipped")
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["camera_model"], "pinhole")
            self.assertEqual(payload["distortion_coefficients"], [])
            self.assertEqual(payload["resolution"], [384, 384])
            self.assertEqual(
                payload["K"],
                [
                    [260.0, 0.0, 191.5],
                    [0.0, 260.0, 191.5],
                    [0.0, 0.0, 1.0],
                ],
            )
            self.assertEqual(sibling_video.read_bytes(), b"keep-video")
            self.assertFalse((output_directory / "_SUCCESS").exists())

    def test_accepts_matching_legacy_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode_directory = _make_episode(root)
            output_directory = episode_directory / "undistorted"
            output_directory.mkdir()
            output_json = output_directory / "camera_intrinsics.json"
            output_json.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "undistorted": {
                            "resolution": [384, 384],
                            "K": [
                                [260.0, 0.0, 191.5],
                                [0.0, 260.0, 191.5],
                                [0.0, 0.0, 1.0],
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = process_episode(
                discover_episodes(root)[0],
                output=build_pinhole_calibration(_arguments()),
                overwrite=False,
            )

            self.assertEqual(result.status, "skipped")

    def test_different_existing_k_requires_overwrite_and_preserves_video(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode_directory = _make_episode(root)
            output_directory = episode_directory / "undistorted"
            output_directory.mkdir()
            output_json = output_directory / "camera_intrinsics.json"
            output_json.write_text(
                json.dumps(
                    {
                        "resolution": [384, 384],
                        "K": [
                            [100.0, 0.0, 191.5],
                            [0.0, 100.0, 191.5],
                            [0.0, 0.0, 1.0],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sibling_video = output_directory / "video.mp4"
            sibling_video.write_bytes(b"keep-video")
            episode = discover_episodes(root)[0]
            requested = build_pinhole_calibration(_arguments())

            with self.assertRaises(FileExistsError):
                process_episode(
                    episode,
                    output=requested,
                    overwrite=False,
                )
            result = process_episode(
                episode,
                output=requested,
                overwrite=True,
            )

            self.assertEqual(result.status, "completed")
            self.assertEqual(sibling_video.read_bytes(), b"keep-video")
            self.assertEqual(
                json.loads(output_json.read_text(encoding="utf-8"))["K"][0][0],
                260.0,
            )


if __name__ == "__main__":
    unittest.main()
