from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import av
import cv2
import numpy as np

from human_to_robot.video_io import (
    source_encoding_profile,
    video_metadata,
    write_video,
)


class SourceEncodingTests(unittest.TestCase):
    @staticmethod
    def _write_source(path: Path) -> list[np.ndarray]:
        size = (96, 64)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            12.0,
            size,
        )
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV mp4v encoder is unavailable")
        frames = [
            np.full((size[1], size[0], 3), index * 30, dtype=np.uint8)
            for index in range(4)
        ]
        for frame in frames:
            writer.write(frame)
        writer.release()
        return frames

    def test_output_inherits_source_codec_pixel_format_and_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source_frames = self._write_source(source)
            output_frames = [
                cv2.resize(frame, (48, 32)) for frame in source_frames
            ]
            output = root / "output.mp4"

            profile = write_video(
                output,
                output_frames,
                1.0,
                source_video=source,
                hardware_encoding=False,
            )

            self.assertEqual(profile.source_codec, "mpeg4")
            self.assertEqual(profile.encoder, "mpeg4")
            self.assertEqual(profile.codec_profile, "Simple Profile")
            self.assertEqual(profile.pixel_format, "yuv420p")
            self.assertEqual(float(profile.frame_rate), 12.0)
            self.assertEqual(profile.output_resolution, (48, 32))
            self.assertIsNotNone(profile.bit_rate)
            self.assertEqual(video_metadata(output)[:3], (48, 32, 4))
            self.assertAlmostEqual(video_metadata(output)[3], 12.0)
            with av.open(str(output)) as container:
                stream = container.streams.video[0]
                self.assertEqual(stream.codec_context.name, "mpeg4")
                self.assertEqual(
                    stream.codec_context.profile,
                    profile.codec_profile,
                )
                self.assertEqual(stream.codec_context.format.name, "yuv420p")
                self.assertEqual(stream.average_rate, profile.frame_rate)
                self.assertEqual(
                    int(stream.codec_context.color_primaries),
                    profile.color_primaries,
                )

    def test_bit_rate_scales_with_output_pixel_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.mp4"
            self._write_source(source)

            full = source_encoding_profile(
                source,
                output_size=(96, 64),
                hardware_encoding=False,
            )
            quarter = source_encoding_profile(
                source,
                output_size=(48, 32),
                hardware_encoding=False,
            )

            self.assertIsNotNone(full.bit_rate)
            self.assertIsNotNone(quarter.bit_rate)
            assert full.bit_rate is not None
            assert quarter.bit_rate is not None
            self.assertGreaterEqual(quarter.bit_rate, 100_000)
            self.assertLessEqual(quarter.bit_rate, full.bit_rate)

    def test_hardware_encoding_selects_nvenc_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.mp4"
            self._write_source(source)

            with mock.patch(
                "human_to_robot.video_io._nvenc_encoder_available",
                return_value=True,
            ):
                profile = source_encoding_profile(
                    source,
                    output_size=(96, 64),
                    hardware_encoding=True,
                )

            self.assertTrue(profile.hardware_encoding_requested)
            self.assertTrue(profile.hardware_encoding_active)
            self.assertEqual(profile.encoder, "h264_nvenc")
            self.assertIsNone(profile.codec_profile)

    def test_hardware_encoding_falls_back_when_nvenc_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.mp4"
            self._write_source(source)

            with mock.patch(
                "human_to_robot.video_io._nvenc_encoder_available",
                return_value=False,
            ):
                profile = source_encoding_profile(
                    source,
                    output_size=(96, 64),
                    hardware_encoding=True,
                )

            self.assertTrue(profile.hardware_encoding_requested)
            self.assertFalse(profile.hardware_encoding_active)
            self.assertEqual(profile.encoder, "mpeg4")


if __name__ == "__main__":
    unittest.main()
