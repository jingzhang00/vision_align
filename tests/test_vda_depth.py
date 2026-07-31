from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from repaint.vda_depth import (
    DecodedVideoFrames,
    decode_contiguous_video_frames_rgb,
    depth_m_to_uint16_mm,
    infer_metric_vda_sequence,
    scale_and_resize_metric_depths,
    validate_frame_indices,
)


class VdaVideoDecodeTests(unittest.TestCase):
    def test_contiguous_subset_starts_at_ten_and_resizes_for_inference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "frames.avi"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"MJPG"),
                30.0,
                (32, 24),
            )
            self.assertTrue(writer.isOpened())
            for frame_index in range(14):
                frame = np.empty((24, 32, 3), dtype=np.uint8)
                frame[..., 0] = frame_index * 10
                frame[..., 1] = 50
                frame[..., 2] = 200 - frame_index * 5
                writer.write(frame)
            writer.release()

            decoded = decode_contiguous_video_frames_rgb(
                video,
                frame_indices=np.asarray([10, 11, 12], dtype=np.int64),
                max_resolution=16,
            )

        self.assertEqual(decoded.source_size, (32, 24))
        self.assertEqual(decoded.inference_size, (16, 12))
        self.assertEqual(decoded.frames_rgb.shape, (3, 12, 16, 3))
        self.assertEqual(decoded.video_frame_count, 14)
        np.testing.assert_array_equal(
            decoded.frame_indices,
            [10, 11, 12],
        )
        expected_rgb = np.asarray(
            [
                [150, 50, 100],
                [145, 50, 110],
                [140, 50, 120],
            ],
            dtype=np.float64,
        )
        observed_rgb = decoded.frames_rgb.mean(axis=(1, 2))
        np.testing.assert_allclose(observed_rgb, expected_rgb, atol=5.0)

    def test_frame_selection_must_be_contiguous(self) -> None:
        for invalid in (
            np.asarray([], dtype=np.int64),
            np.asarray([-1, 0], dtype=np.int64),
            np.asarray([10, 12], dtype=np.int64),
            np.asarray([[10, 11]], dtype=np.int64),
        ):
            with self.subTest(invalid=invalid.tolist()):
                with self.assertRaises(ValueError):
                    validate_frame_indices(invalid)


class VdaMetricDepthTests(unittest.TestCase):
    def test_scene_scale_is_applied_once_with_nearest_resize(self) -> None:
        raw_depth_m = np.asarray(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ],
            dtype=np.float32,
        )

        resized = scale_and_resize_metric_depths(
            raw_depth_m,
            scene_scale=0.5,
            mesh_size=(4, 4),
        )

        expected = np.asarray(
            [
                [
                    [0.5, 0.5, 1.0, 1.0],
                    [0.5, 0.5, 1.0, 1.0],
                    [1.5, 1.5, 2.0, 2.0],
                    [1.5, 1.5, 2.0, 2.0],
                ]
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(resized, expected)
        np.testing.assert_array_equal(
            depth_m_to_uint16_mm(resized[0]),
            expected[0] * 1000,
        )

    def test_vda_metric_scale_is_independent_from_scene_scale(self) -> None:
        raw_depth_m = np.full((1, 2, 2), 2.0, dtype=np.float32)

        resized = scale_and_resize_metric_depths(
            raw_depth_m,
            scene_scale=0.8,
            vda_metric_scale=0.525,
            mesh_size=(2, 2),
        )

        np.testing.assert_allclose(resized, 0.84, rtol=0.0, atol=1e-7)

    def test_vda_metric_scale_must_be_positive_and_finite(self) -> None:
        raw_depth_m = np.ones((1, 2, 2), dtype=np.float32)
        for invalid in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "vda_metric_scale",
                ):
                    scale_and_resize_metric_depths(
                        raw_depth_m,
                        scene_scale=0.8,
                        vda_metric_scale=invalid,
                        mesh_size=(2, 2),
                    )

    def test_depth_conversion_rounds_mm_and_zeros_invalid_values(self) -> None:
        depth_m = np.asarray(
            [
                [0.0014, 0.0016, 1.2346],
                [0.0, -1.0, np.nan],
            ],
            dtype=np.float32,
        )

        depth_mm = depth_m_to_uint16_mm(depth_m)

        self.assertEqual(depth_mm.dtype, np.uint16)
        np.testing.assert_array_equal(
            depth_mm,
            [
                [1, 2, 1235],
                [0, 0, 0],
            ],
        )

    def test_uint16_millimeter_overflow_is_rejected(self) -> None:
        with self.assertRaisesRegex(OverflowError, "uint16-millimeter"):
            scale_and_resize_metric_depths(
                np.full((1, 2, 2), 65.536, dtype=np.float32),
                scene_scale=1.0,
                mesh_size=(2, 2),
            )

        accepted = scale_and_resize_metric_depths(
            np.full((1, 2, 2), 65.0, dtype=np.float32),
            scene_scale=1.0,
            mesh_size=(2, 2),
        )
        np.testing.assert_array_equal(
            depth_m_to_uint16_mm(accepted[0]),
            np.full((2, 2), 65000, dtype=np.uint16),
        )

    def test_sparse_depth_is_rejected_before_writing(self) -> None:
        sparse = np.ones((1, 10, 10), dtype=np.float32)
        sparse[0, 0, :2] = 0.0

        with self.assertRaisesRegex(RuntimeError, "not dense"):
            scale_and_resize_metric_depths(
                sparse,
                scene_scale=1.0,
                mesh_size=(10, 10),
            )


class VdaInferenceValidationTests(unittest.TestCase):
    def _decoded(self) -> DecodedVideoFrames:
        return DecodedVideoFrames(
            frames_rgb=np.zeros((2, 6, 8, 3), dtype=np.uint8),
            fps=30.0,
            source_size=(16, 12),
            inference_size=(8, 6),
            frame_indices=np.asarray([10, 11], dtype=np.int64),
            video_frame_count=20,
        )

    def _infer(self, output: np.ndarray, fps: float = 30.0) -> np.ndarray:
        class FakeModel:
            def infer_video_depth(self, *_args, **_kwargs):
                return output, fps

        with (
            patch(
                "repaint.vda_depth.validate_metric_checkpoint",
                return_value=Path("/fake/metric_video_depth_anything_vitl.pth"),
            ),
            patch(
                "repaint.vda_depth.load_metric_vda_model",
                return_value=(FakeModel(), "cpu"),
            ),
            patch("torch.cuda.empty_cache"),
        ):
            return infer_metric_vda_sequence(
                self._decoded(),
                checkpoint=Path("/unused/checkpoint.pth"),
                encoder="vitl",
                input_size=518,
                fp32=True,
            )

    def test_dense_output_at_inference_resolution_is_accepted(self) -> None:
        output = np.full((2, 6, 8), 1.25, dtype=np.float32)
        np.testing.assert_array_equal(self._infer(output), output)

    def test_output_shape_must_match_inference_frames(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Unexpected VDA output shape"):
            self._infer(np.ones((2, 12, 16), dtype=np.float32))

    def test_nonfinite_or_insufficiently_dense_output_is_rejected(self) -> None:
        invalid_outputs = []
        nonfinite = np.ones((2, 6, 8), dtype=np.float32)
        nonfinite[0, 0, 0] = np.nan
        invalid_outputs.append(nonfinite)
        sparse = np.ones((2, 6, 8), dtype=np.float32)
        sparse[1, 0, 0] = 0.0
        invalid_outputs.append(sparse)

        for output in invalid_outputs:
            with self.subTest(nonfinite=not np.isfinite(output).all()):
                with self.assertRaisesRegex(RuntimeError, "finite dense depth"):
                    self._infer(output)

    def test_inference_must_preserve_fps(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "changed FPS"):
            self._infer(
                np.ones((2, 6, 8), dtype=np.float32),
                fps=29.0,
            )


if __name__ == "__main__":
    unittest.main()
