from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import h5py
import numpy as np

from repaint.vda_scale_validation import (
    _anchor_summary_is_supported,
    _analyze,
    _bootstrap_product,
    _depth_cache_key,
    _frame_equal_summary,
    _load_valid_depth_cache,
    _load_hdf5_geometry,
    _scaled_intrinsic,
    _write_depth_cache,
    estimate_handless_metric_scale,
)


class VdaScaleValidationTests(unittest.TestCase):
    def test_frame_equal_summary_does_not_weight_dense_frame_more(self) -> None:
        summary, frame_medians = _frame_equal_summary(
            [
                np.full(1000, 1.0),
                np.asarray([3.0]),
                np.asarray([]),
            ]
        )

        self.assertEqual(summary["supporting_frames"], 2)
        self.assertEqual(summary["raw_sample_count"], 1001)
        self.assertEqual(summary["p50"], 2.0)
        np.testing.assert_allclose(frame_medians[:2], [1.0, 3.0])
        self.assertTrue(np.isnan(frame_medians[2]))

    def test_independent_frame_bootstrap_centers_on_product_of_medians(
        self,
    ) -> None:
        first = np.linspace(0.49, 0.51, 100)
        second = np.linspace(0.89, 0.91, 100)

        result = _bootstrap_product(first, second, samples=500)

        self.assertEqual(result["first_frame_count"], 100)
        self.assertEqual(result["second_frame_count"], 100)
        self.assertAlmostEqual(result["p50"], 0.45, places=3)
        self.assertLess(result["p025"], 0.45)
        self.assertGreater(result["p975"], 0.45)

    def test_intrinsic_scales_with_non_square_resize(self) -> None:
        intrinsic = np.asarray(
            [[800.0, 0.0, 960.0], [0.0, 810.0, 540.0], [0.0, 0.0, 1.0]]
        )

        scaled = _scaled_intrinsic(
            intrinsic,
            source_size=(1920, 1080),
            target_size=(640, 360),
        )

        np.testing.assert_allclose(
            scaled,
            [[800.0 / 3.0, 0.0, 320.0],
             [0.0, 270.0, 180.0],
             [0.0, 0.0, 1.0]],
        )

    def test_root_anchor_selection_uses_final_dynamic_support_gate(self) -> None:
        sparse_root = {
            "supporting_frames": 10,
            "raw_sample_count": 40,
            "log_mad": 0.01,
        }
        supported_root = {
            **sparse_root,
            "supporting_frames": 25,
        }

        self.assertFalse(
            _anchor_summary_is_supported(
                sparse_root,
                selected_frame_count=100,
            )
        )
        self.assertTrue(
            _anchor_summary_is_supported(
                supported_root,
                selected_frame_count=100,
            )
        )

    def test_depth_cache_is_bound_to_complete_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.bin"
            checkpoint = root / "checkpoint.bin"
            video.write_bytes(b"video-a")
            checkpoint.write_bytes(b"checkpoint-a")
            metadata = (128, 96, 3, 30.0)
            key = _depth_cache_key(
                video=video,
                video_metadata=metadata,
                video_sha256="video-sha",
                frame_count=3,
                max_resolution=1280,
                checkpoint=checkpoint,
                checkpoint_sha256="checkpoint-sha",
                encoder="vitl",
                input_size=518,
                fp32=False,
            )
            output = root / "depth.npy"
            _write_depth_cache(
                output=output,
                depths=np.ones((3, 4, 5), dtype=np.float32),
                cache_key=key,
            )

            reused = _load_valid_depth_cache(
                output=output,
                expected_key=key,
                frame_count=3,
            )
            self.assertIsNotNone(reused)
            changed_key = {
                **key,
                "input_size": 384,
            }
            self.assertIsNone(
                _load_valid_depth_cache(
                    output=output,
                    expected_key=changed_key,
                    frame_count=3,
                )
            )

    def test_selected_subset_recovers_handless_metric_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_video = root / "original.avi"
            handless_video = root / "handless.avi"
            original_writer = cv2.VideoWriter(
                str(original_video),
                cv2.VideoWriter_fourcc(*"MJPG"),
                30.0,
                (128, 96),
            )
            handless_writer = cv2.VideoWriter(
                str(handless_video),
                cv2.VideoWriter_fourcc(*"MJPG"),
                30.0,
                (128, 96),
            )
            self.assertTrue(original_writer.isOpened())
            self.assertTrue(handless_writer.isOpened())
            for _ in range(20):
                original = np.zeros((96, 128, 3), dtype=np.uint8)
                original[35:62, 22:43] = 255
                original[35:62, 85:106] = 255
                original_writer.write(original)
                handless_writer.write(
                    np.zeros((96, 128, 3), dtype=np.uint8)
                )
            # A declared HDF5/handless frame-zero prefix may be shorter than
            # its original video (the real Nymeria c024 case has 450 vs 460).
            original_writer.write(original)
            original_writer.release()
            handless_writer.release()

            hdf5_path = root / "source.hdf5"
            camera = np.repeat(np.eye(4)[None], 20, axis=0)
            left = camera.copy()
            right = camera.copy()
            left[:, :3, 3] = [-0.4, 0.0, 1.0]
            right[:, :3, 3] = [0.4, 0.0, 1.0]
            with h5py.File(hdf5_path, "w") as handle:
                camera_group = handle.create_group("camera")
                camera_group.create_dataset(
                    "intrinsic",
                    data=np.asarray(
                        [
                            [80.0, 0.0, 64.0],
                            [0.0, 80.0, 48.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ),
                )
                transforms = handle.create_group("transforms")
                transforms.create_dataset("camera", data=camera)
                transforms.create_dataset("leftHand", data=left)
                transforms.create_dataset("rightHand", data=right)
                metadata = handle.create_group("metadata")
                metadata.attrs["source_video"] = str(original_video.resolve())

            frame_indices = np.arange(5, 20, dtype=np.int64)
            report = estimate_handless_metric_scale(
                original_video=original_video,
                handless_video=handless_video,
                source_hdf5=hdf5_path,
                frame_indices=frame_indices,
                original_depth_raw_m=np.full(
                    (15, 96, 128),
                    2.0,
                    dtype=np.float32,
                ),
                handless_depth_raw_m=np.full(
                    (15, 96, 128),
                    4.0,
                    dtype=np.float32,
                ),
                source_size=(128, 96),
                inference_size=(128, 96),
            )
            da3_root = root / "da3"
            da3_root.mkdir()
            for source_frame in frame_indices:
                self.assertTrue(
                    cv2.imwrite(
                        str(da3_root / f"{source_frame:05d}.png"),
                        np.full((48, 64), 1000, dtype=np.uint16),
                    )
                )
            (
                intrinsic,
                _camera,
                points_camera,
                confidence,
                labels,
                _baseline,
                hdf5_frame_count,
            ) = _load_hdf5_geometry(
                hdf5_path,
                frame_indices=frame_indices,
            )
            self.assertEqual(hdf5_frame_count, 20)
            da3_analysis = _analyze(
                original_video=original_video,
                handless_video=handless_video,
                mask_dir=None,
                original_depth=np.full(
                    (15, 96, 128),
                    2.0,
                    dtype=np.float32,
                ),
                handless_depth=np.full(
                    (15, 96, 128),
                    4.0,
                    dtype=np.float32,
                ),
                source_size=(128, 96),
                inference_size=(128, 96),
                intrinsic=intrinsic,
                points_camera=points_camera,
                confidence=confidence,
                labels=labels,
                frame_indices=frame_indices,
                da3_depth_dir=da3_root,
                da3_scene_scale=1.0,
            )
            with h5py.File(hdf5_path, "a") as handle:
                transforms = handle["transforms"]
                del transforms["camera"]
                del transforms["leftHand"]
                del transforms["rightHand"]
                longer = np.repeat(np.eye(4)[None], 21, axis=0)
                transforms.create_dataset("camera", data=longer)
                transforms.create_dataset("leftHand", data=longer)
                transforms.create_dataset("rightHand", data=longer)
            with self.assertRaisesRegex(
                ValueError,
                "Handless video and source HDF5 total frame counts disagree",
            ):
                estimate_handless_metric_scale(
                    original_video=original_video,
                    handless_video=handless_video,
                    source_hdf5=hdf5_path,
                    frame_indices=frame_indices,
                    original_depth_raw_m=np.full(
                        (15, 96, 128),
                        2.0,
                        dtype=np.float32,
                    ),
                    handless_depth_raw_m=np.full(
                        (15, 96, 128),
                        4.0,
                        dtype=np.float32,
                    ),
                    source_size=(128, 96),
                    inference_size=(128, 96),
                )

        self.assertEqual(report["decision"]["status"], "validated")
        self.assertAlmostEqual(
            report["resolved_vda_metric_scale"],
            0.25,
            places=6,
        )
        self.assertEqual(
            report["analysis"]["selected_frame_indices"],
            frame_indices.tolist(),
        )
        self.assertEqual(
            da3_analysis["selected_frame_indices"].tolist(),
            frame_indices.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
