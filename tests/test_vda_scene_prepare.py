from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import shutil

import cv2
import numpy as np

from repaint.vda_depth import DecodedVideoFrames
from repaint.vda_scene_prepare import (
    _auto_scale_sources,
    _load_template,
    _resolve_metric_scale,
    _temporary_output,
    _write_rgb_frames,
    build_parser,
)


class VdaScenePrepareTests(unittest.TestCase):
    def _write_template(self, root: Path, *, frame_count: int = 2) -> None:
        poses = np.repeat(
            np.asarray(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            frame_count,
            axis=0,
        )
        np.savez_compressed(
            root / "episode_data.npz",
            fps=np.asarray(30.0),
            source_size=np.asarray([32, 24]),
            mesh_size=np.asarray([8, 6]),
            frame_indices=np.arange(frame_count),
            camera_pose_cv_xyzw=poses,
            intrinsic=np.asarray(
                [[20.0, 0.0, 16.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]]
            ),
            mesh_intrinsic=np.asarray(
                [[5.0, 0.0, 4.0], [0.0, 5.0, 3.0], [0.0, 0.0, 1.0]]
            ),
            scene_scale=np.asarray(0.8),
        )

    def test_template_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_template(root)
            payload = _load_template(root)
            self.assertEqual(payload["frame_indices"].tolist(), [0, 1])
            self.assertEqual(float(payload["scene_scale"]), 0.8)

    def test_template_requires_multiple_contiguous_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_template(root, frame_count=1)
            with self.assertRaisesRegex(ValueError, "at least two contiguous"):
                _load_template(root)

    def test_temporary_output_is_sibling_and_refuses_existing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output = parent / "prepared"
            temporary_root, rgb_root, depth_root = _temporary_output(output)
            try:
                self.assertEqual(temporary_root.parent, output.parent)
                self.assertTrue(rgb_root.is_dir())
                self.assertTrue(depth_root.is_dir())
            finally:
                shutil.rmtree(temporary_root)

            output.mkdir()
            with self.assertRaises(FileExistsError):
                _temporary_output(output)

    def test_rgb_frames_are_decoded_from_input_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "input.mp4"
            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (32, 24),
            )
            self.assertTrue(writer.isOpened())
            writer.write(np.full((24, 32, 3), 30, dtype=np.uint8))
            writer.write(np.full((24, 32, 3), 220, dtype=np.uint8))
            writer.release()

            output = root / "rgb"
            output.mkdir()
            _write_rgb_frames(
                video,
                output,
                expected_count=2,
                expected_size=(32, 24),
            )
            frames = [
                cv2.imread(str(output / f"{index:05d}.png"))
                for index in range(2)
            ]
            self.assertTrue(all(frame is not None for frame in frames))
            self.assertLess(float(frames[0].mean()), 50.0)
            self.assertGreater(float(frames[1].mean()), 200.0)

    def test_auto_scale_is_default_but_fixed_keeps_legacy_template(self) -> None:
        base = [
            "--input-video",
            "handless.mp4",
            "--template-prepared",
            "template",
            "--output",
            "output",
        ]
        automatic = build_parser().parse_args(base)
        self.assertEqual(automatic.vda_scale_mode, "auto")
        with self.assertRaisesRegex(KeyError, "source_video"):
            _auto_scale_sources(
                args=automatic,
                template={},
            )

        fixed = build_parser().parse_args(
            [*base, "--vda-scale-mode", "fixed"]
        )
        self.assertEqual(fixed.vda_scale_mode, "fixed")
        self.assertEqual(fixed.vda_metric_scale, 0.525)

    def test_auto_scale_failure_requires_explicit_fixed_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original.mp4"
            handless = root / "handless.mp4"
            source_hdf5 = root / "source.hdf5"
            checkpoint = root / "checkpoint.pth"
            for path in (original, handless, source_hdf5, checkpoint):
                path.touch()
            base = [
                "--input-video",
                str(handless),
                "--template-prepared",
                str(root),
                "--output",
                str(root / "output"),
                "--vda-scale-source-video",
                str(original),
                "--vda-scale-source-hdf5",
                str(source_hdf5),
            ]
            decoded = DecodedVideoFrames(
                frames_rgb=np.empty((0, 0, 0, 3), dtype=np.uint8),
                fps=30.0,
                source_size=(32, 24),
                inference_size=(32, 24),
                frame_indices=np.arange(2),
                video_frame_count=2,
            )
            template = {"frame_indices": np.arange(2)}

            def rejected_report() -> dict[str, object]:
                return {
                    "decision": {
                        "status": "needs_fallback_or_review",
                        "reasons": ["no valid hand-anchor scale"],
                    },
                    "resolved_vda_metric_scale": None,
                }

            automatic = build_parser().parse_args(base)
            with (
                patch(
                    "repaint.vda_scene_prepare."
                    "infer_and_estimate_handless_metric_scale",
                    return_value=rejected_report(),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "failed quality gates",
                ),
            ):
                _resolve_metric_scale(
                    args=automatic,
                    template=template,
                    handless_decoded=decoded,
                    handless_depth_raw_m=np.ones((2, 24, 32)),
                    checkpoint=checkpoint,
                )

            fallback = build_parser().parse_args(
                [*base, "--vda-scale-failure-policy", "fixed"]
            )
            with patch(
                "repaint.vda_scene_prepare."
                "infer_and_estimate_handless_metric_scale",
                return_value=rejected_report(),
            ):
                scale, report = _resolve_metric_scale(
                    args=fallback,
                    template=template,
                    handless_decoded=decoded,
                    handless_depth_raw_m=np.ones((2, 24, 32)),
                    checkpoint=checkpoint,
                )

        self.assertEqual(scale, 0.525)
        self.assertEqual(report["resolved_mode"], "fixed_fallback")
        self.assertIsNone(report["automatic_candidate_vda_metric_scale"])
        self.assertEqual(
            report["fallback_reason"],
            ["no valid hand-anchor scale"],
        )


if __name__ == "__main__":
    unittest.main()
