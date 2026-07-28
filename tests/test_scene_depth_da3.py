from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from human_to_robot.pipeline import _scene_depth_complete
from human_to_robot.scene_depth_da3 import (
    _depth_to_millimeters,
    _metric_depth,
    _validate_local_model,
)


REPOSITORY = Path(__file__).resolve().parents[1]


class SceneDepthDa3Tests(unittest.TestCase):
    def test_repository_checkpoint_is_native_metric(self) -> None:
        payload = _validate_local_model(
            REPOSITORY / "models/DA3METRIC-LARGE"
        )
        self.assertEqual(payload["model_name"], "da3metric-large")

    def test_metric_conversion_uses_processed_focal_and_trajectory_scale(
        self,
    ) -> None:
        raw = np.ones((2, 2, 3), dtype=np.float32)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
        intrinsics[0, 0, 0] = 300.0
        intrinsics[0, 1, 1] = 300.0
        intrinsics[1, 0, 0] = 600.0
        intrinsics[1, 1, 1] = 600.0

        depth = _metric_depth(
            raw,
            intrinsics,
            trajectory_scale=0.84,
        )

        np.testing.assert_allclose(depth[0], 0.84)
        np.testing.assert_allclose(depth[1], 1.68)

    def test_depth_cache_is_uint16_mm_with_zero_for_invalid(self) -> None:
        depth = np.asarray(
            [[0.8, 0.8006], [np.nan, -1.0]],
            dtype=np.float32,
        )

        depth_mm = _depth_to_millimeters(
            depth,
            output_size=(2, 2),
        )

        self.assertEqual(depth_mm.dtype, np.uint16)
        np.testing.assert_array_equal(
            depth_mm,
            [[800, 801], [0, 0]],
        )

    def test_scene_depth_cache_validates_every_frame_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_video = root / "inpaint.mp4"
            scene_video.write_bytes(b"clean-background")
            depth_root = root / "scene_depth_mm"
            depth_root.mkdir()
            digest = hashlib.sha256()
            for frame_id, value in enumerate((100, 200)):
                path = depth_root / f"{frame_id:05d}.png"
                self.assertTrue(
                    cv2.imwrite(
                        str(path),
                        np.full((3, 4), value, dtype=np.uint16),
                    )
                )
                digest.update(path.name.encode("ascii"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
            stat = scene_video.stat()
            report = {
                "schema_version": 2,
                "model": "DA3METRIC-LARGE",
                "native_metric": True,
                "frame_count": 2,
                "source_resolution": [4, 3],
                "scene_video": str(scene_video.resolve()),
                "scene_video_fingerprint": {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                },
                "depth_png_sha256": digest.hexdigest(),
            }
            report_path = root / "scene_depth_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            self.assertTrue(
                _scene_depth_complete(
                    depth_root,
                    report_path,
                    2,
                    expected_size=(4, 3),
                    scene_video=scene_video,
                )
            )
            self.assertTrue(
                cv2.imwrite(
                    str(depth_root / "00001.png"),
                    np.full((3, 4), 201, dtype=np.uint16),
                )
            )
            self.assertFalse(
                _scene_depth_complete(
                    depth_root,
                    report_path,
                    2,
                    expected_size=(4, 3),
                    scene_video=scene_video,
                )
            )


if __name__ == "__main__":
    unittest.main()
