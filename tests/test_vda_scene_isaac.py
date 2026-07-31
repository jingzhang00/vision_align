from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from repaint.vda_scene_isaac import (
    _next_playback_frame,
    _resolve_run_options,
    load_prepared_scene,
)


def _args(
    *,
    interactive: bool = False,
    loop: bool = False,
    headless: bool = True,
    output: Path | None = None,
    playback_fps: float | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        interactive=interactive,
        loop=loop,
        headless=headless,
        output=output,
        playback_fps=playback_fps,
    )


class VdaSceneIsaacRunOptionsTests(unittest.TestCase):
    def test_batch_mode_keeps_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "scene.mp4"
            options = _resolve_run_options(
                _args(output=output),
                source_fps=30.0,
            )

        self.assertFalse(options.interactive)
        self.assertFalse(options.loop)
        self.assertEqual(options.playback_fps, 30.0)
        self.assertEqual(options.output_video, output.resolve())
        self.assertEqual(
            options.report_path,
            output.with_suffix(".json").resolve(),
        )

    def test_batch_mode_requires_new_mp4_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "--output is required"):
            _resolve_run_options(_args(), source_fps=30.0)
        with self.assertRaisesRegex(ValueError, "must have an .mp4 suffix"):
            _resolve_run_options(
                _args(output=Path("scene.mov")),
                source_fps=30.0,
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing.mp4"
            output.touch()
            with self.assertRaises(FileExistsError):
                _resolve_run_options(
                    _args(output=output),
                    source_fps=30.0,
                )

    def test_interactive_mode_defaults_to_source_fps(self) -> None:
        options = _resolve_run_options(
            _args(
                interactive=True,
                loop=True,
                headless=False,
            ),
            source_fps=30.0,
        )
        self.assertTrue(options.interactive)
        self.assertTrue(options.loop)
        self.assertEqual(options.playback_fps, 30.0)
        self.assertIsNone(options.output_video)
        self.assertIsNone(options.report_path)

    def test_interactive_mode_constraints(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --no-headless"):
            _resolve_run_options(
                _args(interactive=True),
                source_fps=30.0,
            )
        with self.assertRaisesRegex(ValueError, "omit --output"):
            _resolve_run_options(
                _args(
                    interactive=True,
                    headless=False,
                    output=Path("scene.mp4"),
                ),
                source_fps=30.0,
            )
        with self.assertRaisesRegex(ValueError, "requires --interactive"):
            _resolve_run_options(
                _args(loop=True, output=Path("scene.mp4")),
                source_fps=30.0,
            )

    def test_playback_fps_validation(self) -> None:
        options = _resolve_run_options(
            _args(
                interactive=True,
                headless=False,
                playback_fps=15.0,
            ),
            source_fps=30.0,
        )
        self.assertEqual(options.playback_fps, 15.0)
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    _resolve_run_options(
                        _args(
                            interactive=True,
                            headless=False,
                            playback_fps=invalid,
                        ),
                        source_fps=30.0,
                    )


class VdaSceneIsaacPlaybackTests(unittest.TestCase):
    def test_next_frame_advances_stops_or_wraps(self) -> None:
        self.assertEqual(
            _next_playback_frame(0, frame_count=3, loop=False),
            (1, False, False),
        )
        self.assertEqual(
            _next_playback_frame(2, frame_count=3, loop=False),
            (2, True, True),
        )
        self.assertEqual(
            _next_playback_frame(2, frame_count=3, loop=True),
            (0, True, False),
        )
        self.assertEqual(
            _next_playback_frame(0, frame_count=1, loop=True),
            (0, True, False),
        )

    def test_next_frame_rejects_invalid_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            _next_playback_frame(0, frame_count=0, loop=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            _next_playback_frame(3, frame_count=3, loop=True)


class VdaSceneIsaacPreparedDataTests(unittest.TestCase):
    def _write_prepared(self, root: Path) -> None:
        (root / "rgb").mkdir()
        (root / "depth_mm").mkdir()
        for frame_id in range(2):
            self.assertTrue(
                cv2.imwrite(
                    str(root / "rgb" / f"{frame_id:05d}.png"),
                    np.full((4, 6, 3), 64 + frame_id, dtype=np.uint8),
                )
            )
            self.assertTrue(
                cv2.imwrite(
                    str(root / "depth_mm" / f"{frame_id:05d}.png"),
                    np.full((2, 3), 1000 + frame_id, dtype=np.uint16),
                )
            )
        urdf = root / "g1.urdf"
        urdf.write_text("<robot name='g1'/>", encoding="utf-8")
        poses = np.asarray(
            [
                [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 2.0],
                [0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        np.savez_compressed(
            root / "episode_data.npz",
            fps=np.asarray(30.0),
            source_size=np.asarray([6, 4]),
            mesh_size=np.asarray([3, 2]),
            frame_indices=np.asarray([4, 5]),
            camera_pose_cv_xyzw=poses,
            intrinsic=np.asarray(
                [[4.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
            ),
            mesh_intrinsic=np.asarray(
                [[2.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
            ),
            joint_names=np.asarray(["joint_a", "joint_b"]),
            joint_positions=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            root_pose_xyzw=poses,
            g1_urdf=np.asarray(str(urdf)),
        )

    def test_loads_frame_aligned_robot_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_prepared(root)
            scene = load_prepared_scene(root)

        self.assertEqual(scene.frame_count, 2)
        self.assertEqual(scene.joint_names, ("joint_a", "joint_b"))
        self.assertEqual(scene.joint_positions.shape, (2, 2))
        self.assertEqual(scene.root_poses_xyzw.shape, (2, 7))
        np.testing.assert_allclose(
            np.linalg.norm(scene.root_poses_xyzw[:, 3:7], axis=1),
            1.0,
        )

    def test_rejects_misaligned_robot_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_prepared(root)
            archive_path = root / "episode_data.npz"
            with np.load(archive_path, allow_pickle=False) as archive:
                payload = {
                    key: np.asarray(archive[key])
                    for key in archive.files
                }
            payload["joint_positions"] = np.zeros((1, 2))
            np.savez_compressed(archive_path, **payload)
            with self.assertRaisesRegex(
                ValueError,
                "joint_positions must have shape",
            ):
                load_prepared_scene(root)


if __name__ == "__main__":
    unittest.main()
