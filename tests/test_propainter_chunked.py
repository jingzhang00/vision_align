from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import av
import cv2
import numpy as np

from human_to_robot.pipeline import build_parser as build_pipeline_parser
from human_to_robot.propainter_chunked import (
    BackgroundReference,
    ChunkPlan,
    MaskFrameInfo,
    _apply_background_cleanup,
    compose_crop,
    plan_chunks,
    plan_transitions,
    run_chunked_propainter,
    validate_chunk_plan,
)
from human_to_robot.propainter_worker import run_manifest
from human_to_robot.video_io import video_metadata


class ChunkPlannerTest(unittest.TestCase):
    def test_transition_uses_shared_context_at_adjacent_core_boundary(
        self,
    ) -> None:
        chunks = [
            ChunkPlan(0, 20, 0, 25, (0, 0, 64, 64), 102_400),
            ChunkPlan(20, 40, 15, 45, (8, 0, 72, 64), 122_880),
            ChunkPlan(60, 70, 55, 75, (0, 0, 64, 64), 81_920),
        ]

        transitions = plan_transitions(chunks, transition_frames=16)

        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].previous_chunk, 0)
        self.assertEqual(transitions[0].next_chunk, 1)
        self.assertEqual(
            (transitions[0].start, transitions[0].end),
            (15, 25),
        )

    def test_planner_splits_active_ranges_and_respects_budgets(self) -> None:
        infos = [
            MaskFrameInfo(
                index=index,
                path=Path(f"{index:05d}.png"),
                bbox=(8, 8, 24, 24) if 2 <= index < 18 else None,
                area=256 if 2 <= index < 18 else 0,
            )
            for index in range(20)
        ]

        chunks = plan_chunks(
            infos,
            frame_count=20,
            target_size=(128, 96),
            context_frames=2,
            roi_margin=4,
            max_chunk_pixel_frames=4 * 128 * 96,
            max_clip_frames=8,
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].core_start, 2)
        self.assertEqual(chunks[-1].core_end, 18)
        self.assertTrue(all(chunk.clip_length <= 8 for chunk in chunks))
        self.assertTrue(
            all(chunk.pixel_frames <= 4 * 128 * 96 for chunk in chunks)
        )
        validate_chunk_plan(
            infos,
            chunks,
            frame_count=20,
            target_size=(128, 96),
            max_chunk_pixel_frames=4 * 128 * 96,
            max_clip_frames=8,
        )
        self.assertEqual(
            [(chunk.core_start, chunk.core_end) for chunk in chunks],
            [(2, 6), (6, 10), (10, 14), (14, 18)],
        )

    def test_planner_returns_no_chunks_for_empty_masks(self) -> None:
        infos = [
            MaskFrameInfo(index=index, path=Path(f"{index:05d}.png"), bbox=None, area=0)
            for index in range(5)
        ]

        chunks = plan_chunks(
            infos,
            frame_count=5,
            target_size=(128, 96),
        )

        self.assertEqual(chunks, [])


class RoiCompositionTest(unittest.TestCase):
    def test_compose_crop_changes_only_masked_pixels(self) -> None:
        source = np.full((8, 10, 3), (10, 20, 30), dtype=np.uint8)
        crop = np.full((6, 6, 3), (200, 150, 100), dtype=np.uint8)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[3:5, 4:7] = 255

        result = compose_crop(
            source,
            crop,
            mask,
            roi=(2, 1, 8, 7),
            mask_dilation=0,
        )

        expected = source.copy()
        expected[3:5, 4:7] = (200, 150, 100)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(source, np.full_like(source, (10, 20, 30)))

    def test_compose_crop_honors_mask_dilation(self) -> None:
        source = np.zeros((7, 7, 3), dtype=np.uint8)
        crop = np.full((7, 7, 3), 255, dtype=np.uint8)
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[3, 3] = 255

        result = compose_crop(
            source,
            crop,
            mask,
            roi=(0, 0, 7, 7),
            mask_dilation=1,
        )

        self.assertEqual(int(np.count_nonzero(result[:, :, 0])), 5)
        self.assertEqual(result[3, 3].tolist(), [255, 255, 255])
        self.assertEqual(result[0, 0].tolist(), [0, 0, 0])


class BackgroundCleanupTest(unittest.TestCase):
    def test_replaces_mask_and_connected_neutral_shadow(self) -> None:
        background = np.full((48, 64, 3), (140, 160, 180), dtype=np.uint8)
        source = background.copy()
        source[16:34, 12:28] = (90, 120, 180)
        source[18:32, 28:38] = (115, 135, 155)
        source[5:11, 45:53] = (220, 50, 20)
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[16:34, 12:28] = 1
        reference = BackgroundReference(
            image=background,
            observed=np.ones((48, 64), dtype=bool),
            sample_count=8,
            observed_fraction=1.0,
            median_error=1.0,
            reliable=True,
        )

        result = _apply_background_cleanup(
            source=source,
            inpainted=source.copy(),
            mask=mask,
            reference=reference,
        )

        np.testing.assert_allclose(result[24, 20], background[24, 20], atol=1)
        self.assertLess(
            np.mean(
                np.abs(
                    result[24, 32].astype(np.int16)
                    - background[24, 32].astype(np.int16)
                )
            ),
            np.mean(
                np.abs(
                    source[24, 32].astype(np.int16)
                    - background[24, 32].astype(np.int16)
                )
            ),
        )
        np.testing.assert_array_equal(result[8, 48], source[8, 48])

    def test_unreliable_reference_is_not_applied(self) -> None:
        source = np.full((16, 16, 3), 80, dtype=np.uint8)
        inpainted = np.full_like(source, 120)
        reference = BackgroundReference(
            image=np.full_like(source, 200),
            observed=np.ones((16, 16), dtype=bool),
            sample_count=4,
            observed_fraction=1.0,
            median_error=30.0,
            reliable=False,
        )

        result = _apply_background_cleanup(
            source=source,
            inpainted=inpainted,
            mask=np.ones((16, 16), dtype=np.uint8),
            reference=reference,
        )

        np.testing.assert_array_equal(result, inpainted)

    def test_failed_camera_alignment_keeps_propainter_result(self) -> None:
        source = np.full((16, 16, 3), 80, dtype=np.uint8)
        inpainted = np.full_like(source, 120)
        reference = BackgroundReference(
            image=np.full_like(source, 200),
            observed=np.ones((16, 16), dtype=bool),
            sample_count=1,
            observed_fraction=1.0,
            median_error=1.0,
            reliable=True,
            sample_indices=(0,),
            samples=np.full((1, 16, 16, 3), 200, dtype=np.uint8),
            sample_invalid=np.zeros((1, 16, 16), dtype=bool),
            sample_descriptors=(np.zeros((1, 128), dtype=np.float32),),
        )

        result = _apply_background_cleanup(
            source=source,
            inpainted=inpainted,
            mask=np.ones((16, 16), dtype=np.uint8),
            frame_index=10,
            reference=reference,
        )

        np.testing.assert_array_equal(result, inpainted)


class PipelineDefaultsTest(unittest.TestCase):
    def test_batch_pipeline_does_not_expose_resolution_overrides(self) -> None:
        args = build_pipeline_parser().parse_args(
            ["--folder", "dataset", "--output", "outputs"]
        )
        software_args = build_pipeline_parser().parse_args(
            [
                "--folder",
                "dataset",
                "--output",
                "outputs",
                "--no-hardware-encoding",
            ]
        )

        self.assertFalse(hasattr(args, "width"))
        self.assertFalse(hasattr(args, "height"))
        self.assertTrue(args.hardware_encoding)
        self.assertFalse(software_args.hardware_encoding)


class PersistentWorkerTest(unittest.TestCase):
    def test_manifest_loads_one_engine_for_multiple_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            propainter_root = root / "ProPainter"
            engine_source = (
                propainter_root
                / "web-demos/hugging_face/inpainter/base_inpainter.py"
            )
            engine_source.parent.mkdir(parents=True)
            engine_source.touch()

            jobs: list[dict[str, object]] = []
            expected_colors = [(10, 20, 30), (40, 50, 60)]
            for chunk_index, color in enumerate(expected_colors):
                chunk_root = root / f"chunk_{chunk_index:04d}"
                input_frames = chunk_root / "input_frames"
                input_masks = chunk_root / "input_masks"
                output_frames = chunk_root / "output_frames"
                input_frames.mkdir(parents=True)
                input_masks.mkdir(parents=True)
                frame = np.full((8, 8, 3), color, dtype=np.uint8)
                mask = np.full((8, 8), 255, dtype=np.uint8)
                self.assertTrue(
                    cv2.imwrite(str(input_frames / "00000.png"), frame)
                )
                self.assertTrue(
                    cv2.imwrite(str(input_masks / "00000.png"), mask)
                )
                jobs.append(
                    {
                        "chunk_index": chunk_index,
                        "input_frames": str(input_frames),
                        "input_masks": str(input_masks),
                        "output_frames": str(output_frames),
                        "expected_count": 1,
                    }
                )

            manifest_path = root / "jobs.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "parameters": {
                            "subvideo_length": 20,
                            "neighbor_length": 10,
                            "ref_stride": 10,
                            "raft_iter": 20,
                            "mask_dilation": 0,
                            "fp16": True,
                        },
                        "jobs": jobs,
                    }
                ),
                encoding="utf-8",
            )

            class FakeInpainter:
                load_count = 0
                call_count = 0

                def __init__(self, *_args, **_kwargs) -> None:
                    type(self).load_count += 1

                def inpaint(self, frames, _masks, **_kwargs):
                    type(self).call_count += 1
                    return frames

            inpainter_package = types.ModuleType("inpainter")
            inpainter_module = types.ModuleType(
                "inpainter.base_inpainter"
            )
            inpainter_module.ProInpainter = FakeInpainter
            model_package = types.ModuleType("model")
            misc_module = types.ModuleType("model.misc")
            misc_module.get_device = lambda: "cpu"
            with mock.patch.dict(
                "sys.modules",
                {
                    "inpainter": inpainter_package,
                    "inpainter.base_inpainter": inpainter_module,
                    "model": model_package,
                    "model.misc": misc_module,
                },
            ):
                run_manifest(manifest_path, propainter_root)

            self.assertEqual(FakeInpainter.load_count, 1)
            self.assertEqual(FakeInpainter.call_count, 2)
            for chunk_index, color in enumerate(expected_colors):
                output = cv2.imread(
                    str(
                        root
                        / f"chunk_{chunk_index:04d}"
                        / "output_frames/0000.png"
                    ),
                    cv2.IMREAD_COLOR,
                )
                self.assertIsNotNone(output)
                np.testing.assert_array_equal(output[0, 0], color)


class FakeRunnerEndToEndTest(unittest.TestCase):
    @staticmethod
    def _write_source_video(path: Path, frame_count: int) -> list[np.ndarray]:
        size = (96, 64)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            size,
        )
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV mp4v encoder is unavailable")
        frames: list[np.ndarray] = []
        for index in range(frame_count):
            frame = np.full(
                (size[1], size[0], 3),
                (20 + index * 5, 40 + index * 5, 60 + index * 5),
                dtype=np.uint8,
            )
            frames.append(frame)
            writer.write(frame)
        writer.release()
        return frames

    @staticmethod
    def _fake_runner(
        calls: list[list[str]],
        manifests: list[dict[str, object]],
    ):
        def run(command: list[str], cwd: Path) -> None:
            calls.append(command.copy())

            def value(flag: str) -> Path:
                return Path(command[command.index(flag) + 1])

            manifest = json.loads(
                value("--manifest").read_text(encoding="utf-8")
            )
            manifests.append(manifest)
            for job in manifest["jobs"]:
                input_frames = Path(job["input_frames"])
                saved_frames = Path(job["output_frames"])
                saved_frames.mkdir(parents=True)
                source_paths = sorted(input_frames.glob("*.png"))
                for index, source_path in enumerate(source_paths):
                    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                    if source is None:
                        raise AssertionError(
                            f"Could not read fake input {source_path}"
                        )
                    painted = np.full_like(source, (0, 0, 255))
                    ok = cv2.imwrite(
                        str(saved_frames / f"{index:04d}.png"),
                        painted,
                    )
                    if not ok:
                        raise AssertionError(
                            "Could not write fake ProPainter frame"
                        )

        return run

    def test_small_video_runs_through_chunking_and_composition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "episode.mp4"
            frame_count = 6
            source_frames = self._write_source_video(video, frame_count)

            mask_root = root / "masks"
            mask_root.mkdir()
            for index in range(frame_count):
                mask = np.zeros((64, 96), dtype=np.uint8)
                if index in (2, 3):
                    mask[20:32, 30:44] = 255
                self.assertTrue(
                    cv2.imwrite(str(mask_root / f"{index:05d}.png"), mask)
                )

            propainter_root = root / "ProPainter"
            engine_source = (
                propainter_root
                / "web-demos/hugging_face/inpainter/base_inpainter.py"
            )
            engine_source.parent.mkdir(parents=True)
            engine_source.touch()
            output_root = root / "output"
            calls: list[list[str]] = []
            manifests: list[dict[str, object]] = []

            report = run_chunked_propainter(
                video=video,
                mask_root=mask_root,
                output_root=output_root,
                propainter_root=propainter_root,
                subvideo_length=20,
                context_frames=1,
                roi_margin=2,
                max_chunk_pixel_frames=20 * 96 * 64,
                max_clip_frames=20,
                mask_dilation=0,
                background_cleanup=False,
                fp16=True,
                hardware_encoding=False,
                runner=self._fake_runner(calls, manifests),
            )

            output_video = Path(str(report["output_video"]))
            self.assertTrue(output_video.is_file())
            self.assertEqual(report["resolution"], [96, 64])
            self.assertEqual(report["active_frame_count"], 2)
            self.assertEqual(report["chunk_count"], 1)
            self.assertEqual(report["total_chunk_clip_frames"], 4)
            self.assertEqual(report["total_chunk_pixel_frames"], 16_384)
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0][:3],
                [
                    str(Path(calls[0][0])),
                    "-m",
                    "human_to_robot.propainter_worker",
                ],
            )
            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(manifests[0]["jobs"]), 1)
            self.assertTrue(manifests[0]["parameters"]["fp16"])
            self.assertEqual(video_metadata(output_video)[:3], (96, 64, 6))
            with av.open(str(output_video)) as container:
                self.assertEqual(
                    container.streams.video[0].codec_context.name,
                    "mpeg4",
                )

            report_path = (
                output_root
                / video.stem
                / "chunked_propainter_report.json"
            )
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["resolution"], [96, 64])
            self.assertEqual(saved_report["frame_count"], frame_count)
            self.assertEqual(
                saved_report["encoding"]["source_codec"],
                "mpeg4",
            )

            capture = cv2.VideoCapture(str(output_video))
            decoded: list[np.ndarray] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded.append(frame)
            capture.release()
            self.assertEqual(len(decoded), frame_count)

            for index in (2, 3):
                self.assertGreater(int(decoded[index][25, 35, 2]), 220)
                self.assertLess(int(decoded[index][25, 35, 1]), 40)
                np.testing.assert_allclose(
                    decoded[index][5, 5],
                    source_frames[index][5, 5],
                    atol=12,
                )
            for index in (0, 1, 4, 5):
                np.testing.assert_allclose(
                    decoded[index][25, 35],
                    source_frames[index][25, 35],
                    atol=12,
                )


if __name__ == "__main__":
    unittest.main()
