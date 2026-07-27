from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from human_to_robot.pipeline import (
    EpisodeInterrupted,
    _frame_progress_display,
    _format_duration,
    _load_pipeline_state,
    _mark_running_stage_terminal,
    _progress_timing,
    _set_stage_state,
    _signature_subset_matches,
    _signatures_match,
    _stage_complete,
    build_parser,
)


class PipelineStateTests(unittest.TestCase):
    def test_signature_matching_ignores_selected_code_and_profile_mtime(
        self,
    ) -> None:
        recorded = {
            "signature_version": 1,
            "inputs": {
                "video": {"path": "0.mp4", "size": 10, "mtime_ns": 1},
                "pipeline_code": {
                    "path": "pipeline.py",
                    "size": 100,
                    "mtime_ns": 1,
                },
            },
            "task_profile": {
                "path": "lighting.json",
                "size": 20,
                "mtime_ns": 1,
                "sha256": "same-content",
            },
            "parameters": {"backend": "isaacsim", "debug_artifacts": True},
        }
        expected = {
            "signature_version": 1,
            "inputs": {
                "video": {"path": "0.mp4", "size": 10, "mtime_ns": 1},
                "pipeline_code": {
                    "path": "pipeline.py",
                    "size": 120,
                    "mtime_ns": 2,
                },
            },
            "task_profile": {
                "path": "lighting.json",
                "size": 20,
                "mtime_ns": 2,
                "sha256": "same-content",
            },
            "parameters": {"debug_artifacts": True},
        }

        self.assertFalse(_signatures_match(recorded, expected))
        self.assertTrue(
            _signatures_match(
                recorded,
                expected,
                ignored_inputs=frozenset({"pipeline_code"}),
            )
        )
        self.assertTrue(
            _signature_subset_matches(
                recorded,
                expected,
                input_names=frozenset({"video"}),
                parameter_names=frozenset({"debug_artifacts"}),
            )
        )

    def test_progress_timing_separates_stage_and_total(self) -> None:
        self.assertEqual(_format_duration(509.89), "08:29")
        self.assertEqual(
            _progress_timing("finish", 53.05, 509.89),
            "finish=00:53 total=09:22",
        )
        self.assertEqual(
            _frame_progress_display("render", 100, 1000, 20.0),
            "render=100/1000 (10.0%) eta=00:45 rate=20.0fps",
        )
        self.assertEqual(
            _frame_progress_display("composite", 1000, 1000, 20.0),
            "composite=1000/1000 (100.0%) eta=00:00",
        )

    def test_runtime_error_marks_running_stage_failed_and_preserves_others(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "pipeline_state.json"
            state: dict[str, object] = {"state_version": 1, "stages": {}}
            other_signature = {"input": "other"}
            prepare_signature = {"input": "episode"}

            _set_stage_state(
                state_path,
                state,
                stage="finish",
                status="completed",
                signature=other_signature,
                generation=7,
            )
            _set_stage_state(
                state_path,
                state,
                stage="prepare",
                status="running",
                signature=prepare_signature,
                started_at_ns=123,
            )

            _mark_running_stage_terminal(
                state_path,
                stage="prepare",
                error=RuntimeError("CUDA out of memory"),
                elapsed_seconds=12.5,
            )

            loaded = _load_pipeline_state(state_path)
            prepare = loaded["stages"]["prepare"]
            self.assertEqual(prepare["status"], "failed")
            self.assertEqual(prepare["signature"], prepare_signature)
            self.assertEqual(prepare["started_at_ns"], 123)
            self.assertIsInstance(prepare["finished_at_ns"], int)
            self.assertEqual(prepare["elapsed_seconds"], 12.5)
            self.assertEqual(
                prepare["error"],
                "RuntimeError: CUDA out of memory",
            )
            self.assertEqual(
                loaded["stages"]["finish"],
                {
                    "status": "completed",
                    "signature": other_signature,
                    "generation": 7,
                },
            )
            self.assertFalse(
                _stage_complete(loaded, "prepare", prepare_signature)
            )

    def test_interruptions_are_recorded_as_interrupted(self) -> None:
        for exception in (
            KeyboardInterrupt(),
            SystemExit(130),
            EpisodeInterrupted("child exited with SIGINT"),
        ):
            with self.subTest(exception=type(exception).__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    state_path = Path(temporary) / "pipeline_state.json"
                    signature = {"episode": "0"}
                    state: dict[str, object] = {
                        "state_version": 1,
                        "stages": {},
                    }
                    _set_stage_state(
                        state_path,
                        state,
                        stage="prepare",
                        status="running",
                        signature=signature,
                    )

                    _mark_running_stage_terminal(
                        state_path,
                        stage="prepare",
                        error=exception,
                        elapsed_seconds=3.0,
                    )

                    loaded = _load_pipeline_state(state_path)
                    self.assertEqual(
                        loaded["stages"]["prepare"]["status"],
                        "interrupted",
                    )
                    self.assertFalse(
                        _stage_complete(loaded, "prepare", signature)
                    )

    def test_completed_overwrite_clears_error_and_is_cacheable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "pipeline_state.json"
            signature = {"episode": "0", "parameters": {"width": 1280}}
            other_record = {
                "status": "completed",
                "signature": {"task": "lighting"},
                "generation": 4,
            }
            state: dict[str, object] = {
                "state_version": 1,
                "stages": {"lighting": other_record.copy()},
            }
            _set_stage_state(
                state_path,
                state,
                stage="prepare",
                status="failed",
                signature=signature,
                error="RuntimeError: first attempt failed",
            )

            _set_stage_state(
                state_path,
                state,
                stage="prepare",
                status="completed",
                signature=signature,
                generation=2,
                elapsed_seconds=8.0,
            )

            loaded = _load_pipeline_state(state_path)
            prepare = loaded["stages"]["prepare"]
            self.assertNotIn("error", prepare)
            self.assertEqual(prepare["status"], "completed")
            self.assertEqual(prepare["generation"], 2)
            self.assertTrue(_stage_complete(loaded, "prepare", signature))
            self.assertEqual(loaded["stages"]["lighting"], other_record)


class PipelineParserDefaultsTests(unittest.TestCase):
    def test_resolution_and_chunking_defaults(self) -> None:
        args = build_parser().parse_args(
            ["--folder", "dataset/test", "--output", "outputs/test"]
        )

        self.assertEqual((args.width, args.height), (1280, 720))
        self.assertEqual(args.propainter_subvideo_length, 80)
        self.assertEqual(args.propainter_context_frames, 20)
        self.assertEqual(args.propainter_roi_margin, 64)
        self.assertEqual(args.propainter_max_clip_frames, 240)
        self.assertEqual(args.propainter_transition_frames, 16)
        self.assertTrue(args.binary_hands)
        self.assertTrue(args.hardware_encoding)
        self.assertFalse(hasattr(args, "backend"))

    def test_legacy_isaacsim_backend_argument_is_still_accepted(self) -> None:
        args = build_parser().parse_args(
            [
                "--folder",
                "dataset/test",
                "--output",
                "outputs/test",
                "--backend",
                "isaacsim",
            ]
        )

        self.assertEqual(args._legacy_backend, "isaacsim")


if __name__ == "__main__":
    unittest.main()
