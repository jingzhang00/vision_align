from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py

from human_to_robot.pipeline import (
    Episode,
    EpisodeInterrupted,
    IsaacRuntime,
    PreparedEpisode,
    _discard_episode_workspace,
    _persist_failure_diagnostics,
    _publish_episode_outputs,
    _render_episode,
    _resolve_isaacsim_runtime,
    _run_logged,
    _run_pipeline,
    build_parser,
    discover_episodes,
    run_pipeline,
)


class PipelineOutputTests(unittest.TestCase):
    def test_non_debug_routes_render_and_composite_through_one_process(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_task = root / "source" / "task_a"
            source_task.mkdir(parents=True)
            video = source_task / "0.mp4"
            hdf5 = source_task / "0.hdf5"
            video.touch()
            hdf5.touch()
            episode_root = root / "workspace" / "task_a" / "0"
            inpaint = episode_root / "propainter" / "inpaint.mp4"
            trajectory = episode_root / "g1" / "g1_trajectory.npz"
            inpaint.parent.mkdir(parents=True)
            trajectory.parent.mkdir(parents=True)
            inpaint.touch()
            trajectory.touch()
            scene_depth_report = (
                episode_root
                / "depth_anything3/scene_depth_report.json"
            )
            scene_depth_report.parent.mkdir(parents=True)
            scene_depth_report.write_text("{}", encoding="utf-8")
            episode = Episode(
                task_dir=source_task,
                task_relative=Path("task_a"),
                task_name="task_a",
                name="0",
                video=video,
                hdf5=hdf5,
                output_root=episode_root,
            )
            prepared = PreparedEpisode(
                episode=episode,
                frame_count=2,
                inpaint_video=inpaint,
                sam_root=episode_root / "sam3",
                g1_root=trajectory.parent,
                lighting_observation={},
                prepare_seconds=1.0,
                prepare_generation=1,
            )
            task_profile = root / "lighting_profile.json"
            task_profile.write_text("{}", encoding="utf-8")
            urdf = root / "g1.urdf"
            urdf.touch()
            args = build_parser().parse_args(
                [
                    "--folder",
                    str(source_task),
                    "--output",
                    str(root / "published"),
                    "--no-hardware-encoding",
                    "--visualize",
                ]
            )
            commands: list[list[str]] = []

            def fake_run_logged(
                command: list[str],
                **_kwargs: object,
            ) -> None:
                commands.append(command)
                render_root = episode_root / "g1_isaacsim"
                final_root = episode_root / "final"
                render_root.mkdir(parents=True)
                final_root.mkdir(parents=True)
                (render_root / "isaacsim_render_report.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                (final_root / "composite_report.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                (final_root / "synthetic_g1.mp4").touch()

            with (
                mock.patch(
                    "human_to_robot.pipeline.video_metadata",
                    return_value=(48, 32, 2, 12.0),
                ),
                mock.patch(
                    "human_to_robot.pipeline._video_complete",
                    return_value=True,
                ),
                mock.patch(
                    "human_to_robot.pipeline._run_logged",
                    side_effect=fake_run_logged,
                ),
                mock.patch(
                    "human_to_robot.pipeline._publish_episode_outputs",
                    return_value=(
                        root / "published/task_a/0.mp4",
                        root / "published/task_a/0.hdf5",
                    ),
                ),
            ):
                _render_episode(
                    prepared,
                    args=args,
                    repo=Path.cwd(),
                    urdf=urdf,
                    isaac_runtime=IsaacRuntime(
                        command=("isaac-python",),
                        environment={},
                        description="test",
                    ),
                    task_profile_path=task_profile,
                    progress=mock.Mock(),
                )

            self.assertEqual(len(commands), 1)
            command = commands[0]
            self.assertIn("human_to_robot.render_isaacsim", command)
            self.assertIn("--inpaint-video", command)
            self.assertIn("--composite-output", command)
            self.assertIn("--scene-depths", command)
            self.assertIn("--no-headless", command)
            self.assertNotIn("human_to_robot.composite", command)
            self.assertNotIn("--width", command)
            self.assertNotIn("--height", command)
            for directory_name in (
                "robot_rgb",
                "robot_masks",
                "robot_depth_mm",
            ):
                self.assertFalse(
                    (episode_root / "g1_isaacsim" / directory_name).exists()
                )

    def test_explicit_isaacsim_root_uses_standalone_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "python.sh"
            launcher.touch()
            args = build_parser().parse_args(
                [
                    "--folder",
                    "dataset/test",
                    "--output",
                    "outputs/test",
                    "--isaacsim-root",
                    str(root),
                ]
            )
            with mock.patch(
                "human_to_robot.pipeline._probe_isaacsim_runtime",
                return_value=(True, "simulation_app.simulation_app"),
            ):
                runtime = _resolve_isaacsim_runtime(
                    args,
                    repo=Path.cwd(),
                )

            self.assertEqual(runtime.command, (str(launcher),))
            self.assertEqual(
                runtime.environment["PYTHONEXE"],
                str(Path(sys.executable).absolute()),
            )

    def test_interrupted_child_process_is_reported_distinctly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = mock.Mock()

            with self.assertRaises(EpisodeInterrupted):
                _run_logged(
                    [sys.executable, "-c", "raise SystemExit(130)"],
                    cwd=root,
                    log_path=root / "pipeline.log",
                    progress=progress,
                    current_episode="task_a/0",
                )

    def test_failed_episode_is_cleaned_and_next_episode_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "dataset" / "task_a"
            output = root / "output"
            workspace = root / "workspace"
            folder.mkdir(parents=True)
            episodes = [
                Episode(
                    task_dir=folder,
                    task_relative=Path("task_a"),
                    task_name="task_a",
                    name=str(index),
                    video=folder / f"{index}.mp4",
                    hdf5=folder / f"{index}.hdf5",
                    output_root=workspace / "task_a" / str(index),
                )
                for index in range(3)
            ]
            args = build_parser().parse_args(
                [
                    "--folder",
                    str(folder),
                    "--output",
                    str(output),
                ]
            )

            def fake_prepare(
                episode: Episode,
                **_kwargs: object,
            ) -> PreparedEpisode:
                episode.output_root.mkdir(parents=True, exist_ok=True)
                if episode.name == "0":
                    raise RuntimeError("synthetic prepare failure")
                if episode.name == "1":
                    raise EpisodeInterrupted("synthetic child interruption")
                return PreparedEpisode(
                    episode=episode,
                    frame_count=1,
                    inpaint_video=episode.output_root / "inpaint.mp4",
                    sam_root=episode.output_root / "sam3",
                    g1_root=episode.output_root / "g1",
                    lighting_observation={},
                    prepare_seconds=0.1,
                    prepare_generation=1,
                )

            def fake_render(
                prepared: PreparedEpisode,
                **_kwargs: object,
            ) -> tuple[dict[str, object], float]:
                return (
                    {
                        "task": "task_a",
                        "episode": prepared.episode.name,
                        "status": "completed",
                        "elapsed_seconds": 0.2,
                    },
                    0.1,
                )

            task_profile = {
                "episode_count": 1,
                "confidence": 1.0,
                "warnings": [],
                "isaac": {},
            }
            with (
                mock.patch(
                    "human_to_robot.pipeline._validate_file",
                ),
                mock.patch(
                    "human_to_robot.pipeline._resolve_isaacsim_runtime",
                    return_value=mock.Mock(description="test Isaac runtime"),
                ),
                mock.patch(
                    "human_to_robot.pipeline.discover_episodes",
                    return_value=episodes,
                ),
                mock.patch(
                    "human_to_robot.pipeline._prepare_episode",
                    side_effect=fake_prepare,
                ) as prepare_mock,
                mock.patch(
                    "human_to_robot.pipeline.aggregate_task_lighting_profiles",
                    return_value=task_profile,
                ),
                mock.patch(
                    "human_to_robot.pipeline._render_episode",
                    side_effect=fake_render,
                ),
                mock.patch("human_to_robot.pipeline.tqdm.write"),
            ):
                report = _run_pipeline(
                    args,
                    folder=folder,
                    output=output,
                    workspace_output=workspace,
                )

            self.assertEqual(prepare_mock.call_count, 3)
            self.assertEqual(report["completed_episode_count"], 1)
            self.assertEqual(report["failed_episode_count"], 2)
            self.assertEqual(len(report["episodes"]), 3)
            self.assertNotIn("tasks", report)
            self.assertFalse(
                (workspace / "task_a" / "task_report.json").exists()
            )
            self.assertFalse(episodes[0].output_root.exists())
            self.assertFalse(episodes[1].output_root.exists())
            self.assertFalse(episodes[2].output_root.exists())

    def test_non_debug_discards_episode_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode_root = root / "workspace" / "task_a" / "0"
            episode_root.mkdir(parents=True)
            (episode_root / "intermediate.bin").write_bytes(b"temporary")
            episode = Episode(
                task_dir=root / "source" / "task_a",
                task_relative=Path("task_a"),
                task_name="task_a",
                name="0",
                video=root / "source" / "task_a" / "0.mp4",
                hdf5=root / "source" / "task_a" / "0.hdf5",
                output_root=episode_root,
            )

            _discard_episode_workspace(episode, debug=False)

            self.assertFalse(episode_root.exists())

    def test_debug_keeps_episode_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode_root = root / "workspace" / "task_a" / "0"
            episode_root.mkdir(parents=True)
            episode = Episode(
                task_dir=root / "source" / "task_a",
                task_relative=Path("task_a"),
                task_name="task_a",
                name="0",
                video=root / "source" / "task_a" / "0.mp4",
                hdf5=root / "source" / "task_a" / "0.hdf5",
                output_root=episode_root,
            )

            _discard_episode_workspace(episode, debug=True)

            self.assertTrue(episode_root.is_dir())

    def test_non_debug_persists_failure_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode_root = root / "workspace" / "task_a" / "0"
            episode_root.mkdir(parents=True)
            (episode_root / "pipeline.log").write_text(
                "underlying failure\n",
                encoding="utf-8",
            )
            episode = Episode(
                task_dir=root / "source" / "task_a",
                task_relative=Path("task_a"),
                task_name="task_a",
                name="0",
                video=root / "source" / "task_a" / "0.mp4",
                hdf5=root / "source" / "task_a" / "0.hdf5",
                output_root=episode_root,
            )

            persisted = _persist_failure_diagnostics(
                episode,
                output=root / "published",
                report={"status": "failed", "error": "RuntimeError: boom"},
            )

            failure_root = root / "published" / "task_a" / ".failures" / "0"
            self.assertEqual(
                (failure_root / "pipeline.log").read_text(encoding="utf-8"),
                "underlying failure\n",
            )
            self.assertEqual(
                persisted["log"],
                str(failure_root / "pipeline.log"),
            )
            self.assertTrue((failure_root / "episode_report.json").is_file())

    def test_dataset_root_mirrors_task_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset" / "test"
            task = dataset / "basic_pick_place"
            task.mkdir(parents=True)
            (task / "0.mp4").write_bytes(b"video")
            (task / "0.hdf5").write_bytes(b"hdf5")
            workspace = root / "workspace"

            [episode] = discover_episodes(dataset, workspace)

            self.assertEqual(episode.task_relative, Path("basic_pick_place"))
            self.assertEqual(
                episode.output_root,
                workspace.resolve() / "basic_pick_place" / "0",
            )

    def test_single_task_input_still_creates_task_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / "dataset" / "test" / "fry_bread"
            task.mkdir(parents=True)
            (task / "0.mp4").write_bytes(b"video")
            (task / "0.hdf5").write_bytes(b"hdf5")
            workspace = root / "workspace"

            [episode] = discover_episodes(task, workspace)

            self.assertEqual(episode.task_relative, Path("fry_bread"))
            self.assertEqual(
                episode.output_root,
                workspace.resolve() / "fry_bread" / "0",
            )

    def test_publish_uses_episode_stem_and_copies_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_task = root / "source" / "task_a"
            source_task.mkdir(parents=True)
            source_video = source_task / "12.mp4"
            source_hdf5 = source_task / "12.hdf5"
            synthetic_video = root / "workspace" / "synthetic_g1.mp4"
            synthetic_video.parent.mkdir()
            source_video.write_bytes(b"original video")
            source_hdf5.write_bytes(b"original hdf5")
            synthetic_video.write_bytes(b"synthetic video")
            episode = Episode(
                task_dir=source_task,
                task_relative=Path("task_a"),
                task_name="task_a",
                name="12",
                video=source_video,
                hdf5=source_hdf5,
                output_root=root / "workspace" / "task_a" / "12",
            )

            published_video, published_hdf5 = _publish_episode_outputs(
                episode,
                synthetic_video=synthetic_video,
                output=root / "outputs",
            )

            self.assertEqual(
                published_video, root / "outputs" / "task_a" / "12.mp4"
            )
            self.assertEqual(
                published_hdf5, root / "outputs" / "task_a" / "12.hdf5"
            )
            self.assertEqual(published_video.read_bytes(), b"synthetic video")
            self.assertEqual(published_hdf5.read_bytes(), b"original hdf5")

    def test_publish_keeps_source_hdf5_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_task = root / "source" / "task_a"
            source_task.mkdir(parents=True)
            source_video = source_task / "0.mp4"
            source_hdf5 = source_task / "0.hdf5"
            synthetic_video = root / "workspace" / "synthetic_g1.mp4"
            synthetic_video.parent.mkdir()
            source_video.write_bytes(b"original")
            synthetic_video.write_bytes(b"synthetic")
            with h5py.File(source_hdf5, "w") as handle:
                handle.create_dataset("source", data=[0, 1, 2])
            source_bytes = source_hdf5.read_bytes()
            episode = Episode(
                task_dir=source_task,
                task_relative=Path("task_a"),
                task_name="task_a",
                name="0",
                video=source_video,
                hdf5=source_hdf5,
                output_root=root / "workspace" / "task_a" / "0",
            )

            _, published_hdf5 = _publish_episode_outputs(
                episode,
                synthetic_video=synthetic_video,
                output=root / "outputs",
            )

            self.assertEqual(published_hdf5.read_bytes(), source_bytes)
            with h5py.File(published_hdf5, "r") as handle:
                self.assertIn("source", handle)
                self.assertNotIn("human_to_robot", handle)

    def test_default_workspace_is_removed_but_debug_workspace_is_kept(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "input"
            folder.mkdir()

            def fake_run(
                _args: argparse.Namespace,
                *,
                folder: Path,
                output: Path,
                workspace_output: Path,
            ) -> dict[str, object]:
                del folder, output
                (workspace_output / "scratch").mkdir(parents=True)
                seen_workspaces.append(workspace_output)
                return {"status": "completed"}

            seen_workspaces: list[Path] = []
            normal_output = root / "normal"
            normal_args = argparse.Namespace(
                folder=folder,
                output=normal_output,
                debug=False,
            )
            with mock.patch(
                "human_to_robot.pipeline._run_pipeline",
                side_effect=fake_run,
            ):
                run_pipeline(normal_args)
            self.assertFalse(seen_workspaces[-1].exists())

            debug_output = root / "debug"
            debug_args = argparse.Namespace(
                folder=folder,
                output=debug_output,
                debug=True,
            )
            with mock.patch(
                "human_to_robot.pipeline._run_pipeline",
                side_effect=fake_run,
            ):
                run_pipeline(debug_args)
            self.assertEqual(seen_workspaces[-1], debug_output.resolve())
            self.assertTrue((debug_output / "scratch").is_dir())

    def test_debug_flag_is_opt_in(self) -> None:
        parser = build_parser()
        default_args = parser.parse_args(
            ["--folder", "dataset/test", "--output", "outputs"]
        )
        debug_args = parser.parse_args(
            [
                "--folder",
                "dataset/test",
                "--output",
                "outputs",
                "--debug",
            ]
        )

        self.assertFalse(default_args.debug)
        self.assertFalse(default_args.visualize)
        self.assertTrue(debug_args.debug)


if __name__ == "__main__":
    unittest.main()
