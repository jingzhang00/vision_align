#!/usr/bin/env python3
"""Run the linear EgoDex -> LeRobot -> VDA -> Isaac pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.vda_pipeline_common import (
    natural_key,
    read_json,
    remove_exact,
    render_complete,
)


DEFAULT_TASKS = (
    "basic_pick_place",
    "flip_coin",
    "insert_remove_usb",
    "pour",
)


def _run(python: Path, script: str, *arguments: object) -> None:
    command = [
        str(python),
        str(WORKSPACE_ROOT / script),
        *(str(value) for value in arguments),
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)


def _source_episodes(
    source: Path,
    stems: set[str] | None,
) -> list[tuple[int, str]]:
    hdf5_paths = sorted(
        source.rglob("*.hdf5"),
        key=lambda path: natural_key(str(path.relative_to(source))),
    )
    episodes: list[tuple[int, str]] = []
    for index, hdf5_path in enumerate(hdf5_paths):
        if not hdf5_path.with_suffix(".mp4").is_file():
            raise FileNotFoundError(hdf5_path.with_suffix(".mp4"))
        stem = hdf5_path.relative_to(source).with_suffix("").as_posix()
        if stems is None or stem in stems:
            episodes.append((index, stem))
    if not episodes:
        raise ValueError(f"No EgoDex episodes selected under {source}")
    if stems and {stem for _, stem in episodes} != stems:
        raise ValueError(f"Missing EgoDex stems under {source}: {sorted(stems)}")
    return episodes


def _run_batch(
    tasks: list[str] | tuple[str, ...],
    *,
    dataset_root: Path,
    output_root: Path,
    stems: set[str] | None,
    python_executable: Path,
    headless: bool,
) -> list[Path]:
    outputs: list[Path] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        source = dataset_root / task
        root = output_root / task
        root.mkdir(parents=True, exist_ok=True)
        for stale in ("lerobot", "handless", "prepared", ".batch_work"):
            remove_exact(root / stale, owner_root=root)
        try:
            episodes = _source_episodes(source, stems)
        except Exception as exc:
            print(f"[failed] {task}: {exc}", file=sys.stderr, flush=True)
            continue
        for index, stem in episodes:
            label = f"episode_{index:06d}"
            output_video = root / "final" / label / (Path(stem).name + ".mp4")
            if render_complete(output_video):
                print(f"[skip] completed: {task}/{stem}", flush=True)
                outputs.append(output_video)
                continue
            for partial in (output_video, output_video.with_suffix(".json")):
                remove_exact(partial, owner_root=root)
            work = root / ".batch_work" / label
            try:
                robot = work / "lerobot"
                handless = work / "handless"
                prepared = work / "prepared"
                output_video.parent.mkdir(parents=True, exist_ok=True)
                _run(
                    python_executable,
                    "repaint/egodex/egodex_to_robot.py",
                    "--src-root",
                    source,
                    "--stem",
                    stem,
                    "--output-root",
                    robot,
                )
                _run(
                    python_executable,
                    "repaint/remove_hands.py",
                    "--input_dir",
                    robot,
                    "--output_dir",
                    handless,
                )
                _run(
                    python_executable,
                    "repaint/rgbd_prepare.py",
                    "--dataset",
                    robot,
                    "--source-root",
                    source,
                    "--removed-root",
                    handless,
                    "--episode",
                    0,
                    "--output",
                    prepared,
                )
                _run(
                    python_executable,
                    "repaint/vda_scene_isaac.py",
                    "--prepared",
                    prepared,
                    "--output",
                    output_video,
                    "--capture-view",
                    "robot-camera",
                    "--headless" if headless else "--no-headless",
                )
                if not render_complete(output_video):
                    raise RuntimeError("Isaac returned without a completed render")
                outputs.append(output_video)
                print(f"[done] {task}/{stem}", flush=True)
            except Exception as exc:
                print(f"[failed] {task}/{stem}: {exc}", file=sys.stderr, flush=True)
                for partial in (output_video, output_video.with_suffix(".json")):
                    remove_exact(partial, owner_root=root)
            finally:
                remove_exact(work, owner_root=root)
        remove_exact(root / ".batch_work", owner_root=root)
    return outputs


def run_pipeline(
    tasks: list[str] | tuple[str, ...],
    *,
    dataset_root: Path,
    output_root: Path,
    python_executable: Path = Path(sys.executable),
    headless: bool = True,
    overwrite: bool = False,
    batch: bool = False,
    stems: set[str] | None = None,
) -> list[Path]:
    """Run the pipeline; single runs retain intermediates, batch runs do not."""

    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if batch:
        return _run_batch(
            tasks,
            dataset_root=dataset_root,
            output_root=output_root,
            stems=stems,
            python_executable=python_executable,
            headless=headless,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for task in tasks:
        source = dataset_root / task
        root = output_root / task
        if root.exists():
            if not overwrite:
                raise FileExistsError(f"Output already exists: {root}")
            remove_exact(root, owner_root=output_root)
        robot = root / "lerobot"
        handless = root / "handless"
        prepared = root / "prepared"
        final = root / "final"
        root.mkdir(parents=True)

        _run(
            python_executable,
            "repaint/egodex/egodex_to_robot.py",
            "--src-root",
            source,
            "--output-root",
            robot,
        )
        _run(
            python_executable,
            "repaint/remove_hands.py",
            "--input_dir",
            robot,
            "--output_dir",
            handless,
        )
        manifest = read_json(robot / "meta/egodex.json")
        for episode in manifest["episodes"]:
            index = int(episode["episode_index"])
            label = f"episode_{index:06d}"
            prepared_episode = prepared / label
            output_video = final / label / (
                Path(str(episode["source_stem"])).name + ".mp4"
            )
            output_video.parent.mkdir(parents=True)
            _run(
                python_executable,
                "repaint/rgbd_prepare.py",
                "--dataset",
                robot,
                "--source-root",
                source,
                "--removed-root",
                handless,
                "--episode",
                index,
                "--output",
                prepared_episode,
            )
            _run(
                python_executable,
                "repaint/vda_scene_isaac.py",
                "--prepared",
                prepared_episode,
                "--output",
                output_video,
                "--capture-view",
                "robot-camera",
                "--headless" if headless else "--no-headless",
            )
            outputs.append(output_video)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--dataset-root", type=Path, default=WORKSPACE_ROOT / "dataset"
    )
    parser.add_argument(
        "--output-root", type=Path, default=WORKSPACE_ROOT / "egodex"
    )
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Resume completed renders, continue failures, and delete intermediates.",
    )
    parser.add_argument("--stem", action="append", help="Batch source stem filter.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        args.tasks,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        headless=args.headless,
        overwrite=args.overwrite,
        batch=args.batch,
        stems=set(args.stem) if args.stem else None,
    )


if __name__ == "__main__":
    main()
