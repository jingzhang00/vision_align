#!/usr/bin/env python3
"""Run the linear Nymeria -> G1 -> VDA -> Isaac pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.nymeria.nymeria_adapter import (
    NymeriaClip,
    discover_nymeria_clips,
    natural_key,
)
from repaint.vda_pipeline_common import read_json, remove_exact, render_complete


def _run(python: Path, script: str, *arguments: object) -> None:
    command = [
        str(python),
        str(WORKSPACE_ROOT / script),
        *(str(value) for value in arguments),
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)


def _select_clips(
    dataset_root: Path,
    sessions: set[str] | None,
    requested: set[str] | None,
) -> list[NymeriaClip]:
    clips = discover_nymeria_clips(dataset_root)
    if sessions:
        clips = [clip for clip in clips if clip.session in sessions]
    if requested:
        selected = []
        for name in sorted(requested, key=natural_key):
            matches = [
                clip
                for clip in clips
                if clip.stem == name or clip.stem.endswith(f"_{name}")
            ]
            if len(matches) != 1:
                raise ValueError(f"Nymeria clip {name!r} matched {len(matches)} clips")
            selected.append(matches[0])
        clips = selected
    if not clips:
        raise ValueError("No Nymeria clips selected")
    return clips


def _run_batch(
    selected: list[NymeriaClip],
    *,
    dataset_root: Path,
    output_root: Path,
    python_executable: Path,
    headless: bool,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    for stale in (
        "camera_alignment",
        "alignment_reports",
        "robot",
        "handless",
        "prepared",
        ".batch_work",
    ):
        remove_exact(output_root / stale, owner_root=output_root)

    outputs: list[Path] = []
    for index, clip in enumerate(selected):
        label = f"episode_{index:06d}"
        output_video = output_root / "final" / label / f"{clip.stem}.mp4"
        if render_complete(output_video):
            print(f"[skip] completed: {clip.stem}", flush=True)
            outputs.append(output_video)
            continue
        for partial in (output_video, output_video.with_suffix(".json")):
            remove_exact(partial, owner_root=output_root)

        work = output_root / ".batch_work" / label
        alignment = work / "camera_alignment"
        reports = work / "alignment_reports"
        robot = work / "robot"
        handless = work / "handless"
        prepared = work / "prepared"
        try:
            output_video.parent.mkdir(parents=True, exist_ok=True)
            _run(
                python_executable,
                "repaint/nymeria/nymeria_camera_alignment.py",
                "--episode-root",
                dataset_root / clip.session,
                "--output",
                reports,
                "--adapter-output",
                alignment,
                "--clip",
                clip.stem,
            )
            _run(
                python_executable,
                "repaint/nymeria/nymeria_to_robot.py",
                "--src-root",
                dataset_root,
                "--output-root",
                robot,
                "--camera-alignment-root",
                alignment,
                "--camera-alignment-report-root",
                reports,
                "--session",
                clip.session,
                "--clip",
                clip.stem,
            )
            _run(
                python_executable,
                "repaint/remove_hands.py",
                "--input_dir",
                robot,
                "--output_dir",
                handless,
                "--min_coverage",
                0,
            )
            _run(
                python_executable,
                "repaint/rgbd_prepare.py",
                "--dataset",
                robot,
                "--source-root",
                dataset_root,
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
            print(f"[done] {clip.stem}", flush=True)
        except Exception as exc:
            print(f"[failed] {clip.stem}: {exc}", file=sys.stderr, flush=True)
            for partial in (output_video, output_video.with_suffix(".json")):
                remove_exact(partial, owner_root=output_root)
        finally:
            remove_exact(work, owner_root=output_root)
    remove_exact(output_root / ".batch_work", owner_root=output_root)
    return outputs


def run_pipeline(
    *,
    dataset_root: Path,
    output_root: Path,
    sessions: set[str] | None = None,
    clips: set[str] | None = None,
    python_executable: Path = Path(sys.executable),
    headless: bool = True,
    overwrite: bool = False,
    batch: bool = False,
) -> list[Path]:
    """Run the pipeline; single runs retain intermediates, batch runs do not."""

    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    selected = _select_clips(dataset_root, sessions, clips)
    if batch:
        return _run_batch(
            selected,
            dataset_root=dataset_root,
            output_root=output_root,
            python_executable=python_executable,
            headless=headless,
        )
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}")
        remove_exact(output_root, owner_root=output_root.parent)
    output_root.mkdir(parents=True)

    alignment = output_root / "camera_alignment"
    reports = output_root / "alignment_reports"
    robot = output_root / "robot"
    handless = output_root / "handless"
    prepared = output_root / "prepared"
    final = output_root / "final"

    grouped = {}
    for clip in selected:
        grouped.setdefault(clip.session, []).append(clip)
    for session in sorted(grouped, key=natural_key):
        arguments: list[object] = [
            "--episode-root",
            dataset_root / session,
            "--output",
            reports,
            "--adapter-output",
            alignment,
        ]
        for clip in grouped[session]:
            arguments.extend(("--clip", clip.stem))
        _run(
            python_executable,
            "repaint/nymeria/nymeria_camera_alignment.py",
            *arguments,
        )

    arguments = [
        "--src-root",
        dataset_root,
        "--output-root",
        robot,
        "--camera-alignment-root",
        alignment,
        "--camera-alignment-report-root",
        reports,
    ]
    for session in sorted(grouped, key=natural_key):
        arguments.extend(("--session", session))
    for clip in selected:
        arguments.extend(("--clip", clip.stem))
    _run(
        python_executable,
        "repaint/nymeria/nymeria_to_robot.py",
        *arguments,
    )
    _run(
        python_executable,
        "repaint/remove_hands.py",
        "--input_dir",
        robot,
        "--output_dir",
        handless,
        "--min_coverage",
        0,
    )

    outputs: list[Path] = []
    manifest = read_json(robot / "manifest.json")
    for episode in manifest["episodes"]:
        index = int(episode["episode_index"])
        label = f"episode_{index:06d}"
        prepared_episode = prepared / label
        output_video = final / label / Path(str(episode["video_path"])).name
        output_video.parent.mkdir(parents=True)
        _run(
            python_executable,
            "repaint/rgbd_prepare.py",
            "--dataset",
            robot,
            "--source-root",
            dataset_root,
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
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE_ROOT / "dataset/nymeria",
    )
    parser.add_argument(
        "--output-root", type=Path, default=WORKSPACE_ROOT / "nymeria/vda"
    )
    parser.add_argument("--session", action="append")
    parser.add_argument("--clip", action="append")
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Resume completed renders, continue failures, and delete intermediates.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        sessions=set(args.session) if args.session else None,
        clips=set(args.clip) if args.clip else None,
        headless=args.headless,
        overwrite=args.overwrite,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
