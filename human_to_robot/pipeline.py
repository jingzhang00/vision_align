from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _validate_file(path: Path, description: str, minimum_bytes: int = 1) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    if path.stat().st_size < minimum_bytes:
        raise ValueError(
            f"{description} is incomplete ({path.stat().st_size} bytes): {path}"
        )


def run_pipeline(args: argparse.Namespace) -> dict[str, str]:
    repo = Path(__file__).resolve().parent.parent
    video = args.video.resolve()
    hdf5 = args.hdf5.resolve()
    output = args.output.resolve()
    sam_checkpoint = (repo / "models/sam3/sam3.pt").resolve()
    mjcf = (repo / "assets/unitree_description/mjcf/g1.xml").resolve()
    propainter = (repo / "ProPainter").resolve()

    _validate_file(video, "Input video")
    _validate_file(hdf5, "HDF5 annotation")
    _validate_file(sam_checkpoint, "SAM3 checkpoint", 3_000_000_000)
    _validate_file(mjcf, "G1 MJCF")
    for name in (
        "raft-things.pth",
        "recurrent_flow_completion.pth",
        "ProPainter.pth",
    ):
        _validate_file(propainter / "weights" / name, f"ProPainter weight {name}")

    sam_root = output / "sam3"
    g1_root = output / "g1"
    propainter_root = output / "propainter"
    final_root = output / "final"
    python = sys.executable

    if not args.reuse or not (sam_root / "arm_masks" / "00125.png").is_file():
        sam_command = [
            python,
            "-m",
            "human_to_robot.segment_sam3",
            "--video",
            str(video),
            "--checkpoint",
            str(sam_checkpoint),
            "--output",
            str(sam_root),
            "--arm-prompt",
            "human arm",
            "--arm-prompt-frame",
            "0",
            "--threshold",
            str(args.sam_threshold),
        ]
        if args.skip_object:
            sam_command.append("--skip-object")
        else:
            sam_command.extend(
                [
                    "--object-prompt",
                    "black stapler",
                    "--object-prompt-frame",
                    "60",
                ]
            )
        _run(sam_command, cwd=repo)

    if not args.reuse or not (g1_root / "g1_trajectory.npz").is_file():
        _run(
            [
                python,
                "-m",
                "human_to_robot.retarget_g1",
                "--video",
                str(video),
                "--hdf5",
                str(hdf5),
                "--mjcf",
                str(mjcf),
                "--output",
                str(g1_root),
                "--render-width",
                str(args.width),
                "--render-height",
                str(args.height),
            ],
            cwd=repo,
        )

    inpaint_video = propainter_root / video.stem / "inpaint_out.mp4"
    if not args.reuse or not inpaint_video.is_file():
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", "/tmp/vision_align_matplotlib")
        environment.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
        _run(
            [
                python,
                "inference_propainter.py",
                "--video",
                str(video),
                "--mask",
                str(sam_root / "arm_masks"),
                "--output",
                str(propainter_root),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--fp16",
                "--subvideo_length",
                "80",
                "--neighbor_length",
                "10",
                "--ref_stride",
                "10",
                "--raft_iter",
                "20",
                "--mask_dilation",
                "4",
                "--save_fps",
                "30",
            ],
            cwd=propainter,
            environment=environment,
        )

    composite_command = [
        python,
        "-m",
        "human_to_robot.composite",
        "--source-video",
        str(video),
        "--inpaint-video",
        str(inpaint_video),
        "--hdf5",
        str(hdf5),
        "--g1-root",
        str(g1_root),
        "--output",
        str(final_root),
    ]
    object_masks = sam_root / "object_masks"
    if object_masks.is_dir():
        composite_command.extend(["--object-masks", str(object_masks)])
    _run(composite_command, cwd=repo)

    artifacts = {
        "synthetic_video": str(final_root / "synthetic_g1.mp4"),
        "comparison_video": str(
            final_root / "comparison_original_inpaint_g1.mp4"
        ),
        "sam3_overlay": str(sam_root / "arm_masks_overlay.mp4"),
        "inpaint_video": str(inpaint_video),
        "ik_overlay": str(g1_root / "ik_projection_overlay.mp4"),
        "trajectory": str(g1_root / "g1_trajectory.npz"),
        "ik_report": str(g1_root / "ik_report.json"),
    }
    print(json.dumps(artifacts, indent=2))
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper §2.3 human-to-G1 synthesis")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--sam-threshold", type=float, default=0.4)
    parser.add_argument("--skip-object", action="store_true")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing SAM3, G1, and ProPainter stage outputs",
    )
    return parser


def main() -> None:
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
