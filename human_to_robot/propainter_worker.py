from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _numbered_pngs(root: Path, expected_count: int) -> list[Path]:
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    expected = [
        root / f"{index:05d}.png" for index in range(expected_count)
    ]
    actual = sorted(root.glob("*.png")) if root.is_dir() else []
    if actual != expected:
        raise ValueError(
            f"Expected {expected_count} numbered PNGs in {root}, "
            f"found {len(actual)}"
        )
    return expected


def _read_job_inputs(
    job: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], Path]:
    expected_count = int(job["expected_count"])
    frame_paths = _numbered_pngs(
        Path(str(job["input_frames"])),
        expected_count,
    )
    mask_paths = _numbered_pngs(
        Path(str(job["input_masks"])),
        expected_count,
    )
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    expected_size: tuple[int, int] | None = None
    for frame_path, mask_path in zip(frame_paths, mask_paths, strict=True):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"Could not read input frame: {frame_path}")
        if mask is None:
            raise ValueError(f"Could not read input mask: {mask_path}")
        size = frame.shape[1::-1]
        if expected_size is None:
            expected_size = size
        if size != expected_size or mask.shape[::-1] != expected_size:
            raise ValueError(
                f"Inconsistent frame or mask geometry in job "
                f"{job['chunk_index']}"
            )
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks.append(mask)
    return frames, masks, Path(str(job["output_frames"]))


def _write_output_frames(
    output_root: Path,
    frames: list[np.ndarray],
    *,
    expected_count: int,
) -> None:
    if len(frames) != expected_count:
        raise RuntimeError(
            f"ProPainter returned {len(frames)} frames, "
            f"expected {expected_count}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(
                f"Invalid ProPainter output shape at frame {index}: "
                f"{frame.shape}"
            )
        destination = output_root / f"{index:04d}.png"
        bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(destination), bgr):
            raise RuntimeError(f"Could not write output frame: {destination}")


def run_manifest(manifest_path: Path, propainter_root: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("Unsupported ProPainter job manifest version")
    parameters = manifest.get("parameters")
    jobs = manifest.get("jobs")
    if not isinstance(parameters, dict) or not isinstance(jobs, list):
        raise ValueError("Invalid ProPainter job manifest")
    if not jobs:
        return

    propainter_root = propainter_root.resolve()
    demo_root = propainter_root / "web-demos" / "hugging_face"
    engine_source = demo_root / "inpainter" / "base_inpainter.py"
    if not engine_source.is_file():
        raise FileNotFoundError(f"ProPainter engine not found: {engine_source}")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vision_align_matplotlib")
    sys.path.insert(0, str(propainter_root))
    sys.path.insert(0, str(demo_root))

    from inpainter.base_inpainter import ProInpainter
    from model.misc import get_device

    weights = propainter_root / "weights"
    device = get_device()
    engine = ProInpainter(
        str(weights / "ProPainter.pth"),
        str(weights / "raft-things.pth"),
        str(weights / "recurrent_flow_completion.pth"),
        device=device,
        use_half=bool(parameters["fp16"]),
    )
    print(
        f"ProPainter worker loaded models once on {device}; "
        f"processing {len(jobs)} chunk(s)",
        flush=True,
    )
    for job_number, raw_job in enumerate(jobs, start=1):
        if not isinstance(raw_job, dict):
            raise ValueError(f"Invalid job {job_number}")
        job = raw_job
        chunk_index = int(job["chunk_index"])
        expected_count = int(job["expected_count"])
        try:
            frames, masks, output_root = _read_job_inputs(job)
            print(
                f"ProPainter chunk {job_number}/{len(jobs)}: "
                f"index={chunk_index}, frames={expected_count}",
                flush=True,
            )
            result = engine.inpaint(
                frames,
                masks,
                dilate_radius=int(parameters["mask_dilation"]),
                raft_iter=int(parameters["raft_iter"]),
                subvideo_length=int(parameters["subvideo_length"]),
                neighbor_length=int(parameters["neighbor_length"]),
                ref_stride=int(parameters["ref_stride"]),
            )
            _write_output_frames(
                output_root,
                result,
                expected_count=expected_count,
            )
        except Exception as error:
            raise RuntimeError(
                f"ProPainter chunk {chunk_index} failed: "
                f"{type(error).__name__}: {error}"
            ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple ProPainter frame jobs with one model load"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--propainter-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_manifest(args.manifest.resolve(), args.propainter_root.resolve())


if __name__ == "__main__":
    main()
