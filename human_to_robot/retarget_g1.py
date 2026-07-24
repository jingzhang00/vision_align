from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .g1 import build_g1_model, render_g1, retarget_to_g1
from .trajectory import load_single_point_trajectory
from .video_io import video_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retarget HDF5 single-point hands to Unitree G1 with Mink"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_width, source_height, frame_count, _ = video_metadata(args.video)
    trajectory = load_single_point_trajectory(args.hdf5)
    if trajectory.frame_count != frame_count:
        raise ValueError(
            f"HDF5 has {trajectory.frame_count} frames; video has {frame_count}"
        )
    fovy = float(
        np.degrees(
            2.0
            * np.arctan(
                source_height / (2.0 * float(trajectory.intrinsic[1, 1]))
            )
        )
    )
    model = build_g1_model(args.mjcf, fovy)
    result = retarget_to_g1(
        model,
        trajectory,
        source_width=source_width,
        source_height=source_height,
    )
    report = render_g1(
        model,
        trajectory,
        result,
        output_root=args.output,
        width=args.render_width,
        height=args.render_height,
        source_video=args.video,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
