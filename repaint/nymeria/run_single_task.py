#!/usr/bin/env python3
"""Run one Nymeria clip visibly while retaining every pipeline artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.nymeria.nymeria_vda_pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Nymeria clip and retain alignment, robot, handless, "
            "prepared, and final artifacts. Isaac is visible by default."
        )
    )
    parser.add_argument(
        "--task",
        "--clip",
        dest="clip",
        required=True,
        help="Exact clip stem or an unambiguous cNNN suffix.",
    )
    parser.add_argument("--session", default=None)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE_ROOT / "dataset/nymeria",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "nymeria/single_task",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hide/show the Isaac window (default: visible).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete this run's existing pipeline output before starting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        sessions={args.session} if args.session else None,
        clips={args.clip},
        python_executable=Path(sys.executable).absolute(),
        headless=args.headless,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
