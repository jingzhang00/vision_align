#!/usr/bin/env python3
"""Run one EgoDex task visibly while retaining every pipeline artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from repaint.egodex.egodex_vda_pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one EgoDex task and retain LeRobot, handless, prepared, "
            "and final artifacts. Isaac is visible by default."
        )
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE_ROOT / "dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "egodex_single_task",
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
        help="Delete this task's existing output before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        (args.task,),
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        python_executable=Path(sys.executable).absolute(),
        headless=args.headless,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
