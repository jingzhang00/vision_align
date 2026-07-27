#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def is_camera_name(name: str) -> bool:
    return "camera" in name.lower()


def summarize(value: object) -> str:
    array = np.asarray(value)
    summary = f"shape={array.shape}, dtype={array.dtype}"
    if array.size <= 16:
        summary += f", value={array.tolist()}"
    elif array.ndim > 0:
        summary += f", first={array[0].tolist()}"
    return summary


def print_camera_fields(path: Path) -> bool:
    found = False
    with h5py.File(path, "r") as h5_file:
        for attr_name, attr_value in h5_file.attrs.items():
            if is_camera_name(attr_name):
                print(f"    - @{attr_name}: {summarize(attr_value)}")
                found = True

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            nonlocal found
            if isinstance(obj, h5py.Dataset) and is_camera_name(name):
                print(f"    - {name}: {summarize(obj)}")
                found = True
            for attr_name, attr_value in obj.attrs.items():
                full_name = f"{name}@{attr_name}"
                if is_camera_name(full_name):
                    print(f"    - {full_name}: {summarize(attr_value)}")
                    found = True

        h5_file.visititems(visitor)
    return found


def hdf5_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
        return [path]
    if path.is_dir():
        return sorted(
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in {".h5", ".hdf5"}
        )
    return []


def inspect_target(path: Path) -> bool:
    print(f"\nCheck: {path}")
    if not path.exists():
        print("  [error] path does not exist")
        return False

    files = hdf5_files(path)
    if not files:
        print("  [error] no .h5/.hdf5 files found")
        return False

    found = False
    for file in files:
        print(f"  [HDF5] {file}")
        try:
            has_camera = print_camera_fields(file)
        except OSError as error:
            print(f"  [error] failed to read {file}: {error}")
            continue

        if not has_camera:
            print("    no camera fields found")
            continue

        found = True
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="recursively check HDF5 files or directories for camera parameters"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="HDF5 file or directory to check",
    )
    args = parser.parse_args()

    results = [
        inspect_target(path.expanduser().resolve()) for path in args.paths
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
