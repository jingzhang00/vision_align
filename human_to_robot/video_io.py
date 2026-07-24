from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def video_metadata(path: str | Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return width, height, count, fps


def transcode_video(
    source: str | Path,
    destination: str | Path,
    *,
    width: int,
    height: int,
) -> tuple[int, float]:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {destination}")

    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
        count += 1
    capture.release()
    writer.release()
    if expected and count != expected:
        raise RuntimeError(f"Decoded {count} frames but metadata advertised {expected}")
    return count, fps


def read_video(path: str | Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return frames, fps


def write_video(path: str | Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("Cannot write an empty video")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError("All video frames must have the same resolution")
        writer.write(frame)
    writer.release()
