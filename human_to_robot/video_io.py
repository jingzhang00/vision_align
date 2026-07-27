from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import av
import cv2
import numpy as np


@dataclass(frozen=True)
class VideoEncodingProfile:
    source_codec: str
    encoder: str
    codec_profile: str | None
    frame_rate: Fraction
    pixel_format: str
    bit_rate: int | None
    color_range: int
    color_space: int
    color_primaries: int
    color_transfer: int
    source_resolution: tuple[int, int]
    output_resolution: tuple[int, int]
    hardware_encoding_requested: bool = False
    hardware_encoding_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame_rate"] = {
            "numerator": self.frame_rate.numerator,
            "denominator": self.frame_rate.denominator,
            "fps": float(self.frame_rate),
        }
        payload["source_resolution"] = list(self.source_resolution)
        payload["output_resolution"] = list(self.output_resolution)
        return payload


def _fractional_rate(value: object) -> Fraction | None:
    if value is None:
        return None
    rate = Fraction(value)
    return rate if rate > 0 else None


@lru_cache(maxsize=None)
def _nvenc_encoder_available(
    encoder: str,
    width: int = 256,
    height: int = 256,
) -> bool:
    """Return whether FFmpeg can open and encode with an NVENC encoder."""

    try:
        context = av.CodecContext.create(encoder, "w")
        context.width = width
        context.height = height
        context.pix_fmt = "yuv420p"
        context.time_base = Fraction(1, 30)
        context.framerate = Fraction(30, 1)
        context.options = {"preset": "p4"}
        context.open()
        frame = av.VideoFrame.from_ndarray(
            np.zeros((height, width, 3), dtype=np.uint8),
            format="bgr24",
        )
        context.encode(frame)
        context.encode(None)
    except (ValueError, av.error.FFmpegError):
        return False
    return True


def _nvenc_encoder_for(source_codec: str) -> str:
    if source_codec in {"hevc", "h265"}:
        return "hevc_nvenc"
    return "h264_nvenc"


def source_encoding_profile(
    source_video: str | Path,
    *,
    output_size: tuple[int, int],
    hardware_encoding: bool = True,
) -> VideoEncodingProfile:
    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        raise ValueError("Output width and height must be positive")
    with av.open(str(source_video)) as container:
        if not container.streams.video:
            raise ValueError(f"Video stream not found: {source_video}")
        stream = container.streams.video[0]
        context = stream.codec_context
        frame_rate = next(
            (
                rate
                for rate in (
                    _fractional_rate(stream.average_rate),
                    _fractional_rate(stream.base_rate),
                    _fractional_rate(stream.guessed_rate),
                )
                if rate is not None
            ),
            None,
        )
        if frame_rate is None:
            raise ValueError(f"Could not determine source FPS: {source_video}")
        source_width = int(context.width)
        source_height = int(context.height)
        if source_width <= 0 or source_height <= 0:
            raise ValueError(
                f"Invalid source resolution {source_width}x{source_height}: "
                f"{source_video}"
            )
        source_codec = context.name or "mpeg4"
        nvenc_encoder = _nvenc_encoder_for(source_codec)
        hardware_encoding_active = (
            hardware_encoding
            and _nvenc_encoder_available(
                nvenc_encoder,
                output_width,
                output_height,
            )
        )
        if hardware_encoding_active:
            encoder_codec = av.codec.Codec(nvenc_encoder, "w")
            codec_profile = None
        else:
            try:
                encoder_codec = av.codec.Codec(source_codec, "w")
                codec_profile = context.profile
            except (ValueError, av.error.FFmpegError):
                encoder_codec = av.codec.Codec("h264", "w")
                codec_profile = None
        encoder = encoder_codec.name
        pixel_format = (
            context.format.name if context.format is not None else "yuv420p"
        )
        supported_formats = {
            video_format.name
            for video_format in (encoder_codec.video_formats or [])
        }
        if supported_formats and pixel_format not in supported_formats:
            pixel_format = (
                "yuv420p"
                if "yuv420p" in supported_formats
                else sorted(supported_formats)[0]
            )
        source_bit_rate = stream.bit_rate or context.bit_rate
        bit_rate = None
        if source_bit_rate is not None and source_bit_rate > 0:
            pixel_ratio = (
                output_width
                * output_height
                / (source_width * source_height)
            )
            bit_rate = max(100_000, int(round(source_bit_rate * pixel_ratio)))
        color_range = int(context.color_range)
        color_space = int(context.colorspace)
        color_primaries = int(context.color_primaries)
        color_transfer = int(context.color_trc)
    return VideoEncodingProfile(
        source_codec=source_codec,
        encoder=encoder,
        hardware_encoding_requested=hardware_encoding,
        hardware_encoding_active=hardware_encoding_active,
        codec_profile=codec_profile,
        frame_rate=frame_rate,
        pixel_format=pixel_format,
        bit_rate=bit_rate,
        color_range=color_range,
        color_space=color_space,
        color_primaries=color_primaries,
        color_transfer=color_transfer,
        source_resolution=(source_width, source_height),
        output_resolution=output_size,
    )


class EncodedVideoWriter:
    def __init__(self, path: str | Path, profile: VideoEncodingProfile) -> None:
        self.path = Path(path)
        self.profile = profile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.path), mode="w")
        self._stream = self._container.add_stream(
            profile.encoder,
            rate=profile.frame_rate,
        )
        self._stream.width, self._stream.height = profile.output_resolution
        self._stream.pix_fmt = profile.pixel_format
        if profile.hardware_encoding_active:
            self._stream.options = {"preset": "p4"}
        if profile.codec_profile is not None:
            self._stream.codec_context.profile = profile.codec_profile
        if profile.bit_rate is not None:
            self._stream.codec_context.bit_rate = profile.bit_rate
        self._stream.codec_context.color_range = profile.color_range
        self._stream.codec_context.colorspace = profile.color_space
        self._stream.codec_context.color_primaries = profile.color_primaries
        self._stream.codec_context.color_trc = profile.color_transfer
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        expected_width, expected_height = self.profile.output_resolution
        if frame.shape != (expected_height, expected_width, 3):
            raise ValueError(
                f"Frame shape {frame.shape} does not match "
                f"{expected_width}x{expected_height} BGR"
            )
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()
            self._closed = True

    def __enter__(self) -> EncodedVideoWriter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        if exception_type is None:
            self.close()
        else:
            self._container.close()
            self._closed = True


@contextmanager
def atomic_video_writer(
    path: str | Path,
    *,
    source_video: str | Path,
    output_size: tuple[int, int],
    hardware_encoding: bool = True,
) -> Iterator[EncodedVideoWriter]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = source_encoding_profile(
        source_video,
        output_size=output_size,
        hardware_encoding=hardware_encoding,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with EncodedVideoWriter(temporary, profile) as writer:
            yield writer
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def write_video(
    path: str | Path,
    frames: list[np.ndarray],
    fps: float,
    *,
    source_video: str | Path | None = None,
    hardware_encoding: bool = True,
) -> VideoEncodingProfile:
    if not frames:
        raise ValueError("Cannot write an empty video")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    if source_video is not None:
        profile = source_encoding_profile(
            source_video,
            output_size=(width, height),
            hardware_encoding=hardware_encoding,
        )
    else:
        hardware_encoding_active = (
            hardware_encoding
            and _nvenc_encoder_available("h264_nvenc", width, height)
        )
        profile = VideoEncodingProfile(
            source_codec="mpeg4",
            encoder=("h264_nvenc" if hardware_encoding_active else "mpeg4"),
            hardware_encoding_requested=hardware_encoding,
            hardware_encoding_active=hardware_encoding_active,
            codec_profile=None,
            frame_rate=Fraction(fps).limit_denominator(100_000),
            pixel_format="yuv420p",
            bit_rate=None,
            color_range=0,
            color_space=2,
            color_primaries=2,
            color_transfer=2,
            source_resolution=(width, height),
            output_resolution=(width, height),
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with EncodedVideoWriter(temporary, profile) as writer:
            for frame in frames:
                writer.write(frame)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return profile
