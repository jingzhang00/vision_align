from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from .lighting_profile import (
    DEFAULT_COMPOSITE_PROFILE,
    load_lighting_profile,
    resolve_profile_value,
)
from .video_io import EncodedVideoWriter, atomic_video_writer, video_metadata


def _numbered_image_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {directory}")
    return paths


def _read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (260, 42), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def _grade_robot(
    frame: np.ndarray,
    *,
    exposure: float,
    rgb_gain: tuple[float, float, float],
    contrast: float,
    saturation: float,
) -> np.ndarray:
    """Apply one fixed, video-editor-style grade to the robot layer."""

    rgb = frame[..., ::-1].astype(np.float32) / 255.0
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    linear *= np.asarray(rgb_gain, dtype=np.float32) * (2.0**exposure)
    rgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
    )
    luma = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    rgb = luma[..., None] + saturation * (rgb - luma[..., None])
    rgb = (rgb - 0.5) * contrast + 0.5
    return np.clip(rgb[..., ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)


def _resolve_robot_grade(
    *,
    lighting_profile_path: Path | None,
    robot_exposure: float | None,
    robot_rgb_gain: tuple[float, float, float] | None,
    robot_contrast: float | None,
    robot_saturation: float | None,
) -> tuple[float, tuple[float, float, float], float, float]:
    lighting_profile = load_lighting_profile(lighting_profile_path)
    explicit_values: dict[str, object | None] = {
        "robot_exposure": robot_exposure,
        "robot_rgb_gain": robot_rgb_gain,
        "robot_contrast": robot_contrast,
        "robot_saturation": robot_saturation,
    }
    resolved_grade = {
        key: resolve_profile_value(
            explicit=explicit,
            profile=lighting_profile,
            section="composite",
            key=key,
            default=DEFAULT_COMPOSITE_PROFILE[key],
        )
        for key, explicit in explicit_values.items()
    }
    rgb_gain = tuple(
        float(value)
        for value in resolved_grade["robot_rgb_gain"]  # type: ignore[union-attr]
    )
    if len(rgb_gain) != 3:
        raise ValueError("robot_rgb_gain must contain exactly three values")
    return (
        float(resolved_grade["robot_exposure"]),
        rgb_gain,
        float(resolved_grade["robot_contrast"]),
        float(resolved_grade["robot_saturation"]),
    )


def _depth_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(
        np.rint(depth_m[valid] * 1000.0),
        1,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    return depth_mm


def _robot_visibility_mask(
    *,
    robot_mask: np.ndarray,
    robot_depth_mm: np.ndarray,
    scene_depth_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not (
        robot_mask.shape
        == robot_depth_mm.shape
        == scene_depth_mm.shape
    ):
        raise ValueError("Robot mask and depth planes must have equal shapes")
    robot = robot_mask > 0
    valid_robot = robot_depth_mm > 0
    valid_scene = scene_depth_mm > 0
    visible = (
        robot
        & valid_robot
        & valid_scene
        & (robot_depth_mm <= scene_depth_mm)
    )
    return visible, valid_scene


def _composite_frame(
    *,
    background: np.ndarray,
    robot: np.ndarray,
    robot_mask: np.ndarray,
    robot_depth: np.ndarray,
    scene_depth_mm: np.ndarray,
    robot_exposure: float,
    robot_rgb_gain: tuple[float, float, float],
    robot_contrast: float,
    robot_saturation: float,
) -> tuple[np.ndarray, float, float, float]:
    robot = _grade_robot(
        robot,
        exposure=robot_exposure,
        rgb_gain=robot_rgb_gain,
        contrast=robot_contrast,
        saturation=robot_saturation,
    )
    visible_robot, valid_scene = _robot_visibility_mask(
        robot_mask=robot_mask,
        robot_depth_mm=_depth_to_uint16_mm(robot_depth),
        scene_depth_mm=scene_depth_mm,
    )
    occluded_robot = (robot_mask > 0) & ~visible_robot
    synthetic = background.copy()
    synthetic[visible_robot] = robot[visible_robot]
    return (
        synthetic,
        float(visible_robot.mean()),
        float(occluded_robot.mean()),
        float(valid_scene.mean()),
    )


class InMemoryVideoCompositor:
    """Composite Isaac frames directly into an atomically published MP4."""

    def __init__(
        self,
        *,
        source_video: Path,
        inpaint_video: Path,
        trajectory_path: Path,
        scene_depth_root: Path,
        output_root: Path,
        g1_root: Path,
        lighting_profile_path: Path | None = None,
        robot_exposure: float | None = None,
        robot_rgb_gain: tuple[float, float, float] | None = None,
        robot_contrast: float | None = None,
        robot_saturation: float | None = None,
        hardware_encoding: bool = True,
    ) -> None:
        self.source_video = source_video.resolve()
        self.inpaint_video = inpaint_video.resolve()
        self.trajectory_path = trajectory_path.resolve()
        self.output_root = output_root.resolve()
        self.g1_root = g1_root.resolve()
        self.lighting_profile_path = (
            lighting_profile_path.resolve()
            if lighting_profile_path is not None
            else None
        )
        (
            self.robot_exposure,
            self.robot_rgb_gain,
            self.robot_contrast,
            self.robot_saturation,
        ) = _resolve_robot_grade(
            lighting_profile_path=self.lighting_profile_path,
            robot_exposure=robot_exposure,
            robot_rgb_gain=robot_rgb_gain,
            robot_contrast=robot_contrast,
            robot_saturation=robot_saturation,
        )

        (
            self.source_width,
            self.source_height,
            self.frame_count,
            self.source_fps,
        ) = video_metadata(self.source_video)
        (
            self.width,
            self.height,
            background_count,
            background_fps,
        ) = video_metadata(self.inpaint_video)
        if self.frame_count != background_count:
            raise ValueError(
                "Source and inpaint frame counts disagree: "
                f"{self.frame_count} != {background_count}"
            )
        if abs(self.source_fps - background_fps) > 0.01:
            raise ValueError(
                f"FPS mismatch: source={self.source_fps}, "
                f"inpaint={background_fps}"
            )
        self.scene_depth_root = scene_depth_root.resolve()
        self.scene_depth_paths = _numbered_image_paths(
            self.scene_depth_root
        )
        if len(self.scene_depth_paths) != self.frame_count:
            raise ValueError(
                "Scene depth and video frame counts disagree: "
                f"{len(self.scene_depth_paths)} != {self.frame_count}"
        )
        self.hardware_encoding = hardware_encoding
        self._background_capture: cv2.VideoCapture | None = None
        self._stack: ExitStack | None = None
        self._writer: EncodedVideoWriter | None = None
        self._next_frame = 0
        self._robot_fractions: list[float] = []
        self._robot_occlusion_fractions: list[float] = []
        self._scene_valid_fractions: list[float] = []
        self._encoding: dict[str, object] | None = None
        self._report: dict[str, object] | None = None

    def __enter__(self) -> InMemoryVideoCompositor:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._background_capture = cv2.VideoCapture(
            str(self.inpaint_video)
        )
        if not self._background_capture.isOpened():
            self._release_captures()
            raise RuntimeError("Failed to open inpaint video")
        self._stack = ExitStack()
        try:
            self._writer = self._stack.enter_context(
                atomic_video_writer(
                    self.output_root / "synthetic_g1.mp4",
                    source_video=self.source_video,
                    output_size=(self.width, self.height),
                    hardware_encoding=self.hardware_encoding,
                )
            )
            self._encoding = self._writer.profile.to_dict()
        except BaseException:
            self._stack.close()
            self._release_captures()
            raise
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            if self._stack is None:
                return
            if exception_type is None and self._next_frame != self.frame_count:
                incomplete = RuntimeError(
                    "In-memory composite received "
                    f"{self._next_frame}/{self.frame_count} frames"
                )
                self._stack.__exit__(
                    RuntimeError,
                    incomplete,
                    incomplete.__traceback__,
                )
                raise incomplete
            self._stack.__exit__(exception_type, exception, traceback)
            if exception_type is None:
                self._report = self._build_report()
                (self.output_root / "composite_report.json").write_text(
                    json.dumps(self._report, indent=2),
                    encoding="utf-8",
                )
        finally:
            self._release_captures()

    def _release_captures(self) -> None:
        if self._background_capture is not None:
            self._background_capture.release()
        self._background_capture = None

    def write(
        self,
        robot: np.ndarray,
        robot_mask: np.ndarray,
        robot_depth: np.ndarray,
    ) -> np.ndarray:
        if (
            self._background_capture is None
            or self._writer is None
        ):
            raise RuntimeError("In-memory compositor is not open")
        frame_id = self._next_frame
        if frame_id >= self.frame_count:
            raise RuntimeError(
                f"Received more than {self.frame_count} rendered frames"
            )
        expected_image_shape = (self.height, self.width, 3)
        expected_plane_shape = (self.height, self.width)
        if robot.shape != expected_image_shape:
            raise ValueError(
                f"Robot frame shape {robot.shape} != {expected_image_shape}"
            )
        if robot_mask.shape != expected_plane_shape:
            raise ValueError(
                f"Robot mask shape {robot_mask.shape} != "
                f"{expected_plane_shape}"
            )
        if robot_depth.shape != expected_plane_shape:
            raise ValueError(
                f"Robot depth shape {robot_depth.shape} != "
                f"{expected_plane_shape}"
            )

        background_ok, background = self._background_capture.read()
        if not background_ok:
            raise RuntimeError(f"Video decode stopped at frame {frame_id}")
        if background.shape != expected_image_shape:
            raise RuntimeError(
                f"Unexpected inpaint frame shape: {background.shape}"
            )
        scene_depth_mm = _read_image(
            self.scene_depth_paths[frame_id],
            cv2.IMREAD_UNCHANGED,
        )
        if (
            scene_depth_mm.dtype != np.uint16
            or scene_depth_mm.shape != expected_plane_shape
        ):
            raise ValueError(
                "Scene depth must be source-resolution uint16 millimeters: "
                f"{self.scene_depth_paths[frame_id]}"
            )
        (
            synthetic,
            robot_fraction,
            occlusion_fraction,
            scene_valid_fraction,
        ) = _composite_frame(
            background=background,
            robot=robot,
            robot_mask=robot_mask,
            robot_depth=robot_depth,
            scene_depth_mm=scene_depth_mm,
            robot_exposure=self.robot_exposure,
            robot_rgb_gain=self.robot_rgb_gain,
            robot_contrast=self.robot_contrast,
            robot_saturation=self.robot_saturation,
        )
        self._writer.write(synthetic)
        self._robot_fractions.append(robot_fraction)
        self._robot_occlusion_fractions.append(occlusion_fraction)
        self._scene_valid_fractions.append(scene_valid_fraction)
        self._next_frame += 1
        return synthetic

    @property
    def report(self) -> dict[str, object]:
        if self._report is None:
            raise RuntimeError("Composite report is not available yet")
        return self._report

    def _build_report(self) -> dict[str, object]:
        if self._encoding is None or not self._robot_fractions:
            raise RuntimeError("Cannot report an empty composite")
        return {
            "frame_count": self.frame_count,
            "fps": self.source_fps,
            "resolution": [self.width, self.height],
            "encoding": self._encoding,
            "comparison_encoding": None,
            "mean_robot_area_fraction": float(
                np.mean(self._robot_fractions)
            ),
            "max_robot_area_fraction": float(
                np.max(self._robot_fractions)
            ),
            "mean_robot_occlusion_fraction": float(
                np.mean(self._robot_occlusion_fractions)
            ),
            "mean_scene_depth_valid_fraction": float(
                np.mean(self._scene_valid_fractions)
            ),
            "source_video": str(self.source_video),
            "inpaint_video": str(self.inpaint_video),
            "g1_root": str(self.g1_root),
            "trajectory": str(self.trajectory_path),
            "scene_depth_root": str(self.scene_depth_root),
            "occlusion_rule": (
                "M=robot_mask & valid(D_robot) & valid(D_scene) & "
                "(D_robot_mm <= D_scene_mm); "
                "I_syn=M*I_robot+(1-M)*I_inpaint"
            ),
            "lighting_profile": (
                str(self.lighting_profile_path)
                if self.lighting_profile_path is not None
                else None
            ),
            "streaming_in_memory": True,
            "robot_grade": {
                "exposure_stops": self.robot_exposure,
                "rgb_gain": list(self.robot_rgb_gain),
                "contrast": self.robot_contrast,
                "saturation": self.robot_saturation,
            },
        }


def compose_video(
    *,
    source_video: Path,
    inpaint_video: Path,
    g1_root: Path,
    trajectory_path: Path | None = None,
    scene_depth_root: Path,
    output_root: Path,
    lighting_profile_path: Path | None = None,
    robot_exposure: float | None = None,
    robot_rgb_gain: tuple[float, float, float] | None = None,
    robot_contrast: float | None = None,
    robot_saturation: float | None = None,
    debug_artifacts: bool = False,
    delete_input_frames: bool = False,
    hardware_encoding: bool = True,
) -> dict[str, object]:
    (
        robot_exposure,
        robot_rgb_gain,
        robot_contrast,
        robot_saturation,
    ) = _resolve_robot_grade(
        lighting_profile_path=lighting_profile_path,
        robot_exposure=robot_exposure,
        robot_rgb_gain=robot_rgb_gain,
        robot_contrast=robot_contrast,
        robot_saturation=robot_saturation,
    )

    source_width, source_height, source_count, source_fps = video_metadata(
        source_video
    )
    width, height, background_count, background_fps = video_metadata(
        inpaint_video
    )
    robot_rgb_paths = _numbered_image_paths(g1_root / "robot_rgb")
    robot_mask_paths = _numbered_image_paths(g1_root / "robot_masks")
    robot_depth_paths = _numbered_image_paths(g1_root / "robot_depth_mm")
    scene_depth_paths = _numbered_image_paths(scene_depth_root)
    counts = {
        source_count,
        background_count,
        len(robot_rgb_paths),
        len(robot_mask_paths),
        len(robot_depth_paths),
        len(scene_depth_paths),
    }
    if len(counts) != 1:
        raise ValueError(f"Input frame counts disagree: {sorted(counts)}")
    if abs(source_fps - background_fps) > 0.01:
        raise ValueError(f"FPS mismatch: source={source_fps}, inpaint={background_fps}")

    resolved_trajectory_path = (
        trajectory_path
        if trajectory_path is not None
        else g1_root / "g1_trajectory.npz"
    )
    robot_fractions: list[float] = []
    robot_occlusion_fractions: list[float] = []
    scene_valid_fractions: list[float] = []
    output_root.mkdir(parents=True, exist_ok=True)
    source_capture = cv2.VideoCapture(str(source_video))
    background_capture = cv2.VideoCapture(str(inpaint_video))
    if not source_capture.isOpened() or not background_capture.isOpened():
        source_capture.release()
        background_capture.release()
        raise RuntimeError("Failed to open source or inpaint video")
    selected = (
        np.linspace(0, source_count - 1, 6, dtype=np.int64).tolist()
        if debug_artifacts
        else []
    )
    selected_frames: dict[int, np.ndarray] = {}
    comparison_encoding: dict[str, object] | None = None
    print(f"Composite processed 0/{source_count}", flush=True)
    try:
        with ExitStack() as stack:
            synthetic_writer = stack.enter_context(
                atomic_video_writer(
                    output_root / "synthetic_g1.mp4",
                    source_video=source_video,
                    output_size=(width, height),
                    hardware_encoding=hardware_encoding,
                )
            )
            comparison_writer = (
                stack.enter_context(
                    atomic_video_writer(
                        output_root / "comparison_original_inpaint_g1.mp4",
                        source_video=source_video,
                        output_size=(width * 3, height),
                        hardware_encoding=hardware_encoding,
                    )
                )
                if debug_artifacts
                else None
            )
            if comparison_writer is not None:
                comparison_encoding = comparison_writer.profile.to_dict()
            for frame_id in range(source_count):
                source_ok, source = source_capture.read()
                background_ok, background = background_capture.read()
                if not source_ok or not background_ok:
                    raise RuntimeError(
                        f"Video decode stopped at frame {frame_id}"
                    )
                if source.shape[:2] != (source_height, source_width):
                    raise RuntimeError(
                        f"Unexpected source frame shape: {source.shape}"
                    )
                source = cv2.resize(
                    source,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
                if background.shape[:2] != (height, width):
                    raise RuntimeError(
                        f"Unexpected inpaint frame shape: {background.shape}"
                    )
                robot = _read_image(
                    robot_rgb_paths[frame_id], cv2.IMREAD_COLOR
                )
                robot_mask = _read_image(
                    robot_mask_paths[frame_id], cv2.IMREAD_GRAYSCALE
                )
                depth_mm = _read_image(
                    robot_depth_paths[frame_id], cv2.IMREAD_UNCHANGED
                )
                depth = depth_mm.astype(np.float32) / 1000.0

                scene_depth_mm = _read_image(
                    scene_depth_paths[frame_id],
                    cv2.IMREAD_UNCHANGED,
                )
                if (
                    scene_depth_mm.dtype != np.uint16
                    or scene_depth_mm.shape != (height, width)
                ):
                    raise ValueError(
                        "Scene depth must be source-resolution uint16 "
                        f"millimeters: {scene_depth_paths[frame_id]}"
                    )
                (
                    synthetic,
                    robot_fraction,
                    occlusion_fraction,
                    scene_valid_fraction,
                ) = (
                    _composite_frame(
                        background=background,
                        robot=robot,
                        robot_mask=robot_mask,
                        robot_depth=depth,
                        scene_depth_mm=scene_depth_mm,
                        robot_exposure=robot_exposure,
                        robot_rgb_gain=robot_rgb_gain,
                        robot_contrast=robot_contrast,
                        robot_saturation=robot_saturation,
                    )
                )
                synthetic_writer.write(synthetic)
                if comparison_writer is not None:
                    comparison_writer.write(
                        np.hstack(
                            [
                                _label(source, "original"),
                                _label(background, "ProPainter"),
                                _label(synthetic, "G1 synthesis"),
                            ]
                        )
                    )
                if frame_id in selected:
                    selected_frames[frame_id] = synthetic.copy()
                robot_fractions.append(robot_fraction)
                robot_occlusion_fractions.append(occlusion_fraction)
                scene_valid_fractions.append(scene_valid_fraction)
                if delete_input_frames:
                    robot_rgb_paths[frame_id].unlink(missing_ok=True)
                    robot_mask_paths[frame_id].unlink(missing_ok=True)
                    robot_depth_paths[frame_id].unlink(missing_ok=True)
                completed = frame_id + 1
                if (
                    completed == 1
                    or completed % 10 == 0
                    or completed == source_count
                ):
                    print(
                        f"Composite processed {completed}/{source_count}",
                        flush=True,
                    )
            synthetic_encoding = synthetic_writer.profile.to_dict()
    finally:
        source_capture.release()
        background_capture.release()

    if debug_artifacts:
        contact = np.vstack(
            [
                np.hstack([selected_frames[index] for index in selected[:3]]),
                np.hstack([selected_frames[index] for index in selected[3:]]),
            ]
        )
        if not cv2.imwrite(
            str(output_root / "synthetic_contact.jpg"), contact
        ):
            raise RuntimeError("Failed to write synthetic contact sheet")
    if delete_input_frames:
        for directory in (
            g1_root / "robot_rgb",
            g1_root / "robot_masks",
            g1_root / "robot_depth_mm",
        ):
            try:
                directory.rmdir()
            except OSError:
                pass

    report = {
        "frame_count": source_count,
        "fps": source_fps,
        "resolution": [width, height],
        "encoding": synthetic_encoding,
        "comparison_encoding": comparison_encoding,
        "mean_robot_area_fraction": float(np.mean(robot_fractions)),
        "max_robot_area_fraction": float(np.max(robot_fractions)),
        "mean_robot_occlusion_fraction": float(
            np.mean(robot_occlusion_fractions)
        ),
        "mean_scene_depth_valid_fraction": float(
            np.mean(scene_valid_fractions)
        ),
        "source_video": str(source_video),
        "inpaint_video": str(inpaint_video),
        "g1_root": str(g1_root),
        "trajectory": str(resolved_trajectory_path),
        "scene_depth_root": str(scene_depth_root),
        "occlusion_rule": (
            "M=robot_mask & valid(D_robot) & valid(D_scene) & "
            "(D_robot_mm <= D_scene_mm); "
            "I_syn=M*I_robot+(1-M)*I_inpaint"
        ),
        "lighting_profile": (
            str(lighting_profile_path) if lighting_profile_path is not None else None
        ),
        "robot_grade": {
            "exposure_stops": robot_exposure,
            "rgb_gain": list(robot_rgb_gain),
            "contrast": robot_contrast,
            "saturation": robot_saturation,
        },
    }
    (output_root / "composite_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Depth-aware compositing of rendered G1 arms"
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--inpaint-video", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Trajectory NPZ; defaults to G1_ROOT/g1_trajectory.npz",
    )
    parser.add_argument(
        "--scene-depths",
        type=Path,
        required=True,
        help="Numbered native-metric DA3 depth PNGs in uint16 millimeters",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lighting-profile",
        type=Path,
        help="Task-level JSON generated by human_to_robot.pipeline",
    )
    parser.add_argument("--robot-exposure", type=float)
    parser.add_argument(
        "--robot-rgb-gain",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
    )
    parser.add_argument("--robot-contrast", type=float)
    parser.add_argument("--robot-saturation", type=float)
    parser.add_argument("--debug-artifacts", action="store_true")
    parser.add_argument("--delete-input-frames", action="store_true")
    parser.add_argument(
        "--hardware-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVIDIA NVENC when available (default: true)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = compose_video(
        source_video=args.source_video.resolve(),
        inpaint_video=args.inpaint_video.resolve(),
        g1_root=args.g1_root.resolve(),
        trajectory_path=(
            args.trajectory.resolve()
            if args.trajectory is not None
            else None
        ),
        scene_depth_root=args.scene_depths.resolve(),
        output_root=args.output.resolve(),
        lighting_profile_path=(
            args.lighting_profile.resolve()
            if args.lighting_profile is not None
            else None
        ),
        robot_exposure=args.robot_exposure,
        robot_rgb_gain=(
            tuple(args.robot_rgb_gain) if args.robot_rgb_gain is not None else None
        ),
        robot_contrast=args.robot_contrast,
        robot_saturation=args.robot_saturation,
        debug_artifacts=args.debug_artifacts,
        delete_input_frames=args.delete_input_frames,
        hardware_encoding=args.hardware_encoding,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
