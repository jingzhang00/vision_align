#!/usr/bin/env python3
"""Prepare processed RGB, metric VDA depth, lighting, and replay data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from human_to_robot.lighting_profile import (  # noqa: E402
    estimate_lighting_profile,
)
from repaint.rgbd_arm_ik import (  # noqa: E402
    repair_arm_positions_to_source_hands,
)
from repaint.rgbd_calibration import (  # noqa: E402
    calibrate_camera_translation_to_hands,
)
from repaint.rgbd_episode import RgbdEpisode, load_rgbd_episode  # noqa: E402
from repaint.rgbd_geometry import scale_intrinsic  # noqa: E402
from repaint.vda_depth import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_VDA_METRIC_SCALE,
    DecodedVideoFrames,
    INFERENCE_OVERLAP_FRAMES,
    INFERENCE_WINDOW_FRAMES,
    MODEL_CONFIGS,
    MODEL_SCALE_NAMES,
    decode_contiguous_video_frames_rgb,
    depth_m_to_uint16_mm,
    file_sha256,
    infer_metric_vda_sequence,
    scale_and_resize_metric_depths,
    validate_metric_checkpoint,
)
from repaint.vda_scale_validation import (  # noqa: E402
    infer_and_estimate_handless_metric_scale,
)
from repaint.trajectory_io import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_G1_URDF,
    SUPPORTED_FIELDS,
)


DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "dataset/insert_remove_usb"


def _open_at(path: Path, frame_index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    if frame_index:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if actual != frame_index:
            capture.release()
            raise RuntimeError(
                f"Could not seek {path} to frame {frame_index}; got {actual}"
            )
    return capture


def _require_output_layout(output_root: Path) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    frame_root = output_root / "rgb"
    depth_root = output_root / "depth_mm"
    for path in (frame_root, depth_root):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"Refusing to mix a new preparation with existing files: {path}"
            )
        path.mkdir(parents=True, exist_ok=True)
    for path in (
        output_root / "episode_data.npz",
        output_root / "lighting_profile.json",
        output_root / "prepare_report.json",
    ):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
    return frame_root, depth_root


def _validate_output_available(output_root: Path) -> None:
    """Reject a conflicting output before either expensive VDA pass."""

    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(output_root)
    for directory in (output_root / "rgb", output_root / "depth_mm"):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"Refusing to mix a new preparation with existing files: "
                f"{directory}"
            )
    for path in (
        output_root / "episode_data.npz",
        output_root / "lighting_profile.json",
        output_root / "prepare_report.json",
    ):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")


def _resolve_episode_metric_scale(
    *,
    args: argparse.Namespace,
    episode: RgbdEpisode,
    handless_decoded: DecodedVideoFrames,
    handless_depth_raw_m: np.ndarray,
    checkpoint: Path,
) -> tuple[float, dict[str, object]]:
    fixed_scale = float(args.vda_metric_scale)
    if not np.isfinite(fixed_scale) or fixed_scale <= 0.0:
        raise ValueError("--vda-metric-scale must be positive and finite")
    if args.vda_scale_mode == "fixed":
        return fixed_scale, {
            "schema_version": 1,
            "requested_mode": "fixed",
            "resolved_mode": "fixed",
            "resolved_vda_metric_scale": fixed_scale,
            "scale_excludes_scene_scale": True,
        }

    calibration = infer_and_estimate_handless_metric_scale(
        original_video=episode.source_video,
        handless_video=episode.removed_video,
        source_hdf5=episode.source_hdf5,
        frame_indices=episode.frame_indices,
        handless_decoded=handless_decoded,
        handless_depth_raw_m=handless_depth_raw_m,
        checkpoint=checkpoint,
        encoder=args.vda_encoder,
        input_size=args.vda_input_size,
        max_resolution=args.vda_max_resolution,
        fp32=args.vda_fp32,
        mask_dir=args.vda_scale_mask_dir,
    )
    status = str(calibration["decision"]["status"])
    if status == "validated":
        calibration["requested_mode"] = "auto"
        calibration["resolved_mode"] = "auto"
        return float(calibration["resolved_vda_metric_scale"]), calibration
    reasons = calibration["decision"]["reasons"]
    if args.vda_scale_failure_policy == "fixed":
        calibration["requested_mode"] = "auto"
        calibration["resolved_mode"] = "fixed_fallback"
        calibration["automatic_candidate_vda_metric_scale"] = calibration[
            "resolved_vda_metric_scale"
        ]
        calibration["resolved_vda_metric_scale"] = fixed_scale
        calibration["fallback_reason"] = reasons
        print(
            "WARNING: automatic VDA scale rejected; using explicitly allowed "
            f"fixed fallback {fixed_scale}: {reasons}",
            flush=True,
        )
        return fixed_scale, calibration
    raise RuntimeError(
        "Automatic VDA metric scale failed quality gates: "
        f"{reasons}. Use --vda-scale-mode fixed only after choosing an "
        "episode-specific scale."
    )


def prepare(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    output_root = args.output.resolve()
    _validate_output_available(output_root)
    episode = load_rgbd_episode(
        dataset_root=args.dataset,
        source_root=args.source_root,
        episode_index=args.episode,
        field=args.field,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        scene_scale=args.scene_scale,
    )
    if (
        args.arm_alignment == "source-hand-position"
        and args.camera_alignment != "source-extrinsic-only"
    ):
        raise ValueError(
            "Arm position repair and hand-based camera translation are "
            "alternative alignments; do not enable both"
        )
    if args.arm_alignment == "source-hand-position":
        arm_repair = repair_arm_positions_to_source_hands(
            episode,
            args.urdf,
        )
        joint_positions = arm_repair.joint_positions
        arm_alignment: dict[str, object] = {
            "method": "independent source 3D hand position IK",
            "optimized_joints": (
                "left/right shoulder pitch, roll, yaw, and elbow"
            ),
            "preserved_joints": "wrists, fingers, torso, and lower body",
            "correspondence_count": arm_repair.correspondence_count,
            "confidence_threshold": arm_repair.confidence_threshold,
            "hand_position_rmse_before_m": arm_repair.rmse_before_m,
            "hand_position_rmse_after_m": arm_repair.rmse_after_m,
            "hand_position_max_error_before_m": (
                arm_repair.max_error_before_m
            ),
            "hand_position_max_error_after_m": (
                arm_repair.max_error_after_m
            ),
            "max_joint_correction_rad": (
                arm_repair.max_joint_correction_rad
            ),
            "max_repaired_joint_step_rad": (
                arm_repair.max_repaired_joint_step_rad
            ),
        }
    else:
        joint_positions = episode.joint_positions
        arm_alignment = {
            "method": "trajectory-only",
            "optimized_joints": [],
        }
    uncalibrated_camera_pose = episode.camera_pose_cv_xyzw
    camera_alignment: dict[str, object]
    if args.camera_alignment == "human-hand-translation":
        full_episode = load_rgbd_episode(
            dataset_root=args.dataset,
            source_root=args.source_root,
            episode_index=args.episode,
            field=args.field,
            start_frame=0,
            end_frame=None,
            scene_scale=args.scene_scale,
        )
        calibration = calibrate_camera_translation_to_hands(
            full_episode,
            args.urdf,
        )
        camera_pose = calibration.camera_pose_cv_xyzw[
            episode.frame_indices
        ]
        camera_alignment = {
            "method": "episode-level human-hand translation",
            "translation_camera_xyz_m": (
                calibration.translation_camera_xyz.tolist()
            ),
            "correspondence_count": calibration.correspondence_count,
            "confidence_threshold": calibration.confidence_threshold,
            "hand_anchor_rmse_before_m": calibration.rmse_before_m,
            "hand_anchor_rmse_after_m": calibration.rmse_after_m,
            "hand_anchor_median_error_before_m": (
                calibration.median_error_before_m
            ),
            "hand_anchor_median_error_after_m": (
                calibration.median_error_after_m
            ),
            "preserves": "source camera rotation, motion, and intrinsic",
        }
    else:
        camera_pose = uncalibrated_camera_pose
        camera_alignment = {
            "method": "source-extrinsic-only",
            "translation_camera_xyz_m": [0.0, 0.0, 0.0],
            "preserves": "source camera rotation, motion, and intrinsic",
        }
    source_width, source_height = episode.source_size
    mesh_width = args.mesh_width
    mesh_height = int(round(source_height * mesh_width / source_width))
    if mesh_height < 2:
        raise ValueError("Derived mesh height is too small")
    mesh_size = (mesh_width, mesh_height)
    mesh_intrinsic = scale_intrinsic(
        episode.intrinsic,
        source_size=episode.source_size,
        target_size=mesh_size,
    )

    depths_mesh_m: np.ndarray | None = None
    depth_backend_report: dict[str, object] | None = None
    resolved_vda_metric_scale: float | None = None
    scale_calibration: dict[str, object] | None = None
    if not args.skip_depth:
        checkpoint = validate_metric_checkpoint(
            args.vda_checkpoint,
            args.vda_encoder,
        )
        decoded = decode_contiguous_video_frames_rgb(
            episode.removed_video,
            frame_indices=episode.frame_indices,
            max_resolution=args.vda_max_resolution,
        )
        if decoded.frame_count != episode.frame_count:
            raise ValueError(
                "VDA/video episode frame mismatch: "
                f"{decoded.frame_count} != {episode.frame_count}"
            )
        if decoded.source_size != episode.source_size:
            raise ValueError(
                "VDA/video episode size mismatch: "
                f"{decoded.source_size} != {episode.source_size}"
            )
        if not np.isclose(decoded.fps, episode.fps, atol=1e-3):
            raise ValueError(
                f"VDA/video episode FPS mismatch: "
                f"{decoded.fps} != {episode.fps}"
            )
        depths_raw_m = infer_metric_vda_sequence(
            decoded,
            checkpoint=checkpoint,
            encoder=args.vda_encoder,
            input_size=args.vda_input_size,
            fp32=args.vda_fp32,
        )
        # The matching original-video pass only needs handless metadata.
        # Avoid retaining two complete RGB sequences at once.
        decoded = decoded.without_rgb_frames()
        resolved_vda_metric_scale, scale_calibration = (
            _resolve_episode_metric_scale(
                args=args,
                episode=episode,
                handless_decoded=decoded,
                handless_depth_raw_m=depths_raw_m,
                checkpoint=checkpoint,
            )
        )
        depths_mesh_m = scale_and_resize_metric_depths(
            depths_raw_m,
            scene_scale=episode.scene_scale,
            vda_metric_scale=resolved_vda_metric_scale,
            mesh_size=mesh_size,
        )
        depth_backend_report = {
            "backend": "video-depth-anything",
            "model": (
                "Metric-Video-Depth-Anything-"
                f"{MODEL_SCALE_NAMES[args.vda_encoder]}"
            ),
            "encoder": args.vda_encoder,
            "metric": True,
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": file_sha256(checkpoint),
            "input_size": args.vda_input_size,
            "max_video_resolution": args.vda_max_resolution,
            "precision": (
                "fp32" if args.vda_fp32 else "autocast-fp16"
            ),
            "window_frames": INFERENCE_WINDOW_FRAMES,
            "window_overlap_frames": INFERENCE_OVERLAP_FRAMES,
            "inference_size": list(decoded.inference_size),
            "video_frame_count": decoded.video_frame_count,
            "selected_frame_indices": decoded.frame_indices.tolist(),
            "selected_frame_start": int(decoded.frame_indices[0]),
            "selected_frame_end_exclusive": int(
                decoded.frame_indices[-1] + 1
            ),
            "selection_is_contiguous": True,
            "input_video_sha256": file_sha256(episode.removed_video),
            "raw_depth": "metric camera-Z meters",
            "vda_metric_scale": resolved_vda_metric_scale,
            "vda_metric_scale_source": (
                "automatic per-episode HDF5 hand anchor plus unchanged "
                "background transfer"
                if scale_calibration["resolved_mode"] == "auto"
                else "explicit fixed value or explicitly allowed fallback"
            ),
            "vda_scale_calibration": scale_calibration,
            "vda_metric_scale_application_count": 1,
            "scene_scale": episode.scene_scale,
            "scene_scale_application_count": 1,
            "effective_raw_depth_to_robot_scale": (
                resolved_vda_metric_scale * episode.scene_scale
            ),
            "depth_resize_interpolation": "nearest",
        }
        del depths_raw_m, decoded

    frame_root, depth_root = _require_output_layout(output_root)
    lighting_path = output_root / "lighting_profile.json"
    lighting = estimate_lighting_profile(
        inpaint_video=episode.removed_video,
        output_path=lighting_path,
        reliability_mask_root=None,
        sample_count=args.lighting_sample_count,
        max_dimension=args.lighting_max_dimension,
        write_artifacts=True,
    )

    source_capture = _open_at(
        episode.source_video,
        int(episode.frame_indices[0]),
    )
    removed_capture = _open_at(
        episode.removed_video,
        int(episode.frame_indices[0]),
    )
    completed = 0
    positive_depth_samples: list[np.ndarray] = []
    try:
        while completed < episode.frame_count:
            expected_source_index = int(episode.frame_indices[completed])
            if (
                completed > 0
                and expected_source_index
                != int(episode.frame_indices[0]) + completed
            ):
                raise ValueError(
                    "Selected source frame indices must be contiguous"
                )
            source_ok, source_frame = source_capture.read()
            removed_ok, removed_frame = removed_capture.read()
            if not source_ok or not removed_ok:
                raise RuntimeError(
                    f"Video decode failed at source frame "
                    f"{expected_source_index}"
                )
            frame_path = frame_root / f"{completed:05d}.png"
            texture_frame = (
                source_frame
                if args.rgb_source == "original"
                else removed_frame
            )
            if not cv2.imwrite(str(frame_path), texture_frame):
                raise RuntimeError(f"Failed to write {frame_path}")
            if depths_mesh_m is not None:
                depth_mm = depth_m_to_uint16_mm(depths_mesh_m[completed])
                depth_path = depth_root / f"{completed:05d}.png"
                if not cv2.imwrite(str(depth_path), depth_mm):
                    raise RuntimeError(f"Failed to write {depth_path}")
                positive = depth_mm[depth_mm > 0]
                if positive.size:
                    stride = max(1, positive.size // 4096)
                    positive_depth_samples.append(positive[::stride])
            completed += 1
            print(
                f"Prepared "
                f"{'RGB' if args.skip_depth else 'RGB-D'} frames "
                f"{completed}/{episode.frame_count}",
                flush=True,
            )
    finally:
        source_capture.release()
        removed_capture.release()

    episode_data_path = output_root / "episode_data.npz"
    np.savez_compressed(
        episode_data_path,
        episode_index=np.asarray(episode.episode_index, dtype=np.int64),
        fps=np.asarray(episode.fps, dtype=np.float64),
        source_size=np.asarray(episode.source_size, dtype=np.int64),
        mesh_size=np.asarray(mesh_size, dtype=np.int64),
        frame_indices=episode.frame_indices,
        joint_names=np.asarray(episode.joint_names),
        joint_positions=joint_positions.astype(np.float32),
        root_pose_xyzw=episode.root_pose_xyzw.astype(np.float32),
        camera_pose_cv_xyzw=camera_pose.astype(np.float32),
        source_camera_pose_cv_xyzw=(
            episode.source_camera_pose_cv_xyzw.astype(np.float32)
        ),
        uncalibrated_camera_pose_cv_xyzw=(
            uncalibrated_camera_pose.astype(np.float32)
        ),
        recorded_camera_pose_cv_xyzw=(
            episode.recorded_camera_pose_cv_xyzw.astype(np.float32)
        ),
        intrinsic=episode.intrinsic.astype(np.float64),
        mesh_intrinsic=mesh_intrinsic.astype(np.float64),
        scene_scale=np.asarray(episode.scene_scale, dtype=np.float64),
        recorded_scene_scale=np.asarray(
            episode.recorded_scene_scale,
            dtype=np.float64,
        ),
        source_to_robot_rotation=(
            episode.source_to_robot_rotation.astype(np.float64)
        ),
        source_video=np.asarray(str(episode.source_video)),
        removed_video=np.asarray(str(episode.removed_video)),
        source_hdf5=np.asarray(str(episode.source_hdf5)),
        g1_urdf=np.asarray(str(args.urdf.resolve())),
        vda_metric_scale=np.asarray(
            (
                resolved_vda_metric_scale
                if resolved_vda_metric_scale is not None
                else np.nan
            ),
            dtype=np.float64,
        ),
        vda_scale_mode=np.asarray(
            (
                str(scale_calibration["resolved_mode"])
                if scale_calibration is not None
                else "disabled"
            )
        ),
    )

    samples = (
        np.concatenate(positive_depth_samples)
        if positive_depth_samples
        else np.empty(0, dtype=np.uint16)
    )
    percentiles = (
        np.percentile(samples, [1, 50, 99]).tolist()
        if samples.size
        else [None, None, None]
    )
    if args.skip_depth:
        depth_report: dict[str, object] = {
            "method": "disabled",
            "reason": "video-plane background does not use depth",
        }
    else:
        assert depth_backend_report is not None
        depth_report = {
            **depth_backend_report,
            "method": "offline_metric_video_depth_anything",
            "input": "removed-arm RGB",
            "output_unit": "uint16 millimeters in robot-scene scale",
            "percentiles_mm": {
                "p01": percentiles[0],
                "p50": percentiles[1],
                "p99": percentiles[2],
            },
        }
    base_camera_position_delta = np.linalg.norm(
        uncalibrated_camera_pose[:, :3]
        - episode.recorded_camera_pose_cv_xyzw[:, :3],
        axis=1,
    )
    calibrated_camera_position_delta = np.linalg.norm(
        camera_pose[:, :3]
        - episode.recorded_camera_pose_cv_xyzw[:, :3],
        axis=1,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "method": (
            "independent_video_plane"
            if args.skip_depth
            else "independent_rgb_texture_processed-depth_rgbd-mesh"
        ),
        "episode_index": episode.episode_index,
        "source_stem": episode.source_stem,
        "source_video": str(episode.source_video),
        "removed_video_for_depth_and_lighting": str(
            episode.removed_video
        ),
        "rgb_texture": {
            "source": args.rgb_source,
            "video": str(
                episode.source_video
                if args.rgb_source == "original"
                else episode.removed_video
            ),
        },
        "frame_indices": episode.frame_indices.tolist(),
        "frame_count": episode.frame_count,
        "fps": episode.fps,
        "source_size": list(episode.source_size),
        "mesh_size": list(mesh_size),
        "intrinsic": episode.intrinsic.tolist(),
        "mesh_intrinsic": mesh_intrinsic.tolist(),
        "depth": depth_report,
        "camera": {
            "source": (
                "source HDF5 extrinsic recomputed in robot world using "
                "scene_scale"
            ),
            "recorded_scene_scale": episode.recorded_scene_scale,
            "base_mean_position_delta_from_recorded_m": float(
                base_camera_position_delta.mean()
            ),
            "base_max_position_delta_from_recorded_m": float(
                base_camera_position_delta.max()
            ),
            "final_mean_position_delta_from_recorded_m": float(
                calibrated_camera_position_delta.mean()
            ),
            "final_max_position_delta_from_recorded_m": float(
                calibrated_camera_position_delta.max()
            ),
            "alignment": camera_alignment,
        },
        "robot_arm_alignment": arm_alignment,
        "lighting": {
            "input": "removed-arm RGB",
            "affects": "robot only; RGB-D mesh must be emissive",
            "profile": str(lighting_path),
            "isaac": lighting["isaac"],
            "confidence": lighting["confidence"],
            "warnings": lighting["warnings"],
        },
        "episode_data": str(episode_data_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_root / "prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one LeRobot G1 episode for independent video-plane or "
            "RGB-D Isaac repainting without human_to_robot.pipeline."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--field",
        choices=SUPPORTED_FIELDS,
        default="observation.state",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_G1_URDF)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=1.0,
        help=(
            "Source-to-robot translation scale, also applied once to metric "
            "Video Depth Anything depth "
            "when depth preparation is enabled (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--arm-alignment",
        choices=("source-hand-position", "trajectory-only"),
        default="trajectory-only",
        help=(
            "Repair shoulder/elbow positions from source HDF5 3D hands, "
            "or retain the supplied robot trajectory unchanged"
        ),
    )
    parser.add_argument(
        "--camera-alignment",
        choices=("human-hand-translation", "source-extrinsic-only"),
        default="source-extrinsic-only",
        help=(
            "Keep source camera rotation/intrinsic and optionally correct "
            "one episode-level translation using source 3D hand anchors"
        ),
    )
    parser.add_argument("--mesh-width", type=int, default=480)
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help=(
            "Prepare RGB, trajectory, extrinsics, and lighting only for the "
            "Isaac video-plane background"
        ),
    )
    parser.add_argument(
        "--vda-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Metric Video Depth Anything checkpoint",
    )
    parser.add_argument(
        "--vda-encoder",
        choices=tuple(MODEL_CONFIGS),
        default="vitl",
    )
    parser.add_argument(
        "--vda-input-size",
        type=int,
        default=518,
        help="Video Depth Anything network input size (default: 518)",
    )
    parser.add_argument(
        "--vda-max-resolution",
        type=int,
        default=1280,
        help=(
            "Maximum source-frame dimension retained for VDA inference; "
            "0 disables the cap (default: 1280)"
        ),
    )
    parser.add_argument(
        "--vda-scale-mode",
        choices=("auto", "fixed"),
        default="auto",
        help=(
            "Estimate one per-episode VDA-to-source metric scale from the "
            "original video and HDF5 hand anchors, or use a fixed value "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--vda-metric-scale",
        type=float,
        default=DEFAULT_VDA_METRIC_SCALE,
        help=(
            "Fixed scale used by --vda-scale-mode fixed, or by an explicitly "
            "allowed fixed fallback (default: legacy 0.525)"
        ),
    )
    parser.add_argument(
        "--vda-scale-failure-policy",
        choices=("error", "fixed"),
        default="error",
        help=(
            "Fail when automatic calibration is unreliable, or explicitly "
            "fall back to --vda-metric-scale (default: error)"
        ),
    )
    parser.add_argument(
        "--vda-scale-mask-dir",
        type=Path,
        help=(
            "Optional hand-mask PNG directory indexed by source frame; "
            "default derives the changed region from original/handless RGB"
        ),
    )
    parser.add_argument(
        "--vda-fp32",
        action="store_true",
        help="Disable CUDA autocast for VDA inference",
    )
    parser.add_argument(
        "--rgb-source",
        choices=("original", "processed"),
        default="processed",
        help=(
            "Visible video-plane/RGB-D texture; optional depth and lighting "
            "always use the processed repaint video"
        ),
    )
    parser.add_argument("--lighting-sample-count", type=int, default=16)
    parser.add_argument("--lighting-max-dimension", type=int, default=384)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()
