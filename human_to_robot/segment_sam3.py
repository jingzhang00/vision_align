from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .video_io import read_video, video_metadata, write_video


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def union_output_masks(
    outputs: dict[str, Any] | None, height: int, width: int
) -> np.ndarray:
    if not outputs or "out_binary_masks" not in outputs:
        return np.zeros((height, width), dtype=np.uint8)
    masks = _as_numpy(outputs["out_binary_masks"])
    if masks.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    masks = np.squeeze(masks)
    if masks.ndim == 2:
        union = masks.astype(bool)
    else:
        union = np.any(masks.reshape((-1,) + masks.shape[-2:]), axis=0)
    union = union.astype(np.uint8) * 255
    if union.shape != (height, width):
        union = cv2.resize(union, (width, height), interpolation=cv2.INTER_NEAREST)
    return union


def _write_masks(directory: Path, masks: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(masks):
        if not cv2.imwrite(str(directory / f"{index:05d}.png"), mask):
            raise RuntimeError(f"Failed to write mask frame {index}")


def _write_overlay(
    video_path: Path,
    masks: list[np.ndarray],
    output_path: Path,
    color: tuple[int, int, int],
    *,
    hardware_encoding: bool,
) -> None:
    frames, fps = read_video(video_path)
    if len(frames) != len(masks):
        raise ValueError(f"Video has {len(frames)} frames; masks have {len(masks)}")
    overlay_frames: list[np.ndarray] = []
    color_image = np.asarray(color, dtype=np.float32)[None, None, :]
    for frame, mask in zip(frames, masks, strict=True):
        alpha = (mask.astype(np.float32) / 255.0 * 0.5)[..., None]
        overlay = frame.astype(np.float32) * (1.0 - alpha) + color_image * alpha
        overlay_frames.append(np.clip(overlay, 0, 255).astype(np.uint8))
    write_video(
        output_path,
        overlay_frames,
        fps,
        source_video=video_path,
        hardware_encoding=hardware_encoding,
    )


def _segment_concept(
    predictor: Any,
    video_path: Path,
    *,
    prompt: str,
    prompt_frame: int,
    frame_count: int,
    height: int,
    width: int,
    threshold: float,
) -> list[np.ndarray]:
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(video_path),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": False,
        }
    )
    session_id = response["session_id"]
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(frame_count)]
    try:
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame,
                "text": prompt,
                "output_prob_thresh": threshold,
            }
        )
        frame_index = int(response["frame_index"])
        masks[frame_index] = union_output_masks(
            response.get("outputs"), height, width
        )
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both",
            "start_frame_index": prompt_frame,
            "output_prob_thresh": threshold,
        }
        for response in predictor.handle_stream_request(request):
            frame_index = int(response["frame_index"])
            if 0 <= frame_index < frame_count:
                masks[frame_index] = union_output_masks(
                    response.get("outputs"), height, width
                )
    finally:
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id}
        )
    return masks


def run_sam3_segmentation(
    *,
    video_path: Path,
    checkpoint_path: Path,
    output_root: Path,
    arm_prompt: str = "human arm",
    arm_prompt_frame: int = 0,
    object_prompt: str | None = None,
    object_prompt_frame: int | None = None,
    threshold: float = 0.45,
    debug_artifacts: bool = False,
    hardware_encoding: bool = True,
) -> dict[str, Any]:
    from sam3.model_builder import build_sam3_video_predictor

    width, height, frame_count, fps = video_metadata(video_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")

    predictor = build_sam3_video_predictor(
        checkpoint_path=str(checkpoint_path),
        gpus_to_use=[0],
    )
    try:
        arm_masks = _segment_concept(
            predictor,
            video_path,
            prompt=arm_prompt,
            prompt_frame=arm_prompt_frame,
            frame_count=frame_count,
            height=height,
            width=width,
            threshold=threshold,
        )
        _write_masks(output_root / "arm_masks", arm_masks)
        if debug_artifacts:
            _write_overlay(
                video_path,
                arm_masks,
                output_root / "arm_masks_overlay.mp4",
                (40, 40, 255),
                hardware_encoding=hardware_encoding,
            )

        object_masks: list[np.ndarray] | None = None
        resolved_object_frame: int | None = None
        if object_prompt:
            resolved_object_frame = (
                object_prompt_frame
                if object_prompt_frame is not None
                else arm_prompt_frame
            )
            object_masks = _segment_concept(
                predictor,
                video_path,
                prompt=object_prompt,
                prompt_frame=resolved_object_frame,
                frame_count=frame_count,
                height=height,
                width=width,
                threshold=threshold,
            )
            _write_masks(output_root / "object_masks", object_masks)
            if debug_artifacts:
                _write_overlay(
                    video_path,
                    object_masks,
                    output_root / "object_masks_overlay.mp4",
                    (40, 210, 40),
                    hardware_encoding=hardware_encoding,
                )
    finally:
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()
        del predictor
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "video": str(video_path),
        "checkpoint": str(checkpoint_path),
        "resolution": [width, height],
        "frame_count": frame_count,
        "fps": fps,
        "arm_prompt": arm_prompt,
        "arm_prompt_frame": arm_prompt_frame,
        "object_prompt": object_prompt,
        "object_prompt_frame": resolved_object_frame,
        "threshold": threshold,
        "hardware_encoding": hardware_encoding,
        "arm_area_fraction": [
            float(np.count_nonzero(mask) / mask.size) for mask in arm_masks
        ],
        "object_area_fraction": (
            [float(np.count_nonzero(mask) / mask.size) for mask in object_masks]
            if object_masks is not None
            else None
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "segmentation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment arms with local SAM3")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-prompt", default="human arm")
    parser.add_argument("--arm-prompt-frame", type=int, default=0)
    parser.add_argument("--object-prompt")
    parser.add_argument("--object-prompt-frame", type=int)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--debug-artifacts", action="store_true")
    parser.add_argument(
        "--hardware-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVIDIA NVENC for debug videos when available (default: true)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_sam3_segmentation(
        video_path=args.video.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        output_root=args.output.resolve(),
        arm_prompt=args.arm_prompt,
        arm_prompt_frame=args.arm_prompt_frame,
        object_prompt=args.object_prompt,
        object_prompt_frame=args.object_prompt_frame,
        threshold=args.threshold,
        debug_artifacts=args.debug_artifacts,
        hardware_encoding=args.hardware_encoding,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
