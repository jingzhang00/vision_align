#!/usr/bin/env python3
"""Repaint egocentric human arms as Unitree G1 arms with PowerPaint v2-1.

This is an inference-only wrapper around the unmodified upstream PowerPaint
checkout.  It deliberately keeps the source silhouette/pose and composites
the generated pixels only inside the supplied human-arm mask.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
POWERPAINT_ROOT = WORKSPACE_ROOT / "PowerPaint"
DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "PowerPaint-v2-1"
DEFAULT_DATASET = WORKSPACE_ROOT / "dataset" / "basic_pick_place"
DEFAULT_MASK_ROOT = (
    WORKSPACE_ROOT
    / "output_new"
    / "basic_pick_place"
    / "videos_handless"
    / "_masks"
    / "observation.images.ego"
    / "chunk-000"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"

DEFAULT_PROMPT = (
    "first-person real Unitree G1 robot arms, simple industrial design, smooth "
    "matte ivory-white rounded plastic covers, sparse black seams, dark gray "
    "cylindrical joints, small matte black robotic hands, unlit, photorealistic, "
    "same pose, same silhouette, same perspective, original lighting"
)
DEFAULT_NEGATIVE_PROMPT = (
    "human skin, flesh, bare arm, sleeve, cloth, organic hand, extra arm, "
    "extra fingers, deformed fingers, sci-fi, cyberpunk, complex armor, "
    "glowing, LED, chrome, text, logo, watermark, background changes"
)

# Keep all model/cache writes within the requested draw workspace or /tmp.
os.environ.setdefault(
    "HF_HOME", str(Path(__file__).resolve().parent / ".hf_cache")
)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vision-align-powerpaint-mpl")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

powerpaint_path = str(POWERPAINT_ROOT)
if powerpaint_path not in sys.path:
    sys.path.insert(0, powerpaint_path)

from diffusers import UniPCMultistepScheduler  # noqa: E402
from powerpaint.models.BrushNet_CA import BrushNetModel  # noqa: E402
from powerpaint.models.unet_2d_condition import (  # noqa: E402
    UNet2DConditionModel,
)
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import (  # noqa: E402
    StableDiffusionPowerPaintBrushNetPipeline,
)
from powerpaint.utils.utils import TokenizerWrapper, add_tokens  # noqa: E402
from safetensors.torch import load_model  # noqa: E402
from transformers import CLIPTextModel  # noqa: E402


@dataclass(frozen=True)
class Episode:
    name: str
    video_path: Path
    mask_dir: Path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int
    fps: float


def set_seed(seed: int) -> None:
    """Use the same seeding behavior as upstream PowerPaint."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_shape_task(
    prompt: str, negative_prompt: str
) -> tuple[str, str, str, str]:
    """Return the task-token prompts used by upstream PowerPaint v2."""

    return (
        prompt + " P_shape",
        prompt + " P_ctxt",
        negative_prompt + "P_shape",
        negative_prompt + "P_ctxt",
    )


def video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(width, height, frame_count) <= 0 or fps <= 0:
        raise ValueError(
            f"Invalid video metadata for {path}: "
            f"{width}x{height}, {frame_count} frames, {fps} fps"
        )
    return VideoInfo(width, height, frame_count, fps)


def numeric_video_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return sys.maxsize, path.name


def discover_episodes(
    dataset_root: Path, mask_root: Path, requested: set[str] | None
) -> list[Episode]:
    videos = sorted(dataset_root.glob("*.mp4"), key=numeric_video_sort_key)
    mask_dirs = sorted(
        (path for path in mask_root.glob("file-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found under {dataset_root}")
    if len(mask_dirs) != len(videos):
        raise ValueError(
            f"Expected one mask directory per source video, found "
            f"{len(videos)} videos and {len(mask_dirs)} mask directories"
        )

    episodes: list[Episode] = []
    for video, mask_dir in zip(videos, mask_dirs, strict=True):
        if requested is None or video.stem in requested:
            episodes.append(Episode(video.stem, video, mask_dir))
    if not episodes:
        raise ValueError(
            f"No requested episodes found. Available: "
            f"{', '.join(path.stem for path in videos)}"
        )
    return episodes


def mask_paths(mask_dir: Path) -> list[Path]:
    # The existing SAM masks use six digits. Five-digit masks are also accepted
    # so the wrapper can consume human_to_robot.segment_sam3 output directly.
    paths = sorted(mask_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG masks found under {mask_dir}")
    return paths


def validate_episode(episode: Episode) -> VideoInfo:
    info = video_info(episode.video_path)
    masks = mask_paths(episode.mask_dir)
    if len(masks) != info.frame_count:
        raise ValueError(
            f"{episode.name}: video has {info.frame_count} frames but "
            f"{episode.mask_dir} has {len(masks)} masks"
        )
    sample = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
    if sample is None:
        raise ValueError(f"Could not read mask: {masks[0]}")
    if sample.shape != (info.height, info.width):
        raise ValueError(
            f"{episode.name}: mask is {sample.shape[1]}x{sample.shape[0]}, "
            f"video is {info.width}x{info.height}"
        )
    return info


def inference_size(width: int, height: int, short_side: int) -> tuple[int, int]:
    """Match upstream PowerPaint's short-side resize and /8 truncation."""

    if width < height:
        resized_width = short_side
        resized_height = int(height / width * short_side)
    else:
        resized_width = int(width / height * short_side)
        resized_height = short_side
    return resized_width - resized_width % 8, resized_height - resized_height % 8


def prepare_mask(
    mask: np.ndarray, dilate_pixels: int
) -> np.ndarray:
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    if dilate_pixels > 0:
        size = 2 * dilate_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.dilate(binary, kernel)
    return binary


def inside_feather_alpha(mask: np.ndarray, feather_pixels: float) -> np.ndarray:
    """Feather inward while keeping alpha exactly zero outside the mask."""

    binary = mask > 0
    if feather_pixels <= 0:
        return binary.astype(np.float32)
    distance = cv2.distanceTransform(
        binary.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    alpha = np.clip(distance / feather_pixels, 0.0, 1.0)
    alpha[~binary] = 0.0
    return alpha.astype(np.float32)


def strict_mask_composite(
    original_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    mask: np.ndarray,
    feather_pixels: float,
) -> np.ndarray:
    alpha = inside_feather_alpha(mask, feather_pixels)[..., None]
    composite = (
        generated_rgb.astype(np.float32) * alpha
        + original_rgb.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(np.rint(composite), 0, 255).astype(np.uint8)


def make_comparison(
    original_rgb: np.ndarray, mask: np.ndarray, output_rgb: np.ndarray
) -> np.ndarray:
    overlay = original_rgb.astype(np.float32).copy()
    tint = np.zeros_like(overlay)
    tint[..., 0] = 255
    alpha = (mask.astype(np.float32) / 255.0 * 0.45)[..., None]
    overlay = overlay * (1.0 - alpha) + tint * alpha
    return np.concatenate(
        [
            original_rgb,
            np.clip(overlay, 0, 255).astype(np.uint8),
            output_rgb,
        ],
        axis=1,
    )


class PowerPaintV21:
    """Local-only loader using the unmodified upstream model classes."""

    def __init__(self, checkpoint_dir: Path, device: str = "cuda") -> None:
        if device != "cuda":
            raise ValueError(
                "Upstream PowerPaint v2-1 inference requires CUDA; "
                "this wrapper intentionally does not patch it for CPU."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not visible. Run outside the device-restricted "
                "sandbox on a CUDA host."
            )

        checkpoint_dir = checkpoint_dir.resolve()
        base_model = checkpoint_dir / "realisticVisionV60B1_v51VAE"
        brushnet_dir = checkpoint_dir / "PowerPaint_Brushnet"
        required = [
            base_model / "model_index.json",
            brushnet_dir / "diffusion_pytorch_model.safetensors",
            brushnet_dir / "pytorch_model.bin",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Incomplete PowerPaint-v2-1 checkpoint: " + ", ".join(missing)
            )

        dtype = torch.float16
        # Upstream app.py initializes these two modules from SD 1.5. The local
        # v2-1 bundle contains the same architecture under its bundled base
        # model, so using that directory keeps inference fully offline.
        init_unet = UNet2DConditionModel.from_pretrained(
            base_model,
            subfolder="unet",
            revision=None,
            torch_dtype=dtype,
            local_files_only=True,
        )
        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            base_model,
            subfolder="text_encoder",
            revision=None,
            torch_dtype=dtype,
            local_files_only=True,
        )
        brushnet = BrushNetModel.from_unet(init_unet)
        del init_unet

        pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
            base_model,
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
            local_files_only=True,
        )
        pipe.unet = UNet2DConditionModel.from_pretrained(
            base_model,
            subfolder="unet",
            revision=None,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipe.tokenizer = TokenizerWrapper(
            from_pretrained=base_model,
            subfolder="tokenizer",
            revision=None,
            local_files_only=True,
        )
        add_tokens(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )
        load_model(
            pipe.brushnet,
            str(brushnet_dir / "diffusion_pytorch_model.safetensors"),
        )
        learned_text_encoder = torch.load(
            brushnet_dir / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        pipe.text_encoder_brushnet.load_state_dict(
            learned_text_encoder, strict=False
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(device)
        self.device = device

    def _validate_prompt_lengths(
        self, prompt: str, negative_prompt: str
    ) -> None:
        prompt_a, prompt_b, negative_a, negative_b = add_shape_task(
            prompt, negative_prompt
        )
        maximum = int(self.pipe.tokenizer.model_max_length)
        for label, text in (
            ("shape prompt", prompt_a),
            ("context prompt", prompt_b),
            ("shape negative prompt", negative_a),
            ("context negative prompt", negative_b),
        ):
            length = len(
                self.pipe.tokenizer(text, truncation=False).input_ids
            )
            if length > maximum:
                raise ValueError(
                    f"{label} expands to {length} CLIP tokens; PowerPaint "
                    f"allows at most {maximum}. Shorten the prompt because "
                    "each PowerPaint task token expands to 10 learned tokens."
                )

    @torch.inference_mode()
    def repaint(
        self,
        original_rgb: np.ndarray,
        mask: np.ndarray,
        *,
        prompt: str,
        negative_prompt: str,
        fitting_degree: float,
        steps: int,
        guidance_scale: float,
        seed: int,
        short_side: int,
        feather_pixels: float,
    ) -> np.ndarray:
        original_height, original_width = original_rgb.shape[:2]
        self._validate_prompt_lengths(prompt, negative_prompt)
        width, height = inference_size(
            original_width, original_height, short_side
        )
        image = Image.fromarray(original_rgb).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        resized_mask = Image.fromarray(mask).resize(
            (width, height), Image.Resampling.NEAREST
        )
        mask_rgb = np.repeat(
            np.asarray(resized_mask, dtype=np.uint8)[..., None], 3, axis=2
        )
        masked = np.asarray(image, dtype=np.uint8) * (
            1.0 - mask_rgb.astype(np.float32) / 255.0
        )
        masked_image = Image.fromarray(masked.astype(np.uint8), mode="RGB")
        mask_image = Image.fromarray(mask_rgb, mode="RGB")

        prompt_a, prompt_b, negative_a, negative_b = add_shape_task(
            prompt, negative_prompt
        )
        set_seed(seed)
        result = self.pipe(
            promptA=prompt_a,
            promptB=prompt_b,
            promptU=prompt,
            tradoff=fitting_degree,
            tradoff_nag=fitting_degree,
            image=masked_image,
            mask=mask_image,
            num_inference_steps=steps,
            generator=torch.Generator(self.device).manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            negative_promptA=negative_a,
            negative_promptB=negative_b,
            negative_promptU=negative_prompt,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        ).images[0]
        generated = np.asarray(
            result.resize(
                (original_width, original_height), Image.Resampling.LANCZOS
            ),
            dtype=np.uint8,
        )
        return strict_mask_composite(
            original_rgb, generated, mask, feather_pixels
        )


def parse_frame_spec(
    value: str | None, frame_count: int
) -> set[int] | None:
    if value is None:
        return None
    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            index += frame_count
        if not 0 <= index < frame_count:
            raise ValueError(
                f"Frame {item} resolves to {index}, outside [0, {frame_count})"
            )
        selected.add(index)
    if not selected:
        raise ValueError("--frames did not select any frames")
    return selected


def start_video_writer(
    output_path: Path, info: VideoInfo
) -> subprocess.Popen[bytes]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{info.width}x{info.height}",
        "-r",
        f"{info.fps:.12g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def process_episode(
    model: PowerPaintV21,
    episode: Episode,
    info: VideoInfo,
    args: argparse.Namespace,
) -> dict[str, object]:
    selected = parse_frame_spec(args.frames, info.frame_count)
    is_full_video = selected is None
    selected = selected or set(range(info.frame_count))
    masks = mask_paths(episode.mask_dir)

    episode_output = args.output_dir / episode.name
    frames_output = episode_output / "frames"
    comparisons_output = episode_output / "comparisons"
    effective_masks_output = episode_output / "effective_masks"
    for directory in (
        frames_output,
        comparisons_output,
        effective_masks_output,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    temporary_video = episode_output / f".{episode.name}_g1_powerpaint.tmp.mp4"
    final_video = episode_output / f"{episode.name}_g1_powerpaint.mp4"
    writer = start_video_writer(temporary_video, info) if is_full_video else None
    capture = cv2.VideoCapture(str(episode.video_path))
    processed = 0
    try:
        for frame_index in tqdm(
            range(info.frame_count),
            desc=f"episode {episode.name}",
            unit="frame",
            dynamic_ncols=True,
        ):
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"{episode.name}: failed to decode frame {frame_index}"
                )
            original_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            output_rgb = original_rgb
            if frame_index in selected:
                mask = cv2.imread(
                    str(masks[frame_index]), cv2.IMREAD_GRAYSCALE
                )
                if mask is None:
                    raise RuntimeError(
                        f"Could not read mask {masks[frame_index]}"
                    )
                effective_mask = prepare_mask(mask, args.mask_dilate)
                frame_seed = (
                    args.seed + frame_index
                    if args.seed_mode == "frame"
                    else args.seed
                )
                output_rgb = model.repaint(
                    original_rgb,
                    effective_mask,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    fitting_degree=args.fitting_degree,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=frame_seed,
                    short_side=args.short_side,
                    feather_pixels=args.mask_feather,
                )
                Image.fromarray(output_rgb).save(
                    frames_output / f"{frame_index:06d}.png"
                )
                Image.fromarray(effective_mask).save(
                    effective_masks_output / f"{frame_index:06d}.png"
                )
                comparison = make_comparison(
                    original_rgb, effective_mask, output_rgb
                )
                Image.fromarray(comparison).save(
                    comparisons_output / f"{frame_index:06d}.jpg",
                    quality=92,
                    subsampling=0,
                )
                processed += 1
            if writer is not None:
                assert writer.stdin is not None
                writer.stdin.write(np.ascontiguousarray(output_rgb).tobytes())
    finally:
        capture.release()
        if writer is not None:
            assert writer.stdin is not None
            writer.stdin.close()
            return_code = writer.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"ffmpeg failed with exit status {return_code}"
                )
    if writer is not None:
        temporary_video.replace(final_video)

    return {
        "episode": episode.name,
        "source_video": str(episode.video_path),
        "mask_dir": str(episode.mask_dir),
        "frame_count": info.frame_count,
        "processed_frames": processed,
        "video": str(final_video) if is_full_video else None,
        "frames_dir": str(frames_output),
        "comparisons_dir": str(comparisons_output),
    }


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.fitting_degree <= 1.0:
        raise ValueError("--fitting-degree must be in [0, 1]")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.guidance_scale <= 0:
        raise ValueError("--guidance-scale must be positive")
    if args.short_side < 64 or args.short_side % 8:
        raise ValueError("--short-side must be >= 64 and divisible by 8")
    if args.mask_dilate < 0:
        raise ValueError("--mask-dilate must be non-negative")
    if args.mask_feather < 0:
        raise ValueError("--mask-feather must be non-negative")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use unmodified PowerPaint v2-1 modules to repaint masked human "
            "arms with a Unitree G1 material appearance."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--episodes",
        nargs="+",
        help="Source video stems to process, e.g. --episodes 9 10",
    )
    parser.add_argument(
        "--frames",
        help=(
            "Comma-separated preview frame indices. Omit to process and encode "
            "the complete video; negative indices count from the end."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT
    )
    parser.add_argument("--fitting-degree", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=20240731)
    parser.add_argument(
        "--seed-mode",
        choices=("fixed", "frame"),
        default="fixed",
        help="fixed reduces frame-to-frame material drift",
    )
    parser.add_argument("--short-side", type=int, default=640)
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=3,
        help="Pixels added around the SAM arm mask to cover skin-colored edges",
    )
    parser.add_argument(
        "--mask-feather",
        type=float,
        default=3.0,
        help="Inward-only compositing feather width in source-resolution pixels",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate video/mask pairing without loading the model",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    args.dataset_root = args.dataset_root.resolve()
    args.mask_root = args.mask_root.resolve()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()

    requested = set(args.episodes) if args.episodes else None
    episodes = discover_episodes(
        args.dataset_root, args.mask_root, requested
    )
    episode_info = {
        episode.name: validate_episode(episode) for episode in episodes
    }
    print(
        "Validated episodes: "
        + ", ".join(
            f"{episode.name} ({episode_info[episode.name].frame_count} frames)"
            for episode in episodes
        )
    )
    if args.validate_only:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = PowerPaintV21(args.model)
    results = [
        process_episode(model, episode, episode_info[episode.name], args)
        for episode in episodes
    ]
    manifest = {
        "method": "PowerPaint v2-1 shape-guided arm appearance replacement",
        "powerpaint_checkout": str(POWERPAINT_ROOT),
        "powerpaint_commit": "5b4c3d52291709fcec2a1870d987da693fd3549c",
        "model": str(args.model),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "fitting_degree": args.fitting_degree,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "seed_mode": args.seed_mode,
        "short_side": args.short_side,
        "mask_dilate": args.mask_dilate,
        "mask_feather": args.mask_feather,
        "episodes": results,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
