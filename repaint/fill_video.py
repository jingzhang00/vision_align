"""Fill white view holes in every video with LaMa or diffusion inpainting.

Example (run from the repository root):
    .venv/bin/python -m repaint.fill_video \
        --input_dir /path/to/videos \
        --output_dir /path/to/filled_videos

The hole color is inferred from bright, low-saturation, flat regions in the
first frames and matched in Lab color space. Pass ``--no-include_interior`` to
restrict filling to holes touching a frame edge.
"""

import argparse
import json
import math
import subprocess
from functools import partial
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from tqdm import tqdm


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
INPAINT_DIR = Path(__file__).resolve().parents[1] / "Inpaint-Anything"
DEFAULT_SD_MODEL = Path(__file__).resolve().parents[1] / "stable-diffusion-2-inpainting"
DEFAULT_FLUX_MODEL = INPAINT_DIR / "pretrained_models/FLUX.1-Fill-dev"
DEFAULT_LAMA_CONFIG = INPAINT_DIR / "lama/configs/prediction/default.yaml"
DEFAULT_LAMA_CKPT = INPAINT_DIR / "pretrained_models/big-lama"


def infer_hole_lab(frame, min_value=200, max_saturation=45, min_area=256):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    candidates = ((hsv[..., 2] >= min_value)
                  & (hsv[..., 1] <= max_saturation)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    edge_labels = set(np.concatenate(
        (labels[0], labels[-1], labels[:, 0], labels[:, -1]))) - {0}
    choices = [label for label in edge_labels
               if stats[label, cv2.CC_STAT_AREA] >= min_area]
    if not choices:
        choices = [label for label in range(1, count)
                   if stats[label, cv2.CC_STAT_AREA] >= min_area]
    if not choices:
        return None

    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    best_score, best_color = -1, None
    for label in choices:
        pixels = lab[labels == label]
        color = np.median(pixels, axis=0)
        spread = np.percentile(np.linalg.norm(pixels - color, axis=1), 90)
        area = stats[label, cv2.CC_STAT_AREA]
        score = area / (spread + 1) ** 2
        if score > best_score:
            best_score, best_color = score, color
    return best_color


def white_hole_mask(frame, hole_lab, color_tolerance=8, min_area=256,
                    dilate=5, include_interior=True):
    if hole_lab is None:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    white = (np.linalg.norm(lab - hole_lab, axis=2)
             <= color_tolerance).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    height, width = white.shape
    mask = np.zeros_like(white)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_edge = x == 0 or y == 0 or x + w == width or y + h == height
        if area >= min_area and (include_interior or touches_edge):
            mask[labels == label] = 255
    if dilate and mask.any():
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    return mask


def find_videos(input_dir, output_dir):
    output_dir = output_dir.resolve()
    output_is_nested = input_dir.resolve() in output_dir.parents
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        and (not output_is_nested or output_dir not in path.resolve().parents)
    )


def letterbox_hole_mask(metadata, width, height):
    box = metadata.get("letterbox", {})
    if box.get("output_size") != [width, height]:
        return None
    offsets = box.get("offsets_px", {})
    try:
        left, top, right, bottom = (
            int(offsets[key]) for key in ("left", "top", "right", "bottom"))
    except (KeyError, TypeError, ValueError):
        return None
    if min(left, top, right, bottom) < 0 or left + right >= width or top + bottom >= height:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, :left] = mask[:, width - right:] = 255
    mask[:top] = mask[height - bottom:] = 255
    return mask


def load_render_hole_masks(input_path, width, height, frame_count):
    mask_path = input_path.with_suffix(".mask.npz")
    if not mask_path.is_file():
        return None
    with np.load(mask_path) as archive:
        masks = archive["masks"]
        shape = tuple(int(value) for value in archive["shape"])
    if shape != (frame_count, height, width) or masks.shape[0] != frame_count:
        raise ValueError(
            f"Mask metadata {shape} does not match video "
            f"{(frame_count, height, width)}: {mask_path}"
        )
    return masks


def process_video(input_path, output_path, args, fill_frame):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = args.fps
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        capture.release()
        raise ValueError(
            f"Video input must have a positive even size, got {width}x{height}: "
            f"{input_path}")

    render_masks = load_render_hole_masks(
        input_path, width, height, frame_count)
    metadata_mask = None
    sidecar = input_path.with_suffix(".json")
    if render_masks is None and sidecar.is_file():
        metadata_mask = letterbox_hole_mask(
            json.loads(sidecar.read_text(encoding="utf-8")), width, height)
        if metadata_mask is not None and not metadata_mask.any():
            metadata_mask = None

    hole_colors = []
    if render_masks is None and metadata_mask is None:
        probe = cv2.VideoCapture(str(input_path))
        for _ in range(min(frame_count, 5) if frame_count else 5):
            ok, bgr = probe.read()
            if not ok:
                break
            color = infer_hole_lab(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.min_value,
                args.max_saturation, args.min_area)
            if color is not None:
                hole_colors.append(color)
        probe.release()
    hole_lab = np.median(hole_colors, axis=0) if hole_colors else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
        "-r", f"{fps:.12g}", "-i", "pipe:0", "-c:v", "libx264", "-crf", "18",
        "-preset", "medium", "-pix_fmt", "yuv420p", str(output_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    filled_frames = processed_frames = 0
    try:
        with tqdm(total=frame_count or None, desc=input_path.name,
                  unit="frame") as progress:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if render_masks is not None:
                    mask = np.unpackbits(
                        render_masks[processed_frames],
                        count=height * width,
                    ).reshape(height, width).astype(np.uint8) * 255
                    if args.dilate:
                        mask = cv2.dilate(
                            mask, np.ones((args.dilate, args.dilate), np.uint8))
                else:
                    mask = metadata_mask if metadata_mask is not None else white_hole_mask(
                        frame, hole_lab, args.color_tolerance, args.min_area,
                        args.dilate, args.include_interior)
                if mask.any():
                    frame = fill_frame(frame, mask)
                    filled_frames += 1
                encoder.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                processed_frames += 1
                progress.update()
        capture.release()
        encoder.stdin.close()
        error = encoder.stderr.read().decode("utf-8", errors="replace")
        if encoder.wait() != 0:
            raise RuntimeError(f"FFmpeg failed for {input_path}:\n{error}")
        if processed_frames == 0:
            raise RuntimeError(f"Video contains no readable frames: {input_path}")
    except Exception:
        capture.release()
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
        encoder.terminate()
        encoder.wait()
        output_path.unlink(missing_ok=True)
        raise
    return processed_frames, filled_frames


def self_check():
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    frame[:2, :3] = (230, 221, 206)
    frame[6:8, 6:8] = (230, 221, 206)
    hole_lab = infer_hole_lab(frame, min_area=4)
    border = white_hole_mask(
        frame, hole_lab, min_area=4, dilate=0, include_interior=False)
    all_holes = white_hole_mask(
        frame, hole_lab, min_area=4, dilate=0)
    assert border[:2, :3].all() and not border[6:8, 6:8].any()
    assert all_holes[:2, :3].all() and all_holes[6:8, 6:8].all()

    from repaint.inpaint_adapter import crop_for_filling, paste_filling
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    mask[1002:1080, 244:1482] = 255
    cropped_image, cropped_mask, geometry = crop_for_filling(image, mask, 512)
    assert cropped_mask.shape == (512, 512) and cropped_mask.any()
    filled = paste_filling(
        image, mask, np.full_like(cropped_image, 255), geometry)
    assert filled[mask != 0].all() and not filled[mask == 0].any()

    metadata = {"letterbox": {
        "output_size": [12, 8],
        "offsets_px": {"left": 2, "top": 1, "right": 3, "bottom": 0},
    }}
    mask = letterbox_hole_mask(metadata, 12, 8)
    assert mask[:, :2].all() and mask[:, -3:].all() and mask[:1].all()
    assert not mask[2:-1, 3:-4].any()

    packed = np.packbits(np.eye(4, dtype=bool).reshape(-1))[None]
    unpacked = np.unpackbits(packed[0], count=16).reshape(4, 4)
    assert np.array_equal(unpacked, np.eye(4, dtype=np.uint8))
    print("self-check passed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--inpaint_model", choices=("lama", "sd", "flux"),
                        default="lama")
    parser.add_argument("--model", "--sd_model", "--flux_model", dest="model",
                        help="diffusion model directory or HuggingFace id")
    parser.add_argument("--lama_config", default=str(DEFAULT_LAMA_CONFIG))
    parser.add_argument("--lama_ckpt", default=str(DEFAULT_LAMA_CKPT))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt",
                        default="blurry, low quality, distorted")
    parser.add_argument("--min_value", "--white_threshold", type=int,
                        default=200, help="minimum HSV brightness for a seed")
    parser.add_argument("--max_saturation", type=int, default=45,
                        help="maximum HSV saturation for a seed")
    parser.add_argument("--color_tolerance", type=float, default=8,
                        help="Lab distance from the inferred hole color")
    parser.add_argument("--min_area", type=int, default=256)
    parser.add_argument("--dilate", type=int, default=5)
    parser.add_argument("--include_interior", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--fps", type=float, default=25,
                        help="fallback when video metadata has no FPS")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        return args
    if args.input_dir is None or args.output_dir is None:
        parser.error("--input_dir and --output_dir are required")
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("input and output directories must be different")
    if not 0 <= args.min_value <= 255 or not 0 <= args.max_saturation <= 255:
        parser.error("--min_value and --max_saturation must be between 0 and 255")
    if args.color_tolerance < 0:
        parser.error("--color_tolerance cannot be negative")
    if args.min_area < 1 or args.dilate < 0 or args.fps <= 0:
        parser.error("--min_area and --fps must be positive; --dilate cannot be negative")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.guidance_scale is not None and args.guidance_scale < 0:
        parser.error("--guidance_scale cannot be negative")
    return args


def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.inpaint_model == "sd":
        args.steps = args.steps or 20
        if args.guidance_scale is None:
            args.guidance_scale = 7.5

    try:
        import sys
        sys.path.insert(0, str(INPAINT_DIR))
        import torch
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"Missing dependency '{error.name}'. Run: "
            "python -m pip install -r requirements.txt") from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.inpaint_model == "lama":
        from lama_inpaint import inpaint_img_with_builded_lama
        from repaint.inpaint_adapter import build_lama_model
        model = build_lama_model(args.lama_config, args.lama_ckpt, device=device)
        fill_frame = partial(
            inpaint_img_with_builded_lama, model, device=device)
    else:
        if device != "cuda":
            raise SystemExit("Diffusion inpainting requires a CUDA GPU.")
        from repaint.inpaint_adapter import fill_img_with_sd
        model = args.model or str(
            DEFAULT_SD_MODEL if args.inpaint_model == "sd" else DEFAULT_FLUX_MODEL)
        fill_frame = partial(
            fill_img_with_sd, text_prompt=args.prompt, device=device,
            model_id=model, steps=args.steps, seed=args.seed,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            full_frame=args.inpaint_model == "sd")

    videos = find_videos(args.input_dir, args.output_dir)
    if not videos:
        raise SystemExit(f"No supported videos found under {args.input_dir}")

    for index, input_path in enumerate(videos, 1):
        relative = input_path.relative_to(args.input_dir).with_suffix(".mp4")
        output_path = args.output_dir / relative
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(videos)}] skip existing: {output_path}")
            continue
        print(f"[{index}/{len(videos)}] {input_path} -> {output_path}")
        frames, filled = process_video(
            input_path, output_path, args, fill_frame)
        print(f"  done: {filled}/{frames} frames contained holes")


if __name__ == "__main__":
    main()
