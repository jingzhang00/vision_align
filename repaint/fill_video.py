"""Fill white view holes in every video under a directory with LaMa or FLUX.

Example (run from the repository root):
    .venv/bin/python -m repaint.fill_video \
        --input_dir /path/to/videos \
        --output_dir /path/to/filled_videos

The hole color is inferred from bright, low-saturation, flat regions in the
first frames and matched in Lab color space. Pass ``--no-include_interior`` to
restrict filling to holes touching a frame edge.
"""

import argparse
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


def process_video(input_path, output_path, args, fill_frame, torch):
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

    probe = cv2.VideoCapture(str(input_path))
    hole_colors = []
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
                mask = white_hole_mask(
                    frame, hole_lab, args.color_tolerance, args.min_area,
                    args.dilate, args.include_interior)
                if mask.any():
                    torch.manual_seed(args.seed)
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
    print("self-check passed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--inpaint_model", choices=("lama", "flux"),
                        default="lama")
    parser.add_argument("--flux_model", "--model", dest="flux_model",
                        default=str(DEFAULT_FLUX_MODEL),
                        help="FLUX directory or HuggingFace id")
    parser.add_argument("--lama_config", default=str(DEFAULT_LAMA_CONFIG))
    parser.add_argument("--lama_ckpt", default=str(DEFAULT_LAMA_CKPT))
    parser.add_argument(
        "--prompt", default="continue the surrounding scene naturally, "
        "photorealistic, seamless background")
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
    parser.add_argument("--seed", type=int, default=0)
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
    return args


def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return

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
        from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
        model = build_lama_model(args.lama_config, args.lama_ckpt, device=device)
        fill_frame = partial(
            inpaint_img_with_builded_lama, model, device=device)
    else:
        if device != "cuda":
            raise SystemExit("FLUX.1-Fill requires a CUDA GPU (about 20 GB VRAM).")
        from stable_diffusion_inpaint import fill_img_with_sd
        fill_frame = partial(
            fill_img_with_sd, text_prompt=args.prompt, device=device,
            model_id=args.flux_model)

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
            input_path, output_path, args, fill_frame, torch)
        print(f"  done: {filled}/{frames} frames contained white holes")


if __name__ == "__main__":
    main()
