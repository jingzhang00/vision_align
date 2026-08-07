"""Project-specific adapters for the unmodified Inpaint-Anything checkout."""

from functools import partial
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image


def build_lama_model(config_p, ckpt_p, device="cuda"):
    import lama_inpaint

    # ponytail: this scoped patch assumes single-threaded model startup; copy
    # the upstream loader if concurrent startup is ever needed.
    load_checkpoint = partial(lama_inpaint.load_checkpoint, map_location="cpu")
    with patch.object(lama_inpaint, "load_checkpoint", load_checkpoint):
        return lama_inpaint.build_lama_model(config_p, ckpt_p, device=device)


def _build_pipe(model_id, device):
    from diffusers.utils import logging as diffusers_logging
    from huggingface_hub import utils as hub_utils
    from transformers.utils import logging as transformers_logging

    diffusers_logging.disable_progress_bar()
    hub_utils.disable_progress_bars()
    transformers_logging.disable_progress_bar()
    import stable_diffusion_inpaint as upstream

    model_id = upstream.resolve_model_id(model_id)
    key = (model_id, str(device))
    checkpoint = Path(model_id) / "512-inpainting-ema.safetensors"
    if key not in upstream._PIPE_CACHE and checkpoint.is_file():
        from diffusers import StableDiffusionInpaintPipeline

        upstream._PIPE_CACHE[key] = (
            StableDiffusionInpaintPipeline.from_single_file(
                str(checkpoint.resolve()),
                config=str(checkpoint.parent.resolve()),
                local_files_only=True,
                torch_dtype=upstream.torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)
        )
    pipe = upstream._PIPE_CACHE.get(key)
    if pipe is None:
        pipe = upstream.build_sd_inpaint_pipe(model_id, device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, upstream.model_kind(model_id)


def _run_pipe(pipe, kind, prompt, image, mask, steps, guidance_scale,
              negative_prompt, generator):
    from sam3_utils import no_autocast

    with no_autocast():
        if kind == "flux":
            return pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                height=image.height,
                width=image.width,
                guidance_scale=30.0,
                num_inference_steps=steps or 50,
                max_sequence_length=512,
                generator=generator,
            ).images[0]
        kwargs = {"height": image.height, "width": image.width}
        if steps is not None:
            kwargs["num_inference_steps"] = steps
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        kwargs["generator"] = generator
        return pipe(prompt=prompt, image=image, mask_image=mask,
                    **kwargs).images[0]


def crop_for_filling(image, mask, crop_size):
    """Return a square model crop and enough geometry to paste it back."""
    x, y, width, height = cv2.boundingRect(mask)
    side = max(crop_size, width, height)
    center_x, center_y = x + width // 2, y + height // 2
    left, top = center_x - side // 2, center_y - side // 2
    right, bottom = left + side, top + side
    image_height, image_width = image.shape[:2]
    pad = (max(0, -left), max(0, -top),
           max(0, right - image_width), max(0, bottom - image_height))
    padded_image = cv2.copyMakeBorder(
        image, pad[1], pad[3], pad[0], pad[2], cv2.BORDER_CONSTANT)
    padded_mask = cv2.copyMakeBorder(
        mask, pad[1], pad[3], pad[0], pad[2], cv2.BORDER_CONSTANT)
    left += pad[0]
    top += pad[1]
    image_crop = padded_image[top:top + side, left:left + side]
    mask_crop = padded_mask[top:top + side, left:left + side]
    if side != crop_size:
        image_crop = cv2.resize(image_crop, (crop_size, crop_size))
        mask_crop = cv2.resize(
            mask_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    return image_crop, mask_crop, (left, top, side, pad)


def paste_filling(image, mask, filled_crop, geometry):
    left, top, side, (pad_left, pad_top, pad_right, pad_bottom) = geometry
    if filled_crop.shape[:2] != (side, side):
        filled_crop = cv2.resize(filled_crop, (side, side))
    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
    padded_mask = cv2.copyMakeBorder(
        mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
    region = padded_image[top:top + side, left:left + side]
    crop_mask = padded_mask[top:top + side, left:left + side] != 0
    region[crop_mask] = filled_crop[crop_mask]
    height, width = image.shape[:2]
    return padded_image[pad_top:pad_top + height, pad_left:pad_left + width]


def fill_img_with_sd(img, mask, text_prompt, device="cuda", model_id=None,
                     steps=None, guidance_scale=None, negative_prompt=None,
                     full_frame=False, seed=42):
    import torch

    pipe, kind = _build_pipe(model_id, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    if full_frame:
        height, width = img.shape[:2]
        size = (width - width % 8, height - height % 8)
        image = Image.fromarray(cv2.resize(img, size))
        mask_image = Image.fromarray(cv2.resize(mask, size))
        result = _run_pipe(pipe, kind, text_prompt, image, mask_image, steps,
                           guidance_scale, negative_prompt, generator)
        if result.size != (width, height):
            return cv2.resize(np.asarray(result), (width, height))
        return np.asarray(result)

    crop_size = 512 if kind == "sd" else 1024
    image_crop, mask_crop, geometry = crop_for_filling(img, mask, crop_size)
    result = _run_pipe(
        pipe, kind, text_prompt, Image.fromarray(image_crop),
        Image.fromarray(mask_crop), steps, guidance_scale, negative_prompt,
        generator)
    return paste_filling(img, mask, np.asarray(result), geometry)
