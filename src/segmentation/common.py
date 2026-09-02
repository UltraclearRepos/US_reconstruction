"""Shared supervised segmentation helpers."""

from pathlib import Path
import json

import numpy as np
from PIL import Image

def load_model_meta(meta_path):
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_model_input_size(meta, fallback_size=(256, 256)):
    size = (meta.get("input", {}).get("image_size")
            or meta.get("input", {}).get("example_trace_size"))
    return fallback_size if size is None else (int(size[0]), int(size[1]))


def normalize_frame_for_unet(frame, mode="auto"):
    frame = frame.astype(np.float32)
    if mode == "minmax":
        minimum, maximum = float(np.nanmin(frame)), float(np.nanmax(frame))
        if maximum - minimum < 1e-8:
            return np.zeros_like(frame)
        return (frame - minimum) / (maximum - minimum)
    if mode == "divide_255":
        return np.clip(frame, 0.0, 255.0) / 255.0
    if mode == "none":
        return np.clip(frame, 0.0, 1.0)
    if float(np.nanmax(frame)) > 1.5:
        return np.clip(frame, 0.0, 255.0) / 255.0
    return np.clip(frame, 0.0, 1.0)


def resize_frame_float(frame01, size_hw):
    h, w = size_hw
    image = Image.fromarray(
        np.clip(frame01 * 255.0, 0, 255).astype(np.uint8), mode="L"
    )
    return np.asarray(
        image.resize((w, h), resample=Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def resize_mask_nearest(mask, size_hw):
    h, w = size_hw
    image = Image.fromarray(mask.astype(np.uint8), mode="L")
    return np.asarray(image.resize((w, h), resample=Image.NEAREST), dtype=np.uint8)
