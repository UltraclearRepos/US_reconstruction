"""Preprocessing used by supervised segmentation inference."""

import numpy as np
from PIL import Image


def prepare_frame(frame, size_hw):
    """Match training preprocessing: grayscale, bilinear resize, then /255."""
    height, width = size_hw
    image = Image.fromarray(
        np.clip(frame, 0, 255).astype(np.uint8),
        mode="L",
    )
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def resize_mask(mask, size_hw):
    height, width = size_hw
    image = Image.fromarray(mask.astype(np.uint16))
    resized = image.resize((width, height), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint16)
