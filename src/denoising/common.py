"""Shared denoising helpers."""

import numpy as np


def normalize_to_float01(img, eps=1e-8):
    img = img.astype(np.float32)
    mn = float(np.nanmin(img))
    mx = float(np.nanmax(img))

    if mx - mn < eps:
        return np.zeros_like(img, dtype=np.float32), mn, mx

    out = (img - mn) / (mx - mn)
    return out.astype(np.float32), mn, mx


def restore_from_float01(img01, mn, mx):
    return (img01.astype(np.float32) * (mx - mn) + mn).astype(np.float32)


def apply_slice_wise(volume_thw, fn):
    out = np.empty_like(volume_thw, dtype=np.float32)

    for i in range(volume_thw.shape[0]):
        out[i] = fn(volume_thw[i].astype(np.float32))

        if (i + 1) % 50 == 0 or i == volume_thw.shape[0] - 1:
            print(f"[INFO] denoised slices: {i + 1}/{volume_thw.shape[0]}")

    return out
