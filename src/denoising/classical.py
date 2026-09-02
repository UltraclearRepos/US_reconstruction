"""Classical denoising algorithms."""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

try:
    from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma, denoise_tv_chambolle
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


def denoise_gaussian_2d(image, sigma=1.0):
    return gaussian_filter(
        image.astype(np.float32),
        sigma=float(sigma),
    ).astype(np.float32)


def denoise_median_2d(image, size=3):
    return median_filter(
        image.astype(np.float32),
        size=int(size),
    ).astype(np.float32)


def denoise_bilateral_2d(
    image,
    sigma_color=0.05,
    sigma_spatial=3.0,
):
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image jest wymagany dla bilateral filter.")

    img = image.astype(np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))

    if mx - mn < 1e-8:
        return img

    img01 = (img - mn) / (mx - mn)

    out01 = denoise_bilateral(
        img01,
        sigma_color=float(sigma_color),
        sigma_spatial=float(sigma_spatial),
        channel_axis=None,
    )

    return (out01 * (mx - mn) + mn).astype(np.float32)


def denoise_tv_2d(
    image,
    weight=0.08,
    eps=0.0002,
    max_num_iter=200,
):
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image jest wymagany dla total variation denoising.")

    img = image.astype(np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))

    if mx - mn < 1e-8:
        return img

    img01 = (img - mn) / (mx - mn)

    out01 = denoise_tv_chambolle(
        img01,
        weight=float(weight),
        eps=float(eps),
        max_num_iter=int(max_num_iter),
        channel_axis=None,
    )

    return (out01 * (mx - mn) + mn).astype(np.float32)
