"""RDPAD denoising."""

import numpy as np
from scipy.ndimage import convolve


EPS = 1e-8


def _shift_reflect(img, dy, dx):
    padded = np.pad(img, ((2, 2), (2, 2)), mode="reflect")
    h, w = img.shape

    y0 = 2 + dy
    x0 = 2 + dx

    return padded[y0:y0 + h, x0:x0 + w]


def _rdpad_q2_modified_13_point(img):
    """
    Modified 5x5 / 13-point estimator inspired by RDPAD paper.

    Points:
      center,
      distance 1 cross + diagonals,
      distance 2 cross.

    q^2 = (1/12 * sum pairwise differences^2) / (sum values)^2

    This follows the idea of Eq. 10 from the paper, but implemented in a
    numerically stable way.
    """
    img = np.maximum(img.astype(np.float32), EPS)

    offsets = [
        (0, 0),
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
    ]

    vals = np.stack([_shift_reflect(img, dy, dx) for dy, dx in offsets], axis=0)

    s = np.sum(vals, axis=0)
    sum_sq_diff = np.zeros_like(img, dtype=np.float32)

    n = vals.shape[0]
    for m in range(n - 1):
        diff = vals[m + 1:] - vals[m:m + 1]
        sum_sq_diff += np.sum(diff * diff, axis=0)

    q2 = (sum_sq_diff / 12.0) / (s * s + EPS)

    return np.maximum(q2, 0.0).astype(np.float32)


def _estimate_q0_squared(q2, mode="median", percentile=25.0):
    """
    q0^2 estimate for homogeneous background/speckle level.

    The paper uses coefficient of variation to distinguish homogeneous
    areas from edges. In practical automated use, robust image-level
    estimates are safer than a hand-selected ROI.
    """
    q2_flat = q2[np.isfinite(q2)]

    if q2_flat.size == 0:
        return np.float32(1e-4)

    if mode == "mean":
        val = float(np.mean(q2_flat))
    elif mode == "percentile":
        val = float(np.percentile(q2_flat, percentile))
    else:
        val = float(np.median(q2_flat))

    return np.float32(max(val, 1e-8))


def _rdpad_diffusion_coefficient(q2, q0_2):
    """
    Robust DPAD coefficient based on the paper's RDPAD idea.

    R = (q^2 - q0^2) / (q0^2 * (1 + q^2))

    c = 0.5 * (1 - R^2)^2 if R <= 1, else 0

    We additionally clamp R >= 0 for stable diffusion in very homogeneous
    regions, so c does not increase above 0.5.
    """
    q0_2 = np.float32(max(float(q0_2), 1e-8))

    r = (q2 - q0_2) / (q0_2 * (1.0 + q2) + EPS)
    r = np.maximum(r, 0.0)

    c = np.zeros_like(q2, dtype=np.float32)

    mask = r <= 1.0
    c[mask] = 0.5 * np.square(1.0 - np.square(r[mask]))

    return np.clip(c, 0.0, 0.5).astype(np.float32)


def _divergence_4n(img, c):
    """
    4-neighbour anisotropic diffusion divergence.
    """
    north = np.roll(img, 1, axis=0) - img
    south = np.roll(img, -1, axis=0) - img
    west = np.roll(img, 1, axis=1) - img
    east = np.roll(img, -1, axis=1) - img

    c_n = np.roll(c, 1, axis=0)
    c_s = np.roll(c, -1, axis=0)
    c_w = np.roll(c, 1, axis=1)
    c_e = np.roll(c, -1, axis=1)

    div = (
        c_n * north
        + c_s * south
        + c_w * west
        + c_e * east
    )

    div[0, :] = 0.0
    div[-1, :] = 0.0
    div[:, 0] = 0.0
    div[:, -1] = 0.0

    return div.astype(np.float32)


def denoise_rdpad_2d(
    image,
    iterations=50,
    timestep=0.15,
    q0_mode="median",
    q0_percentile=25.0,
    preserve_range=True,
):
    """
    RDPAD-like speckle reduction for one 2D ultrasound frame.

    Parameters
    ----------
    image:
        2D image.
    iterations:
        Number of diffusion iterations.
    timestep:
        Explicit diffusion step. Keep <= 0.25 for stability.
    q0_mode:
        "median", "mean", or "percentile".
    q0_percentile:
        Used only when q0_mode == "percentile".
    preserve_range:
        Clip output to original min/max.
    """
    img = image.astype(np.float32)

    original_min = float(np.nanmin(img))
    original_max = float(np.nanmax(img))

    # RDPAD assumes positive image values because q is multiplicative.
    work = img - original_min
    work = work + 1e-3

    work_max = float(np.max(work))
    if work_max > 0:
        work = work / work_max

    timestep = float(timestep)
    if timestep <= 0.0 or timestep > 0.25:
        raise ValueError("timestep powinien być w zakresie (0, 0.25].")

    for _ in range(int(iterations)):
        q2 = _rdpad_q2_modified_13_point(work)
        q0_2 = _estimate_q0_squared(
            q2,
            mode=q0_mode,
            percentile=q0_percentile,
        )

        c = _rdpad_diffusion_coefficient(q2, q0_2)
        div = _divergence_4n(work, c)

        work = work + timestep * div
        work = np.maximum(work, 0.0)

    if work_max > 0:
        out = work * work_max
    else:
        out = work

    out = out - 1e-3 + original_min

    if preserve_range:
        out = np.clip(out, original_min, original_max)

    return out.astype(np.float32)
