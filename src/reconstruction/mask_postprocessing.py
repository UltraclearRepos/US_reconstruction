"""Connect, clean and smooth reconstructed labels in physical 3D space."""

import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as ndi
from scipy.ndimage import generate_binary_structure


CONNECTIVITY = generate_binary_structure(3, 3)


def largest_component(mask):
    """Keep the largest 26-connected component; ties use scan order."""
    components, count = ndi.label(mask, structure=CONNECTIVITY)
    if count <= 1:
        return mask.copy(), count
    sizes = cp.bincount(components.ravel())
    sizes[0] = 0
    return components == sizes.argmax(), count


def physical_ball(radius_mm, spacing_zyx):
    """Build a spherical footprint, including for anisotropic voxels."""
    extent = np.ceil(radius_mm / spacing_zyx).astype(int)
    axes = [np.arange(-n, n + 1) * s for n, s in zip(extent, spacing_zyx)]
    z, y, x = np.meshgrid(*axes, indexing="ij", sparse=True)
    return z * z + y * y + x * x <= radius_mm * radius_mm + 1e-12


def mask_bounds(mask, margin):
    """Find a padded bounding box without allocating foreground coordinates."""
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        bounds.append(slice(max(0, int(occupied[0]) - int(margin[axis])),
                            min(mask.shape[axis], int(occupied[-1]) + int(margin[axis]) + 1)))
    return tuple(bounds)


def postprocess_segmentation_labels(
    labels,
    spacing_zyx,
    closing_radius_mm=0.6,
    smoothing_sigma_mm=0.4,
    smoothing_threshold=0.5,
):
    """Clean one foreground class and return integer labels with a report.

    Background is zero; the single nonzero class keeps its original ID.
    Closing is followed by largest-component selection, Gaussian smoothing
    and thresholding, then another largest-component selection.
    Processing uses CuPy on CUDA; input/output and crop preparation use NumPy.
    Multiclass input is rejected. No input is mutated.
    """
    labels = np.asarray(labels)
    spacing = np.asarray(spacing_zyx, dtype=float)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected a 3D integer label volume")
    if np.any(labels < 0):
        raise ValueError("Labels must be nonnegative; zero is background")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three positive finite values")
    if not np.isfinite(closing_radius_mm) or closing_radius_mm < 0:
        raise ValueError("closing_radius_mm must be finite and nonnegative")
    if not np.isfinite(smoothing_sigma_mm) or smoothing_sigma_mm < 0:
        raise ValueError("smoothing_sigma_mm must be finite and nonnegative")
    if not np.isfinite(smoothing_threshold) or not 0 < smoothing_threshold < 1:
        raise ValueError("smoothing_threshold must be between zero and one")

    class_ids = np.unique(labels)
    class_ids = class_ids[class_ids != 0]
    if class_ids.size > 1:
        raise ValueError("Expected background zero and at most one nonzero class")
    processed_labels = np.zeros_like(labels)
    if class_ids.size == 0:
        return processed_labels, []
    class_id = class_ids[0]

    sigma_voxels = smoothing_sigma_mm / spacing
    closing_footprint = physical_ball(closing_radius_mm, spacing)
    closing_padding = np.asarray(closing_footprint.shape) // 2
    crop_margin = closing_padding + np.ceil(4 * sigma_voxels).astype(int) + 1

    mask = labels == class_id
    region = mask_bounds(mask, crop_margin)
    mask = mask[region].copy()
    original_voxels = int(mask.sum())
    device_id = cp.cuda.runtime.getDevice()
    print(f"[MASK POSTPROCESS] device=cuda:{device_id}", flush=True)
    mask = cp.asarray(mask)
    closing_footprint = cp.asarray(closing_footprint)
    components_before = int(ndi.label(mask, structure=CONNECTIVITY)[1])

    if closing_radius_mm > 0:
        original_region = tuple(
            slice(int(n), int(n) + size)
            for n, size in zip(closing_padding, mask.shape)
        )
        padding = [(int(n), int(n)) for n in closing_padding]
        mask = cp.pad(mask, padding, mode="constant")
        mask = ndi.binary_closing(mask, structure=closing_footprint)
        mask = mask[original_region]

    mask, components_after_closing = largest_component(mask)

    mask_before_smoothing = mask.copy()
    if smoothing_sigma_mm > 0:
        scores = ndi.gaussian_filter(
            mask.astype(np.float32),
            sigma=sigma_voxels,
            mode="constant",
            cval=0.0,
        )
    else:
        scores = mask.astype(np.float32)

    mask = scores >= smoothing_threshold
    smoothing_fallback = not bool(mask.any().item())
    if smoothing_fallback:
        mask = mask_before_smoothing

    mask, _ = largest_component(mask)

    # Wait for GPU work and download the final mask before saving.
    mask = cp.asnumpy(mask, blocking=True)

    processed_labels[region] = np.where(mask, class_id, 0)

    voxels_after = int(mask.sum())
    report = [{
        "class_id": int(class_id),
        "backend": "cupy",
        "device": f"cuda:{device_id}",
        "voxels_before": original_voxels,
        "components_before": int(components_before),
        "components_after_closing": int(components_after_closing),
        "smoothing_fallback": bool(smoothing_fallback),
        "voxels_after": voxels_after,
        "components_after": int(voxels_after > 0),
    }]

    return processed_labels, report
