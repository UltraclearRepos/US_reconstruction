"""Connect, clean and smooth reconstructed labels in physical 3D space."""

import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as ndi
from scipy.ndimage import find_objects, generate_binary_structure


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


def smooth_mask(mask, sigma_voxels, threshold):
    """Smooth a GPU mask, retaining the input if thresholding erases it."""
    if np.any(sigma_voxels > 0):
        scores = ndi.gaussian_filter(
            mask.astype(np.float32), sigma=sigma_voxels, mode="constant", cval=0.0,
        )
        smoothed = scores >= threshold
    else:
        smoothed = mask.copy()
    fallback = not bool(smoothed.any().item())
    return (mask.copy() if fallback else smoothed), fallback


def smooth_nodules(mask, sigma_voxels, threshold):
    """Smooth each 26-connected nodule independently, without size filtering."""
    components, count = ndi.label(mask, structure=CONNECTIVITY)
    bounds = find_objects(cp.asnumpy(components, blocking=True))
    margin = np.ceil(4 * sigma_voxels).astype(int) + 1
    result = cp.zeros_like(mask)
    fallback_count = 0
    for component_id, box in enumerate(bounds, start=1):
        region = tuple(
            slice(max(0, axis.start - int(pad)), min(size, axis.stop + int(pad)))
            for axis, pad, size in zip(box, margin, mask.shape)
        )
        nodule = components[region] == component_id
        smoothed, fallback = smooth_mask(nodule, sigma_voxels, threshold)
        result[region] |= smoothed
        fallback_count += int(fallback)
    return result, int(count), fallback_count


def postprocess_segmentation_labels(
    labels,
    spacing_zyx,
    closing_radius_mm=0.6,
    smoothing_sigma_mm=0.4,
    smoothing_threshold=0.5,
    nodule_smoothing_sigma_mm=0.4,
):
    """Clean thyroid (1) and independently smooth nodules (2) on CUDA.

    Thyroid uses closing, largest-component selection, smoothing and another
    largest-component selection. Nodules use only per-component smoothing.
    Nodules take priority where output masks overlap. Input is not mutated.
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
    if not np.isfinite(nodule_smoothing_sigma_mm) or nodule_smoothing_sigma_mm < 0:
        raise ValueError("nodule_smoothing_sigma_mm must be finite and nonnegative")
    if not np.isfinite(smoothing_threshold) or not 0 < smoothing_threshold < 1:
        raise ValueError("smoothing_threshold must be between zero and one")

    class_ids = np.unique(labels)
    class_ids = class_ids[class_ids != 0]
    if np.any(~np.isin(class_ids, [1, 2])):
        raise ValueError("Expected background 0, thyroid 1 and nodules 2")
    processed_labels = np.zeros_like(labels)
    if class_ids.size == 0:
        return processed_labels, []
    closing_footprint = physical_ball(closing_radius_mm, spacing)
    closing_padding = np.asarray(closing_footprint.shape) // 2
    device_id = cp.cuda.runtime.getDevice()
    print(f"[MASK POSTPROCESS] device=cuda:{device_id}", flush=True)
    closing_footprint = cp.asarray(closing_footprint)
    report = []
    # Reserve nodules first, then write thyroid only outside those voxels.
    for class_id in sorted(class_ids, reverse=True):
        sigma_mm = nodule_smoothing_sigma_mm if class_id == 2 else smoothing_sigma_mm
        sigma_voxels = sigma_mm / spacing
        padding = closing_padding if class_id == 1 else np.zeros(3, dtype=int)
        crop_margin = padding + np.ceil(4 * sigma_voxels).astype(int) + 1
        mask = labels == class_id
        region = mask_bounds(mask, crop_margin)
        mask = mask[region].copy()
        original_voxels = int(mask.sum())
        mask = cp.asarray(mask)

        if class_id == 2:
            mask, components_before, fallback_count = smooth_nodules(
                mask, sigma_voxels, smoothing_threshold,
            )
            components_after_closing = components_before
        else:
            components_before = int(ndi.label(mask, structure=CONNECTIVITY)[1])
            if closing_radius_mm > 0:
                original_region = tuple(
                    slice(int(n), int(n) + size)
                    for n, size in zip(closing_padding, mask.shape)
                )
                mask = cp.pad(mask, [(int(n), int(n)) for n in closing_padding])
                mask = ndi.binary_closing(mask, structure=closing_footprint)
                mask = mask[original_region]
            mask, components_after_closing = largest_component(mask)
            mask, fallback = smooth_mask(mask, sigma_voxels, smoothing_threshold)
            fallback_count = int(fallback)
            mask, _ = largest_component(mask)
            mask &= cp.asarray(processed_labels[region] != 2)

        components_after = int(ndi.label(mask, structure=CONNECTIVITY)[1])
        # Synchronize and download the final mask before assembly and saving.
        mask = cp.asnumpy(mask, blocking=True)
        processed_labels[region][mask] = class_id
        report.append({
            "class_id": int(class_id),
            "backend": "cupy",
            "device": f"cuda:{device_id}",
            "smoothing_sigma_mm": float(sigma_mm),
            "voxels_before": original_voxels,
            "components_before": components_before,
            "components_after_closing": int(components_after_closing),
            "smoothing_fallback": bool(fallback_count),
            "smoothing_fallback_count": fallback_count,
            "voxels_after": int(mask.sum()),
            "components_after": components_after,
        })

    return processed_labels, sorted(report, key=lambda item: item["class_id"])
