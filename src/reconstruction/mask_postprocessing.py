"""Connect, clean and smooth reconstructed labels in physical 3D space."""

import numpy as np
from scipy import ndimage as ndi


CONNECTIVITY = ndi.generate_binary_structure(3, 3)


def largest_component(mask):
    """Keep the largest 26-connected component; ties use scan order."""
    components, count = ndi.label(mask, structure=CONNECTIVITY)
    if count <= 1:
        return mask.copy(), count
    sizes = np.bincount(components.ravel())
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
    """Return integer labels and a per-class processing report.

    Each nonzero class undergoes closing, largest-component selection and
    Gaussian smoothing followed by thresholding. A final component check
    prevents smoothing or class conflicts from leaving detached fragments.
    Classes compete by their smoothed scores only in original background;
    they cannot grow into another class's original voxels. No input is mutated.
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

    sigma = smoothing_sigma_mm / spacing
    footprint = physical_ball(closing_radius_mm, spacing)
    closing_pad = np.asarray(footprint.shape) // 2
    margin = closing_pad + np.ceil(4 * sigma).astype(int) + 1
    output = np.zeros_like(labels)
    best_scores = np.zeros(labels.shape, dtype=np.float32)
    report = []

    for class_id in np.unique(labels):
        if class_id == 0:
            continue
        full_mask = labels == class_id
        region = mask_bounds(full_mask, margin)
        mask = full_mask[region].copy()
        del full_mask
        original_voxels = int(mask.sum())
        _, components_before = ndi.label(mask, structure=CONNECTIVITY)
        available = (labels[region] == 0) | (labels[region] == class_id)

        if closing_radius_mm > 0:
            # Pad beyond the volume edge so closing does not erode edge objects.
            pad = [(int(n), int(n)) for n in closing_pad]
            padded = np.pad(mask, pad, mode="constant")
            closed = ndi.binary_closing(padded, structure=footprint)
            unpad = tuple(slice(int(n), int(n) + size)
                          for n, size in zip(closing_pad, mask.shape))
            mask = closed[unpad] & available

        mask, components_after_closing = largest_component(mask)
        if smoothing_sigma_mm > 0:
            scores = ndi.gaussian_filter(mask.astype(np.float32), sigma=sigma,
                                         mode="constant", cval=0.0)
        else:
            scores = mask.astype(np.float32)
        candidate = (scores >= smoothing_threshold) & available
        smoothing_fallback = not candidate.any()
        if smoothing_fallback:
            # A tiny class can vanish under Gaussian thresholding. Retain it.
            candidate = mask.copy()
            scores = mask.astype(np.float32)
        candidate, _ = largest_component(candidate)
        output_region = output[region]
        score_region = best_scores[region]
        replace = candidate & (scores > score_region)
        output_region[replace] = class_id
        score_region[replace] = scores[replace]
        report.append({
            "class_id": int(class_id),
            "voxels_before": original_voxels,
            "components_before": int(components_before),
            "components_after_closing": int(components_after_closing),
            "smoothing_fallback": bool(smoothing_fallback),
        })

    # Resolving overlapping class expansions can split a previously connected mask.
    for item in report:
        class_id = item["class_id"]
        full_mask = output == class_id
        if full_mask.any():
            region = mask_bounds(full_mask, np.zeros(3, dtype=int))
            mask = full_mask[region]
            kept, _ = largest_component(mask)
            output_region = output[region]
            output_region[mask & ~kept] = 0
            item["voxels_after"] = int(kept.sum())
            item["components_after"] = 1
        else:
            item["voxels_after"] = 0
            item["components_after"] = 0
    return output, report
