"""Segmentation volume metrics using a freshly computed surface alignment."""

import argparse
from dataclasses import dataclass
from itertools import product
import json

import numpy as np
from scipy.ndimage import affine_transform

from src.registration.align_surfaces import compute_case_alignment, transform_points
from config import (
    ALIGN_OVERLAP, ALIGN_SAMPLES,
    EVALUATION_DIR, EVAL_CLASS_ID, EVAL_VOLUME_CHUNK_DEPTH,
    SEGMENTATION_MODEL_NAME, SEGMENTATION_RECONSTRUCTION_METHOD,
)
from src.utils.logging import timed_stage
from src.utils.paths import ground_truth_labels_path, segmentation_reconstruction_path


@dataclass
class ClassMask:
    mask: np.ndarray
    spacing_zyx: np.ndarray
    start_zyx: np.ndarray
    original_shape_zyx: tuple
    foreground_voxels: int


def crop_mask(mask, spacing_zyx):
    """Retain original voxel indices while storing only the foreground bounding box."""
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    if mask.ndim != 3 or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("A 3D mask and three positive finite ZYX spacings are required.")
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(np.any(mask, axis=tuple(i for i in range(3) if i != axis)))
        if not len(occupied):
            raise ValueError("Requested segmentation class has no foreground voxels.")
        bounds.append((int(occupied[0]), int(occupied[-1]) + 1))
    cropped = mask[tuple(slice(lo, hi) for lo, hi in bounds)].copy()
    return ClassMask(cropped, spacing, np.array([lo for lo, _ in bounds]),
                     tuple(mask.shape), int(np.count_nonzero(cropped)))


def load_class_mask(path, class_id):
    with np.load(path, allow_pickle=False) as data:
        key = "volume" if "volume" in data else "labels"
        labels = data[key]
        if labels.ndim != 3 or not np.isfinite(labels).all():
            raise ValueError(f"Invalid segmentation labels: {path}")
        spacing = data["voxel_size"]
        mask = labels == class_id
        del labels
    return crop_mask(mask, spacing)


def resampling_geometry(gt, pred, transform_xyz):
    """Return GT-grid union bounds and the inverse map into padded source crop indices.

    Mesh extraction uses index*spacing local coordinates, ignoring NPZ origins.
    Therefore this exact same convention must be used for the surface transform.
    """
    transform = np.asarray(transform_xyz, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], (0, 0, 0, 1))):
        raise ValueError("A finite homogeneous 4x4 rigid transform is required.")
    rotation, translation = transform[:3, :3], transform[:3, 3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError("Volume evaluation requires rigid rotation without scale or reflection.")
    pred_lo = pred.start_zyx - 0.5
    pred_hi = pred.start_zyx + np.array(pred.mask.shape) - 0.5
    corners_zyx = np.array(list(product(*zip(pred_lo, pred_hi))))
    corners_xyz = (corners_zyx * pred.spacing_zyx)[:, ::-1]
    moved_index_zyx = transform_points(corners_xyz, transform)[:, ::-1] / gt.spacing_zyx
    # Extend the GT lattice to the union of objects: no clipping of prediction outside GT.
    lower = np.minimum(gt.start_zyx - 1, np.floor(moved_index_zyx.min(axis=0)).astype(int) - 1)
    upper = np.maximum(gt.start_zyx + np.array(gt.mask.shape) + 1,
                       np.ceil(moved_index_zyx.max(axis=0)).astype(int) + 2)
    swap = np.eye(3)[::-1]
    inverse_spacing = np.diag(1.0 / pred.spacing_zyx)
    matrix = inverse_spacing @ swap @ rotation.T @ swap @ np.diag(gt.spacing_zyx)
    offset = (-inverse_spacing @ swap @ rotation.T @ translation
              + matrix @ lower - (pred.start_zyx - 1))
    return lower, upper - lower, matrix, offset


def aligned_overlap_counts(gt, pred, transform_xyz, chunk_depth=EVAL_VOLUME_CHUNK_DEPTH):
    """Resample in slabs; count all GT and aligned prediction foreground, including outliers."""
    if chunk_depth < 1:
        raise ValueError("Chunk depth must be positive.")
    lower, shape, matrix, offset = resampling_geometry(gt, pred, transform_xyz)
    padded_pred = np.pad(pred.mask, 1, mode="constant")
    pred_count, gt_count, intersection = 0, 0, 0
    gt_upper = gt.start_zyx + np.array(gt.mask.shape)
    for start_z in range(0, int(shape[0]), chunk_depth):
        depth = min(chunk_depth, int(shape[0]) - start_z)
        block_shape = (depth, int(shape[1]), int(shape[2]))
        moved = affine_transform(
            padded_pred, matrix, offset=offset + matrix[:, 0] * start_z,
            output_shape=block_shape, output=np.bool_, order=0,
            mode="constant", cval=0, prefilter=False,
        )
        target = np.zeros(block_shape, dtype=bool)
        block_lower = lower + (start_z, 0, 0)
        block_upper = block_lower + block_shape
        common_lower = np.maximum(block_lower, gt.start_zyx)
        common_upper = np.minimum(block_upper, gt_upper)
        if np.all(common_upper > common_lower):
            target_slices = tuple(slice(int(lo), int(hi)) for lo, hi in
                                  zip(common_lower - block_lower, common_upper - block_lower))
            gt_slices = tuple(slice(int(lo), int(hi)) for lo, hi in
                              zip(common_lower - gt.start_zyx, common_upper - gt.start_zyx))
            target[target_slices] = gt.mask[gt_slices]
        pred_count += int(np.count_nonzero(moved))
        gt_count += int(np.count_nonzero(target))
        intersection += int(np.count_nonzero(moved & target))
    if gt_count != gt.foreground_voxels:
        raise RuntimeError("Evaluation grid did not retain all GT foreground voxels.")
    return pred_count, gt_count, intersection, lower, shape


def overlap_metrics(pred_count, gt_count, intersection):
    if min(pred_count, gt_count, intersection) < 0 or intersection > min(pred_count, gt_count):
        raise ValueError("Invalid foreground/intersection counts.")
    union = pred_count + gt_count - intersection
    total = pred_count + gt_count
    return {
        "dice": 2.0 * intersection / total if total else 1.0,
        "iou": intersection / union if union else 1.0,
        "precision": intersection / pred_count if pred_count else float(gt_count == 0),
        "recall": intersection / gt_count if gt_count else float(pred_count == 0),
        "true_positive_voxels": intersection,
        "false_positive_voxels": pred_count - intersection,
        "false_negative_voxels": gt_count - intersection,
    }


def volume_metrics(gt, pred, transform_xyz, chunk_depth=EVAL_VOLUME_CHUNK_DEPTH):
    pred_count, gt_count, intersection, lower, shape = aligned_overlap_counts(
        gt, pred, transform_xyz, chunk_depth,
    )
    voxel_volume = float(np.prod(gt.spacing_zyx))
    gt_volume = gt_count * voxel_volume
    pred_native_volume = pred.foreground_voxels * float(np.prod(pred.spacing_zyx))
    pred_resampled_volume = pred_count * voxel_volume
    metrics = overlap_metrics(pred_count, gt_count, intersection)
    metrics.update({
        "gt_volume_mm3": gt_volume,
        "pred_native_volume_mm3": pred_native_volume,
        "absolute_volume_error_mm3": abs(pred_native_volume - gt_volume),
        "relative_volume_error": abs(pred_native_volume - gt_volume) / gt_volume,
        "signed_volume_bias_mm3": pred_native_volume - gt_volume,
        "pred_resampled_volume_mm3": pred_resampled_volume,
        "resampling_volume_change_mm3": pred_resampled_volume - pred_native_volume,
        "false_positive_volume_mm3": (pred_count - intersection) * voxel_volume,
        "false_negative_volume_mm3": (gt_count - intersection) * voxel_volume,
        "symmetric_difference_volume_mm3": (pred_count + gt_count - 2 * intersection) * voxel_volume,
    })
    grid = {"lower_index_zyx": lower.tolist(), "shape_zyx": shape.tolist(),
            "spacing_zyx_mm": gt.spacing_zyx.tolist()}
    return metrics, grid


def evaluate_volume_case(
    case_name, model_name=SEGMENTATION_MODEL_NAME,
    reconstruction_method=SEGMENTATION_RECONSTRUCTION_METHOD, class_id=EVAL_CLASS_ID,
    chunk_depth=EVAL_VOLUME_CHUNK_DEPTH, alignment_samples=ALIGN_SAMPLES,
    overlap=ALIGN_OVERLAP,
):
    # Always recompute from the original meshes, independently of visualization/surface metrics.
    alignment = compute_case_alignment(case_name, model_name, reconstruction_method,
                                       class_id, alignment_samples, overlap)
    with timed_stage("VOLUME METRICS"):
        gt_path = ground_truth_labels_path(case_name)
        pred_path = segmentation_reconstruction_path(case_name, model_name, reconstruction_method)
        gt = load_class_mask(gt_path, class_id)
        pred = load_class_mask(pred_path, class_id)
        metrics, grid = volume_metrics(gt, pred, alignment.transform_xyz, chunk_depth)
        report = {
            "case_name": case_name, "model_name": model_name,
            "reconstruction_method": reconstruction_method, "class_id": int(class_id),
            "gt_labels_file": str(gt_path), "pred_labels_file": str(pred_path),
            "gt_surface_file": str(alignment.gt_path), "pred_surface_file": str(alignment.pred_path),
            "alignment": "fresh rigid predicted local XYZ mm -> GT local XYZ mm",
            "transform_xyz": alignment.transform_xyz.tolist(),
            "alignment_samples": int(alignment_samples), "alignment_overlap": float(overlap),
            "sample_distance_before_alignment_mm": alignment.before_mm,
            "sample_distance_after_alignment_mm": alignment.after_mm,
            "coordinate_convention": "local index*spacing, matching mesh extraction; NPZ origins are not applied twice",
            "resampling": "inverse rigid transform, nearest neighbour, GT lattice extended to union of objects",
            "volume_error_definition": "native foreground counts times each input spacing product; rigid invariant",
            "evaluation_grid": grid, "chunk_depth": int(chunk_depth),
            "gt_original_shape_zyx": list(gt.original_shape_zyx),
            "pred_original_shape_zyx": list(pred.original_shape_zyx),
            "pred_spacing_zyx_mm": pred.spacing_zyx.tolist(),
            "metrics": metrics,
        }
        output = (EVALUATION_DIR / "volume" / case_name / model_name
                  / reconstruction_method / f"class_{class_id}_metrics.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name, value in metrics.items():
            print(f"[VOLUME] {name}: {value:.6f}")
        print(f"[VOLUME] Report: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", default=SEGMENTATION_MODEL_NAME)
    parser.add_argument("--method", default=SEGMENTATION_RECONSTRUCTION_METHOD)
    parser.add_argument("--class-id", type=int, default=EVAL_CLASS_ID)
    parser.add_argument("--chunk-depth", type=int, default=EVAL_VOLUME_CHUNK_DEPTH)
    parser.add_argument("--alignment-samples", type=int, default=ALIGN_SAMPLES)
    parser.add_argument("--overlap", type=float, default=ALIGN_OVERLAP)
    args = parser.parse_args()
    evaluate_volume_case(args.case, args.model, args.method, args.class_id,
                         args.chunk_depth, args.alignment_samples, args.overlap)


if __name__ == "__main__":
    main()
