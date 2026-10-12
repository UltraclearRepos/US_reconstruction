"""Surface metrics after a fresh rigid alignment from align_surfaces.py."""

import argparse
import json

import numpy as np

from src.registration.align_surfaces import DEFAULT_OVERLAP, DEFAULT_SAMPLES, compute_case_alignment
from config import (
    EVALUATION_DIR, EVAL_CLASS_ID, EVAL_SURFACE_TOLERANCE_MM,
    SEGMENTATION_MODEL_NAME, SEGMENTATION_RECONSTRUCTION_METHOD,
)
from src.utils.logging import timed_stage


def vertex_area_weights(vertices, faces):
    """Distribute every triangle's physical area equally among its three vertices."""
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0],
    ), axis=1)
    weights = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(weights, faces.ravel(), np.repeat(areas / 3.0, 3))
    return weights


def weighted_percentile(distances, weights, percentile):
    """Inverse empirical CDF, with vertex contributions weighted by mesh area."""
    positive = weights > 0
    values, areas = distances[positive], weights[positive]
    order = np.argsort(values)
    cumulative = np.cumsum(areas[order])
    index = np.searchsorted(cumulative, cumulative[-1] * percentile / 100.0, side="left")
    return float(values[order[min(int(index), len(order) - 1)]])


def point_to_mesh_distances(points, vertices, faces):
    """Exact nearest triangle distance for each point, including triangle interiors."""
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import reference, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkGenericCell, vtkPolyData, vtkStaticCellLocator

    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=np.float64), deep=True))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 3 * (len(faces) + 1), 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.asarray(faces, dtype=np.int64).ravel(), deep=True),
    )
    mesh = vtkPolyData()
    mesh.SetPoints(vtk_points)
    mesh.SetPolys(cells)
    locator = vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    cell = vtkGenericCell()
    cell_id, sub_id, distance_squared = reference(0), reference(0), reference(0.0)
    closest = [0.0, 0.0, 0.0]
    distances = np.empty(len(points), dtype=np.float64)
    for index, point in enumerate(points):
        locator.FindClosestPoint(point, closest, cell, cell_id, sub_id, distance_squared)
        distances[index] = np.sqrt(max(float(distance_squared), 0.0))
    return distances


def surface_metrics(gt_mesh, pred_mesh, tolerance_mm=EVAL_SURFACE_TOLERANCE_MM):
    if not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError("Surface tolerance must be finite and nonnegative.")
    gt_weights = vertex_area_weights(*gt_mesh)
    pred_weights = vertex_area_weights(*pred_mesh)
    gt_area = float(gt_weights.sum())
    pred_area = float(pred_weights.sum())
    if min(gt_area, pred_area) <= 0:
        raise ValueError("Surface metrics require two nonempty surfaces with positive area.")
    pred_to_gt = point_to_mesh_distances(pred_mesh[0], *gt_mesh)
    gt_to_pred = point_to_mesh_distances(gt_mesh[0], *pred_mesh)
    total_area = gt_area + pred_area
    pred_mean = float(np.dot(pred_weights, pred_to_gt) / pred_area)
    gt_mean = float(np.dot(gt_weights, gt_to_pred) / gt_area)
    return {
        "mean_pred_to_gt_mm": pred_mean,
        "mean_gt_to_pred_mm": gt_mean,
        "assd_mm": (pred_area * pred_mean + gt_area * gt_mean) / total_area,
        "rmse_mm": float(np.sqrt((np.dot(pred_weights, pred_to_gt ** 2)
                                   + np.dot(gt_weights, gt_to_pred ** 2)) / total_area)),
        "hd95_mm": max(weighted_percentile(pred_to_gt, pred_weights, 95),
                       weighted_percentile(gt_to_pred, gt_weights, 95)),
        "vertex_hausdorff_mm": float(max(pred_to_gt.max(), gt_to_pred.max())),
        "surface_dice": float((pred_weights[pred_to_gt <= tolerance_mm].sum()
                               + gt_weights[gt_to_pred <= tolerance_mm].sum()) / total_area),
        "pred_surface_area_mm2": pred_area,
        "gt_surface_area_mm2": gt_area,
        "absolute_surface_area_error_mm2": abs(pred_area - gt_area),
        "relative_surface_area_error": abs(pred_area - gt_area) / gt_area,
    }


def evaluate_surface_case(
    case_name, model_name=SEGMENTATION_MODEL_NAME,
    reconstruction_method=SEGMENTATION_RECONSTRUCTION_METHOD, class_id=EVAL_CLASS_ID,
    tolerance_mm=EVAL_SURFACE_TOLERANCE_MM,
    alignment_samples=DEFAULT_SAMPLES, overlap=DEFAULT_OVERLAP,
):
    # Recompute even when an aligned NPZ or a previous metrics report exists.
    alignment = compute_case_alignment(case_name, model_name, reconstruction_method,
                                       class_id, alignment_samples, overlap)
    with timed_stage("SURFACE METRICS"):
        metrics = surface_metrics(alignment.gt_mesh,
                                  (alignment.aligned_vertices, alignment.pred_mesh[1]),
                                  tolerance_mm)
        report = {
            "case_name": case_name, "model_name": model_name,
            "reconstruction_method": reconstruction_method, "class_id": int(class_id),
            "gt_surface_file": str(alignment.gt_path),
            "pred_surface_file": str(alignment.pred_path),
            "alignment": "fresh rigid predicted local XYZ mm -> GT local XYZ mm",
            "transform_xyz": alignment.transform_xyz.tolist(),
            "alignment_samples": int(alignment_samples), "alignment_overlap": float(overlap),
            "sample_distance_before_alignment_mm": alignment.before_mm,
            "sample_distance_after_alignment_mm": alignment.after_mm,
            "distance_method": "all vertices of both meshes to exact nearest opposite mesh triangles, no sampling or trimming",
            "pred_vertices_evaluated": int(len(alignment.pred_mesh[0])),
            "gt_vertices_evaluated": int(len(alignment.gt_mesh[0])),
            "pred_triangles": int(len(alignment.pred_mesh[1])),
            "gt_triangles": int(len(alignment.gt_mesh[1])),
            "vertex_area_weighting": "one third of each incident triangle area per vertex",
            "surface_dice_tolerance_mm": float(tolerance_mm),
            "assd_definition": "area-weighted mean of both directional distances",
            "hd95_definition": "maximum of the two area-weighted directional 95th percentiles",
            "hausdorff_definition": "maximum vertex-to-mesh distance in both directions; not a continuous triangle-interior maximum",
            "metrics": metrics,
        }
        output = (EVALUATION_DIR / "surface" / case_name / model_name
                  / reconstruction_method / f"class_{class_id}_metrics.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name, value in metrics.items():
            print(f"[SURFACE] {name}: {value:.6f}")
        print(f"[SURFACE] Report: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", default=SEGMENTATION_MODEL_NAME)
    parser.add_argument("--method", default=SEGMENTATION_RECONSTRUCTION_METHOD)
    parser.add_argument("--class-id", type=int, default=EVAL_CLASS_ID)
    parser.add_argument("--tolerance-mm", type=float, default=EVAL_SURFACE_TOLERANCE_MM)
    parser.add_argument("--alignment-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    evaluate_surface_case(args.case, args.model, args.method, args.class_id,
                          args.tolerance_mm, args.alignment_samples, args.overlap)


if __name__ == "__main__":
    main()
