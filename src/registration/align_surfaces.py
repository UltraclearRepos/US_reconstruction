"""Try rigid surface alignment, save the result, and display it in Napari.

Run independently of reconstruction:
    .venv/bin/python -m src.registration.align_surfaces --case 003_01
"""

import argparse
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from config import (
    SEGMENTATION_MODEL_NAME,
    SEGMENTATION_RECONSTRUCTION_METHOD,
    SURFACES_DIR,
)
from src.utils.logging import log_io, timed_stage
from src.utils.paths import ground_truth_surface_dir, predicted_surface_dir
from src.visualization.qt_environment import prepare_qt_environment


DEFAULT_SAMPLES = 8000
DEFAULT_OVERLAP = 0.9


@dataclass
class CaseAlignment:
    gt_path: Path
    pred_path: Path
    gt_mesh: tuple
    pred_mesh: tuple
    aligned_vertices: np.ndarray
    transform_xyz: np.ndarray
    before_mm: float
    after_mm: float


def load_surface(path):
    """Read existing pipeline meshes; their vertices are in local XYZ mm."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Surface not found: {path}. Run surface extraction for this case first."
        )
    with np.load(path, allow_pickle=False) as data:
        vertices = np.asarray(data["verts"], dtype=np.float64)
        raw_faces = np.asarray(data["faces"])
    if (vertices.ndim != 2 or vertices.shape[1] != 3
            or len(vertices) < 3 or not np.isfinite(vertices).all()):
        raise ValueError(f"Invalid or empty surface vertices: {path}")
    if (raw_faces.ndim != 2 or raw_faces.shape[1] != 3
            or not np.issubdtype(raw_faces.dtype, np.integer)
            or len(raw_faces) == 0
            or raw_faces.min() < 0 or raw_faces.max() >= len(vertices)):
        raise ValueError(f"Invalid or empty triangle faces: {path}")
    return vertices, raw_faces.astype(np.int32)


def sample_surface(vertices, faces, count, rng):
    """Sample triangle interiors by area, independently of mesh vertex density."""
    triangles = vertices[faces]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]), axis=1,
    )
    total = areas.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Surface has no nondegenerate triangles.")
    chosen = triangles[rng.choice(len(faces), count, p=areas / total)]
    u, v = rng.random((2, count))
    root_u = np.sqrt(u)
    return (chosen[:, 0] * (1 - root_u)[:, None]
            + chosen[:, 1] * (root_u * (1 - v))[:, None]
            + chosen[:, 2] * (root_u * v)[:, None])


def transform_points(points, transform):
    """Convention: target_xyz = R @ source_xyz + t (column vectors)."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def rigid_fit(source, target):
    """Least-squares rotation and translation, forbidding reflection and scale."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def pca_frame(points):
    center = points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((points - center).T))
    axes = axes[:, ::-1].copy()
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return center, axes


def initial_transforms(source, target):
    """24 proper PCA axis permutations/signs, plus unchanged and centered starts."""
    source_center, source_axes = pca_frame(source)
    target_center, target_axes = pca_frame(target)
    yield np.eye(4)
    centered = np.eye(4)
    centered[:3, 3] = target_center - source_center
    yield centered
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            axis_map = np.eye(3)[:, order] @ np.diag(signs)
            if np.linalg.det(axis_map) < 0:
                continue
            rotation = target_axes @ axis_map @ source_axes.T
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = target_center - rotation @ source_center
            yield transform


def nearest_fraction(distances, overlap):
    count = max(3, int(np.ceil(len(distances) * overlap)))
    return np.argpartition(distances, count - 1)[:count]


def alignment_score(source, target, target_tree, transform, overlap):
    """Symmetric trimmed RMS on samples, used only to rank alignment attempts."""
    moved = transform_points(source, transform)
    forward = target_tree.query(moved)[0]
    backward = cKDTree(moved).query(target)[0]
    return float(np.sqrt(0.5 * (
        np.mean(forward[nearest_fraction(forward, overlap)] ** 2)
        + np.mean(backward[nearest_fraction(backward, overlap)] ** 2)
    )))


def icp(source, target, initial, overlap, iterations):
    """Bidirectional point-to-point ICP with trimmed correspondences."""
    target_tree = cKDTree(target)
    transform = initial.copy()
    best = transform.copy()
    best_score = alignment_score(source, target, target_tree, transform, overlap)
    previous = best_score
    tolerance = max(float(np.linalg.norm(np.ptp(target, axis=0))) * 1e-6, 1e-7)
    for _ in range(iterations):
        moved = transform_points(source, transform)
        forward, target_indices = target_tree.query(moved)
        backward, source_indices = cKDTree(moved).query(target)
        keep_source = nearest_fraction(forward, overlap)
        keep_target = nearest_fraction(backward, overlap)
        paired_source = np.concatenate((moved[keep_source],
                                        moved[source_indices[keep_target]]))
        paired_target = np.concatenate((target[target_indices[keep_source]],
                                        target[keep_target]))
        delta = rigid_fit(paired_source, paired_target)
        transform = delta @ transform
        score = alignment_score(source, target, target_tree, transform, overlap)
        if score < best_score:
            best_score, best = score, transform.copy()
        if abs(previous - score) < tolerance:
            break
        previous = score
    return best_score, best


def align_point_clouds(source, target, overlap=DEFAULT_OVERLAP, verbose=True):
    """Refine all PCA starts coarsely, then the best four at full sample density."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (source.ndim != 2 or target.ndim != 2
            or source.shape[1] != 3 or target.shape[1] != 3
            or min(len(source), len(target)) < 3
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Alignment needs finite (N, 3) point arrays with N >= 3.")
    if not np.isfinite(overlap) or not 0 < overlap <= 1:
        raise ValueError("Overlap must be in (0, 1].")
    coarse_source = source[:min(1500, len(source))]
    coarse_target = target[:min(1500, len(target))]
    starts = list(initial_transforms(source, target))
    candidates = []
    for index, initial in enumerate(starts, 1):
        candidates.append(icp(coarse_source, coarse_target, initial, overlap, 60))
        if verbose and (index % 5 == 0 or index == len(starts)):
            print(f"[ALIGN] Initial orientations: {index}/{len(starts)}", flush=True)
    candidates.sort(key=lambda result: result[0])
    target_tree = cKDTree(target)
    best = (alignment_score(source, target, target_tree, np.eye(4), overlap),
            np.eye(4))
    for index, (_, initial) in enumerate(candidates[:4], 1):
        candidate = icp(source, target, initial, overlap, 150)
        if candidate[0] < best[0]:
            best = candidate
        if verbose:
            print(f"[ALIGN] Fine ICP: {index}/4", flush=True)
    return best[1]


def mean_sample_distance(source, target, transform):
    moved = transform_points(source, transform)
    return float(0.5 * (cKDTree(target).query(moved)[0].mean()
                        + cKDTree(moved).query(target)[0].mean()))


def show_alignment(gt_mesh, pred_mesh, aligned_vertices, case_name):
    prepare_qt_environment()
    import napari

    viewer = napari.Viewer(ndisplay=3, title=f"Surface alignment: {case_name}")
    viewer.dims.axis_labels = ("z (mm)", "y (mm)", "x (mm)")
    # Pipeline mesh files use XYZ; Napari spatial axes follow ZYX.
    for name, vertices, faces, color, visible in (
        ("GT", *gt_mesh, "green", True),
        ("Reconstruction aligned", aligned_vertices, pred_mesh[1], "magenta", True),
        ("Reconstruction before alignment", *pred_mesh, "cyan", False),
    ):
        viewer.add_surface(
            (vertices[:, ::-1].astype(np.float32), faces,
             np.ones(len(vertices), dtype=np.float32)),
            name=name, colormap=color, contrast_limits=(0, 1),
            opacity=0.55, shading="smooth", visible=visible,
        )
    viewer.reset_view()
    napari.run()


def compute_case_alignment(
    case_name,
    model_name=SEGMENTATION_MODEL_NAME,
    reconstruction_method=SEGMENTATION_RECONSTRUCTION_METHOD,
    class_id=1,
    samples=DEFAULT_SAMPLES,
    overlap=DEFAULT_OVERLAP,
):
    """Always compute from original meshes; never read a saved alignment."""
    if samples < 100:
        raise ValueError("Use at least 100 surface samples.")
    if not np.isfinite(overlap) or not 0 < overlap <= 1:
        raise ValueError("Overlap must be in (0, 1].")
    filename = f"class_{class_id}_surface.npz"
    gt_path = ground_truth_surface_dir(case_name) / filename
    pred_path = predicted_surface_dir(case_name, model_name, reconstruction_method) / filename
    log_io("ALIGN SURFACES (fresh)", {"GT surface": gt_path, "predicted surface": pred_path})
    with timed_stage("ALIGN SURFACES"):
        gt_mesh = load_surface(gt_path)
        pred_mesh = load_surface(pred_path)
        rng = np.random.default_rng(0)
        target = sample_surface(*gt_mesh, samples, rng)
        source = sample_surface(*pred_mesh, samples, rng)
        before = mean_sample_distance(source, target, np.eye(4))
        transform = align_point_clouds(source, target, overlap)
        aligned_vertices = transform_points(pred_mesh[0], transform)
        after = mean_sample_distance(source, target, transform)
        print(f"[ALIGN] Symmetric mean sample distance: {before:.3f} -> {after:.3f} mm")
        print("[ALIGN] Rigid transform predicted XYZ -> GT XYZ (no scale or reflection):")
        print(np.array2string(transform, precision=6, suppress_small=True))
        print("[ALIGN] Geometric fit attempt; inspect the overlap in Napari.")
    return CaseAlignment(gt_path, pred_path, gt_mesh, pred_mesh,
                         aligned_vertices, transform, before, after)


def align_and_view_case(
    case_name,
    model_name=SEGMENTATION_MODEL_NAME,
    reconstruction_method=SEGMENTATION_RECONSTRUCTION_METHOD,
    class_id=1,
    samples=DEFAULT_SAMPLES,
    overlap=DEFAULT_OVERLAP,
    show=True,
):
    result = compute_case_alignment(case_name, model_name, reconstruction_method,
                                    class_id, samples, overlap)
    output = (SURFACES_DIR / "aligned" / case_name / model_name
              / reconstruction_method / f"class_{class_id}_surface.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, verts=result.aligned_vertices.astype(np.float32), faces=result.pred_mesh[1],
        values=np.ones(len(result.aligned_vertices), dtype=np.float32),
        transform_xyz=result.transform_xyz, class_id=np.int32(class_id),
        source_file=np.array(str(result.pred_path)), gt_file=np.array(str(result.gt_path)),
        samples=np.int32(samples), overlap=np.float64(overlap),
    )
    print(f"[ALIGN] Saved: {output}")
    if show:
        with timed_stage("NAPARI SURFACE ALIGNMENT"):
            show_alignment(result.gt_mesh, result.pred_mesh, result.aligned_vertices, case_name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Acquisition name, e.g. 003_01")
    parser.add_argument("--model", default=SEGMENTATION_MODEL_NAME)
    parser.add_argument("--method", default=SEGMENTATION_RECONSTRUCTION_METHOD)
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                        help="Nearest correspondence fraction retained in each direction (0, 1]")
    parser.add_argument("--no-view", action="store_true", help="Save alignment without opening Napari")
    args = parser.parse_args()
    align_and_view_case(args.case, args.model, args.method, args.class_id,
                        args.samples, args.overlap, show=not args.no_view)


if __name__ == "__main__":
    main()
