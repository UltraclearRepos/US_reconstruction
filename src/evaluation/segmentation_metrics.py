from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    center_of_mass,
)
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.transform import resize

from config import (
    EVALUATION_DIR,
    EVAL_SEGMENTATION_CLASSES,
    EVAL_SURFACE_DICE_TOLERANCE_MM,
    EVAL_HIST_BINS,
    EVAL_EPS,
)
from src.utils.paths import (
    ground_truth_labels_path,
    segmentation_reconstruction_path,
)

from src.visualization.common import load_volume
from src.utils.logging import log_io


# ============================================================
# HELPERS
# ============================================================

def normalize_case_names(case_names):
    if case_names is None:
        return []
    if isinstance(case_names, str):
        return [case_names]
    return list(case_names)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        if x is None:
            return None
        x = float(x)
        if not np.isfinite(x):
            return None
        return x
    except Exception:
        return None


def find_gt_segmentation(case_name):
    path = ground_truth_labels_path(case_name)
    return path if path.exists() else None


def load_labels_npz(path):
    data = np.load(path, allow_pickle=True)

    for key in ["labels", "volume", "arr_0", "data", "mask", "masks"]:
        if key in data:
            return np.asarray(data[key])

    return np.asarray(data[data.files[0]])
def mask_surface(mask):
    mask = mask.astype(bool)

    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3)), border_value=0)
    return mask & (~eroded)


def surface_points(mask, spacing):
    surf = mask_surface(mask)
    pts = np.argwhere(surf)

    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    spacing = np.asarray(spacing, dtype=np.float32)
    return pts.astype(np.float32) * spacing[None, :]


def surface_distances(mask_a, mask_b, spacing):
    pts_a = surface_points(mask_a, spacing)
    pts_b = surface_points(mask_b, spacing)

    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return None, None

    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)

    tree_a = cKDTree(pts_a)
    d_ba, _ = tree_a.query(pts_b, k=1)

    return d_ab.astype(np.float32), d_ba.astype(np.float32)


# ============================================================
# SEGMENTATION METRICS
# ============================================================

def dice_score(pred_mask, gt_mask, eps=1e-8):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    inter = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()

    if denom == 0:
        return 1.0

    return 2.0 * inter / (denom + eps)


def jaccard_index(pred_mask, gt_mask, eps=1e-8):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 1.0

    return inter / (union + eps)


def surface_dice(pred_mask, gt_mask, spacing, tolerance_mm=1.0):
    d_pg, d_gp = surface_distances(pred_mask, gt_mask, spacing)

    if d_pg is None:
        if not np.any(pred_mask) and not np.any(gt_mask):
            return 1.0
        return 0.0

    n_close_pred = np.sum(d_pg <= tolerance_mm)
    n_close_gt = np.sum(d_gp <= tolerance_mm)

    denom = d_pg.size + d_gp.size

    if denom == 0:
        return 1.0

    return (n_close_pred + n_close_gt) / denom


def hd95(pred_mask, gt_mask, spacing):
    d_pg, d_gp = surface_distances(pred_mask, gt_mask, spacing)

    if d_pg is None:
        return None

    d = np.concatenate([d_pg, d_gp])

    if d.size == 0:
        return None

    return np.percentile(d, 95)


def assd(pred_mask, gt_mask, spacing):
    d_pg, d_gp = surface_distances(pred_mask, gt_mask, spacing)

    if d_pg is None:
        return None

    d = np.concatenate([d_pg, d_gp])

    if d.size == 0:
        return None

    return np.mean(d)


def center_of_mass_distance(pred_mask, gt_mask, spacing):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if not np.any(pred_mask) or not np.any(gt_mask):
        return None

    c_pred = np.asarray(center_of_mass(pred_mask), dtype=np.float32)
    c_gt = np.asarray(center_of_mass(gt_mask), dtype=np.float32)

    spacing = np.asarray(spacing, dtype=np.float32)

    return np.linalg.norm((c_pred - c_gt) * spacing)


def absolute_volume_error(pred_mask, gt_mask, spacing):
    voxel_volume = float(np.prod(spacing))

    v_pred = float(np.sum(pred_mask > 0) * voxel_volume)
    v_gt = float(np.sum(gt_mask > 0) * voxel_volume)

    return abs(v_pred - v_gt)


def relative_volume_error(pred_mask, gt_mask, spacing, eps=1e-8):
    voxel_volume = float(np.prod(spacing))

    v_pred = float(np.sum(pred_mask > 0) * voxel_volume)
    v_gt = float(np.sum(gt_mask > 0) * voxel_volume)

    return abs(v_pred - v_gt) / (v_gt + eps)


def pca_main_axes(mask, spacing):
    pts = np.argwhere(mask.astype(bool))

    if pts.shape[0] < 3:
        return None, None, None

    spacing = np.asarray(spacing, dtype=np.float32)
    pts = pts.astype(np.float32) * spacing[None, :]

    center = np.mean(pts, axis=0)
    x = pts - center

    cov = np.cov(x.T)
    vals, vecs = np.linalg.eigh(cov)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    return center.astype(np.float32), vals.astype(np.float32), vecs.astype(np.float32)


def orientation_error_pca(pred_mask, gt_mask, spacing):
    pred_pca = pca_main_axes(pred_mask, spacing)
    gt_pca = pca_main_axes(gt_mask, spacing)

    if pred_pca[0] is None or gt_pca[0] is None:
        return None

    _, _, vec_pred = pred_pca
    _, _, vec_gt = gt_pca

    a = vec_pred[:, 0]
    b = vec_gt[:, 0]

    cosang = abs(float(np.dot(a, b)))
    cosang = np.clip(cosang, -1.0, 1.0)

    angle_rad = np.arccos(cosang)
    return np.degrees(angle_rad)


def surface_area(mask, spacing):
    mask = mask.astype(np.float32)

    if np.count_nonzero(mask) < 8:
        return 0.0

    try:
        verts, faces, _, _ = marching_cubes(
            mask,
            level=0.5,
            spacing=spacing,
        )
        return float(mesh_surface_area(verts, faces))
    except Exception:
        return None


def surface_area_error(pred_mask, gt_mask, spacing):
    a_pred = surface_area(pred_mask, spacing)
    a_gt = surface_area(gt_mask, spacing)

    if a_pred is None or a_gt is None:
        return None

    return abs(a_pred - a_gt)


# ============================================================
# PLOTS
# ============================================================

def save_inside_outside_histograms(image_volume, pred_mask, gt_mask, out_path, bins=128):
    """
    Jeżeli masz dostępny wolumen intensywności, ta funkcja robi histogramy
    wewnątrz i na zewnątrz obiektu. W tym pliku używamy masek jako fallback,
    ale funkcję można później podpiąć pod rekonstrukcję/GT volume.
    """
    img = np.asarray(image_volume, dtype=np.float32)

    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    plt.figure(figsize=(10, 5))

    if np.any(gt_mask):
        plt.hist(img[gt_mask], bins=bins, alpha=0.5, label="inside GT")

    if np.any(~gt_mask):
        plt.hist(img[~gt_mask], bins=bins, alpha=0.5, label="outside GT")

    if np.any(pred_mask):
        plt.hist(img[pred_mask], bins=bins, alpha=0.4, label="inside pred")

    plt.xlabel("Intensity / label value")
    plt.ylabel("Count")
    plt.title("Inside/outside object histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_pca_axes_npz(pred_mask, gt_mask, spacing, out_path):
    pred_center, pred_vals, pred_vecs = pca_main_axes(pred_mask, spacing)
    gt_center, gt_vals, gt_vecs = pca_main_axes(gt_mask, spacing)

    payload = {}

    if pred_center is not None:
        payload["pred_center"] = pred_center
        payload["pred_values"] = pred_vals
        payload["pred_axes"] = pred_vecs

    if gt_center is not None:
        payload["gt_center"] = gt_center
        payload["gt_values"] = gt_vals
        payload["gt_axes"] = gt_vecs

    if payload:
        np.savez_compressed(out_path, **payload)


def save_orthogonal_slices(pred_mask, gt_mask, out_path):
    pred_mask = pred_mask.astype(np.float32)
    gt_mask = gt_mask.astype(np.float32)

    shape = gt_mask.shape
    z = shape[0] // 2
    y = shape[1] // 2
    x = shape[2] // 2

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(gt_mask[z], cmap="gray")
    plt.title("GT axial")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(gt_mask[:, y, :], cmap="gray")
    plt.title("GT coronal")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(gt_mask[:, :, x], cmap="gray")
    plt.title("GT sagittal")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(pred_mask[z], cmap="gray")
    plt.title("Pred axial")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(pred_mask[:, y, :], cmap="gray")
    plt.title("Pred coronal")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(pred_mask[:, :, x], cmap="gray")
    plt.title("Pred sagittal")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# CASE EVALUATION
# ============================================================

def evaluate_segmentation_class(case_name, cls, pred_labels, gt_labels, spacing, out_dir):
    pred_mask = pred_labels == cls
    gt_mask = gt_labels == cls

    row = {
        "case_name": case_name,
        "class_id": int(cls),
        "dice": safe_float(dice_score(pred_mask, gt_mask, eps=EVAL_EPS)),
        "jaccard": safe_float(jaccard_index(pred_mask, gt_mask, eps=EVAL_EPS)),
        "surface_dice": safe_float(surface_dice(
            pred_mask,
            gt_mask,
            spacing=spacing,
            tolerance_mm=EVAL_SURFACE_DICE_TOLERANCE_MM,
        )),
        "hd95_mm": safe_float(hd95(pred_mask, gt_mask, spacing=spacing)),
        "assd_mm": safe_float(assd(pred_mask, gt_mask, spacing=spacing)),
        "center_of_mass_distance_mm": safe_float(center_of_mass_distance(pred_mask, gt_mask, spacing=spacing)),
        "absolute_volume_error_mm3": safe_float(absolute_volume_error(pred_mask, gt_mask, spacing=spacing)),
        "relative_volume_error": safe_float(relative_volume_error(pred_mask, gt_mask, spacing=spacing, eps=EVAL_EPS)),
        "orientation_error_deg": safe_float(orientation_error_pca(pred_mask, gt_mask, spacing=spacing)),
        "surface_area_error_mm2": safe_float(surface_area_error(pred_mask, gt_mask, spacing=spacing)),
        "pred_voxels": int(np.sum(pred_mask)),
        "gt_voxels": int(np.sum(gt_mask)),
    }

    cls_dir = out_dir / f"class_{cls}"
    ensure_dir(cls_dir)

    with open(cls_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    save_pca_axes_npz(
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        spacing=spacing,
        out_path=cls_dir / "pca_axes_for_napari.npz",
    )

    save_orthogonal_slices(
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        out_path=cls_dir / "orthogonal_slices_gt_vs_pred.png",
    )

    # Fallback: histogram z labeli/masek.
    # Jeżeli chcesz histogramy intensywności wewnątrz/na zewnątrz obiektu,
    # najlepiej podać tutaj rekonstrukcję albo GT volume.
    save_inside_outside_histograms(
        image_volume=gt_labels.astype(np.float32),
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        out_path=cls_dir / "inside_outside_histogram_labels_fallback.png",
        bins=EVAL_HIST_BINS,
    )

    print(f"[OK] Segmentation metrics saved: {cls_dir / 'metrics.json'}")

    return row


def write_summary_csv(rows, out_path):
    if not rows:
        return

    keys = sorted(set().union(*[row.keys() for row in rows]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_segmentation_case(
    case_name,
    model_name,
    reconstruction_method,
):
    gt_path = find_gt_segmentation(case_name)

    if gt_path is None:
        print(f"[WARN] No GT segmentation found for case: {case_name}")
        return []

    recon_path = segmentation_reconstruction_path(
        case_name,
        model_name,
        reconstruction_method,
    )
    if not recon_path.exists():
        print(f"[WARN] No reconstructed segmentation found: {recon_path}")
        return []

    gt_labels = load_labels_npz(gt_path)
    pred_labels, spacing = load_volume(recon_path)
    pred_labels = np.rint(pred_labels).astype(np.uint16)
    if gt_labels.shape != pred_labels.shape:
        gt_labels = resize(
            gt_labels.astype(np.float32),
            output_shape=pred_labels.shape,
            order=0,
            mode="edge",
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.uint16)

    out_dir = EVALUATION_DIR.joinpath(
        "segmentation",
        case_name,
        model_name,
        reconstruction_method,
    )
    log_io(
        "EVALUATION segmentation",
        {"reconstruction": recon_path, "GT labels": gt_path},
        {"report": out_dir},
    )
    ensure_dir(out_dir)

    rows = []

    classes = EVAL_SEGMENTATION_CLASSES
    if classes is None:
        classes = sorted([int(x) for x in np.unique(gt_labels) if int(x) != 0])

    for cls in classes:
        cls = int(cls)

        try:
            row = evaluate_segmentation_class(
                case_name=case_name,
                cls=cls,
                pred_labels=pred_labels,
                gt_labels=gt_labels,
                spacing=spacing,
                out_dir=out_dir,
            )
            row["reconstructed_labels_file"] = str(recon_path)
            row["ground_truth_labels_file"] = str(gt_path)
            rows.append(row)

        except Exception as e:
            print(f"[WARN] Failed segmentation evaluation for class {cls}: {e}")

    write_summary_csv(rows, out_dir / "summary_segmentation.csv")

    return rows


def evaluate_segmentation_cases(
    case_names,
    model_name=None,
    reconstruction_method="voxel_nearest",
):
    case_names = normalize_case_names(case_names)

    if not case_names:
        print("[WARN] No cases selected for segmentation evaluation.")
        return []

    all_rows = []

    for case_name in case_names:
        rows = evaluate_segmentation_case(
            case_name,
            model_name,
            reconstruction_method,
        )
        all_rows.extend(rows)

    out_dir = EVALUATION_DIR / "segmentation" / "summary"
    ensure_dir(out_dir)
    write_summary_csv(all_rows, out_dir / "all_segmentation.csv")

    return all_rows
