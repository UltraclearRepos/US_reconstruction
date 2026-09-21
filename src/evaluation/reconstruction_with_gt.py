from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

try:
    from sewar.full_ref import msssim
    SEWAR_AVAILABLE = True
except Exception:
    SEWAR_AVAILABLE = False

from config import (
    EVALUATION_DIR,
    EVAL_EPS,
    EVAL_HIST_BINS,
    EVAL_SSIM_AXIS,
    EVAL_MS_SSIM_ENABLED,
)
from src.utils.paths import (
    ground_truth_volume_path,
    intensity_reconstruction_path,
)

from src.visualization.common import load_volume
from src.utils.logging import log_io


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_case_names(case_names):
    if case_names is None:
        return []
    if isinstance(case_names, str):
        return [case_names]
    return list(case_names)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_reconstruction_files(case_name, methods):
    return [
        path
        for method in normalize_case_names(methods)
        if (path := intensity_reconstruction_path(case_name, method)).exists()
    ]


def find_gt_volume(case_name):
    path = ground_truth_volume_path(case_name)
    return path if path.exists() else None
def valid_pair(pred, gt):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    mask = np.isfinite(pred) & np.isfinite(gt)

    pred = pred[mask]
    gt = gt[mask]

    return pred, gt


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


def data_range_from_gt(gt):
    mn = float(np.nanmin(gt))
    mx = float(np.nanmax(gt))
    dr = mx - mn
    if dr <= 0:
        dr = 1.0
    return dr


def normalize01(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


# ============================================================
# METRICS
# ============================================================

def mae(pred, gt):
    p, g = valid_pair(pred, gt)
    if p.size == 0:
        return None
    return np.mean(np.abs(p - g))


def rmse(pred, gt):
    p, g = valid_pair(pred, gt)
    if p.size == 0:
        return None
    return np.sqrt(np.mean((p - g) ** 2))


def psnr(pred, gt, eps=1e-8):
    r = rmse(pred, gt)
    if r is None:
        return None
    dr = data_range_from_gt(gt)
    return 20.0 * np.log10(dr / (r + eps))


def normalized_cross_correlation(pred, gt, eps=1e-8):
    p, g = valid_pair(pred, gt)

    if p.size == 0:
        return None

    p = p - np.mean(p)
    g = g - np.mean(g)

    denom = np.sqrt(np.sum(p ** 2) * np.sum(g ** 2)) + eps
    return np.sum(p * g) / denom


def mutual_information(pred, gt, bins=128, eps=1e-8):
    p, g = valid_pair(pred, gt)

    if p.size == 0:
        return None

    hist_2d, _, _ = np.histogram2d(p, g, bins=bins)

    pxy = hist_2d / (np.sum(hist_2d) + eps)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_py = px[:, None] * py[None, :]
    nz = pxy > 0

    return np.sum(pxy[nz] * np.log((pxy[nz] + eps) / (px_py[nz] + eps)))


def wasserstein_intensity(pred, gt):
    p, g = valid_pair(pred, gt)
    if p.size == 0:
        return None
    return wasserstein_distance(p, g)


def jensen_shannon_intensity(pred, gt, bins=128, eps=1e-8):
    p, g = valid_pair(pred, gt)

    if p.size == 0:
        return None

    lo = min(float(np.min(p)), float(np.min(g)))
    hi = max(float(np.max(p)), float(np.max(g)))

    if hi <= lo:
        return 0.0

    hp, _ = np.histogram(p, bins=bins, range=(lo, hi), density=False)
    hg, _ = np.histogram(g, bins=bins, range=(lo, hi), density=False)

    hp = hp.astype(np.float64) + eps
    hg = hg.astype(np.float64) + eps

    hp /= np.sum(hp)
    hg /= np.sum(hg)

    return jensenshannon(hp, hg, base=2.0)


def intensity_bias(pred, gt):
    p, g = valid_pair(pred, gt)
    if p.size == 0:
        return None
    return np.mean(p - g)


def relative_intensity_bias(pred, gt, eps=1e-8):
    p, g = valid_pair(pred, gt)
    if p.size == 0:
        return None
    return np.mean(p - g) / (np.mean(g) + eps)


def ssim_slice_wise(pred, gt, axis=0):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    pred01 = normalize01(pred)
    gt01 = normalize01(gt)

    pred_moved = np.moveaxis(pred01, axis, 0)
    gt_moved = np.moveaxis(gt01, axis, 0)

    vals = []

    for i in range(pred_moved.shape[0]):
        p = pred_moved[i]
        g = gt_moved[i]

        if p.shape[0] < 7 or p.shape[1] < 7:
            continue

        try:
            val = ssim(
                g,
                p,
                data_range=1.0,
            )
            vals.append(float(val))
        except Exception:
            continue

    if not vals:
        return None

    return float(np.mean(vals))


def ms_ssim_slice_wise(pred, gt, axis=0):
    if not SEWAR_AVAILABLE:
        return None

    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    pred01 = normalize01(pred)
    gt01 = normalize01(gt)

    pred_moved = np.moveaxis(pred01, axis, 0)
    gt_moved = np.moveaxis(gt01, axis, 0)

    vals = []

    for i in range(pred_moved.shape[0]):
        p = pred_moved[i]
        g = gt_moved[i]

        if p.shape[0] < 32 or p.shape[1] < 32:
            continue

        try:
            val = msssim(g, p, MAX=1.0)
            vals.append(float(np.real(val)))
        except Exception:
            continue

    if not vals:
        return None

    return float(np.mean(vals))


# ============================================================
# PLOTS
# ============================================================

def save_histogram_plot(pred, gt, out_path, bins=128):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    p, g = valid_pair(pred, gt)

    if p.size == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(g.ravel(), bins=bins, alpha=0.5, label="GT")
    plt.hist(p.ravel(), bins=bins, alpha=0.5, label="Reconstruction")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.title("Intensity histogram: GT vs reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_difference_slice_plot(pred, gt, out_path):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    z = pred.shape[0] // 2
    diff = pred - gt

    vmin = np.percentile(gt, 1)
    vmax = np.percentile(gt, 99)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(gt[z], cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred[z], cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Reconstruction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff[z], cmap="coolwarm")
    plt.title("Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# CASE EVALUATION
# ============================================================

def evaluate_reconstruction_file(case_name, recon_path, gt_path, out_dir):
    log_io(
        "EVALUATION intensity",
        {"reconstruction": recon_path, "GT volume": gt_path},
        {"report": out_dir},
    )
    recon, recon_voxel = load_volume(recon_path)
    gt, gt_voxel = load_volume(gt_path)

    recon = np.asarray(recon, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    stem = recon_path.stem
    ensure_dir(out_dir)

    # ------------------------------------------------------------
    # WAŻNE:
    # Reconstruction-with-GT liczymy tylko wtedy, gdy rekonstrukcja
    # i GT są na tej samej siatce voxelowej.
    #
    # Nie robimy automatycznego resize 3D, bo przy dużych wolumenach
    # może to zabić RAM i dodatkowo dawać metryki trudne do interpretacji.
    # ------------------------------------------------------------
    if recon.shape != gt.shape:
        print(
            f"[WARN] Shape mismatch for {recon_path.name}: "
            f"recon={recon.shape}, gt={gt.shape}"
        )
        print("[INFO] Resampling GT volume to reconstruction shape for full-reference metrics.")

        gt = resize(
            gt.astype(np.float32),
            output_shape=recon.shape,
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

    # ------------------------------------------------------------
    # Opcjonalna kontrola voxel size.
    # Shape może być takie samo, ale spacing może się różnić.
    # Wtedy metryki intensywności nadal można policzyć, ale zapisujemy warning.
    # ------------------------------------------------------------
    voxel_warning = None

    recon_voxel_arr = np.asarray(recon_voxel, dtype=np.float32)
    gt_voxel_arr = np.asarray(gt_voxel, dtype=np.float32)

    if recon_voxel_arr.shape == gt_voxel_arr.shape:
        if not np.allclose(recon_voxel_arr, gt_voxel_arr, rtol=1e-3, atol=1e-5):
            voxel_warning = (
                f"Voxel size mismatch: recon={tuple(recon_voxel_arr)}, "
                f"gt={tuple(gt_voxel_arr)}"
            )
            print(f"[WARN] {voxel_warning}")

    # ------------------------------------------------------------
    # Właściwe metryki full-reference dla rekonstrukcji intensywności.
    # ------------------------------------------------------------
    metrics = {
        "case_name": case_name,
        "reconstruction_file": str(recon_path),
        "ground_truth_file": str(gt_path),
        "recon_shape": list(recon.shape),
        "gt_shape": list(gt.shape),
        "recon_voxel_size": list(recon_voxel),
        "gt_voxel_size": list(gt_voxel),
        "voxel_warning": voxel_warning,
        "status": "ok",

        "mae": safe_float(mae(recon, gt)),
        "rmse": safe_float(rmse(recon, gt)),
        "psnr": safe_float(psnr(recon, gt, eps=EVAL_EPS)),
        "ssim": safe_float(ssim_slice_wise(recon, gt, axis=EVAL_SSIM_AXIS)),
        "ms_ssim": safe_float(
            ms_ssim_slice_wise(recon, gt, axis=EVAL_SSIM_AXIS)
        ) if EVAL_MS_SSIM_ENABLED else None,
        "mutual_information": safe_float(
            mutual_information(recon, gt, bins=EVAL_HIST_BINS, eps=EVAL_EPS)
        ),
        "ncc": safe_float(
            normalized_cross_correlation(recon, gt, eps=EVAL_EPS)
        ),
        "wasserstein_distance": safe_float(
            wasserstein_intensity(recon, gt)
        ),
        "jensen_shannon_divergence": safe_float(
            jensen_shannon_intensity(
                recon,
                gt,
                bins=EVAL_HIST_BINS,
                eps=EVAL_EPS,
            )
        ),
        "intensity_bias": safe_float(intensity_bias(recon, gt)),
        "relative_intensity_bias": safe_float(
            relative_intensity_bias(recon, gt, eps=EVAL_EPS)
        ),
    }

    json_path = out_dir / f"{stem}_reconstruction_with_gt_metrics.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_histogram_plot(
        pred=recon,
        gt=gt,
        out_path=out_dir / f"{stem}_hist_gt_vs_recon.png",
        bins=EVAL_HIST_BINS,
    )

    save_difference_slice_plot(
        pred=recon,
        gt=gt,
        out_path=out_dir / f"{stem}_middle_slice_difference.png",
    )

    print(f"[OK] Reconstruction metrics saved: {json_path}")

    return metrics


def write_summary_csv(rows, out_path):
    if not rows:
        return

    keys = sorted(set().union(*[row.keys() for row in rows]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_reconstruction_case_with_gt(case_name, methods):
    gt_path = find_gt_volume(case_name)

    if gt_path is None:
        print(f"[WARN] No GT volume found for case: {case_name}")
        return []

    recon_files = find_reconstruction_files(case_name, methods)

    if not recon_files:
        print(f"[WARN] No reconstruction files found for case: {case_name}")
        return []

    out_dir = EVALUATION_DIR / "intensity" / case_name

    rows = []

    for recon_path in recon_files:
        try:
            method_dir = out_dir / recon_path.parent.name
            row = evaluate_reconstruction_file(
                case_name=case_name,
                recon_path=recon_path,
                gt_path=gt_path,
                out_dir=method_dir,
            )
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed reconstruction evaluation for {recon_path}: {e}")

    write_summary_csv(rows, out_dir / "summary_reconstruction_with_gt.csv")

    return rows


def evaluate_reconstruction_cases_with_gt(case_names, methods=None):
    case_names = normalize_case_names(case_names)

    if not case_names:
        print("[WARN] No cases selected for reconstruction evaluation.")
        return []

    all_rows = []

    for case_name in case_names:
        rows = evaluate_reconstruction_case_with_gt(case_name, methods)
        all_rows.extend(rows)

    out_dir = EVALUATION_DIR / "intensity" / "summary"
    ensure_dir(out_dir)
    write_summary_csv(all_rows, out_dir / "all_reconstruction_with_gt.csv")

    return all_rows
