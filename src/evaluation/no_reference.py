from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import (
    sobel,
    binary_dilation,
    binary_erosion,
)
from skimage.measure import shannon_entropy

from config import (
    EVALUATION_DIR,
    EVAL_OBJECT_THRESHOLD,
    EVAL_NO_REFERENCE_USE_MASK,
    EVAL_HIST_BINS,
    EVAL_EPS,
)
from src.utils.paths import (
    ground_truth_labels_path,
    intensity_reconstruction_path,
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


def find_reconstruction_files(case_name, methods):
    return [
        path
        for method in normalize_case_names(methods)
        if (path := intensity_reconstruction_path(case_name, method)).exists()
    ]
def find_mask_for_case(case_name):
    gt_path = ground_truth_labels_path(case_name)

    if gt_path.exists():
        data = np.load(gt_path, allow_pickle=True)
        if "volume" in data:
            labels = data["volume"]
        elif "labels" in data:
            labels = data["labels"]
        else:
            labels = data[data.files[0]]
        return labels > 0

    return None


def resize_mask_nearest(mask, target_shape):
    from skimage.transform import resize

    if mask is None:
        return None

    if tuple(mask.shape) == tuple(target_shape):
        return mask.astype(bool)

    out = resize(
        mask.astype(np.float32),
        output_shape=target_shape,
        order=0,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )

    return out > 0.5


def normalize01(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


# ============================================================
# NO-REFERENCE METRICS
# ============================================================

def contrast_to_noise_ratio(volume, object_mask=None, eps=1e-8):
    x = np.asarray(volume, dtype=np.float32)

    if object_mask is None:
        thr = float(np.percentile(x, 75))
        object_mask = x > thr

    object_mask = object_mask.astype(bool)
    background_mask = ~object_mask

    if not np.any(object_mask) or not np.any(background_mask):
        return None

    mu_obj = float(np.mean(x[object_mask]))
    mu_bg = float(np.mean(x[background_mask]))
    std_bg = float(np.std(x[background_mask]))

    return abs(mu_obj - mu_bg) / (std_bg + eps)


def generalized_cnr(volume, object_mask=None, bins=128, eps=1e-8):
    """
    gCNR = 1 - overlap(P_object, P_background)
    Wartość bliska 1: dobra separacja rozkładów.
    Wartość bliska 0: duże nakładanie się rozkładów.
    """
    x = np.asarray(volume, dtype=np.float32)

    if object_mask is None:
        thr = float(np.percentile(x, 75))
        object_mask = x > thr

    object_mask = object_mask.astype(bool)
    background_mask = ~object_mask

    if not np.any(object_mask) or not np.any(background_mask):
        return None

    obj = x[object_mask]
    bg = x[background_mask]

    lo = float(min(np.min(obj), np.min(bg)))
    hi = float(max(np.max(obj), np.max(bg)))

    if hi <= lo:
        return 0.0

    h_obj, _ = np.histogram(obj, bins=bins, range=(lo, hi), density=True)
    h_bg, edges = np.histogram(bg, bins=bins, range=(lo, hi), density=True)

    bin_width = float(edges[1] - edges[0])
    overlap = np.sum(np.minimum(h_obj, h_bg)) * bin_width

    return 1.0 - overlap


def noise_std(volume, object_mask=None):
    x = np.asarray(volume, dtype=np.float32)

    if object_mask is None:
        # Zakładamy, że dolne 25% intensywności to głównie tło.
        thr = float(np.percentile(x, 25))
        bg = x[x <= thr]
    else:
        bg = x[~object_mask.astype(bool)]

    if bg.size == 0:
        return None

    return np.std(bg)


def total_variation(volume):
    x = np.asarray(volume, dtype=np.float32)

    tv = 0.0

    for axis in range(x.ndim):
        diff = np.diff(x, axis=axis)
        tv += np.sum(np.abs(diff))

    return tv / x.size


def volume_entropy(volume):
    x = normalize01(volume)
    return shannon_entropy(x)


def boundary_mask_from_object(object_mask):
    object_mask = object_mask.astype(bool)

    if not np.any(object_mask):
        return None

    dil = binary_dilation(object_mask, iterations=1)
    ero = binary_erosion(object_mask, iterations=1)

    return dil & (~ero)


def boundary_sharpness(volume, object_mask=None):
    """
    Średni gradient na granicy obiektu.
    Im większy, tym ostrzejsza granica.
    """
    x = np.asarray(volume, dtype=np.float32)

    if object_mask is None:
        thr = float(np.percentile(x, 75))
        object_mask = x > thr

    boundary = boundary_mask_from_object(object_mask)

    if boundary is None or not np.any(boundary):
        return None

    gx = sobel(x, axis=0)
    gy = sobel(x, axis=1)
    gz = sobel(x, axis=2)

    grad = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    return np.mean(grad[boundary])


def gradient_energy(volume):
    x = np.asarray(volume, dtype=np.float32)

    gx = sobel(x, axis=0)
    gy = sobel(x, axis=1)
    gz = sobel(x, axis=2)

    grad = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    return np.mean(grad)


# ============================================================
# PLOTS
# ============================================================

def save_no_reference_histogram(volume, object_mask, out_path, bins=128):
    x = np.asarray(volume, dtype=np.float32)

    plt.figure(figsize=(8, 5))

    if object_mask is not None and np.any(object_mask):
        plt.hist(x[object_mask], bins=bins, alpha=0.5, label="object")
        plt.hist(x[~object_mask], bins=bins, alpha=0.5, label="background")
    else:
        plt.hist(x.ravel(), bins=bins, alpha=0.8, label="all voxels")

    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.title("No-reference intensity histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_middle_slice_quality_plot(volume, object_mask, out_path):
    x = np.asarray(volume, dtype=np.float32)

    z = x.shape[0] // 2

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(x[z], cmap="gray")
    plt.title("Middle slice")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if object_mask is not None:
        plt.imshow(object_mask[z].astype(np.float32), cmap="gray")
        plt.title("Object mask")
    else:
        gx = sobel(x, axis=0)
        gy = sobel(x, axis=1)
        gz = sobel(x, axis=2)
        grad = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        plt.imshow(grad[z], cmap="gray")
        plt.title("Gradient magnitude")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================================================
# CASE EVALUATION
# ============================================================

def evaluate_no_reference_file(case_name, recon_path, object_mask, out_dir):
    log_io(
        "EVALUATION no-reference",
        {"reconstruction": recon_path},
        {"report": out_dir},
    )
    volume, voxel = load_volume(recon_path)
    volume = np.asarray(volume, dtype=np.float32)

    mask = None

    if EVAL_NO_REFERENCE_USE_MASK and object_mask is not None:
        mask = resize_mask_nearest(object_mask, volume.shape)

    if mask is None:
        mask = volume > EVAL_OBJECT_THRESHOLD

    row = {
        "case_name": case_name,
        "reconstruction_file": str(recon_path),
        "shape": list(volume.shape),
        "cnr": safe_float(contrast_to_noise_ratio(volume, object_mask=mask, eps=EVAL_EPS)),
        "gcnr": safe_float(generalized_cnr(volume, object_mask=mask, bins=EVAL_HIST_BINS, eps=EVAL_EPS)),
        "boundary_sharpness": safe_float(boundary_sharpness(volume, object_mask=mask)),
        "noise_std": safe_float(noise_std(volume, object_mask=mask)),
        "total_variation": safe_float(total_variation(volume)),
        "entropy": safe_float(volume_entropy(volume)),
        "gradient_energy": safe_float(gradient_energy(volume)),
        "object_voxels": int(np.sum(mask)) if mask is not None else None,
    }

    stem = recon_path.stem
    ensure_dir(out_dir)

    json_path = out_dir / f"{stem}_no_reference_metrics.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    save_no_reference_histogram(
        volume=volume,
        object_mask=mask,
        out_path=out_dir / f"{stem}_hist_object_background.png",
        bins=EVAL_HIST_BINS,
    )

    save_middle_slice_quality_plot(
        volume=volume,
        object_mask=mask,
        out_path=out_dir / f"{stem}_middle_slice_quality.png",
    )

    print(f"[OK] No-reference metrics saved: {json_path}")

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


def evaluate_no_reference_case(case_name, methods):
    recon_files = find_reconstruction_files(case_name, methods)

    if not recon_files:
        print(f"[WARN] No reconstruction files found for no-reference evaluation: {case_name}")
        return []

    object_mask = find_mask_for_case(case_name)

    out_dir = EVALUATION_DIR / "no_reference" / case_name

    rows = []

    for recon_path in recon_files:
        try:
            method_dir = out_dir / recon_path.parent.name
            row = evaluate_no_reference_file(
                case_name=case_name,
                recon_path=recon_path,
                object_mask=object_mask,
                out_dir=method_dir,
            )
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed no-reference evaluation for {recon_path}: {e}")

    write_summary_csv(rows, out_dir / "summary_no_reference.csv")

    return rows


def evaluate_no_reference_cases(case_names, methods=None):
    case_names = normalize_case_names(case_names)

    if not case_names:
        print("[WARN] No cases selected for no-reference evaluation.")
        return []

    all_rows = []

    for case_name in case_names:
        rows = evaluate_no_reference_case(case_name, methods)
        all_rows.extend(rows)

    out_dir = EVALUATION_DIR / "no_reference" / "summary"
    ensure_dir(out_dir)
    write_summary_csv(all_rows, out_dir / "all_no_reference.csv")

    return all_rows
