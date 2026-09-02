"""U-Net training entry point."""

from pathlib import Path
import csv
import random
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from src.utils.logging import log_io


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

TRAINING_ROOT = PROJECT_ROOT / "data" / "segmentation_training"

ORIGINAL_IMAGES_DIR = TRAINING_ROOT / "images"
MASKS_DIR = TRAINING_ROOT / "masks"

DENOISED_ROOT = TRAINING_ROOT / "denoised"

MODELS_ROOT = PROJECT_ROOT / "models" / "segmentation"


# ============================================================
# DATA VARIANTS
# ============================================================

TRAIN_VARIANTS = [
    "original",
    "rdpad",
    "gaussian",
    "median",
    "bilateral",
    "tv",
]

DENOISING_VARIANTS = [
    "rdpad",
    "gaussian",
    "median",
    "bilateral",
    "tv",
]


# ============================================================
# TRAINING SETTINGS
# ============================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

NUM_CLASSES = 6
IN_CHANNELS = 1

IMAGE_SIZE = (256, 256)  # (H, W)
TRAIN_RATIO = 0.80

BATCH_SIZE = 8
NUM_WORKERS = 2

EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BASE_CHANNELS = 32
DROPOUT = 0.10

USE_AMP = True

# Loss
CE_WEIGHT_POWER = 0.5
DICE_LOSS_WEIGHT = 1.0
CE_LOSS_WEIGHT = 1.0
FOCAL_LOSS_WEIGHT = 0.5
FOCAL_GAMMA = 2.0

# Oversampling obrazów z klasami mniejszościowymi
USE_WEIGHTED_SAMPLER = True
MINORITY_SAMPLE_POWER = 0.7

# Augmentacje
USE_AUGMENTATION = True
AUG_FLIP_PROB = 0.5
AUG_ROT90_PROB = 0.5
AUG_INTENSITY_PROB = 0.5
AUG_NOISE_PROB = 0.25

# Save
SAVE_EVERY_LAST = True
SAVE_TORCHSCRIPT_BEST = True
SAVE_FULL_PYTORCH_BEST = True

# Denoising
PREPARE_DENOISED_DATA = True
OVERWRITE_DENOISED_DATA = False

RDPAD_ITERATIONS = 100
RDPAD_TIMESTEP = 0.20
RDPAD_Q0_MODE = "median"
RDPAD_Q0_PERCENTILE = 25.0

GAUSSIAN_SIGMA = 1.0
MEDIAN_SIZE = 3
BILATERAL_SIGMA_COLOR = 0.05
BILATERAL_SIGMA_SPATIAL = 3.0
TV_WEIGHT = 0.08
TV_MAX_NUM_ITER = 200


# ============================================================
# OPTIONAL PROJECT DENOISING IMPORTS
# ============================================================

try:
    from src.denoising.rdp_ad import denoise_rdpad_2d
except Exception:
    denoise_rdpad_2d = None

try:
    from src.denoising.classical import (
        denoise_gaussian_2d,
        denoise_median_2d,
        denoise_bilateral_2d,
        denoise_tv_2d,
    )
except Exception:
    denoise_gaussian_2d = None
    denoise_median_2d = None
    denoise_bilateral_2d = None
    denoise_tv_2d = None


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path) -> List[Path]:
    files = []

    if not path.exists():
        return files

    for ext in IMAGE_EXTENSIONS:
        files.extend(sorted(path.glob(f"*{ext}")))

    return sorted(files)


def load_grayscale_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img).astype(np.float32)


def load_mask_raw(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    return np.asarray(mask)


def save_grayscale_uint8(arr: np.ndarray, path: Path):
    ensure_dir(path.parent)
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def normalize_to_uint8_like_input(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Denoising może zwrócić float w zakresie 0..1 albo 0..255.
    Zapisujemy zawsze jako uint8, zachowując typowy zakres intensywności.
    """
    x = np.asarray(x, dtype=np.float32)

    if np.nanmax(x) <= 1.5:
        x = x * 255.0

    return np.clip(x, 0, 255).astype(np.uint8)


def resize_image_np(arr: np.ndarray, size_hw: Tuple[int, int], is_mask: bool) -> np.ndarray:
    h, w = size_hw

    if is_mask:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
        img = img.resize((w, h), resample=Image.NEAREST)
        return np.asarray(img)

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    img = img.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(img).astype(np.float32)


def find_matching_mask(image_path: Path, masks_dir: Path) -> Path:
    stem = image_path.stem

    for ext in IMAGE_EXTENSIONS:
        candidate = masks_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    candidates = list(masks_dir.glob(f"{stem}.*"))
    candidates = [p for p in candidates if p.suffix.lower() in IMAGE_EXTENSIONS]

    if candidates:
        return sorted(candidates)[0]

    raise FileNotFoundError(f"Nie znaleziono maski dla obrazu: {image_path.name}")


def build_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Brak folderu obrazów: {images_dir}")

    if not masks_dir.exists():
        raise FileNotFoundError(f"Brak folderu masek: {masks_dir}")

    images = list_images(images_dir)

    if not images:
        raise RuntimeError(f"Nie znaleziono obrazów w: {images_dir}")

    pairs = []

    for img_path in images:
        mask_path = find_matching_mask(img_path, masks_dir)
        pairs.append((img_path, mask_path))

    return pairs


def discover_mask_values(pairs: List[Tuple[Path, Path]]) -> List[int]:
    values = set()

    for _, mask_path in pairs:
        mask = load_mask_raw(mask_path)
        unique = np.unique(mask)

        for v in unique:
            values.add(int(v))

    values = sorted(values)

    if len(values) > NUM_CLASSES:
        print("[WARN] Liczba unikalnych wartości w maskach > NUM_CLASSES.")
        print("[WARN] Unique mask values:", values)
        print("[WARN] NUM_CLASSES:", NUM_CLASSES)

    return values


def build_label_mapping(mask_values: List[int]) -> Dict[int, int]:
    mapping = {}

    for idx, val in enumerate(mask_values):
        if idx >= NUM_CLASSES:
            break
        mapping[int(val)] = int(idx)

    if 0 not in mapping:
        print("[WARN] W maskach nie znaleziono wartości 0. Tło może nie być klasą 0.")

    return mapping


def remap_mask(mask: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.int64)

    for raw_value, class_id in mapping.items():
        out[mask == raw_value] = class_id

    return out.astype(np.int64)


def split_pairs(pairs: List[Tuple[Path, Path]]):
    pairs = pairs.copy()
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    return train_pairs, val_pairs


def compute_class_counts(pairs: List[Tuple[Path, Path]], mapping: Dict[int, int]) -> np.ndarray:
    counts = np.zeros((NUM_CLASSES,), dtype=np.float64)

    for _, mask_path in pairs:
        mask = load_mask_raw(mask_path)

        if IMAGE_SIZE is not None:
            mask = resize_image_np(mask, IMAGE_SIZE, is_mask=True)

        mask = remap_mask(mask, mapping)

        for c in range(NUM_CLASSES):
            counts[c] += np.sum(mask == c)

    return counts


def compute_class_weights(class_counts: np.ndarray) -> torch.Tensor:
    counts = class_counts.copy()
    counts[counts <= 0] = 1.0

    freq = counts / counts.sum()
    weights = 1.0 / np.power(freq, CE_WEIGHT_POWER)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(
    pairs: List[Tuple[Path, Path]],
    mapping: Dict[int, int],
    class_weights: torch.Tensor,
):
    cw = class_weights.detach().cpu().numpy()
    sample_weights = []

    for _, mask_path in pairs:
        mask = load_mask_raw(mask_path)

        if IMAGE_SIZE is not None:
            mask = resize_image_np(mask, IMAGE_SIZE, is_mask=True)

        mask = remap_mask(mask, mapping)

        present = np.unique(mask)
        present = present[present < NUM_CLASSES]

        if len(present) == 0:
            sample_weights.append(1.0)
            continue

        w = float(np.mean(cw[present]))
        w = np.power(w, MINORITY_SAMPLE_POWER)
        sample_weights.append(w)

    return torch.tensor(sample_weights, dtype=torch.float32)


# ============================================================
# DENOISING PREPARATION
# ============================================================

def get_denoise_function(method: str):
    if method == "rdpad":
        if denoise_rdpad_2d is None:
            raise ImportError("Brak denoising.rdp_ad.denoise_rdpad_2d")

        return lambda frame: denoise_rdpad_2d(
            frame,
            iterations=RDPAD_ITERATIONS,
            timestep=RDPAD_TIMESTEP,
            q0_mode=RDPAD_Q0_MODE,
            q0_percentile=RDPAD_Q0_PERCENTILE,
        )

    if method == "gaussian":
        if denoise_gaussian_2d is None:
            raise ImportError("Brak denoising.classical.denoise_gaussian_2d")

        return lambda frame: denoise_gaussian_2d(
            frame,
            sigma=GAUSSIAN_SIGMA,
        )

    if method == "median":
        if denoise_median_2d is None:
            raise ImportError("Brak denoising.classical.denoise_median_2d")

        return lambda frame: denoise_median_2d(
            frame,
            size=MEDIAN_SIZE,
        )

    if method == "bilateral":
        if denoise_bilateral_2d is None:
            raise ImportError("Brak denoising.classical.denoise_bilateral_2d")

        return lambda frame: denoise_bilateral_2d(
            frame,
            sigma_color=BILATERAL_SIGMA_COLOR,
            sigma_spatial=BILATERAL_SIGMA_SPATIAL,
        )

    if method == "tv":
        if denoise_tv_2d is None:
            raise ImportError("Brak denoising.classical.denoise_tv_2d")

        return lambda frame: denoise_tv_2d(
            frame,
            weight=TV_WEIGHT,
            max_num_iter=TV_MAX_NUM_ITER,
        )

    raise ValueError(f"Nieznana metoda denoisingu: {method}")


def prepare_denoised_variant(method: str):
    src_dir = ORIGINAL_IMAGES_DIR
    dst_dir = DENOISED_ROOT / method

    if not src_dir.exists():
        print(f"[WARN] Brak folderu obrazów oryginalnych: {src_dir}")
        return

    images = list_images(src_dir)

    if not images:
        print(f"[WARN] Brak obrazów do denoisingu w: {src_dir}")
        return

    if dst_dir.exists() and not OVERWRITE_DENOISED_DATA:
        existing = list_images(dst_dir)
        if len(existing) == len(images):
            print(f"[SKIP] Denoised variant already exists: {dst_dir}")
            return

    ensure_dir(dst_dir)

    fn = get_denoise_function(method)

    print("============================================================")
    print(f"[INFO] Preparing denoised data: {method}")
    print(f"[INFO] Source: {src_dir}")
    print(f"[INFO] Output: {dst_dir}")
    print(f"[INFO] Images: {len(images)}")

    for idx, img_path in enumerate(images, start=1):
        image = load_grayscale_image(img_path)
        denoised = fn(image)
        denoised_u8 = normalize_to_uint8_like_input(denoised, image)

        out_path = dst_dir / img_path.name
        save_grayscale_uint8(denoised_u8, out_path)

        if idx % 50 == 0:
            print(f"[INFO] {method}: processed {idx}/{len(images)}")

    print(f"[OK] Denoised data ready: {dst_dir}")


def prepare_all_denoised_variants():
    if not PREPARE_DENOISED_DATA:
        return

    for method in DENOISING_VARIANTS:
        try:
            prepare_denoised_variant(method)
        except Exception as e:
            print(f"[WARN] Nie udało się przygotować wariantu '{method}': {e}")


def get_images_dir_for_variant(variant_name: str) -> Path:
    if variant_name == "original":
        return ORIGINAL_IMAGES_DIR

    return DENOISED_ROOT / variant_name


# ============================================================
# DATASET
# ============================================================

class SegmentationDataset(Dataset):
    def __init__(self, pairs, mapping, augment=False):
        self.pairs = pairs
        self.mapping = mapping
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _augment(self, image, mask):
        if not USE_AUGMENTATION:
            return image, mask

        if random.random() < AUG_FLIP_PROB:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if random.random() < AUG_FLIP_PROB:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        if random.random() < AUG_ROT90_PROB:
            k = random.randint(0, 3)
            image = np.rot90(image, k=k).copy()
            mask = np.rot90(mask, k=k).copy()

        if random.random() < AUG_INTENSITY_PROB:
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-0.05, 0.05)
            image = image * gain + bias
            image = np.clip(image, 0.0, 1.0)

        if random.random() < AUG_NOISE_PROB:
            noise = np.random.normal(0.0, 0.02, size=image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return image, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = load_grayscale_image(img_path)
        mask_raw = load_mask_raw(mask_path)

        if IMAGE_SIZE is not None:
            image = resize_image_np(image, IMAGE_SIZE, is_mask=False)
            mask_raw = resize_image_np(mask_raw, IMAGE_SIZE, is_mask=True)

        mask = remap_mask(mask_raw, self.mapping)

        image = image.astype(np.float32) / 255.0
        image = np.clip(image, 0.0, 1.0)

        if self.augment:
            image, mask = self._augment(image, mask)

        image_t = torch.from_numpy(image[None, :, :]).float()
        mask_t = torch.from_numpy(mask).long()

        return image_t, mask_t


# ============================================================
# MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.block(x))


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, num_classes=6, base=32, dropout=0.10):
        super().__init__()

        self.enc1 = DoubleConv(in_ch, base, dropout=0.0)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base, base * 2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base * 2, base * 4, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base, dropout=0.0)

        self.out = nn.Conv2d(base, num_classes, kernel_size=1)

    def _crop_or_pad(self, x, ref):
        _, _, h, w = x.shape
        _, _, rh, rw = ref.shape

        if h == rh and w == rw:
            return x

        dh = rh - h
        dw = rw - w

        if dh > 0 or dw > 0:
            x = F.pad(
                x,
                [
                    max(dw // 2, 0),
                    max(dw - dw // 2, 0),
                    max(dh // 2, 0),
                    max(dh - dh // 2, 0),
                ],
            )

        _, _, h, w = x.shape

        if h > rh:
            y0 = (h - rh) // 2
            x = x[:, :, y0:y0 + rh, :]

        if w > rw:
            x0 = (w - rw) // 2
            x = x[:, :, :, x0:x0 + rw]

        return x

    def forward(self, x):
        e1 = self.enc1(x)

        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self._crop_or_pad(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._crop_or_pad(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._crop_or_pad(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ============================================================
# LOSSES
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5, include_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if not self.include_background and self.num_classes > 1:
            dice = dice[1:]

        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )

        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce

        return focal.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(weight=class_weights)

        self.dice = DiceLoss(
            num_classes=NUM_CLASSES,
            include_background=False,
        )

        self.focal = FocalLoss(
            gamma=FOCAL_GAMMA,
            weight=class_weights,
        )

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)

        loss = (
            CE_LOSS_WEIGHT * ce_loss
            + DICE_LOSS_WEIGHT * dice_loss
            + FOCAL_LOSS_WEIGHT * focal_loss
        )

        return loss, {
            "ce": float(ce_loss.detach().cpu()),
            "dice_loss": float(dice_loss.detach().cpu()),
            "focal": float(focal_loss.detach().cpu()),
        }


# ============================================================
# METRICS
# ============================================================

@torch.no_grad()
def compute_metrics(logits, targets, num_classes):
    preds = torch.argmax(logits, dim=1)

    dices = []
    ious = []

    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c

        inter = torch.sum(pred_c & target_c).float()
        pred_sum = torch.sum(pred_c).float()
        target_sum = torch.sum(target_c).float()
        union = torch.sum(pred_c | target_c).float()

        dice = (2.0 * inter + 1e-5) / (pred_sum + target_sum + 1e-5)
        iou = (inter + 1e-5) / (union + 1e-5)

        dices.append(float(dice.cpu()))
        ious.append(float(iou.cpu()))

    mean_dice_fg = float(np.mean(dices[1:])) if num_classes > 1 else dices[0]
    mean_iou_fg = float(np.mean(ious[1:])) if num_classes > 1 else ious[0]

    return {
        "dice_per_class": dices,
        "iou_per_class": ious,
        "mean_dice_fg": mean_dice_fg,
        "mean_iou_fg": mean_iou_fg,
    }


def aggregate_metrics(metric_list):
    if not metric_list:
        return None

    num_classes = len(metric_list[0]["dice_per_class"])

    dice_arr = np.array([m["dice_per_class"] for m in metric_list], dtype=np.float64)
    iou_arr = np.array([m["iou_per_class"] for m in metric_list], dtype=np.float64)

    return {
        "dice_per_class": dice_arr.mean(axis=0).tolist(),
        "iou_per_class": iou_arr.mean(axis=0).tolist(),
        "mean_dice_fg": float(np.mean(dice_arr[:, 1:])) if num_classes > 1 else float(np.mean(dice_arr)),
        "mean_iou_fg": float(np.mean(iou_arr[:, 1:])) if num_classes > 1 else float(np.mean(iou_arr)),
    }


# ============================================================
# SAVE MODEL EXPORTS
# ============================================================

def save_best_model_exports(
    model,
    out_dir: Path,
    variant_name: str,
    epoch: int,
    best_score: float,
    mapping: Dict[int, int],
    mask_values: List[int],
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    checkpoint_path = out_dir / "best_model.pt"

    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "best_score": float(best_score),
            "variant_name": variant_name,
            "label_mapping": mapping,
            "mask_values": mask_values,
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "base_channels": BASE_CHANNELS,
            "dropout": DROPOUT,
            "in_channels": IN_CHANNELS,
        },
        checkpoint_path,
    )

    was_training = model.training

    model_cpu = model.to("cpu")
    model_cpu.eval()

    full_path = out_dir / "best_model_full.pt"
    traced_path = out_dir / "best_model_traced.pt"

    if SAVE_FULL_PYTORCH_BEST:
        torch.save(model_cpu, full_path)

    if IMAGE_SIZE is None:
        example_h, example_w = 256, 256
    else:
        example_h, example_w = IMAGE_SIZE

    if SAVE_TORCHSCRIPT_BEST:
        example_input = torch.randn(
            1,
            IN_CHANNELS,
            example_h,
            example_w,
            dtype=torch.float32,
        )

        with torch.no_grad():
            traced = torch.jit.trace(model_cpu, example_input)
            traced.save(str(traced_path))

    export_meta = {
        "variant_name": variant_name,
        "epoch": int(epoch),
        "best_score": float(best_score),
        "files": {
            "checkpoint": str(checkpoint_path),
            "full_pytorch_model": str(full_path) if SAVE_FULL_PYTORCH_BEST else None,
            "torchscript_traced_model": str(traced_path) if SAVE_TORCHSCRIPT_BEST else None,
        },
        "input": {
            "in_channels": int(IN_CHANNELS),
            "image_size": list(IMAGE_SIZE) if IMAGE_SIZE is not None else None,
            "example_trace_size": [int(example_h), int(example_w)],
        },
        "output": {
            "num_classes": int(NUM_CLASSES),
            "logits_shape_for_trace": [1, int(NUM_CLASSES), int(example_h), int(example_w)],
        },
        "model": {
            "base_channels": int(BASE_CHANNELS),
            "dropout": float(DROPOUT),
        },
        "label_mapping": mapping,
        "mask_values": mask_values,
    }

    with open(out_dir / "best_model_export_meta.json", "w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2, ensure_ascii=False)

    model.to(DEVICE)

    if was_training:
        model.train()
    else:
        model.eval()

    print(f"[OK] Saved best checkpoint   : {checkpoint_path}")

    if SAVE_FULL_PYTORCH_BEST:
        print(f"[OK] Saved full PyTorch model: {full_path}")

    if SAVE_TORCHSCRIPT_BEST:
        print(f"[OK] Saved TorchScript model : {traced_path}")

    print(f"[OK] Saved metadata          : {out_dir / 'best_model_export_meta.json'}")


# ============================================================
# TRAINING
# ============================================================

def make_loaders(train_pairs, val_pairs, mapping, class_weights):
    train_ds = SegmentationDataset(
        pairs=train_pairs,
        mapping=mapping,
        augment=True,
    )

    val_ds = SegmentationDataset(
        pairs=val_pairs,
        mapping=mapping,
        augment=False,
    )

    if USE_WEIGHTED_SAMPLER:
        sample_weights = compute_sample_weights(
            train_pairs,
            mapping,
            class_weights,
        )

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_variant(variant_name: str, images_dir: Path):
    print("============================================================")
    print(f"[INFO] Training variant: {variant_name}")
    print(f"[INFO] Images dir: {images_dir}")
    print(f"[INFO] Masks dir : {MASKS_DIR}")
    print(f"[INFO] Device    : {DEVICE}")

    out_dir = MODELS_ROOT / variant_name
    log_io(
        f"TRAIN supervised/{variant_name}",
        {"images": images_dir, "masks": MASKS_DIR},
        {"model": out_dir},
    )
    ensure_dir(out_dir)

    pairs = build_pairs(images_dir, MASKS_DIR)

    mask_values = discover_mask_values(pairs)
    mapping = build_label_mapping(mask_values)

    train_pairs, val_pairs = split_pairs(pairs)

    print(f"[INFO] Total pairs : {len(pairs)}")
    print(f"[INFO] Train pairs : {len(train_pairs)}")
    print(f"[INFO] Val pairs   : {len(val_pairs)}")
    print(f"[INFO] Mask values : {mask_values}")
    print(f"[INFO] Label map   : {mapping}")

    class_counts = compute_class_counts(train_pairs, mapping)
    class_weights = compute_class_weights(class_counts).to(DEVICE)

    print(f"[INFO] Class counts : {class_counts.tolist()}")
    print(f"[INFO] Class weights: {class_weights.detach().cpu().numpy().tolist()}")

    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant_name": variant_name,
                "images_dir": str(images_dir),
                "masks_dir": str(MASKS_DIR),
                "num_pairs": len(pairs),
                "num_train": len(train_pairs),
                "num_val": len(val_pairs),
                "mask_values": mask_values,
                "label_mapping": mapping,
                "class_counts": class_counts.tolist(),
                "class_weights": class_weights.detach().cpu().numpy().tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    train_loader, val_loader = make_loaders(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        mapping=mapping,
        class_weights=class_weights.detach().cpu(),
    )

    model = UNet2D(
        in_ch=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base=BASE_CHANNELS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = CombinedLoss(class_weights=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))

    metrics_csv = out_dir / "metrics.csv"

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_mean_dice_fg",
            "val_mean_iou_fg",
            "lr",
        ]

        for c in range(NUM_CLASSES):
            header.append(f"val_dice_class_{c}")

        for c in range(NUM_CLASSES):
            header.append(f"val_iou_class_{c}")

        writer.writerow(header)

    best_score = -1.0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_losses = []

        for images, masks in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE == "cuda")):
                logits = model(images)
                loss, _ = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))

        model.eval()

        val_losses = []
        val_metric_list = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                logits = model(images)
                loss, _ = criterion(logits, masks)

                val_losses.append(float(loss.detach().cpu()))
                val_metric_list.append(compute_metrics(logits, masks, NUM_CLASSES))

        val_loss = float(np.mean(val_losses))
        val_metrics = aggregate_metrics(val_metric_list)

        val_score = val_metrics["mean_dice_fg"]

        scheduler.step(val_score)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{variant_name}] "
            f"epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice_fg={val_metrics['mean_dice_fg']:.4f} | "
            f"val_iou_fg={val_metrics['mean_iou_fg']:.4f} | "
            f"lr={lr:.2e}"
        )

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            row = [
                epoch,
                train_loss,
                val_loss,
                val_metrics["mean_dice_fg"],
                val_metrics["mean_iou_fg"],
                lr,
            ]

            row.extend(val_metrics["dice_per_class"])
            row.extend(val_metrics["iou_per_class"])

            writer.writerow(row)

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch

            save_best_model_exports(
                model=model,
                out_dir=out_dir,
                variant_name=variant_name,
                epoch=epoch,
                best_score=best_score,
                mapping=mapping,
                mask_values=mask_values,
            )

            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": float(best_score),
                    "variant_name": variant_name,
                    "label_mapping": mapping,
                    "mask_values": mask_values,
                    "num_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "base_channels": BASE_CHANNELS,
                    "dropout": DROPOUT,
                    "in_channels": IN_CHANNELS,
                },
                out_dir / "best_training_checkpoint.pt",
            )

            print(f"[OK] Saved best training checkpoint: {out_dir / 'best_training_checkpoint.pt'}")

        if SAVE_EVERY_LAST:
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": float(best_score),
                    "variant_name": variant_name,
                    "label_mapping": mapping,
                    "mask_values": mask_values,
                    "num_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "base_channels": BASE_CHANNELS,
                    "dropout": DROPOUT,
                    "in_channels": IN_CHANNELS,
                },
                out_dir / "last_model.pt",
            )

    print("------------------------------------------------------------")
    print(f"[DONE] Variant: {variant_name}")
    print(f"[BEST] epoch={best_epoch}, val_mean_dice_fg={best_score:.4f}")
    print(f"[OUT] {out_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    ensure_dir(MODELS_ROOT)

    print("============================================================")
    print(f"[INFO] PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"[INFO] TRAINING_ROOT: {TRAINING_ROOT}")
    print(f"[INFO] IMAGES       : {ORIGINAL_IMAGES_DIR}")
    print(f"[INFO] MASKS        : {MASKS_DIR}")
    print(f"[INFO] MODELS_ROOT  : {MODELS_ROOT}")
    print(f"[INFO] DEVICE       : {DEVICE}")

    if DEVICE == "cuda":
        print(f"[INFO] CUDA device  : {torch.cuda.get_device_name(0)}")

    print("============================================================")

    if not ORIGINAL_IMAGES_DIR.exists():
        print(f"[WARN] Brak folderu obrazów: {ORIGINAL_IMAGES_DIR}")
        print("[WARN] Utwórz strukturę:")
        print("       data/segmentation_training/images/")
        print("       data/segmentation_training/masks/")
        return

    if not MASKS_DIR.exists():
        print(f"[WARN] Brak folderu masek: {MASKS_DIR}")
        print("[WARN] Utwórz strukturę:")
        print("       data/segmentation_training/images/")
        print("       data/segmentation_training/masks/")
        return

    prepare_all_denoised_variants()

    for variant_name in TRAIN_VARIANTS:
        images_dir = get_images_dir_for_variant(variant_name)

        if not images_dir.exists():
            print(f"[WARN] Pomijam wariant '{variant_name}', brak folderu: {images_dir}")
            continue

        try:
            train_one_variant(
                variant_name=variant_name,
                images_dir=images_dir,
            )
        except Exception as e:
            print("============================================================")
            print(f"[ERROR] Training failed for variant: {variant_name}")
            print(f"[ERROR] {e}")
            print("============================================================")

    print("============================================================")
    print("[OK] Training script finished.")
    print("============================================================")


if __name__ == "__main__":
    main()
