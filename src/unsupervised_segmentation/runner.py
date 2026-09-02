from pathlib import Path
from datetime import datetime
import os
import shutil
import subprocess

import h5py
import numpy as np
from PIL import Image

from config import (
    UNSUPERVISED_REPO_DIR,
    UNSUPERVISED_REPO_URL,
    UNSUPERVISED_DINOV2_REPO_URL,
    UNSUPERVISED_ALLOW_GIT_CLONE,
    UNSUPERVISED_DATASET_NAME,
    UNSUPERVISED_CASE_NAME,
    UNSUPERVISED_IMAGE_KEY,
    UNSUPERVISED_FRAME_START,
    UNSUPERVISED_FRAME_END,
    UNSUPERVISED_FRAME_STEP,
    UNSUPERVISED_PNG_NORMALIZATION,
    UNSUPERVISED_P_LOW,
    UNSUPERVISED_P_HIGH,
    UNSUPERVISED_N_CLASSES,
    UNSUPERVISED_SEGMENTS_NUM,
    UNSUPERVISED_CLUSTERS_NUM,
    UNSUPERVISED_WANDB_PROJECT,
    UNSUPERVISED_WANDB_ENTITY,
    UNSUPERVISED_WANDB_MODE,
    UNSUPERVISED_WANDB_KEY,
    UNSUPERVISED_WANDB_TAG,
    UNSUPERVISED_USE_WANDB_DISABLED,
    UNSUPERVISED_PYTHON_EXECUTABLE,
    UNSUPERVISED_OVERWRITE_DATASET,
    UNSUPERVISED_OVERWRITE_CONFIGS,
    UNSUPERVISED_BATCH_SIZE,

    UNSUPERVISED_NORM,
    UNSUPERVISED_INV,
    UNSUPERVISED_GAUSS_BLUR,
    UNSUPERVISED_GAUSS_TETA,
    UNSUPERVISED_HIST_EQ,
    UNSUPERVISED_IMAGE_DOWNSAMPLE_FACTOR,
    UNSUPERVISED_C_DINO,
    UNSUPERVISED_C_SSD_KNN,
    UNSUPERVISED_C_VAR_KNN,
    UNSUPERVISED_C_POS_KNN,
    UNSUPERVISED_C_SSD,
    UNSUPERVISED_C_NCC,
    UNSUPERVISED_C_LNCC,
    UNSUPERVISED_C_SSIM,
    UNSUPERVISED_C_MI,
    UNSUPERVISED_C_SAM,
)

from src.data_loading.h5_utils import list_h5_files, read_h5_meta, guess_keys
from src.utils.paths import source_sweep_path, unsupervised_run_dir
from src.utils.logging import log_io


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_case_names(case_names):
    if case_names is None:
        return None

    if isinstance(case_names, str):
        return [case_names]

    return list(case_names)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd, cwd=None, env=None, check=True):
    print("============================================================")
    print("[CMD]", " ".join(str(x) for x in cmd))

    if cwd is not None:
        print("[CWD]", cwd)

    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=False,
    )

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")

    return result.returncode


def yaml_list(values):
    return "[" + ", ".join(str(v) for v in values) + "]"


def write_text_if_needed(path: Path, content: str):
    path = Path(path)

    if path.exists() and not UNSUPERVISED_OVERWRITE_CONFIGS:
        print(f"[OK] config exists, not overwritten: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] wrote: {path}")


# ============================================================
# EXTERNAL REPO SETUP
# ============================================================

def ensure_external_repo():
    repo_dir = Path(UNSUPERVISED_REPO_DIR)

    if repo_dir.exists():
        print(f"[OK] External repo exists: {repo_dir}")
        return repo_dir

    if not UNSUPERVISED_ALLOW_GIT_CLONE:
        raise FileNotFoundError(
            f"Brak external repo: {repo_dir}\n"
            f"UNSUPERVISED_ALLOW_GIT_CLONE=False, więc runner nie będzie używał git.\n"
            f"Skopiuj repo ręcznie do:\n"
            f"  {repo_dir}"
        )

    ensure_dir(repo_dir.parent)

    run_command(
        [
            "git",
            "clone",
            UNSUPERVISED_REPO_URL,
            str(repo_dir),
        ],
        cwd=repo_dir.parent,
        check=True,
    )

    return repo_dir


def find_dinov2_root(repo_dir: Path):
    candidates = [
        repo_dir / "dinov2_with_attention_extraction",
        repo_dir / "deep-spectral-segmentation" / "dino2_models" / "dinov2_with_attention_extraction",
        repo_dir / "deep-spectral-segmentation" / "dinov2_with_attention_extraction",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def ensure_dinov2_repo(repo_dir: Path):
    dinov2_root = find_dinov2_root(repo_dir)

    if dinov2_root.exists():
        print(f"[OK] DINOv2 repo exists: {dinov2_root}")
    else:
        if not UNSUPERVISED_ALLOW_GIT_CLONE:
            raise FileNotFoundError(
                f"Brak DINOv2 repo: {dinov2_root}\n"
                f"UNSUPERVISED_ALLOW_GIT_CLONE=False, więc runner nie będzie używał git.\n"
                f"Skopiuj repo ręcznie do jednej z lokalizacji, np.:\n"
                f"  {repo_dir / 'dinov2_with_attention_extraction'}"
            )

        run_command(
            [
                "git",
                "clone",
                UNSUPERVISED_DINOV2_REPO_URL,
                str(dinov2_root),
            ],
            cwd=repo_dir,
            check=True,
        )

    init_paths = [
        dinov2_root / "__init__.py",
        dinov2_root / "dinov2" / "__init__.py",
        dinov2_root / "dinov2" / "models" / "__init__.py",
        dinov2_root / "dinov2" / "layers" / "__init__.py",
    ]

    for path in init_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        print(f"[OK] touch {path}")

    return dinov2_root


def get_dss_root(repo_dir: Path):
    dss_root = repo_dir / "deep-spectral-segmentation"

    if not dss_root.exists():
        raise FileNotFoundError(
            f"Brak katalogu deep-spectral-segmentation w external repo:\n"
            f"  {dss_root}"
        )

    return dss_root


# ============================================================
# H5 -> PNG DATASET
# ============================================================

def load_h5_image(path: Path, image_key=None):
    meta = read_h5_meta(path)

    if image_key is None:
        image_key, _ = guess_keys(meta)

    if image_key is None:
        raise RuntimeError(f"Nie znaleziono datasetu z obrazami w H5: {path}")

    with h5py.File(path, "r") as f:
        img = np.asarray(f[image_key])

    if img.ndim == 2:
        img = img[None, ...]

    if img.ndim != 3:
        raise ValueError(f"Oczekiwano obrazu (N,H,W), dostałem: {img.shape}")

    return img.astype(np.float32), image_key


def normalize_frame_to_uint8(frame):
    x = frame.astype(np.float32)

    if UNSUPERVISED_PNG_NORMALIZATION == "raw":
        return np.clip(x, 0, 255).astype(np.uint8)

    if UNSUPERVISED_PNG_NORMALIZATION == "minmax":
        mn = float(np.nanmin(x))
        mx = float(np.nanmax(x))

        if mx - mn < 1e-8:
            return np.zeros_like(x, dtype=np.uint8)

        y = (x - mn) / (mx - mn) * 255.0
        return np.clip(y, 0, 255).astype(np.uint8)

    # percentile
    lo = float(np.percentile(x, UNSUPERVISED_P_LOW))
    hi = float(np.percentile(x, UNSUPERVISED_P_HIGH))

    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)

    y = (np.clip(x, lo, hi) - lo) / (hi - lo) * 255.0

    return np.clip(y, 0, 255).astype(np.uint8)


def center_crop_to_multiple(frame, multiple=14):
    h, w = frame.shape[:2]

    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple

    if new_h <= 0 or new_w <= 0:
        raise ValueError(
            f"Nie można przyciąć obrazu shape={frame.shape} do wielokrotności {multiple}"
        )

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    return frame[top:top + new_h, left:left + new_w]
    


def get_external_dataset_dir(repo_dir: Path):
    return repo_dir / "data" / UNSUPERVISED_DATASET_NAME / UNSUPERVISED_CASE_NAME


def write_list_file(path: Path, filenames):
    with open(path, "w", encoding="utf-8") as f:
        for name in filenames:
            f.write(f"{name}\n")


def prepare_dataset_from_h5(
    repo_dir: Path,
    case_name: str,
    source="prepared",
    denoising_method=None,
):
    h5_path = source_sweep_path(case_name, source, denoising_method)

    dataset_dir = get_external_dataset_dir(repo_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    lists_dir = dataset_dir / "lists"

    if dataset_dir.exists() and UNSUPERVISED_OVERWRITE_DATASET:
        shutil.rmtree(dataset_dir)

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    lists_dir.mkdir(parents=True, exist_ok=True)

    img, resolved_key = load_h5_image(
        h5_path,
        image_key=UNSUPERVISED_IMAGE_KEY,
    )

    n_frames = img.shape[0]

    start = int(UNSUPERVISED_FRAME_START)
    end = n_frames if UNSUPERVISED_FRAME_END is None else min(int(UNSUPERVISED_FRAME_END), n_frames)
    step = int(UNSUPERVISED_FRAME_STEP)

    filenames = []

    print("============================================================")
    print("[INFO] Preparing unsupervised dataset")
    print(f"[INFO] case_name      : {case_name}")
    print(f"[INFO] source h5      : {h5_path}")
    print(f"[INFO] image key      : {resolved_key}")
    print(f"[INFO] image shape    : {img.shape}")
    print(f"[INFO] target dataset : {dataset_dir}")
    print(f"[INFO] frames         : {start}:{end}:{step}")

    out_idx = 0

    for frame_idx in range(start, end, step):
        frame = img[frame_idx]
    
        # DINOv2 ViT-S/14 wymaga zgodności z siatką patchy 14x14.
        # Dla Twoich danych 697x397 daje to crop do 686x392.
        frame = center_crop_to_multiple(frame, multiple=14)
    
        frame_u8 = normalize_frame_to_uint8(frame)
    
        filename = f"frame_{out_idx:06d}.png"
        out_path = images_dir / filename
    
        Image.fromarray(frame_u8, mode="L").save(out_path)

        filenames.append(filename)
        out_idx += 1

        if out_idx % 50 == 0:
            print(f"[INFO] exported frames: {out_idx}")

    write_list_file(lists_dir / "images.txt", filenames)
    write_list_file(lists_dir / "all.txt", filenames)
    write_list_file(lists_dir / "train.txt", filenames)
    write_list_file(lists_dir / "val.txt", filenames)
    write_list_file(lists_dir / "test.txt", filenames)

    print(f"[OK] exported PNG frames: {len(filenames)}")
    print(f"[OK] dataset dir: {dataset_dir}")

    return dataset_dir


# ============================================================
# CONFIG GENERATION
# ============================================================

def get_config_roots(repo_dir: Path):
    """
    Pipeline ma dekorator:

        @hydra.main(config_path="../../configs", config_name="defaults")

    Plik pipeline_sweep_subfolders.py jest w:
        deep-spectral-segmentation/pipeline/

    Czyli ../../configs wskazuje na:
        external/UnsupervisedSegmentor4Ultrasound/configs/

    Dlatego generujemy configi tylko tam.
    """
    config_root = repo_dir / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    return [config_root]

def generate_main_defaults_config(config_root: Path, output_dir: Path):
    """
    Główny config Hydry:
      configs/defaults.yaml

    Pipeline ma:
      @hydra.main(config_path="../../configs", config_name="defaults")

    Najważniejsze ustawienia:
      - dataset_type: single
      - norm: string, np. "none", nie True/False
      - loader.batch_size: 1, bo eigen step wymaga B == 1
      - clustering1: kmeans_eigen, bo extract.py nie obsługuje "kmeans"
      - image_downsample_factor: 4, żeby uniknąć OOM
    """
    norm_value = str(UNSUPERVISED_NORM)

    allowed_norms = {"imagenet", "custom", "custom_global", "none"}
    if norm_value not in allowed_norms:
        raise ValueError(
            f"UNSUPERVISED_NORM={UNSUPERVISED_NORM!r} jest niepoprawne. "
            f"Dozwolone wartości: {sorted(allowed_norms)}"
        )

    segments_default = int(UNSUPERVISED_SEGMENTS_NUM[0])
    clusters_default = int(UNSUPERVISED_CLUSTERS_NUM[0])

    content = f"""defaults:
  - dataset: {UNSUPERVISED_CASE_NAME}
  - sweep: {UNSUPERVISED_CASE_NAME}_clusters
  - wandb: defaults
  - vis: selected
  - pipeline_steps: defaults
  - _self_

custom_path_to_save_data: "{output_dir}"
norm: {norm_value}
inv: {str(UNSUPERVISED_INV).lower()}
gauss_blur: {str(UNSUPERVISED_GAUSS_BLUR).lower()}
gauss_teta: {float(UNSUPERVISED_GAUSS_TETA)}
hist_eq: {str(UNSUPERVISED_HIST_EQ).lower()}
preprocessed_data: raw

model:
  name: dinov2_vits14
  checkpoint: ""

loader:
  batch_size: {UNSUPERVISED_BATCH_SIZE}

spectral_clustering:
  K: {segments_default}
  which_matrix: laplacian
  which_color_matrix: none
  which_features: k
  normalize: true
  threshold_at_zero: true
  lapnorm: true
  image_downsample_factor: {UNSUPERVISED_IMAGE_DOWNSAMPLE_FACTOR}
  image_color_lambda: 0.0
  multiprocessing: false
  C_dino: {UNSUPERVISED_C_DINO}
  C_ssd_knn: {UNSUPERVISED_C_SSD_KNN}
  C_var_knn: {UNSUPERVISED_C_VAR_KNN}
  C_pos_knn: {UNSUPERVISED_C_POS_KNN}
  max_knn_neigbors: 10
  C_ssd: {UNSUPERVISED_C_SSD}
  C_ncc: {UNSUPERVISED_C_NCC}
  C_lncc: {UNSUPERVISED_C_LNCC}
  C_ssim: {UNSUPERVISED_C_SSIM}
  C_mi: {UNSUPERVISED_C_MI}
  C_sam: {UNSUPERVISED_C_SAM}
  patch_size: 5
  aff_sigma: 0.1
  distance_weight1: 1.0
  distance_weight2: 1.0
  use_transform: false

multi_region_segmentation:
  adaptive: false
  non_adaptive_num_segments: {segments_default}
  infer_bg_index: false
  clustering1: kmeans_eigen
  num_eigenvectors: {segments_default}
  multiprocessing: false

bbox:
  num_clusters: {clusters_default}
  num_erode: 0
  num_dilate: 0
  skip_bg_index: false
  downsample_factor: 1
  C_pos: 1.0
  C_mask: 1.0
  feat_comb_method: concat
  apply_mask: true
  seed: 1
  pca_dim: 50
  clustering: kmeans
  should_use_siamese: false
  should_use_ae: false
  is_sparse_graph: false
  spectral_n_nbg: 10

crf:
  num_classes: {clusters_default}
  downsample_factor: 1
  multiprocessing: false
  w1: 15
  alpha: 7
  beta: 10
  w2: 5
  gamma: 5
  it: 10

precomputed:
  mode: ""
  features: ""
  eig: ""
  multi_region_segmentation: ""
  crf_multi_region: ""
  bboxes: ""
  bbox_features: ""
  bbox_clusters: ""
  segmaps: ""
  crf_segmaps: ""

eval:
  eval_per_image: false
  iou_thresh: 0.5
  void_label: 255
  vis_rand_k: 20
"""

    write_text_if_needed(
        config_root / "defaults.yaml",
        content,
    )


def generate_pipeline_steps_defaults(config_root: Path):
    content = """dino_features: true
eigen: true
segments: true
bbox: true
bbox_features: true
clusters: true
sem_segm: true
crf_segm: true
crf_multi_region: false
eval: false
"""

    write_text_if_needed(
        config_root / "pipeline_steps" / "defaults.yaml",
        content,
    )


def generate_wandb_defaults(config_root: Path):
    content = f"""setup:
    project: {UNSUPERVISED_WANDB_PROJECT}
    entity: {UNSUPERVISED_WANDB_ENTITY}
    mode: disabled
key: ""
tag: '{UNSUPERVISED_WANDB_TAG}'
watch:
    log: all
    log_freq: 1
mode: disabled
"""
    write_text_if_needed(
        config_root / "wandb" / "defaults.yaml",
        content,
    )


def generate_sweep_config(config_root: Path):
    sweep_name = f"{UNSUPERVISED_CASE_NAME}_clusters"

    content = f"""name: {sweep_name}
seg_for_eval: ['segmaps']
method: grid
count: 1
simple: True
sweep_id: null
config:
    segments_num: {yaml_list(UNSUPERVISED_SEGMENTS_NUM)}
    clusters_num: {yaml_list(UNSUPERVISED_CLUSTERS_NUM)}

    crf:
        num_classes: {yaml_list(UNSUPERVISED_CLUSTERS_NUM)}
        w1: [15]
        alpha: [7]
        beta: [10]
        w2: [5]
        gamma: [5]
        it: [10]
"""
    write_text_if_needed(
        config_root / "sweep" / f"{sweep_name}.yaml",
        content,
    )

    return sweep_name


def generate_dataset_config(config_root: Path):
    dataset_root = f"{UNSUPERVISED_DATASET_NAME}/{UNSUPERVISED_CASE_NAME}"

    content = f"""name: {UNSUPERVISED_CASE_NAME}
dataset_root: {dataset_root}
dataset_type: single
images_root: images
list: lists/images.txt
gt_dir: labels
pred_dir: ""
n_classes: {UNSUPERVISED_N_CLASSES}
features_dir: ""
preprocessed_dir: ""
derained_dir: ""
eigenseg_dir: ""
"""

    write_text_if_needed(
        config_root / "dataset" / f"{UNSUPERVISED_CASE_NAME}.yaml",
        content,
    )

    return UNSUPERVISED_CASE_NAME


def generate_vis_selected_config(config_root: Path):
    content = """segmaps: true
eigen: false
crf_segmaps: true
multiregion_segmaps: false
crf_multi_region: false
dino_attn_maps: false
"""
    write_text_if_needed(
        config_root / "vis" / "selected.yaml",
        content,
    )


def generate_all_configs(repo_dir: Path, output_dir: Path):
    config_roots = get_config_roots(repo_dir)

    sweep_config_name = f"{UNSUPERVISED_CASE_NAME}_clusters"
    dataset_config_name = UNSUPERVISED_CASE_NAME

    print("============================================================")
    print("[INFO] Config roots:")
    for root in config_roots:
        print("  -", root)

    for config_root in config_roots:
        generate_main_defaults_config(config_root, output_dir)
        generate_pipeline_steps_defaults(config_root)
        generate_wandb_defaults(config_root)
        sweep_config_name = generate_sweep_config(config_root)
        dataset_config_name = generate_dataset_config(config_root)
        generate_vis_selected_config(config_root)

    run_script = generate_run_pipeline_script(
        repo_dir=repo_dir,
        dataset_config_name=dataset_config_name,
        sweep_config_name=sweep_config_name,
    )

    return {
        "config_roots": [str(x) for x in config_roots],
        "sweep_config_name": sweep_config_name,
        "dataset_config_name": dataset_config_name,
        "run_script": str(run_script),
    }


def generate_run_pipeline_script(repo_dir: Path, dataset_config_name: str, sweep_config_name: str):
    if UNSUPERVISED_USE_WANDB_DISABLED:
        wandb_exports = """export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_API_KEY=
"""
    else:
        wandb_exports = f"""export WANDB_MODE={UNSUPERVISED_WANDB_MODE}
export WANDB_DISABLED=false
export WANDB_API_KEY={UNSUPERVISED_WANDB_KEY}
"""

    script = f"""#!/usr/bin/env bash
set -e

cd deep-spectral-segmentation

{wandb_exports}
export WANDB_CONFIG_DIR=/tmp/
export WANDB_CACHE_DIR=/tmp/
export WANDB_AGENT_MAX_INITIAL_FAILURE=20
export WANDB__SERVICE_WAIT=600
export XFORMERS_DISABLED=True

{UNSUPERVISED_PYTHON_EXECUTABLE} -m pipeline.pipeline_sweep_subfolders \\
    vis=selected \\
    pipeline_steps=defaults \\
    pipeline_steps.crf_segm=true \\
    pipeline_steps.crf_multi_region=false \\
    dataset={dataset_config_name} \\
    wandb=defaults \\
    wandb.tag={UNSUPERVISED_WANDB_TAG} \\
    sweep={sweep_config_name}
"""

    path = repo_dir / "run_pipeline.sh"
    write_text_if_needed(path, script)

    try:
        path.chmod(0o755)
    except Exception:
        pass

    return path


# ============================================================
# RUN EXTERNAL PIPELINE
# ============================================================

def run_external_pipeline(repo_dir: Path, dinov2_root: Path):
    dss_root = get_dss_root(repo_dir)

    env = os.environ.copy()

    pythonpath_parts = [
        str(dss_root),
        str(dinov2_root),
    ]

    old_pythonpath = env.get("PYTHONPATH", "")

    if old_pythonpath:
        pythonpath_parts.append(old_pythonpath)

    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    if UNSUPERVISED_USE_WANDB_DISABLED:
        env["WANDB_MODE"] = "disabled"
        env["WANDB_DISABLED"] = "true"
        env["WANDB_API_KEY"] = ""
    else:
        env["WANDB_MODE"] = str(UNSUPERVISED_WANDB_MODE)
        env["WANDB_DISABLED"] = "false"
        env["WANDB_API_KEY"] = str(UNSUPERVISED_WANDB_KEY)

    env["WANDB_CONFIG_DIR"] = "/tmp/"
    env["WANDB_CACHE_DIR"] = "/tmp/"
    env["WANDB_AGENT_MAX_INITIAL_FAILURE"] = "20"
    env["WANDB__SERVICE_WAIT"] = "600"
    env["XFORMERS_DISABLED"] = "True"

    print("============================================================")
    print("[INFO] Running external unsupervised pipeline")
    print(f"[INFO] repo       : {repo_dir}")
    print(f"[INFO] dss_root   : {dss_root}")
    print(f"[INFO] dinov2     : {dinov2_root}")
    print(f"[INFO] PYTHONPATH : {env['PYTHONPATH']}")
    print(f"[INFO] WANDB_MODE : {env.get('WANDB_MODE')}")
    print(f"[INFO] WANDB_DISABLED: {env.get('WANDB_DISABLED')}")
    print("============================================================")

    run_command(
        ["bash", "run_pipeline.sh"],
        cwd=repo_dir,
        env=env,
        check=True,
    )

    print("[OK] Unsupervised pipeline finished")


def make_run_output_dir(case_name, source, denoising_method):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return (
        unsupervised_run_dir(case_name, source, denoising_method)
        / f"{UNSUPERVISED_WANDB_TAG}-{timestamp}"
    )
        
# ============================================================
# MAIN ENTRY
# ============================================================

def run_unsupervised_segmentation(
    case_names,
    source="prepared",
    denoising_method=None,
):
    case_names = normalize_case_names(case_names)

    if not case_names:
        raise RuntimeError("Brak case_names do unsupervised segmentation.")

    repo_dir = ensure_external_repo()
    dinov2_root = ensure_dinov2_repo(repo_dir)
    outputs = []
    for case_name in case_names:
        output_dir = make_run_output_dir(
            case_name, source, denoising_method
        )
        log_io(
            "UNSUPERVISED SEGMENTATION",
            {"sweep": source_sweep_path(case_name, source, denoising_method)},
            {"run": output_dir},
        )
        prepare_dataset_from_h5(
            repo_dir,
            case_name,
            source,
            denoising_method,
        )
        generate_all_configs(repo_dir, output_dir)
        run_external_pipeline(repo_dir, dinov2_root)
        outputs.append(output_dir)
    return outputs
