from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    UNSUPERVISED_FRAME_END,
    UNSUPERVISED_FRAME_START,
    UNSUPERVISED_FRAME_STEP,
    UNSUPERVISED_IMPORT_MODEL_NAME,
    UNSUPERVISED_IMPORT_USE_CRF,
)
from src.data_loading.sweep import load_sweep_h5, save_sweep_h5
from src.utils.paths import (
    segmentation_path,
    source_sweep_path,
    unsupervised_run_dir,
)
from src.utils.logging import log_io


def normalize_list(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return list(values)


def center_crop_info(shape_hw, multiple=14):
    height, width = shape_hw
    crop_height = height // multiple * multiple
    crop_width = width // multiple * multiple
    return (
        (height - crop_height) // 2,
        (width - crop_width) // 2,
        crop_height,
        crop_width,
    )


def has_segmentation_outputs(path):
    path = Path(path)
    return any(
        candidate.exists()
        for candidate in (
            path / "segmaps",
            path / "crf_segmaps",
            path / "semantic_segmentations" / "laplacian" / "segmaps",
            path / "semantic_segmentations" / "laplacian" / "crf_segmaps",
        )
    )


def find_latest_run(case_name, source, denoising_method):
    root = unsupervised_run_dir(case_name, source, denoising_method)
    candidates = [path for path in root.rglob("*") if path.is_dir() and has_segmentation_outputs(path)]
    if has_segmentation_outputs(root):
        candidates.append(root)
    if not candidates:
        raise FileNotFoundError(f"No unsupervised output found under: {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_run_dir(run_dir):
    run_dir = Path(run_dir)
    if has_segmentation_outputs(run_dir):
        return run_dir
    candidates = [
        path
        for path in run_dir.rglob("*")
        if path.is_dir() and has_segmentation_outputs(path)
    ]
    if not candidates:
        raise FileNotFoundError(f"No unsupervised output found under: {run_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def get_segmentation_png_dir(run_dir, use_crf=True):
    run_dir = Path(run_dir)
    candidates = []
    if run_dir.name in {"segmaps", "crf_segmaps"}:
        candidates.append(run_dir)
    candidates.extend(
        [
            run_dir / "semantic_segmentations" / "laplacian" / (
                "crf_segmaps" if use_crf else "segmaps"
            ),
            run_dir / ("crf_segmaps" if use_crf else "segmaps"),
            run_dir / "semantic_segmentations" / "laplacian" / "segmaps",
            run_dir / "segmaps",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No segmentation PNG directory in: {run_dir}")


def selected_frame_indices(frame_count):
    end = frame_count if UNSUPERVISED_FRAME_END is None else min(
        int(UNSUPERVISED_FRAME_END), frame_count
    )
    return list(
        range(
            int(UNSUPERVISED_FRAME_START),
            end,
            int(UNSUPERVISED_FRAME_STEP),
        )
    )


def read_unsupervised_png_stack(seg_dir, source_shape):
    png_files = sorted(Path(seg_dir).glob("*.png"))
    frame_count, height, width = source_shape
    frame_indices = selected_frame_indices(frame_count)
    labels = np.zeros(source_shape, dtype=np.uint16)
    top, left, crop_height, crop_width = center_crop_info((height, width))

    for png_path, frame_index in zip(png_files, frame_indices):
        label = np.asarray(Image.open(png_path))
        if label.ndim == 3:
            label = label[:, :, 0]
        if label.shape != (crop_height, crop_width):
            label = np.asarray(
                Image.fromarray(label).resize(
                    (crop_width, crop_height),
                    resample=Image.NEAREST,
                )
            )
        labels[
            frame_index,
            top : top + crop_height,
            left : left + crop_width,
        ] = label.astype(np.uint16)
    return labels


def import_one_unsupervised_output(
    case_name,
    source="prepared",
    denoising_method=None,
    run_dir=None,
    use_crf=UNSUPERVISED_IMPORT_USE_CRF,
    model_name=UNSUPERVISED_IMPORT_MODEL_NAME,
):
    source_h5 = source_sweep_path(case_name, source, denoising_method)
    frames, poses, spacing_xy, _ = load_sweep_h5(source_h5)
    run_dir = (
        resolve_run_dir(run_dir)
        if run_dir
        else find_latest_run(case_name, source, denoising_method)
    )
    png_dir = get_segmentation_png_dir(run_dir, use_crf)
    labels = read_unsupervised_png_stack(png_dir, frames.shape)
    output_h5 = segmentation_path(
        case_name, source, denoising_method, model_name
    )
    log_io(
        "IMPORT UNSUPERVISED",
        {"segmentation PNGs": png_dir, "source sweep": source_h5},
        {"labels": output_h5},
    )
    save_sweep_h5(
        output_h5,
        labels,
        poses,
        spacing_xy,
        data_key="labels",
        dtype=np.uint16,
        attrs={
            "content_type": "segmentation_labels",
            "model_name": model_name,
            "source_h5": str(source_h5),
            "source_run": str(run_dir),
            "source_png_dir": str(png_dir),
            "classes": np.unique(labels).astype(int),
        },
    )
    print(f"[IMPORT UNSUPERVISED] {png_dir} -> {output_h5}")
    return output_h5


def import_unsupervised_outputs(
    case_names,
    source="prepared",
    denoising_method=None,
    run_dirs=None,
    model_name=UNSUPERVISED_IMPORT_MODEL_NAME,
):
    run_dirs = normalize_list(run_dirs)
    outputs = []
    for index, case_name in enumerate(normalize_list(case_names)):
        run_dir = run_dirs[index] if index < len(run_dirs) else None
        outputs.append(
            import_one_unsupervised_output(
                case_name,
                source,
                denoising_method,
                run_dir,
                model_name=model_name,
            )
        )
    return outputs
