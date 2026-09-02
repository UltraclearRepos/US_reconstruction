"""Canonical artifact paths for all pipeline stages."""

from pathlib import Path

from config import (
    PREPARED_DIR,
    DENOISED_DIR,
    SEGMENTED_DIR,
    RECONSTRUCTED_INTENSITY_DIR,
    RECONSTRUCTED_SEGMENTATION_DIR,
    GROUND_TRUTH_DIR,
    PREDICTED_SURFACES_DIR,
    GROUND_TRUTH_SURFACES_DIR,
    UNSUPERVISED_RUNS_DIR,
)


def source_parts(source="prepared", denoising_method=None):
    if source == "prepared":
        return ("prepared",)
    if source == "denoised":
        if not denoising_method:
            raise ValueError("denoising_method is required for source='denoised'")
        return ("denoised", str(denoising_method))
    raise ValueError("source must be 'prepared' or 'denoised'")


def source_sweep_path(case_name, source="prepared", denoising_method=None):
    if source == "prepared":
        return PREPARED_DIR / case_name / "sweep.h5"
    return DENOISED_DIR / case_name / source_parts(source, denoising_method)[1] / "sweep.h5"


def denoised_sweep_path(case_name, method):
    return DENOISED_DIR / case_name / method / "sweep.h5"


def segmentation_dir(case_name, source, denoising_method, model_name):
    return SEGMENTED_DIR.joinpath(case_name, *source_parts(source, denoising_method), model_name)


def segmentation_path(case_name, source, denoising_method, model_name):
    return segmentation_dir(case_name, source, denoising_method, model_name) / "labels.h5"


def intensity_reconstruction_dir(case_name, source, denoising_method, method):
    return RECONSTRUCTED_INTENSITY_DIR.joinpath(
        case_name, *source_parts(source, denoising_method), method
    )


def intensity_reconstruction_path(case_name, source, denoising_method, method):
    return intensity_reconstruction_dir(case_name, source, denoising_method, method) / "volume.npz"


def segmentation_reconstruction_dir(
    case_name, source, denoising_method, model_name, method="voxel_nearest"
):
    return RECONSTRUCTED_SEGMENTATION_DIR.joinpath(
        case_name, *source_parts(source, denoising_method), model_name, method
    )


def segmentation_reconstruction_path(
    case_name, source, denoising_method, model_name, method="voxel_nearest"
):
    return segmentation_reconstruction_dir(
        case_name, source, denoising_method, model_name, method
    ) / "labels.npz"


def predicted_surface_dir(case_name, source, denoising_method, model_name, method):
    return PREDICTED_SURFACES_DIR.joinpath(
        case_name, *source_parts(source, denoising_method), model_name, method
    )


def ground_truth_volume_path(case_name):
    return GROUND_TRUTH_DIR / case_name / "volume.npz"


def ground_truth_labels_path(case_name):
    return GROUND_TRUTH_DIR / case_name / "labels.npz"


def ground_truth_surface_dir(case_name):
    return GROUND_TRUTH_SURFACES_DIR / case_name


def unsupervised_run_dir(case_name, source, denoising_method):
    return UNSUPERVISED_RUNS_DIR.joinpath(
        case_name, *source_parts(source, denoising_method)
    )


def list_cases(root):
    if not Path(root).exists():
        return []
    return sorted(path.name for path in Path(root).iterdir() if path.is_dir())
