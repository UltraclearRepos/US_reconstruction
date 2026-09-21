"""Canonical artifact paths for all pipeline stages."""

from pathlib import Path

from config import (
    PREPARED_DIR,
    SEGMENTED_DIR,
    RECONSTRUCTED_INTENSITY_DIR,
    RECONSTRUCTED_SEGMENTATION_DIR,
    GROUND_TRUTH_DIR,
    PREDICTED_SURFACES_DIR,
    GROUND_TRUTH_SURFACES_DIR,
    UNSUPERVISED_RUNS_DIR,
)


def prepared_sweep_path(case_name):
    return PREPARED_DIR / case_name / "sweep.h5"


def segmentation_dir(case_name, model_name):
    return SEGMENTED_DIR / case_name / model_name


def segmentation_path(case_name, model_name):
    return segmentation_dir(case_name, model_name) / "labels.h5"


def intensity_reconstruction_dir(case_name, method):
    return RECONSTRUCTED_INTENSITY_DIR / case_name / method


def intensity_reconstruction_path(case_name, method):
    return intensity_reconstruction_dir(case_name, method) / "volume.npz"


def segmentation_reconstruction_dir(case_name, model_name, method="voxel_nearest"):
    return RECONSTRUCTED_SEGMENTATION_DIR / case_name / model_name / method


def segmentation_reconstruction_path(case_name, model_name, method="voxel_nearest"):
    return segmentation_reconstruction_dir(case_name, model_name, method) / "labels.npz"


def predicted_surface_dir(case_name, model_name, method):
    return PREDICTED_SURFACES_DIR / case_name / model_name / method


def ground_truth_volume_path(case_name):
    return GROUND_TRUTH_DIR / case_name / "volume.npz"


def ground_truth_labels_path(case_name):
    return GROUND_TRUTH_DIR / case_name / "labels.npz"


def ground_truth_surface_dir(case_name):
    return GROUND_TRUTH_SURFACES_DIR / case_name


def unsupervised_run_dir(case_name):
    return UNSUPERVISED_RUNS_DIR / case_name


def list_cases(root):
    if not Path(root).exists():
        return []
    return sorted(path.name for path in Path(root).iterdir() if path.is_dir())
