from pathlib import Path

import numpy as np

from config import (
    GROUND_TRUTH_SURFACES_DIR,
    NAPARI_NDISPLAY,
    PREDICTED_SURFACES_DIR,
    SHOW_GROUND_TRUTH_VOLUME,
    SHOW_RECONSTRUCTION,
    SHOW_SEGMENTATION_LABELS,
    SHOW_SURFACES,
)
from src.utils.paths import (
    ground_truth_labels_path,
    ground_truth_volume_path,
    intensity_reconstruction_path,
    predicted_surface_dir,
    segmentation_reconstruction_path,
)
from src.visualization.common import load_volume
from src.utils.logging import log_io


# Display-only offset in viewer coordinates; does not affect saved data or metrics.
GT_SHIFT = (0, 0, 20)


def add_surface(viewer, path, translate=(0, 0, 0)):
    data = np.load(path, allow_pickle=True)
    viewer.add_surface(
        (data["verts"], data["faces"], data["values"]),
        name=path.stem,
        opacity=0.6,
        shading="smooth",
        translate=translate,
    )


def visualize_case_napari(
    case_name,
    model_name=None,
    intensity_methods=("voxel_nearest",),
    segmentation_method="voxel_nearest",
):
    import napari

    log_io(
        "VISUALIZATION napari",
        {
            "case": case_name,
            "intensity root": intensity_reconstruction_path(
                case_name, intensity_methods[0]
            ).parent,
            "segmentation": segmentation_reconstruction_path(
                case_name, model_name, segmentation_method
            ),
        },
        {"display": "napari window"},
    )
    viewer = napari.Viewer(ndisplay=NAPARI_NDISPLAY)

    if SHOW_RECONSTRUCTION:
        for method in intensity_methods:
            path = intensity_reconstruction_path(
                case_name, method
            )
            if path.exists():
                volume, spacing = load_volume(path)
                viewer.add_image(
                    volume,
                    name=f"intensity/{method}",
                    scale=spacing,
                    colormap="gray",
                    rendering="mip",
                )

    predicted_labels = segmentation_reconstruction_path(
        case_name,
        model_name,
        segmentation_method,
    )
    if SHOW_SEGMENTATION_LABELS and predicted_labels.exists():
        labels, spacing = load_volume(predicted_labels)
        viewer.add_labels(
            labels.astype(np.int32),
            name=f"predicted/{model_name}",
            scale=spacing,
        )

    if SHOW_GROUND_TRUTH_VOLUME:
        path = ground_truth_volume_path(case_name)
        if path.exists():
            volume, spacing = load_volume(path)
            viewer.add_image(
                volume,
                name="ground_truth/volume",
                scale=spacing,
                translate=GT_SHIFT,
                colormap="green",
                rendering="mip",
                opacity=0.35,
            )

    if SHOW_SEGMENTATION_LABELS:
        path = ground_truth_labels_path(case_name)
        if path.exists():
            labels, spacing = load_volume(path)
            viewer.add_labels(
                labels.astype(np.int32),
                name="ground_truth/labels",
                scale=spacing,
                translate=GT_SHIFT,
                opacity=0.45,
            )

    if SHOW_SURFACES:
        surface_dirs = [
            GROUND_TRUTH_SURFACES_DIR / case_name,
            predicted_surface_dir(
                case_name,
                model_name,
                segmentation_method,
            ),
        ]
        for directory in surface_dirs:
            shift = (
                GT_SHIFT
                if directory == GROUND_TRUTH_SURFACES_DIR / case_name
                else (0, 0, 0)
            )
            for path in sorted(Path(directory).glob("*_surface.npz")):
                add_surface(viewer, path, translate=shift)

    napari.run()
