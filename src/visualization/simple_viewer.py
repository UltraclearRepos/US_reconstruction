from src.utils.paths import (
    ground_truth_labels_path,
    ground_truth_volume_path,
    intensity_reconstruction_path,
    segmentation_reconstruction_path,
)
from src.visualization.common import load_volume
from src.visualization.qt_environment import prepare_qt_environment
from src.utils.logging import log_io


def visualize_case_simple(
    case_name,
    model_name=None,
    intensity_method="voxel_nearest",
    segmentation_method="voxel_nearest",
):
    prepare_qt_environment()
    import matplotlib.pyplot as plt

    paths = [
        ("intensity reconstruction", intensity_reconstruction_path(
            case_name, intensity_method
        )),
        ("segmentation reconstruction", segmentation_reconstruction_path(
            case_name, model_name, segmentation_method
        )),
        ("ground truth volume", ground_truth_volume_path(case_name)),
        ("ground truth labels", ground_truth_labels_path(case_name)),
    ]
    items = [(name, load_volume(path)[0]) for name, path in paths if path.exists()]
    log_io(
        "VISUALIZATION simple",
        {name: path for name, path in paths if path.exists()},
        {"display": "matplotlib window"},
    )
    if not items:
        raise RuntimeError(f"No volumes found for case: {case_name}")

    figure, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 5))
    axes = [axes] if len(items) == 1 else axes
    for axis, (title, volume) in zip(axes, items):
        z = volume.shape[0] // 2
        axis.imshow(volume[z], cmap="gray")
        axis.set_title(f"{title} | z={z}")
        axis.axis("off")
    plt.tight_layout()
    plt.show()
