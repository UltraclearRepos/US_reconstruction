"""Load inference-only U-Net weights from a Lightning checkpoint."""

from pathlib import Path, PosixPath

import torch

from src.segmentation.model import UNet


def _required(mapping, key, location):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"Checkpoint is missing {location}.{key}")
    return mapping[key]


def load_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint_path}")

    # The saved training config contains pathlib.PosixPath values. Keep
    # weights_only enabled and allow only this standard-library value type.
    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    state_dict = _required(checkpoint, "state_dict", "checkpoint")
    hyperparameters = _required(
        checkpoint,
        "hyper_parameters",
        "checkpoint",
    )
    config = _required(
        hyperparameters,
        "config",
        "checkpoint.hyper_parameters",
    )
    model_config = _required(
        config,
        "model",
        "checkpoint.hyper_parameters.config",
    )
    data_config = _required(
        config,
        "data",
        "checkpoint.hyper_parameters.config",
    )

    image_size = _required(
        data_config,
        "image_size",
        "checkpoint.hyper_parameters.config.data",
    )
    if len(image_size) != 2:
        raise ValueError(f"Expected image_size=[H, W], got: {image_size}")

    in_channels = int(
        _required(
            model_config,
            "in_channels",
            "checkpoint.hyper_parameters.config.model",
        )
    )
    bilinear = bool(
        _required(
            model_config,
            "bilinear",
            "checkpoint.hyper_parameters.config.model",
        )
    )
    n_classes = int(
        _required(
            hyperparameters,
            "num_classes",
            "checkpoint.hyper_parameters",
        )
    )
    class_name_mappings = _required(
        hyperparameters,
        "class_name_mappings",
        "checkpoint.hyper_parameters",
    )
    if len(class_name_mappings) != 1:
        raise ValueError(
            "Inference expects exactly one segmentation task in "
            f"class_name_mappings, got: {sorted(class_name_mappings)}"
        )
    task_name, class_names = next(iter(class_name_mappings.items()))

    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        raise ValueError(f"Checkpoint contains no model.* weights: {checkpoint_path}")

    model = UNet(
        in_channels=in_channels,
        n_classes=n_classes,
        bilinear=bilinear,
    )
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()

    settings = {
        "image_size": tuple(int(value) for value in image_size),
        "in_channels": in_channels,
        "n_classes": n_classes,
        "task_name": task_name,
        "class_names": class_names,
    }
    return model, settings
