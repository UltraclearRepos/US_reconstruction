import numpy as np
import torch

from config import (
    SEGMENTATION_DEVICE,
    SEGMENTATION_IMAGE_KEY,
    SEGMENTATION_MODEL_FILE,
    SEGMENTATION_MODEL_META_FILE,
    SEGMENTATION_MODEL_NAME,
    SEGMENTATION_MODELS_DIR,
    SEGMENTATION_NORMALIZATION,
)
from src.data_loading.sweep import load_sweep_h5, save_sweep_h5
from src.segmentation.common import (
    get_model_input_size,
    load_model_meta,
    normalize_frame_for_unet,
    resize_frame_float,
    resize_mask_nearest,
)
from src.utils.paths import segmentation_path, source_sweep_path
from src.utils.logging import log_io


def normalize_list(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return list(values)


def load_segmentation_model(model_name):
    model_dir = SEGMENTATION_MODELS_DIR / model_name
    model_path = model_dir / SEGMENTATION_MODEL_FILE
    meta_path = model_dir / SEGMENTATION_MODEL_META_FILE
    device = torch.device(
        SEGMENTATION_DEVICE
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[SEGMENT] loading {model_path} on {device}")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model, load_model_meta(meta_path), device, model_path


@torch.no_grad()
def predict_masks_for_sequence(frames, model, model_input_size, device):
    count, height, width = frames.shape
    masks = np.zeros((count, height, width), dtype=np.uint16)
    for index, frame in enumerate(frames):
        normalized = normalize_frame_for_unet(frame, SEGMENTATION_NORMALIZATION)
        resized = resize_frame_float(normalized, model_input_size)
        tensor = torch.from_numpy(resized[None, None]).to(device, torch.float32)
        prediction = torch.argmax(model(tensor), dim=1).squeeze().cpu().numpy()
        masks[index] = resize_mask_nearest(prediction, (height, width))
        if (index + 1) % 50 == 0 or index + 1 == count:
            print(f"[SEGMENT] frames {index + 1}/{count}")
    return masks


def segment_case(
    case_name,
    source="prepared",
    denoising_method=None,
    model_name=SEGMENTATION_MODEL_NAME,
    loaded_model=None,
):
    source_h5 = source_sweep_path(case_name, source, denoising_method)
    output_h5 = segmentation_path(case_name, source, denoising_method, model_name)
    log_io("SUPERVISED SEGMENTATION", {"sweep": source_h5}, {"labels": output_h5})
    model, meta, device, model_path = loaded_model or load_segmentation_model(model_name)
    frames, poses, spacing_xy, _ = load_sweep_h5(
        source_h5, image_key=SEGMENTATION_IMAGE_KEY
    )
    masks = predict_masks_for_sequence(
        frames, model, get_model_input_size(meta), device
    )
    classes = np.unique(masks).astype(int)
    save_sweep_h5(
        output_h5,
        masks,
        poses,
        spacing_xy,
        data_key="labels",
        dtype=np.uint16,
        attrs={
            "content_type": "segmentation_labels",
            "model_name": model_name,
            "model_path": str(model_path),
            "source_h5": str(source_h5),
            "classes": classes,
        },
    )
    print(f"[SEGMENT] {source_h5} -> {output_h5}; classes={classes.tolist()}")
    return output_h5


def segment_cases(
    case_names,
    source="prepared",
    denoising_method=None,
    model_name=SEGMENTATION_MODEL_NAME,
):
    loaded_model = load_segmentation_model(model_name)
    return [
        segment_case(
            case_name,
            source,
            denoising_method,
            model_name,
            loaded_model,
        )
        for case_name in normalize_list(case_names)
    ]
