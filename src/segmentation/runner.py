import numpy as np
import torch
from PIL import Image

from config import (
    SEGMENTATION_CHECKPOINT,
    SEGMENTATION_DEVICE,
    SEGMENTATION_IMAGE_KEY,
    SEGMENTATION_MODEL_NAME,
)
from src.data_loading.sweep import load_sweep_h5, save_sweep_h5
from src.segmentation.checkpoint import load_checkpoint
from src.segmentation.common import prepare_frame, resize_mask
from src.utils.logging import log_io
from src.utils.paths import prepared_sweep_path, segmentation_path


MASK_COLORS = np.asarray(
    [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
    ],
    dtype=np.float32,
)


def normalize_list(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return list(values)


def load_segmentation_model():
    device = torch.device(
        SEGMENTATION_DEVICE
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[SEGMENT] loading {SEGMENTATION_CHECKPOINT} on {device}")
    model, settings = load_checkpoint(SEGMENTATION_CHECKPOINT, device)
    print(
        f"[SEGMENT] image_size={settings['image_size']}, "
        f"in_channels={settings['in_channels']}, "
        f"classes={settings['n_classes']}, "
        f"task={settings['task_name']}"
    )
    return model, settings, device


@torch.inference_mode()
def predict_masks_for_sequence(frames, model, settings, device, images_dir=None):
    count, height, width = frames.shape
    masks = np.zeros((count, height, width), dtype=np.uint16)
    if images_dir is not None:
        images_dir.mkdir(parents=True, exist_ok=True)

    for index, frame in enumerate(frames):
        prepared = prepare_frame(frame, settings["image_size"])
        tensor = torch.from_numpy(prepared[None, None]).to(device, torch.float32)
        tensor = tensor.repeat(1, settings["in_channels"], 1, 1)

        outputs = model(tensor)
        task_name = settings["task_name"]
        if task_name not in outputs:
            raise KeyError(
                f"Model has no '{task_name}' output; available: {sorted(outputs)}"
            )
        prediction = outputs[task_name].argmax(dim=1)[0].cpu().numpy()
        masks[index] = resize_mask(prediction, (height, width))

        if images_dir is not None:
            low, high = np.percentile(frame[np.isfinite(frame)], (1.0, 99.0))
            if high - low < 1e-8:
                gray = np.zeros(frame.shape, dtype=np.uint8)
            else:
                gray = np.clip((frame - low) / (high - low), 0.0, 1.0)
                gray = np.nan_to_num(gray, nan=0.0)
                gray = np.rint(gray * 255.0).astype(np.uint8)

            overlay = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
            for class_id in np.unique(masks[index]):
                if class_id == 0:
                    continue
                pixels = masks[index] == class_id
                color = MASK_COLORS[(int(class_id) - 1) % len(MASK_COLORS)]
                overlay[pixels] = 0.55 * overlay[pixels] + 0.45 * color

            Image.fromarray(np.rint(overlay).astype(np.uint8), mode="RGB").save(
                images_dir / f"frame_{index:06d}.png"
            )

        processed = index + 1
        if processed % 50 == 0 or processed == count:
            print(f"[SEGMENT] frames {processed}/{count}")

    return masks


def segment_case(
    case_name,
    model_name=SEGMENTATION_MODEL_NAME,
    loaded_model=None,
):
    source_h5 = prepared_sweep_path(case_name)
    output_h5 = segmentation_path(case_name, model_name)
    images_dir = output_h5.parent / "images"
    log_io(
        "SUPERVISED SEGMENTATION",
        {"sweep": source_h5},
        {"labels": output_h5, "overlay images": images_dir},
    )

    model, settings, device = loaded_model or load_segmentation_model()
    frames, poses, spacing_xy, _ = load_sweep_h5(
        source_h5,
        image_key=SEGMENTATION_IMAGE_KEY,
    )
    masks = predict_masks_for_sequence(
        frames, model, settings, device, images_dir=images_dir
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
            "checkpoint": str(SEGMENTATION_CHECKPOINT),
            "task": settings["task_name"],
            "model_input_size": settings["image_size"],
            "classes": classes,
            "source_h5": str(source_h5),
        },
    )
    print(f"[SEGMENT] {source_h5} -> {output_h5}; classes={classes.tolist()}")
    return output_h5


def segment_cases(
    case_names,
    model_name=SEGMENTATION_MODEL_NAME,
):
    loaded_model = load_segmentation_model()
    return [
        segment_case(
            case_name,
            model_name,
            loaded_model,
        )
        for case_name in normalize_list(case_names)
    ]
