from pathlib import Path

import h5py
import numpy as np

from src.data_loading.h5_utils import guess_keys, read_h5_meta


def load_sweep_h5(path, image_key=None):
    path = Path(path)
    meta = read_h5_meta(path)
    guessed_image_key, guessed_pose_key = guess_keys(meta)
    image_key = image_key or guessed_image_key
    pose_key = guessed_pose_key or "poses"

    with h5py.File(path, "r") as handle:
        frames = np.asarray(handle[image_key])
        poses = np.asarray(handle[pose_key], dtype=np.float32)
        dataset = handle[image_key]
        sx = float(dataset.attrs.get("spacing_x", 1.0))
        sy = float(dataset.attrs.get("spacing_y", 1.0))
        if "spacing" in dataset.attrs:
            sx, sy = (float(x) for x in dataset.attrs["spacing"][:2])
        elif "spacing_xy" in handle:
            sx, sy = (float(x) for x in np.asarray(handle["spacing_xy"])[:2])

    if frames.ndim == 2:
        frames = frames[None]
    if frames.shape[0] != poses.shape[0] and frames.shape[-1] == poses.shape[0]:
        frames = np.moveaxis(frames, -1, 0)
    if frames.ndim != 3 or frames.shape[0] != poses.shape[0]:
        raise ValueError(
            f"Expected frames (N,H,W) matching poses (N,4,4), "
            f"got {frames.shape} and {poses.shape}"
        )

    return frames.astype(np.float32), poses, (sx, sy), image_key


def save_sweep_h5(
    path,
    frames,
    poses,
    spacing_xy,
    data_key="img",
    attrs=None,
    dtype=np.float32,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sx, sy = spacing_xy

    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            data_key,
            data=np.asarray(frames, dtype=dtype),
            compression="gzip",
        )
        dataset.attrs["spacing_x"] = float(sx)
        dataset.attrs["spacing_y"] = float(sy)
        dataset.attrs["spacing"] = np.asarray([sx, sy], dtype=np.float32)
        handle.create_dataset(
            "poses",
            data=np.asarray(poses, dtype=np.float32),
            compression="gzip",
        )
        for key, value in (attrs or {}).items():
            try:
                handle.attrs[key] = value
            except TypeError:
                handle.attrs[key] = str(value)

    return path
