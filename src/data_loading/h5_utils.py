"""Small helpers for inspecting H5 files."""

from pathlib import Path
import h5py

LIKELY_IMAGE_KEYS = [
    "img", "image", "images", "frames", "sweep", "sweeps", "volume"
]


def list_h5_files(root):
    root = Path(root)
    return sorted(list(root.rglob("*.h5")) + list(root.rglob("*.hdf5")))


def read_h5_meta(path):
    meta = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                meta[name] = (obj.shape, str(obj.dtype))
        f.visititems(visit)
    return meta


def guess_keys(meta):
    image_key = None
    pose_key = None

    for pref in LIKELY_IMAGE_KEYS:
        for key, (shape, _) in meta.items():
            if pref in key.lower() and len(shape) in (2, 3):
                image_key = key
                break
        if image_key is not None:
            break

    if image_key is None:
        for key, (shape, _) in meta.items():
            if len(shape) in (2, 3):
                image_key = key
                break

    for key, (shape, _) in meta.items():
        if len(shape) == 3 and shape[-2:] == (4, 4):
            pose_key = key
            break

    return image_key, pose_key
