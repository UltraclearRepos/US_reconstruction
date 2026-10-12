"""Volume loading shared by viewers."""

from pathlib import Path

import numpy as np


def load_volume(path, dtype=np.float32):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            for key in ("volume", "labels", "final_volume", "appearance_volume", "mask_volume"):
                if key in data:
                    volume = data[key]
                    break
            else:
                volume = data[data.files[0]]
            spacing = tuple(float(x) for x in data.get("voxel_size", (1, 1, 1)))
        return np.asarray(volume, dtype=dtype), spacing

    raise ValueError(f"Unsupported volume: {path}")
