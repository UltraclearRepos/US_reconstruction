"""Volume loading shared by viewers and evaluation."""

from pathlib import Path

import numpy as np


def load_volume(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        for key in ("volume", "labels", "final_volume", "appearance_volume", "mask_volume"):
            if key in data:
                volume = data[key]
                break
        else:
            volume = data[data.files[0]]
        spacing = tuple(float(x) for x in data.get("voxel_size", (1, 1, 1)))
        return np.asarray(volume, dtype=np.float32), spacing

    raise ValueError(f"Unsupported volume: {path}")
