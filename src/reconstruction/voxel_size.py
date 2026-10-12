"""Resolve reconstruction voxel spacing in XYZ order, in millimetres."""

import numpy as np

from src.utils.paths import ground_truth_labels_path, ground_truth_volume_path


def resolve_voxel_size(case_name, voxel_size):
    if voxel_size is not None:
        return voxel_size

    candidates = (ground_truth_labels_path(case_name), ground_truth_volume_path(case_name))
    for path in candidates:
        if not path.exists():
            continue
        # Read only spacing metadata; do not decompress the GT volume.
        with np.load(path, allow_pickle=False) as data:
            if "voxel_size" not in data.files:
                raise ValueError(f"GT file is missing voxel_size: {path}")
            spacing_zyx = np.asarray(data["voxel_size"], dtype=np.float64)
        if (spacing_zyx.shape != (3,) or not np.isfinite(spacing_zyx).all()
                or np.any(spacing_zyx <= 0)):
            raise ValueError(f"GT voxel_size must contain three positive finite values: {path}")
        voxel_xyz = tuple(float(value) for value in spacing_zyx[::-1])
        print(f"[INFO] {case_name}: voxel size XYZ={voxel_xyz} mm from GT: {path}")
        return voxel_xyz

    raise FileNotFoundError(
        f"VOXEL_SIZE=None requires GT for case '{case_name}'. "
        f"Expected {candidates[0]} or {candidates[1]}."
    )
