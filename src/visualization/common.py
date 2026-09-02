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

    if path.suffix.lower() == ".vti":
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        image = reader.GetOutput()
        dimensions = image.GetDimensions()
        volume = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(
            dimensions[2], dimensions[1], dimensions[0]
        )
        spacing_xyz = image.GetSpacing()
        return volume.astype(np.float32), spacing_xyz[::-1]

    raise ValueError(f"Unsupported volume: {path}")
