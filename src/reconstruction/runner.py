from pathlib import Path

import numpy as np

from config import (
    BACKGROUND_THRESHOLD,
    CENTER,
    CHUNK_PIXELS,
    DEVICE,
    DISTANCE_EPS,
    DISTANCE_POWER,
    DWR_DISTANCE_EPS,
    DWR_DISTANCE_POWER,
    DWR_INTERP,
    DWR_MAX_PLANE_DIST,
    DWR_N_NEAREST,
    IMAGE_KEY,
    N_NEAREST,
    PIXEL_SPACING,
    INTENSITY_RECONSTRUCTION_METHODS,
    SEGMENTATION_RECONSTRUCTION_METHOD,
    SPLINE_ORDER,
    VNN_MEAN_N_NEAREST,
    VOXEL_SIZE,
)
from src.data_loading.sweep import load_sweep_h5
from src.reconstruction.bspline import reconstruct_bspline_for_file
from src.reconstruction.common import save_as_vti, save_reconstruction_npz
from src.reconstruction.vnn import (
    reconstruct_voxel_nearest_for_file,
    voxel_nearest_reconstruct_torch,
)
from src.reconstruction.vnn_distance_weighted import reconstruct_distance_weighted_for_file
from src.reconstruction.vnn_mean import reconstruct_voxel_nearest_mean_for_file
from src.utils.paths import (
    intensity_reconstruction_dir,
    prepared_sweep_path,
    segmentation_path,
    segmentation_reconstruction_dir,
)
from src.utils.logging import log_io


METHODS = {
    "bspline": reconstruct_bspline_for_file,
    "voxel_nearest": reconstruct_voxel_nearest_for_file,
    "voxel_nearest_mean": reconstruct_voxel_nearest_mean_for_file,
    "distance_weighted": reconstruct_distance_weighted_for_file,
}


def normalize_list(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return list(values)


def reconstruct_file_with_method(path, out_dir, method):
    fn = METHODS[method]
    common = dict(
        path=path,
        out_root=out_dir,
        voxel=VOXEL_SIZE,
        pixel_spacing=PIXEL_SPACING,
        background_threshold=BACKGROUND_THRESHOLD,
        center=CENTER,
        chunk_pixels=CHUNK_PIXELS,
        image_key=IMAGE_KEY,
        device_str=DEVICE,
    )
    if method == "bspline":
        common.update(
            n_nearest=N_NEAREST,
            spline_order=SPLINE_ORDER,
            distance_eps=DISTANCE_EPS,
            distance_power=DISTANCE_POWER,
        )
    elif method == "voxel_nearest_mean":
        common["n_nearest"] = VNN_MEAN_N_NEAREST
    elif method == "distance_weighted":
        common.update(
            n_nearest=DWR_N_NEAREST,
            distance_power=DWR_DISTANCE_POWER,
            distance_eps=DWR_DISTANCE_EPS,
            interp=DWR_INTERP,
            max_plane_dist=DWR_MAX_PLANE_DIST,
        )
    fn(**common)


def reconstruct_intensity_case(
    case_name,
    methods=None,
):
    input_h5 = prepared_sweep_path(case_name)
    outputs = []
    for method in normalize_list(methods or INTENSITY_RECONSTRUCTION_METHODS):
        out_dir = intensity_reconstruction_dir(case_name, method)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_io(
            f"RECONSTRUCTION intensity/{method}",
            {"sweep": input_h5},
            {"volume": out_dir / "volume.npz"},
        )
        reconstruct_file_with_method(input_h5, out_dir, method)
        generated_npz = out_dir / f"{input_h5.stem}.npz"
        generated_vti = out_dir / f"{input_h5.stem}.vti"
        output_npz = out_dir / "volume.npz"
        output_vti = out_dir / "volume.vti"
        generated_npz.replace(output_npz)
        if generated_vti.exists():
            generated_vti.replace(output_vti)
        outputs.append(output_npz)
    return outputs


def reconstruct_segmentation_case(
    case_name,
    model_name=None,
    method=SEGMENTATION_RECONSTRUCTION_METHOD,
):
    if method != "voxel_nearest":
        raise ValueError("Multiclass labels must use method='voxel_nearest'")

    input_h5 = segmentation_path(case_name, model_name)
    out_dir = segmentation_reconstruction_dir(case_name, model_name, method)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, poses, (sx, sy), _ = load_sweep_h5(input_h5, image_key="labels")
    if PIXEL_SPACING is not None:
        sx, sy = PIXEL_SPACING

    log_io(
        f"RECONSTRUCTION segmentation/{method}",
        {"labels 2D": input_h5},
        {"labels 3D": out_dir / "labels.npz"},
    )
    volume, weights, origin, voxel = voxel_nearest_reconstruct_torch(
        img=labels,
        Ts=poses,
        sx=sx,
        sy=sy,
        origin=None,
        voxel=VOXEL_SIZE,
        background_threshold=None,
        center=CENTER,
        chunk_pixels=CHUNK_PIXELS,
        binary_input=False,
        device_str=DEVICE,
    )
    volume = np.rint(volume).astype(np.uint16)
    output_npz = out_dir / "labels.npz"
    save_reconstruction_npz(
        output_npz,
        volume,
        weights,
        voxel,
        origin,
        method,
        extra={
            "content_type": np.array("segmentation_labels"),
            "model_name": np.array(model_name),
            "classes": np.unique(volume),
        },
    )
    save_as_vti(out_dir / "labels.vti", volume, voxel, origin, array_name="labels")
    return output_npz


def reconstruct_intensity_cases(case_names, methods=None):
    return [
        output
        for case_name in normalize_list(case_names)
        for output in reconstruct_intensity_case(
            case_name, methods
        )
    ]


def reconstruct_segmentation_cases(
    case_names,
    model_name,
    method=SEGMENTATION_RECONSTRUCTION_METHOD,
):
    return [
        reconstruct_segmentation_case(
            case_name, model_name, method
        )
        for case_name in normalize_list(case_names)
    ]
