from config import (
    DENOISING_METHODS,
    RDPAD_ITERATIONS,
    RDPAD_TIMESTEP,
    RDPAD_Q0_MODE,
    RDPAD_Q0_PERCENTILE,
    GAUSSIAN_SIGMA,
    MEDIAN_SIZE,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPATIAL,
    TV_WEIGHT,
    TV_MAX_NUM_ITER,
)
from src.data_loading.sweep import load_sweep_h5, save_sweep_h5
from src.denoising.classical import (
    denoise_bilateral_2d,
    denoise_gaussian_2d,
    denoise_median_2d,
    denoise_tv_2d,
)
from src.denoising.common import apply_slice_wise
from src.denoising.rdp_ad import denoise_rdpad_2d
from src.utils.paths import denoised_sweep_path, source_sweep_path
from src.utils.logging import log_io


def normalize_list(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return list(values)


def get_denoise_function(method):
    if method == "rdpad":
        return lambda frame: denoise_rdpad_2d(
            frame,
            iterations=RDPAD_ITERATIONS,
            timestep=RDPAD_TIMESTEP,
            q0_mode=RDPAD_Q0_MODE,
            q0_percentile=RDPAD_Q0_PERCENTILE,
        )
    if method == "gaussian":
        return lambda frame: denoise_gaussian_2d(frame, sigma=GAUSSIAN_SIGMA)
    if method == "median":
        return lambda frame: denoise_median_2d(frame, size=MEDIAN_SIZE)
    if method == "bilateral":
        return lambda frame: denoise_bilateral_2d(
            frame,
            sigma_color=BILATERAL_SIGMA_COLOR,
            sigma_spatial=BILATERAL_SIGMA_SPATIAL,
        )
    if method == "tv":
        return lambda frame: denoise_tv_2d(
            frame, weight=TV_WEIGHT, max_num_iter=TV_MAX_NUM_ITER
        )
    raise ValueError(f"Unknown denoising method: {method}")


def denoise_case(case_name, method):
    source = source_sweep_path(case_name, "prepared")
    output = denoised_sweep_path(case_name, method)
    log_io("DENOISE", {"sweep": source}, {"denoised sweep": output})

    frames, poses, spacing_xy, _ = load_sweep_h5(source)
    denoised = apply_slice_wise(frames, get_denoise_function(method))
    save_sweep_h5(
        output,
        denoised,
        poses,
        spacing_xy,
        attrs={
            "content_type": "denoised_ultrasound_sweep",
            "denoising_method": method,
            "source_h5": str(source),
        },
    )
    return output


def denoise_cases(case_names, methods=None):
    methods = normalize_list(methods or DENOISING_METHODS)
    return [
        denoise_case(case_name, method)
        for case_name in normalize_list(case_names)
        for method in methods
    ]
