import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from config import (
    INTENSITY_RECONSTRUCTION_METHODS,
    SEGMENTATION_RECONSTRUCTION_METHOD,
    SEGMENTATION_MODEL_NAME,
    EVAL_CLASS_ID,
)

from src.utils.logging import timed_stage


# ============================================================
# WHAT TO RUN
# ============================================================

CASE_NAMES = ["008_02"]

SEGMENTATION_RESULT_MODELS = [SEGMENTATION_MODEL_NAME]

USE_NEW_GT_LOADER = True

RUN_PREPARE_DATA = False
RUN_SEGMENTATION = False
RUN_RECONSTRUCT_INTENSITY = False
RUN_RECONSTRUCT_SEGMENTATION = False
RUN_EXTRACT_SEGMENTATION_SURFACES = False
RUN_ALIGN_SURFACES = True
RUN_NAPARI_VIS = False
RUN_SIMPLE_VIS = False
RUN_EVAL_SURFACE = False
RUN_EVAL_VOLUME = True

VIS_CASE = None


def main():
    with timed_stage("PIPELINE TOTAL"):
        cases = list(CASE_NAMES)
        print(f"[PIPELINE] cases={cases}")

        if RUN_PREPARE_DATA:
            with timed_stage("PREPARE DATA"):
                from src.data_loading.prepare_dicom import prepare_cases
                prepare_cases(cases, use_new_gt_loader=USE_NEW_GT_LOADER)

        if RUN_SEGMENTATION:
            with timed_stage("SEGMENTATION"):
                from src.segmentation.runner import segment_cases
                segment_cases(
                    cases,
                    model_name=SEGMENTATION_MODEL_NAME,
                )

        if RUN_RECONSTRUCT_INTENSITY:
            with timed_stage("RECONSTRUCT INTENSITY"):
                from src.reconstruction.runner import reconstruct_intensity_cases
                reconstruct_intensity_cases(
                    cases,
                    methods=INTENSITY_RECONSTRUCTION_METHODS,
                )

        if RUN_RECONSTRUCT_SEGMENTATION:
            with timed_stage("RECONSTRUCT SEGMENTATION"):
                from src.reconstruction.runner import reconstruct_segmentation_cases
                for model_name in SEGMENTATION_RESULT_MODELS:
                    reconstruct_segmentation_cases(
                        cases,
                        model_name,
                        SEGMENTATION_RECONSTRUCTION_METHOD,
                    )

        if RUN_EXTRACT_SEGMENTATION_SURFACES:
            with timed_stage("EXTRACT SEGMENTATION SURFACES"):
                from src.isosurface_extraction.extract import (
                    extract_segmentation_surfaces_for_cases,
                )
                for model_name in SEGMENTATION_RESULT_MODELS:
                    extract_segmentation_surfaces_for_cases(
                        cases,
                        model_name,
                        SEGMENTATION_RECONSTRUCTION_METHOD,
                    )

        case_for_vis = VIS_CASE or cases[0]
        model_for_vis = SEGMENTATION_RESULT_MODELS[0]
        if RUN_ALIGN_SURFACES:
            from src.registration.align_surfaces import align_and_view_case
            align_and_view_case(
                case_for_vis,
                model_for_vis,
                SEGMENTATION_RECONSTRUCTION_METHOD,
            )
        elif RUN_NAPARI_VIS:
            from src.visualization.napari_viewer import visualize_case_napari
            visualize_case_napari(
                case_for_vis,
                model_for_vis,
                INTENSITY_RECONSTRUCTION_METHODS,
                SEGMENTATION_RECONSTRUCTION_METHOD,
            )
        if RUN_SIMPLE_VIS:
            from src.visualization.simple_viewer import visualize_case_simple
            visualize_case_simple(
                case_for_vis,
                model_for_vis,
                INTENSITY_RECONSTRUCTION_METHODS[0],
                SEGMENTATION_RECONSTRUCTION_METHOD,
            )

        if RUN_EVAL_SURFACE:
            with timed_stage("EVALUATE SURFACES"):
                from src.evaluation.evaluate_surface import evaluate_surface_case
                for case_name in cases:
                    for model_name in SEGMENTATION_RESULT_MODELS:
                        evaluate_surface_case(
                            case_name, model_name, SEGMENTATION_RECONSTRUCTION_METHOD,
                            class_id=EVAL_CLASS_ID,
                        )

        if RUN_EVAL_VOLUME:
            with timed_stage("EVALUATE VOLUMES"):
                from src.evaluation.evaluate_volume import evaluate_volume_case
                for case_name in cases:
                    for model_name in SEGMENTATION_RESULT_MODELS:
                        evaluate_volume_case(
                            case_name, model_name, SEGMENTATION_RECONSTRUCTION_METHOD,
                            class_id=EVAL_CLASS_ID,
                        )


if __name__ == "__main__":
    main()
