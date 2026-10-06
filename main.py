import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from config import (
    INTENSITY_RECONSTRUCTION_METHODS,
    SEGMENTATION_RECONSTRUCTION_METHOD,
    SEGMENTATION_MODEL_NAME,
)

from src.utils.logging import timed_stage


# ============================================================
# WHAT TO RUN
# ============================================================

CASE_NAMES = ["001_01"]

SEGMENTATION_RESULT_MODELS = [SEGMENTATION_MODEL_NAME]

RUN_PREPARE_DATA = True
RUN_SEGMENTATION = True
RUN_RECONSTRUCT_INTENSITY = True
RUN_RECONSTRUCT_SEGMENTATION = True
RUN_EXTRACT_SEGMENTATION_SURFACES = True
RUN_EVALUATION = False
RUN_NAPARI_VIS = True
RUN_SIMPLE_VIS = False

RUN_EVAL_RECONSTRUCTION_WITH_GT = True
RUN_EVAL_SEGMENTATION = True
RUN_EVAL_NO_REFERENCE = True

VIS_CASE = None


def main():
    with timed_stage("PIPELINE TOTAL"):
        cases = list(CASE_NAMES)
        print(f"[PIPELINE] cases={cases}")

        if RUN_PREPARE_DATA:
            with timed_stage("PREPARE DATA"):
                from src.data_loading.prepare_dicom import prepare_cases
                prepare_cases(cases)

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

        if RUN_EVALUATION:
            with timed_stage("EVALUATION TOTAL"):
                if RUN_EVAL_RECONSTRUCTION_WITH_GT:
                    with timed_stage("EVALUATION INTENSITY WITH GT"):
                        from src.evaluation.reconstruction_with_gt import (
                            evaluate_reconstruction_cases_with_gt,
                        )
                        evaluate_reconstruction_cases_with_gt(
                            cases, methods=INTENSITY_RECONSTRUCTION_METHODS
                        )

                if RUN_EVAL_SEGMENTATION:
                    with timed_stage("EVALUATION SEGMENTATION"):
                        from src.evaluation.segmentation_metrics import evaluate_segmentation_cases
                        for model_name in SEGMENTATION_RESULT_MODELS:
                            evaluate_segmentation_cases(
                                cases,
                                model_name,
                                SEGMENTATION_RECONSTRUCTION_METHOD,
                            )

                if RUN_EVAL_NO_REFERENCE:
                    with timed_stage("EVALUATION NO REFERENCE"):
                        from src.evaluation.no_reference import evaluate_no_reference_cases
                        evaluate_no_reference_cases(
                            cases, methods=INTENSITY_RECONSTRUCTION_METHODS
                        )

        case_for_vis = VIS_CASE or cases[0]
        model_for_vis = SEGMENTATION_RESULT_MODELS[0]
        if RUN_NAPARI_VIS:
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


if __name__ == "__main__":
    main()
