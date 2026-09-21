import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from config import (
    RECONSTRUCTION_METHODS,
    SEGMENTATION_MODEL_NAME,
    UNSUPERVISED_IMPORT_MODEL_NAME,
)


# ============================================================
# WHAT TO RUN
# ============================================================

CASE_NAMES = ["kosartur"]

SEGMENTATION_RESULT_MODELS = [SEGMENTATION_MODEL_NAME]

RUN_PREPARE_DATA = False
RUN_SUPERVISED_SEGMENTATION = False
RUN_UNSUPERVISED_SEGMENTATION = False
RUN_IMPORT_UNSUPERVISED = False
RUN_RECONSTRUCT_INTENSITY = False
RUN_RECONSTRUCT_SEGMENTATION = False
RUN_ISOSURFACE_EXTRACTION = False
RUN_EVALUATION = False
RUN_NAPARI_VIS = True
RUN_SIMPLE_VIS = False

RUN_EVAL_RECONSTRUCTION_WITH_GT = True
RUN_EVAL_SEGMENTATION = True
RUN_EVAL_NO_REFERENCE = True

VIS_CASE = None


def main():
    cases = list(CASE_NAMES)
    print(f"[PIPELINE] cases={cases}")

    if RUN_PREPARE_DATA:
        from src.data_loading.prepare_dicom import prepare_cases
        prepare_cases(cases)

    if RUN_SUPERVISED_SEGMENTATION:
        from src.segmentation.runner import segment_cases
        segment_cases(
            cases,
            model_name=SEGMENTATION_MODEL_NAME,
        )

    unsupervised_runs = None
    if RUN_UNSUPERVISED_SEGMENTATION:
        from src.unsupervised_segmentation.runner import run_unsupervised_segmentation
        unsupervised_runs = run_unsupervised_segmentation(cases)

    if RUN_IMPORT_UNSUPERVISED:
        from src.data_loading.unsupervised_output_loader import import_unsupervised_outputs
        import_unsupervised_outputs(
            cases,
            run_dirs=unsupervised_runs,
            model_name=UNSUPERVISED_IMPORT_MODEL_NAME,
        )

    if RUN_RECONSTRUCT_INTENSITY:
        from src.reconstruction.runner import reconstruct_intensity_cases
        reconstruct_intensity_cases(
            cases,
            methods=RECONSTRUCTION_METHODS,
        )

    if RUN_RECONSTRUCT_SEGMENTATION:
        from src.reconstruction.runner import reconstruct_segmentation_cases
        for model_name in SEGMENTATION_RESULT_MODELS:
            reconstruct_segmentation_cases(
                cases,
                model_name,
                RECONSTRUCTION_METHODS[0],
            )

    if RUN_ISOSURFACE_EXTRACTION:
        from src.isosurface_extraction.extract import extract_surfaces_for_cases
        for model_name in SEGMENTATION_RESULT_MODELS:
            extract_surfaces_for_cases(
                cases,
                model_name,
                RECONSTRUCTION_METHODS[0],
            )

    if RUN_EVALUATION:
        if RUN_EVAL_RECONSTRUCTION_WITH_GT:
            from src.evaluation.reconstruction_with_gt import (
                evaluate_reconstruction_cases_with_gt,
            )
            evaluate_reconstruction_cases_with_gt(
                cases, methods=RECONSTRUCTION_METHODS
            )

        if RUN_EVAL_SEGMENTATION:
            from src.evaluation.segmentation_metrics import evaluate_segmentation_cases
            for model_name in SEGMENTATION_RESULT_MODELS:
                evaluate_segmentation_cases(
                    cases,
                    model_name,
                    RECONSTRUCTION_METHODS[0],
                )

        if RUN_EVAL_NO_REFERENCE:
            from src.evaluation.no_reference import evaluate_no_reference_cases
            evaluate_no_reference_cases(
                cases, methods=RECONSTRUCTION_METHODS
            )

    case_for_vis = VIS_CASE or cases[0]
    model_for_vis = SEGMENTATION_RESULT_MODELS[0]
    if RUN_NAPARI_VIS:
        from src.visualization.napari_viewer import visualize_case_napari
        visualize_case_napari(
            case_for_vis,
            model_for_vis,
            RECONSTRUCTION_METHODS,
            RECONSTRUCTION_METHODS[0],
        )
    if RUN_SIMPLE_VIS:
        from src.visualization.simple_viewer import visualize_case_simple
        visualize_case_simple(
            case_for_vis,
            model_for_vis,
            RECONSTRUCTION_METHODS[0],
            RECONSTRUCTION_METHODS[0],
        )


if __name__ == "__main__":
    main()
