from pathlib import Path
import json

import numpy as np
from skimage.measure import marching_cubes

from config import ISO_CLASSES, ISO_MIN_VOXELS, ISO_SAVE_OBJ, ISO_STEP_SIZE
from src.utils.paths import (
    ground_truth_labels_path,
    ground_truth_surface_dir,
    predicted_surface_dir,
    segmentation_reconstruction_path,
)
from src.utils.logging import log_io


def load_labels(path):
    data = np.load(path, allow_pickle=True)
    labels = data["volume"] if "volume" in data else data["labels"]
    spacing_zyx = tuple(float(x) for x in data.get("voxel_size", (1, 1, 1)))
    return np.rint(labels).astype(np.uint16), spacing_zyx


def save_obj(path, vertices, faces):
    with open(path, "w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def extract_surfaces_from_labels(labels_path, output_dir, classes=None):
    labels_path = Path(labels_path)
    output_dir = Path(output_dir)
    labels, spacing_zyx = load_labels(labels_path)
    selected = classes or [int(x) for x in np.unique(labels) if int(x) != 0]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_io("ISOSURFACE", {"labels 3D": labels_path}, {"surfaces": output_dir})
    summary = []

    for class_id in selected:
        mask = labels == int(class_id)
        if int(mask.sum()) < ISO_MIN_VOXELS:
            continue
        vertices_zyx, faces, normals_zyx, values = marching_cubes(
            mask.astype(np.float32),
            level=0.5,
            spacing=spacing_zyx,
            step_size=ISO_STEP_SIZE,
            allow_degenerate=False,
        )
        vertices = vertices_zyx[:, [2, 1, 0]]
        normals = normals_zyx[:, [2, 1, 0]]
        output = output_dir / f"class_{class_id}_surface.npz"
        np.savez_compressed(
            output,
            verts=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            normals=normals.astype(np.float32),
            values=values.astype(np.float32),
            class_id=np.int32(class_id),
            source_file=np.array(str(labels_path)),
        )
        if ISO_SAVE_OBJ:
            save_obj(output.with_suffix(".obj"), vertices, faces)
        summary.append(
            {
                "class_id": int(class_id),
                "surface": output.name,
                "vertices": int(len(vertices)),
                "faces": int(len(faces)),
            }
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[ISOSURFACE] {labels_path} -> {output_dir}")
    return summary


def extract_surfaces_for_case(
    case_name,
    model_name=None,
    reconstruction_method="voxel_nearest",
):
    outputs = {}
    gt_path = ground_truth_labels_path(case_name)
    if gt_path.exists():
        outputs["ground_truth"] = extract_surfaces_from_labels(
            gt_path,
            ground_truth_surface_dir(case_name),
            ISO_CLASSES,
        )

    predicted_path = segmentation_reconstruction_path(
        case_name,
        model_name,
        reconstruction_method,
    )
    if predicted_path.exists():
        outputs["predicted"] = extract_surfaces_from_labels(
            predicted_path,
            predicted_surface_dir(
                case_name,
                model_name,
                reconstruction_method,
            ),
            ISO_CLASSES,
        )
    return outputs


def extract_surfaces_for_cases(
    case_names,
    model_name=None,
    reconstruction_method="voxel_nearest",
):
    return {
        case_name: extract_surfaces_for_case(
            case_name,
            model_name,
            reconstruction_method,
        )
        for case_name in case_names
    }
