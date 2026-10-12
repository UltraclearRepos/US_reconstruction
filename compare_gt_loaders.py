"""Compare GT before and after the DICOM loader fix, without command-line arguments.

Run: .venv/bin/python compare_gt_loaders.py
Settings are below. This script reads raw DICOM and does not write pipeline data.
Loader and viewer code is copied here so this comparison stays independent.
"""

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import iter_pixels


# Settings
CASE_NAME = "003_02"
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
SOFTWARE_RENDERING = True  # Avoid the D3D12 driver crash in this WSL environment.
DISPLAY_GAP_MM = 10.0


def extract_volume_spacing_zyx(ds):
    measures = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
    sy, sx = (float(value) for value in measures.PixelSpacing)
    return float(measures.SliceThickness), sy, sx


def load_segmentation_labels(path, reference_volume=None):
    """Return one ZYX labelmap; higher segment IDs win where masks overlap.

    SEG pixels describe occupancy, not class IDs. SegmentIdentificationSequence
    supplies the class, and ReferencedFrameNumber supplies the source Z index.
    Decode one frame at a time to avoid holding the interleaved pixel array.
    """
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    segmentation_type = getattr(ds, "SegmentationType", None)
    if segmentation_type is None:
        # Preserve the existing path for ordinary, already-labelled volumes.
        pixels = np.asarray(pydicom.dcmread(path).pixel_array)
        if pixels.ndim == 2:
            pixels = pixels[None]
        if pixels.ndim != 3:
            raise ValueError("Expected a 3D ground-truth label volume")
        return np.rint(pixels).astype(np.uint16), extract_volume_spacing_zyx(ds)
    if segmentation_type not in ("BINARY", "FRACTIONAL"):
        raise ValueError(f"Unsupported DICOM SegmentationType: {segmentation_type}")

    segment_ids = {int(item.SegmentNumber) for item in ds.SegmentSequence}
    if not segment_ids or any(number <= 0 or number > 65535 for number in segment_ids):
        raise ValueError("DICOM segment numbers must fit positive uint16 labels")
    frames = ds.PerFrameFunctionalGroupsSequence
    if len(frames) != int(ds.NumberOfFrames):
        raise ValueError("DICOM SEG frame metadata does not match NumberOfFrames")

    mapping = []
    source_uids = set()
    for frame in frames:
        identification = getattr(frame, "SegmentIdentificationSequence", [])
        if len(identification) != 1:
            raise ValueError("Each SEG frame must identify exactly one segment")
        segment_id = int(identification[0].ReferencedSegmentNumber)
        if segment_id not in segment_ids:
            raise ValueError(f"SEG frame references undefined segment {segment_id}")
        references = {
            (str(image.ReferencedSOPInstanceUID), int(image.ReferencedFrameNumber))
            for derivation in getattr(frame, "DerivationImageSequence", [])
            for image in getattr(derivation, "SourceImageSequence", [])
            if hasattr(image, "ReferencedFrameNumber")
        }
        if len(references) != 1:
            raise ValueError("Each SEG frame must reference exactly one source frame")
        source_uid, source_frame = references.pop()
        if source_frame <= 0:
            raise ValueError("ReferencedFrameNumber must be positive (one-based)")
        source_uids.add(source_uid)
        mapping.append((source_frame - 1, segment_id))

    if len(source_uids) != 1 or not mapping:
        raise ValueError("GT import requires SEG frames referencing one source volume")
    spacing = extract_volume_spacing_zyx(ds)
    depth = max(index for index, _ in mapping) + 1
    if reference_volume is not None:
        if source_uids != {str(reference_volume.SOPInstanceUID)}:
            raise ValueError("DICOM SEG references a different source volume")
        if (int(ds.Rows), int(ds.Columns)) != (
                int(reference_volume.Rows), int(reference_volume.Columns)):
            raise ValueError("SEG and source volume must have identical pixel dimensions")
        source_spacing = extract_volume_spacing_zyx(reference_volume)
        if not np.allclose(spacing, source_spacing, rtol=1e-5, atol=1e-7):
            raise ValueError("SEG and source volume must have matching voxel spacing")
        depth = int(reference_volume.NumberOfFrames)
        if any(index >= depth for index, _ in mapping):
            raise ValueError("SEG references a frame outside the source volume")
        spacing = source_spacing
    if not np.isfinite(spacing).all() or np.any(np.asarray(spacing) <= 0):
        raise ValueError("GT voxel spacing must contain positive finite values")

    maximum = 1.0
    if segmentation_type == "FRACTIONAL":
        maximum = float(ds.MaximumFractionalValue)
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("MaximumFractionalValue must be positive and finite")
    labels = np.zeros((depth, int(ds.Rows), int(ds.Columns)), dtype=np.uint16)
    for pixels, (z, segment_id) in zip(iter_pixels(path), mapping, strict=True):
        if pixels.shape != labels.shape[1:] or np.any(pixels > maximum):
            raise ValueError("Invalid SEG pixel dimensions or occupancy values")
        foreground = pixels >= 0.5 * maximum
        target = labels[z]
        # Preserve nodule label 2 even when it also lies inside thyroid label 1.
        target[foreground] = np.maximum(target[foreground], segment_id)
    return labels, spacing


def find_one(directory, pattern):
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {pattern} in {directory}, found {len(paths)}")
    return paths[0]


def load_gt_before(path):
    """Original loader: every SEG frame becomes the next Z slice; IDs are ignored."""
    ds = pydicom.dcmread(path)
    labels = np.asarray(ds.pixel_array)
    if labels.ndim == 2:
        labels = labels[None]
    if labels.ndim != 3:
        raise ValueError(f"Expected 3D SEG pixels, got {labels.shape}")
    if np.issubdtype(labels.dtype, np.floating):
        labels = np.rint(labels)
    return labels.astype(np.uint16), extract_volume_spacing_zyx(ds)


def prepare_qt_environment():
    if SOFTWARE_RENDERING and sys.platform == "linux":
        # These settings apply to this standalone process only.
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "true"
        os.environ["GALLIUM_DRIVER"] = "llvmpipe"
    if sys.platform != "linux":
        return
    if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
    try:
        metadata = runtime.lstat()
        if stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.getuid():
            if stat.S_IMODE(metadata.st_mode) != 0o700:
                runtime.chmod(0o700)
            os.environ.setdefault("XDG_RUNTIME_DIR", str(runtime))
    except OSError:
        pass


def make_surface(labels, spacing, class_id):
    """Marching cubes at full resolution; preserve original ZYX coordinates in mm."""
    from skimage.measure import marching_cubes

    mask = labels == class_id
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        if not len(occupied):
            return None
        bounds.append((int(occupied[0]), int(occupied[-1]) + 1))
    # Crop background before marching cubes and restore its physical offset.
    region = tuple(slice(low, high) for low, high in bounds)
    cropped = np.pad(mask[region], 1, mode="constant").astype(np.float32)
    del mask
    vertices, faces, _, _ = marching_cubes(
        cropped, level=0.5, spacing=spacing, step_size=1, allow_degenerate=False
    )
    offset = (np.array([low for low, _ in bounds]) - 1) * np.asarray(spacing)
    vertices += offset
    return vertices, faces, np.ones(len(vertices), dtype=np.float32)


def describe(name, labels, spacing, segment_ids):
    print(f"[{name}] shape ZYX={labels.shape}; spacing ZYX={spacing} mm", flush=True)
    print(f"[{name}] full extent ZYX={np.round(np.asarray(labels.shape) * spacing, 3)} mm", flush=True)
    for class_id in segment_ids:
        count = int(np.count_nonzero(labels == class_id))
        print(f"[{name}] label={class_id}: {count} voxels, {count * np.prod(spacing):.3f} mm3", flush=True)


def compare_gt():
    case_dir = RAW_DIR / CASE_NAME
    seg_path = find_one(case_dir, "*Segmentation*.dcm")
    volume_path = find_one(case_dir, "*Volume*.dcm")
    reference = pydicom.dcmread(volume_path, stop_before_pixels=True)
    segment = pydicom.dcmread(seg_path, stop_before_pixels=True)
    segment_names = {int(item.SegmentNumber): str(item.SegmentLabel)
                     for item in segment.SegmentSequence}
    print(f"[CASE] {CASE_NAME}", flush=True)
    print(f"[INPUT] {seg_path}", flush=True)
    print("[COMPARE] BEFORE on the left; AFTER on the right. Same scale in mm.", flush=True)

    # Build surfaces before opening Qt, then release the full label volumes.
    before, before_spacing = load_gt_before(seg_path)
    describe("BEFORE", before, before_spacing, [1])
    before_shape = before.shape
    before_surface = make_surface(before, before_spacing, 1)
    del before

    after, after_spacing = load_segmentation_labels(seg_path, reference)
    describe("AFTER", after, after_spacing, sorted(segment_names))
    after_surfaces = {number: make_surface(after, after_spacing, number)
                      for number in sorted(segment_names)}
    del after

    prepare_qt_environment()
    import napari
    from qtpy.QtWidgets import QApplication

    viewer = napari.Viewer(ndisplay=3, title=f"GT BEFORE / AFTER loader - {CASE_NAME}", show=False)
    viewer.dims.axis_labels = ("Z (mm)", "Y (mm)", "X (mm)")
    shift = (0, 0, before_shape[2] * before_spacing[2] + DISPLAY_GAP_MM)
    if before_surface is not None:
        viewer.add_surface(before_surface, name="BEFORE / mixed thyroid + nodule",
                           opacity=1.0, shading="smooth")
    for number, surface in after_surfaces.items():
        if surface is not None:
            viewer.add_surface(surface, name=f"AFTER / {number}: {segment_names[number]}",
                               opacity=1.0, shading="smooth", translate=shift)
    viewer.text_overlay.text = "LEFT: BEFORE loader     |     RIGHT: AFTER loader     |     same mm scale"
    viewer.text_overlay.visible = True
    viewer.reset_view()
    # An oblique, orthographic view exposes the change along Z as well as XY.
    viewer.camera.perspective = 0
    viewer.camera.angles = (20, 35, 90)
    viewer.show()
    # Place this comparison window inside the primary screen's available area.
    window = viewer.window._qt_window
    screen = QApplication.primaryScreen().availableGeometry()
    width, height = min(1200, screen.width()), min(800, screen.height())
    window.showNormal()
    window.resize(width, height)
    window.move(screen.x() + (screen.width() - width) // 2,
                screen.y() + (screen.height() - height) // 2)
    viewer.window.activate()
    print("[DISPLAY] Ready. Toggle layers in Napari to inspect the old/new GT.", flush=True)
    napari.run()


if __name__ == "__main__":
    compare_gt()
