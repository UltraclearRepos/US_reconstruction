"""Import DICOM SEG masks using segment IDs and referenced source frames."""

import numpy as np
import pydicom
from pydicom.pixels import iter_pixels


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
