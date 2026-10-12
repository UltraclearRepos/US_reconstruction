import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SegmentationStorage, generate_uid

from src.data_loading.dicom_segmentation import load_segmentation_labels


class SegmentationImportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "seg.dcm"
        self.reference = Dataset()
        self.reference.SOPInstanceUID = generate_uid()
        self.reference.Rows, self.reference.Columns = 2, 3
        self.reference.NumberOfFrames = 3
        self.reference.SharedFunctionalGroupsSequence = [self.measures()]

    @staticmethod
    def measures():
        shared, measures = Dataset(), Dataset()
        measures.PixelSpacing = [0.2, 0.1]
        measures.SliceThickness = 0.3
        shared.PixelMeasuresSequence = [measures]
        return shared

    def write_seg(self, masks, segments, source_frames, binary=False, maximum=1):
        masks = np.asarray(masks, dtype=np.uint8)
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = SegmentationStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(self.path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Rows, ds.Columns = masks.shape[1:]
        ds.NumberOfFrames = len(masks)
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.BitsAllocated = ds.BitsStored = 1 if binary else 8
        ds.HighBit = ds.BitsStored - 1
        ds.SegmentationType = "BINARY" if binary else "FRACTIONAL"
        if not binary:
            ds.MaximumFractionalValue = maximum
            ds.SegmentationFractionalType = "OCCUPANCY"
        ds.SharedFunctionalGroupsSequence = [self.measures()]
        ds.SegmentSequence = []
        for number in (1, 2):
            segment = Dataset()
            segment.SegmentNumber = number
            segment.SegmentLabel = "Thyroid" if number == 1 else "Nodule"
            ds.SegmentSequence.append(segment)
        ds.PerFrameFunctionalGroupsSequence = []
        for segment_number, frame_number in zip(segments, source_frames):
            frame, identification, derivation, source = (Dataset() for _ in range(4))
            identification.ReferencedSegmentNumber = segment_number
            frame.SegmentIdentificationSequence = [identification]
            source.ReferencedSOPInstanceUID = self.reference.SOPInstanceUID
            source.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.2"
            source.ReferencedFrameNumber = frame_number
            derivation.SourceImageSequence = [source]
            frame.DerivationImageSequence = [derivation]
            ds.PerFrameFunctionalGroupsSequence.append(frame)
        ds.PixelData = (np.packbits(masks.ravel(), bitorder="little").tobytes()
                        if binary else masks.tobytes())
        ds.save_as(self.path, enforce_file_format=True)
        return ds

    def test_interleaved_segments_share_source_slices_and_nodule_wins_overlap(self):
        thyroid = np.ones((2, 3), dtype=np.uint8)
        nodule = np.zeros((2, 3), dtype=np.uint8)
        nodule[0, 1] = 1
        self.write_seg([thyroid, nodule, nodule, thyroid], [1, 2, 2, 1], [1, 1, 2, 2])
        labels, spacing = load_segmentation_labels(self.path, self.reference)
        expected = thyroid.astype(np.uint16)
        expected[0, 1] = 2
        self.assertEqual(labels.shape, (3, 2, 3))
        self.assertEqual(labels.dtype, np.uint16)
        np.testing.assert_array_equal(labels[0], expected)
        np.testing.assert_array_equal(labels[1], expected)
        self.assertFalse(labels[2].any())
        self.assertEqual(spacing, (0.3, 0.2, 0.1))

    def test_single_segment_is_reordered_by_source_frame_numbers(self):
        masks = np.zeros((3, 2, 3), dtype=np.uint8)
        masks[0, 0, 0], masks[1, 0, 1], masks[2, 0, 2] = 1, 1, 1
        self.write_seg(masks, [1, 1, 1], [3, 1, 2])
        labels, _ = load_segmentation_labels(self.path, self.reference)
        np.testing.assert_array_equal(labels, masks[[1, 2, 0]])

    def test_sparse_frames_without_reference_keep_missing_slices_empty(self):
        mask = np.ones((2, 3), dtype=np.uint8)
        self.write_seg([mask, mask], [1, 2], [1, 3])
        labels, _ = load_segmentation_labels(self.path)
        self.assertEqual(labels.shape, (3, 2, 3))
        self.assertFalse(labels[1].any())
        self.assertTrue((labels[2] == 2).all())

    def test_fractional_occupancy_uses_declared_maximum(self):
        self.write_seg([[[0, 127, 128], [255, 1, 254]]], [2], [1], maximum=255)
        labels, _ = load_segmentation_labels(self.path, self.reference)
        np.testing.assert_array_equal(labels[0], [[0, 0, 2], [2, 0, 2]])

    def test_binary_bit_packed_masks_preserve_segment_ids(self):
        self.write_seg([[[1, 0, 1], [0, 1, 0]]], [2], [1], binary=True)
        labels, _ = load_segmentation_labels(self.path, self.reference)
        np.testing.assert_array_equal(labels[0], [[2, 0, 2], [0, 2, 0]])

    def test_invalid_references_are_rejected(self):
        for segment, frame_number, pattern in ((3, 1, "undefined segment"),
                                               (1, 0, "positive"),
                                               (1, 4, "outside")):
            with self.subTest(segment=segment, frame_number=frame_number):
                self.write_seg([np.ones((2, 3))], [segment], [frame_number])
                with self.assertRaisesRegex(ValueError, pattern):
                    load_segmentation_labels(self.path, self.reference)

    def test_missing_source_frame_number_is_rejected(self):
        ds = self.write_seg([np.ones((2, 3))], [1], [1])
        del ds.PerFrameFunctionalGroupsSequence[0].DerivationImageSequence[0].SourceImageSequence[0].ReferencedFrameNumber
        ds.save_as(self.path, enforce_file_format=True)
        with self.assertRaisesRegex(ValueError, "exactly one source frame"):
            load_segmentation_labels(self.path, self.reference)

    def test_wrong_source_uid_or_geometry_is_rejected(self):
        self.write_seg([np.ones((2, 3))], [1], [1])
        original_uid = self.reference.SOPInstanceUID
        self.reference.SOPInstanceUID = generate_uid()
        with self.assertRaisesRegex(ValueError, "different source volume"):
            load_segmentation_labels(self.path, self.reference)
        self.reference.SOPInstanceUID = original_uid
        self.reference.Columns = 4
        with self.assertRaisesRegex(ValueError, "pixel dimensions"):
            load_segmentation_labels(self.path, self.reference)
        self.reference.Columns = 3
        self.reference.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing = [1, 1]
        with self.assertRaisesRegex(ValueError, "matching voxel spacing"):
            load_segmentation_labels(self.path, self.reference)


if __name__ == "__main__":
    unittest.main()
