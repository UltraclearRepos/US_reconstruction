import unittest

import numpy as np
from scipy import ndimage as ndi

from src.reconstruction.mask_postprocessing import (
    CONNECTIVITY,
    postprocess_segmentation_labels,
)


class MaskPostprocessingTests(unittest.TestCase):
    def process(self, labels, **kwargs):
        return postprocess_segmentation_labels(
            labels, spacing_zyx=(0.5, 0.25, 0.25), **kwargs,
        )

    def test_zero_sigma_preserves_all_nodules_without_closing(self):
        labels = np.zeros((24, 24, 24), dtype=np.uint16)
        labels[0, 0, 0] = 2
        labels[6:10, 6:10, 6:10] = 2
        labels[7, 7, 7] = 0
        labels[18:21, 18:21, 18:21] = 2
        result, report = self.process(
            labels, closing_radius_mm=2, nodule_smoothing_sigma_mm=0,
        )
        np.testing.assert_array_equal(result, labels)
        self.assertEqual(report[0]["components_after"], 3)

    def test_nodules_are_smoothed_individually_in_physical_units(self):
        labels = np.zeros((24, 24, 24), dtype=np.uint16)
        labels[3:9, 3:9, 3:9] = 2
        labels[3:9, 3:9, 10:16] = 2
        expected = np.zeros_like(labels)
        for x in (3, 10):
            mask = np.zeros(labels.shape, dtype=bool)
            mask[3:9, 3:9, x:x + 6] = True
            expected[ndi.gaussian_filter(
                mask.astype(np.float32), sigma=(0.8, 1.6, 1.6), mode="constant",
            ) >= 0.5] = 2
        result, report = self.process(labels, nodule_smoothing_sigma_mm=0.4)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(report[0]["components_before"], 2)

    def test_tiny_nodules_survive_smoothing(self):
        labels = np.zeros((20, 20, 20), dtype=np.uint16)
        labels[0, 0, 0] = labels[10, 10, 10] = 2
        result, report = self.process(labels, nodule_smoothing_sigma_mm=1)
        np.testing.assert_array_equal(result, labels)
        self.assertEqual(report[0]["smoothing_fallback_count"], 2)

    def test_thyroid_keeps_only_largest_component(self):
        labels = np.zeros((24, 24, 24), dtype=np.uint16)
        labels[3:13, 3:13, 3:13] = 1
        labels[20, 20, 20] = 1
        result, report = self.process(
            labels, closing_radius_mm=0, smoothing_sigma_mm=0,
        )
        self.assertEqual(result[20, 20, 20], 0)
        self.assertEqual(np.count_nonzero(result), 1000)
        self.assertEqual(report[0]["components_after"], 1)

    def test_nodule_priority_reports_and_separate_sigmas(self):
        labels = np.zeros((24, 24, 24), dtype=np.uint16)
        labels[3:21, 3:21, 3:21] = 1
        labels[9:15, 9:15, 9:15] = 2
        original = labels.copy()
        result, report = self.process(
            labels, closing_radius_mm=1, smoothing_sigma_mm=0.5,
            nodule_smoothing_sigma_mm=0,
        )
        np.testing.assert_array_equal(labels, original)
        self.assertTrue(np.all(result[9:15, 9:15, 9:15] == 2))
        self.assertEqual([item["smoothing_sigma_mm"] for item in report], [0.5, 0])
        for item in report:
            mask = result == item["class_id"]
            self.assertEqual(item["voxels_after"], int(mask.sum()))
            self.assertEqual(item["components_after"], ndi.label(mask, CONNECTIVITY)[1])

    def test_empty_and_invalid_input(self):
        labels = np.zeros((4, 4, 4), dtype=np.uint16)
        result, report = self.process(labels)
        np.testing.assert_array_equal(result, labels)
        self.assertEqual(report, [])
        for sigma in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.process(labels, nodule_smoothing_sigma_mm=sigma)


if __name__ == "__main__":
    unittest.main()
