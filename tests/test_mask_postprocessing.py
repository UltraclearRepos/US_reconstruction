"""GPU postprocessing checks; no reconstruction pipeline is required."""

import unittest

import numpy as np


class GPUMaskPostprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
        except ImportError as exc:
            raise unittest.SkipTest("CuPy is not installed") from exc
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise unittest.SkipTest("CUDA is unavailable") from exc
        if device_count == 0:
            raise unittest.SkipTest("No CUDA devices")
        from src.reconstruction.mask_postprocessing import postprocess_segmentation_labels
        cls.process = staticmethod(postprocess_segmentation_labels)

    def test_largest_component_preserves_position_shape_dtype_and_class_id(self):
        labels = np.zeros((12, 14, 16), dtype=np.uint16)
        labels[1:5, 3:8, 7:12] = 7
        labels[10, 12, 14] = 7
        original = labels.copy()
        output, report = self.process(
            labels, (0.2, 0.2, 0.2),
            closing_radius_mm=0, smoothing_sigma_mm=0,
        )
        expected = original.copy()
        expected[10, 12, 14] = 0
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(labels, original)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.dtype, labels.dtype)
        self.assertEqual(report[0]["components_before"], 2)
        self.assertEqual(report[0]["components_after"], 1)
        self.assertEqual(report[0]["backend"], "cupy")
        self.assertEqual(report[0]["device"], "cuda:0")

    def test_closing_preserves_object_touching_volume_boundary(self):
        labels = np.zeros((15, 16, 17), dtype=np.uint16)
        labels[0:7, 0:8, 0:9] = 7
        output, report = self.process(
            labels, (0.4, 0.2, 0.3),
            closing_radius_mm=0.6, smoothing_sigma_mm=0,
        )
        np.testing.assert_array_equal(output, labels)
        self.assertEqual(report[0]["components_after"], 1)

    def test_smoothing_removes_small_protrusion_and_preserves_object(self):
        labels = np.zeros((25, 25, 25), dtype=np.uint16)
        labels[6:19, 6:19, 6:19] = 7
        labels[19:22, 12, 12] = 7
        output, report = self.process(
            labels, (0.2, 0.2, 0.2),
            closing_radius_mm=0, smoothing_sigma_mm=0.4,
        )
        self.assertEqual(int(output[12, 12, 12]), 7)
        self.assertEqual(int(output[21, 12, 12]), 0)
        self.assertFalse(report[0]["smoothing_fallback"])
        self.assertEqual(report[0]["components_after"], 1)

    def test_smoothing_fallback_preserves_tiny_object(self):
        labels = np.zeros((9, 9, 9), dtype=np.uint16)
        labels[0, 0, 0] = 7
        output, report = self.process(
            labels, (0.2, 0.2, 0.2),
            closing_radius_mm=0, smoothing_sigma_mm=1.0,
        )
        np.testing.assert_array_equal(output, labels)
        self.assertTrue(report[0]["smoothing_fallback"])

    def test_equal_size_components_keep_first_in_scan_order(self):
        labels = np.zeros((14, 14, 14), dtype=np.uint16)
        labels[1:4, 1:4, 1:4] = 7
        labels[9:12, 9:12, 9:12] = 7
        output, _ = self.process(
            labels, (0.2, 0.2, 0.2),
            closing_radius_mm=0, smoothing_sigma_mm=0,
        )
        expected = labels.copy()
        expected[9:12, 9:12, 9:12] = 0
        np.testing.assert_array_equal(output, expected)

    def test_empty_volume_keeps_full_shape(self):
        labels = np.zeros((9, 10, 11), dtype=np.uint16)
        output, report = self.process(labels, (0.4, 0.2, 0.3))
        np.testing.assert_array_equal(output, labels)
        self.assertEqual(report, [])

    def test_multiclass_input_is_rejected(self):
        labels = np.zeros((9, 9, 9), dtype=np.uint16)
        labels[2, 2, 2] = 1
        labels[6, 6, 6] = 2
        with self.assertRaisesRegex(ValueError, "one nonzero class"):
            self.process(labels, (0.2, 0.2, 0.2))


if __name__ == "__main__":
    unittest.main()
