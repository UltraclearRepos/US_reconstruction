import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from src.segmentation import checkpoint, runner


class SegmentationTasksTests(unittest.TestCase):
    def setUp(self):
        self.frames = np.zeros((1, 4, 6), dtype=np.uint8)
        self.settings = {"image_size": (2, 3), "in_channels": 1, "n_classes": 2}
        self.thyroid = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint16)
        self.nodule = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.uint16)
        self.outputs = {
            name: torch.from_numpy(np.stack([1 - mask, mask]).astype(np.float32))[None]
            for name, mask in (("thyroid", self.thyroid), ("nodule", self.nodule))
        }

    def model(self, inputs):
        return self.outputs

    def predict(self, tasks):
        with patch.object(runner, "SEGMENTATION_TASKS", tasks):
            return runner.predict_masks_for_sequence(
                self.frames, self.model, self.settings, torch.device("cpu"),
            )

    def test_single_task_labels(self):
        for task, mask, label in (("thyroid", self.thyroid, 1), ("nodule", self.nodule, 2)):
            with self.subTest(task=task):
                expected = np.repeat(np.repeat(mask * label, 2, axis=0), 2, axis=1)
                result = self.predict([task])
                np.testing.assert_array_equal(result[0], expected)
                self.assertEqual(result.dtype, np.uint16)

    def test_both_tasks_and_overlap_priority_ignore_list_order(self):
        expected = self.thyroid.copy()
        expected[self.nodule != 0] = 2
        expected = np.repeat(np.repeat(expected, 2, axis=0), 2, axis=1)
        for tasks in (["thyroid", "nodule"], ["nodule", "thyroid"]):
            np.testing.assert_array_equal(self.predict(tasks)[0], expected)

    def test_invalid_selection_and_missing_output_fail(self):
        for tasks in ([], ["typo"]):
            with self.assertRaises(ValueError):
                self.predict(tasks)
        del self.outputs["nodule"]
        with self.assertRaisesRegex(KeyError, "nodule"):
            self.predict(["nodule"])
        self.predict(["thyroid"])

    def test_saved_labels_and_task_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "labels.h5"
            sweep = (self.frames, np.eye(4)[None], (0.2, 0.3), "img")
            with patch.object(runner, "SEGMENTATION_TASKS", ["thyroid", "nodule"]), \
                 patch.object(runner, "segmentation_path", return_value=output), \
                 patch.object(runner, "load_sweep_h5", return_value=sweep):
                runner.segment_case("synthetic", loaded_model=(self.model, self.settings, "cpu"))
            with h5py.File(output) as handle:
                np.testing.assert_array_equal(handle["labels"][:], self.predict(["thyroid", "nodule"]))
                self.assertEqual(list(handle.attrs["tasks"]), ["thyroid", "nodule"])
                self.assertNotIn("task", handle.attrs)
            self.assertTrue((output.parent / "images" / "frame_000000.png").is_file())

    def test_checkpoint_does_not_select_or_require_task_mappings(self):
        metadata = {
            "state_dict": {"model.weight": torch.zeros(1)},
            "hyper_parameters": {
                "config": {
                    "model": {"in_channels": 1, "bilinear": True},
                    "data": {"image_size": [2, 3]},
                },
                "num_classes": 2,
            },
        }
        with tempfile.NamedTemporaryFile() as file:
            for mappings in (None, {"thyroid": {}, "nodule": {}}):
                if mappings is not None:
                    metadata["hyper_parameters"]["class_name_mappings"] = mappings
                with patch.object(checkpoint.torch, "load", return_value=metadata), \
                     patch.object(checkpoint, "UNet"):
                    _, settings = checkpoint.load_checkpoint(file.name, "cpu")
                self.assertEqual(settings, self.settings)


if __name__ == "__main__":
    unittest.main()
