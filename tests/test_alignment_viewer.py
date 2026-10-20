import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.registration import align_surfaces as alignment


class AlignmentViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.gt_dir, self.pred_dir = root / "gt", root / "pred"
        self.gt_dir.mkdir()
        self.pred_dir.mkdir()
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        # Two separated nodules in the same class surface.
        self.nodules = np.concatenate((self.vertices + 2, self.vertices + 8))
        self.nodule_faces = np.concatenate((self.faces, self.faces + 3))
        for directory in (self.gt_dir, self.pred_dir):
            np.savez(directory / "class_1_surface.npz", verts=self.vertices, faces=self.faces)
            np.savez(directory / "class_2_surface.npz", verts=self.nodules, faces=self.nodule_faces)
        transform = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]])
        self.result = alignment.CaseAlignment(
            self.gt_dir / "class_1_surface.npz", self.pred_dir / "class_1_surface.npz",
            (self.vertices, self.faces), (self.vertices, self.faces),
            alignment.transform_points(self.vertices, transform), transform, 1.0, 0.0,
        )
        self.viewer = Mock()
        self.viewer.dims = SimpleNamespace()
        self.napari = SimpleNamespace(Viewer=Mock(return_value=self.viewer), run=Mock())
        self.patch_modules = patch.dict("sys.modules", napari=self.napari)
        self.patch_modules.start()
        self.addCleanup(self.patch_modules.stop)
        self.patch_qt = patch.object(alignment, "prepare_qt_environment")
        self.patch_qt.start()
        self.addCleanup(self.patch_qt.stop)

    def layers(self):
        return {call.kwargs["name"]: call for call in self.viewer.add_surface.call_args_list}

    def test_all_classes_colors_and_shared_transform(self):
        alignment.show_alignment(self.result, "synthetic")
        layers = self.layers()
        self.assertEqual(len(layers), 6)
        for name, color in (("GT thyroid", "green"), ("GT nodules", "yellow"),
                            ("Reconstruction aligned thyroid", "magenta"),
                            ("Reconstruction aligned nodules", "blue")):
            self.assertEqual(layers[name].kwargs["colormap"], color)
            self.assertTrue(layers[name].kwargs["visible"])
        for label in ("thyroid", "nodules"):
            self.assertFalse(layers[f"Reconstruction before alignment {label}"].kwargs["visible"])
        moved = alignment.transform_points(self.nodules, self.result.transform_xyz)
        np.testing.assert_allclose(layers["Reconstruction aligned nodules"].args[0][0], moved[:, ::-1])
        np.testing.assert_array_equal(layers["GT nodules"].args[0][0], self.nodules[:, ::-1])
        np.testing.assert_array_equal(layers["Reconstruction before alignment nodules"].args[0][0], self.nodules[:, ::-1])
        self.assertEqual(layers["GT thyroid"].kwargs["opacity"], 0.55)
        self.assertEqual(layers["GT nodules"].kwargs["opacity"], 1.0)

    def test_class_present_on_only_one_side_still_displays(self):
        for directory in (self.gt_dir, self.pred_dir):
            with self.subTest(missing_from=directory):
                path = directory / "class_2_surface.npz"
                path.unlink()
                self.viewer.add_surface.reset_mock()
                alignment.show_alignment(self.result, "synthetic")
                layers = self.layers()
                self.assertEqual("GT nodules" in layers, directory != self.gt_dir)
                self.assertEqual("Reconstruction aligned nodules" in layers, directory != self.pred_dir)
                np.savez(path, verts=self.nodules, faces=self.nodule_faces)

    def test_alignment_is_computed_once_on_thyroid(self):
        with patch.object(alignment, "compute_case_alignment", return_value=self.result) as compute, \
             patch.object(alignment, "SURFACES_DIR", Path(self.tmp.name) / "out"):
            output = alignment.align_and_view_case("synthetic")
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(compute.call_args.args[3], 1)
        with np.load(output) as data:
            np.testing.assert_array_equal(data["transform_xyz"], self.result.transform_xyz)
        self.assertIn("Reconstruction aligned nodules", self.layers())

    def test_viewer_rejects_fitting_nodules(self):
        with self.assertRaisesRegex(ValueError, "thyroid"):
            alignment.align_and_view_case("synthetic", class_id=2)


if __name__ == "__main__":
    unittest.main()
