# Ultrasound reconstruction pipeline

Pipeline converts a prepared 2D ultrasound sweep into an intensity volume or
a reconstructed 3D segmentation. Technical parameters are in `config.py`;
cases and `RUN_*` switches are at the top of `main.py`. Implementation packages
live under `src/`.

## Data contract

`case` always means the acquisition name. The prepared sweep is the single
input used by every downstream stage.

```text
data/
├── raw/<case>/*.dcm
├── prepared/<case>/sweep.h5
├── segmented/<case>/<model>/
│   ├── labels.h5
│   └── images/frame_*.png
├── reconstructed/
│   ├── intensity/<case>/<method>/volume.npz
│   └── segmentation/<case>/<model>/voxel_nearest/labels.npz
├── ground_truth/<case>/
│   ├── volume.npz
│   └── labels.npz
└── surfaces/
    ├── ground_truth/<case>/
    └── predicted/<case>/<model>/voxel_nearest/

reports/evaluation/{intensity,segmentation,no_reference}/
models/segmentation/
```

Every sweep-like H5 contains frames or labels with shape `(N,H,W)`, probe
poses `(N,4,4)`, and pixel spacing. Each stage writes one canonical artifact;
NPZ and per-class H5 copies are not created.

## Running the pipeline

Select cases at the top of `main.py`:

```python
CASE_NAMES = ["kosartur"]
```

Enable only the required stages:

```python
RUN_PREPARE_DATA = True
RUN_SEGMENTATION = True
RUN_RECONSTRUCT_INTENSITY = True
RUN_RECONSTRUCT_SEGMENTATION = True
RUN_EXTRACT_SEGMENTATION_SURFACES = True
```

Every enabled pipeline stage prints `[TIME] <stage>: START` and its elapsed
wall-clock seconds on completion. Evaluation branches are timed separately;
segmentation postprocessing is timed as one complete operation inside
segmentation reconstruction. `PIPELINE TOTAL` includes all enabled stages.
A failing or interrupted stage prints `FAILED` with its elapsed time and
propagates the error. Measurements include imports, data loading and saving
inside each stage. Parent times include their nested stages and should not be
added together. Visualization times include time spent in an open interactive
viewer; they finish when its call returns, usually after the window is closed.

Stages do not have to run in one invocation. Reconstruction can read an
existing segmentation with earlier switches disabled.

Choose the segmentation artifacts used downstream:

```python
SEGMENTATION_RESULT_MODELS = [SEGMENTATION_MODEL_NAME]
```

Segmentation runs the local U-Net with `SEGMENTATION_CHECKPOINT` and writes
`data/segmented/<case>/<model>/labels.h5`. Reconstruction, surface extraction,
visualization and evaluation read this artifact. The pipeline does not require
an external segmentation tool.

Multiclass segmentation uses `voxel_nearest` reconstruction. Averaging methods
are reserved for intensity because averaging class IDs would create invalid
labels.

`RECONSTRUCTION_MAX_PLANE_DISTANCE_MM` in `config.py` limits the perpendicular
distance from a voxel centre to a frame plane for **all reconstruction methods**,
both intensity and segmentation. The initial setting is `1.0` mm; it has not
been tuned on the data. A frame contributes only when the distance is less than
or equal to the limit and the existing image-boundary/background checks pass.
Methods using multiple frames exclude each distant frame before aggregation
and normalize using only accepted contributions. With no valid contribution,
the reconstructed value and `weights` are both zero. Distance uses the pose
coordinate units (expected millimetres and rigid transforms), not voxel counts.
`None` disables the limit; zero accepts only voxel centres on a frame plane.
Negative and nonfinite limits are rejected. This shared setting replaces
`DWR_MAX_PLANE_DIST`. Each output NPZ records `max_plane_dist` in mm (`-1` means
disabled). Too small a limit can leave gaps between sparsely acquired frames.
The limit applies during reconstruction; the subsequent segmentation closing
and smoothing may expand labels into voxels without original sampling support.

Reconstructed segmentation is postprocessed before `labels.npz` is saved.
Postprocessing expects background zero and at most one non-background class;
multiclass input is rejected and the original foreground class ID is preserved.
The pipeline closes small gaps with a spherical 3D footprint, retains the
largest 26-connected component, and applies Gaussian smoothing followed by
thresholding. Another largest-component selection removes fragments split off
by smoothing. If smoothing would erase the class completely, its cleaned mask
is retained. An empty input produces an empty output and an empty class report.

Segmentation postprocessing uses CuPy and `cupyx.scipy.ndimage` on the
current CUDA device (GPU 0 by default). It does not change the active device;
`DEVICE` continues to configure reconstruction.
The sequence of operations, single working `mask` and Gaussian `scores`
are unchanged. Input validation, crop preparation and assembly of the saved
NumPy volume remain on CPU. The cropped mask is uploaded once and the final
mask is downloaded with synchronization, so the stage timer includes completed
GPU work and transfers. The console and class report identify the actual GPU;
the class report is also saved in the NPZ processing metadata.
There is one GPU implementation, requiring CuPy and a working CUDA device.
The first GPU run may include kernel compilation time. Floating-point rounding
near the smoothing threshold can cause small differences from the former
SciPy implementation.

This WSL environment has CUDA Toolkit 12.8; its GPU dependency is
`cupy-cuda12x>=14.0,<15.0` in `requirements.txt`. Install it with:

```bash
.venv/bin/python -m pip install 'cupy-cuda12x>=14.0,<15.0'
```

For a different CUDA Toolkit, use the matching wheel from the
[CuPy installation instructions](https://docs.cupy.dev/en/stable/install.html).
Install only one CuPy wheel variant in an environment.

The controls in `config.py` are `SEGMENTATION_POSTPROCESS_ENABLED` (default
`True`), `SEGMENTATION_CLOSING_RADIUS_MM` (`0.6` mm),
`SEGMENTATION_SMOOTHING_SIGMA_MM` (`0.4` mm), and
`SEGMENTATION_SMOOTHING_THRESHOLD` (`0.5`). A zero radius or sigma skips that
operation. Closing can join nearby fragments, but its radius is not an exact
maximum gap distance: the result also depends on fragment geometry. Increase
the radius cautiously if real parts remain disconnected. Spacing is handled
in physical units, including anisotropic voxels.

Postprocessing changes the saved labels used by surfaces, viewing and metrics.
Intensity and ground truth are unaffected. Reconstruction `weights` still
describe the original sampling support, not smoothing confidence. The NPZ
includes processing parameters and per-class counts in the JSON string
`segmentation_postprocessing`. To refresh existing results, enable
`RUN_RECONSTRUCT_SEGMENTATION` and `RUN_EXTRACT_SEGMENTATION_SURFACES` in
`main.py`; existing 2D segmentations can be reused.

Surface extraction accepts only reconstructed segmentation labels. Intensity
volumes are not an input to this stage.

Run:

```bash
.venv/bin/python main.py
```

On Linux/WSLg with Wayland, the Napari viewer defaults to
`QT_QPA_PLATFORM=xcb` and `PYOPENGL_PLATFORM=glx` before importing Napari.
This pairs Qt's X11 context with PyOpenGL's GLX backend to avoid
`Attempt to retrieve context when no valid context`.
Explicit environment overrides are preserved.
See the [Napari workaround](https://github.com/napari/napari/pull/9009).

If activating a moved virtual environment still loads packages from
`~/.local`, check that `VIRTUAL_ENV` in `.venv/bin/activate` points to the
current project directory. Reactivate it after correcting the path, or use
`.venv/bin/python` directly. Use `.venv/bin/python -m pip` for package installation.

Evaluation is terminal output, not input data, and is written under
`reports/evaluation`. Ground truth is prepared only from separately supplied
DICOM Volume and Segmentation files; it is never generated by a model.
