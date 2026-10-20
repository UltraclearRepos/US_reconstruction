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
    ├── predicted/<case>/<model>/voxel_nearest/
    └── aligned/<case>/<model>/voxel_nearest/

models/segmentation/
```

Every sweep-like H5 contains frames or labels with shape `(N,H,W)`, probe
poses `(N,4,4)`, and pixel spacing. Each stage writes one canonical artifact;
NPZ and per-class H5 copies are not created.

GT DICOM SEG import reads each frame's `ReferencedSegmentNumber` and
`ReferencedFrameNumber`, preserving segment IDs on the referenced source-volume
grid. Interleaved thyroid and nodule frames become one label volume with the
source depth, not a doubled Z stack. For the current files, 0 is background,
1 is thyroid and 2 is nodule. Where masks overlap, the higher segment ID wins,
so nodule voxels remain 2; the whole thyroid including nodules is the union of
labels 1 and 2. Fractional masks are thresholded at half `MaximumFractionalValue`.
Source dimensions, spacing, UID and frame numbers are checked; incompatible
geometry raises an error rather than being resampled. The SEG file must reference
one source volume with an explicit source frame number for each mask frame.
Re-run preparation and GT surface extraction to refresh existing artifacts.
Previously generated alignment and evaluation reports need to be recomputed.

`USE_NEW_GT_LOADER` in `main.py` chooses the GT segmentation importer:
`True` uses the corrected source-frame/segment mapping (default); `False`
reproduces the original raw frame stack with its original pixel values.
The choice is applied during `RUN_PREPARE_DATA`, not when opening existing NPZ
files. Enable `RUN_PREPARE_DATA` and `RUN_EXTRACT_SEGMENTATION_SURFACES` to update
GT labels and meshes after switching. Surface extraction removes obsolete
generated class meshes, so old nodule surfaces do not remain in the viewer.

To compare the original and corrected GT imports directly from the same DICOM,
run `.venv/bin/python compare_gt_loaders.py` without command-line arguments.
Set `CASE_NAME` at the top of that standalone script (default `003_02`).
Napari shows BEFORE on the left and AFTER on the right at the same physical
scale. It displays only surfaces and releases the complete label volumes before
opening the viewer. Software rendering is enabled for this comparison
to avoid the observed WSL D3D12 driver crash. No pipeline data is overwritten.

Newly extracted reconstruction surface NPZ files also contain `mesh_to_mapping`,
a homogeneous 4x4 matrix mapping local mesh XYZ millimetres to the coordinate
frame of the reconstruction poses (the mapping reference, if poses are relative
to its last frame). Apply it with column vectors:
`p_mapping = mesh_to_mapping @ [x, y, z, 1]`.
Its rotation is identity and its translation is
`origin_xyz + 0.5 * voxel_size[::-1]`, accounting for voxel centres and ZYX
volume spacing. Existing `verts`, faces, normals, OBJ coordinates and evaluation
remain local and unchanged. The field is omitted for inputs without
`origin_xyz`, including current GT files, whose relationship to tracking is
unknown. Re-extract existing surfaces to add the field; reconstruction does
not need to be rerun. GT alignment transforms are separate from this mapping.

## Tracking replay video

Set `CASE_NAME` and the video settings at the top of
`src/visualization/tracking_video.py`, then create an MP4 from an existing
predicted surface and prepared sweep:

```bash
.venv/bin/python -m src.visualization.tracking_video
```

The video shows one opaque exterior surface, as in a 3D surface viewer. The
whole mesh moves and rotates relative to the probe using its inverse pose;
the orthographic camera stays fixed throughout the animation. It smoothly
replays recorded translations and rotations forward and backward, using
spherical rotation interpolation. Both poses and mesh are rebased to the last
recorded frame. No reconstruction or GT alignment is run.
Legacy meshes without `mesh_to_mapping` recover it from the reconstruction NPZ.
Output: `outputs/tracking/<case>_tracking_surface.mp4` and a PNG preview.
Settings include `DURATION_SECONDS`, `FPS`, `WIDTH`, `HEIGHT`, `MODEL_NAME`,
`RECONSTRUCTION_METHOD`, `CLASS_ID`, and optional `SURFACE_PATH`, `SWEEP_PATH`,
`VOLUME_PATH`, `OUTPUT_PATH` (use `Path("...")`; `None` selects canonical case
paths). The default output is relative to the project root. Rendering uses VTK;
MP4 encoding requires an `ffmpeg` executable with `libx264` on PATH.

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
wall-clock seconds on completion. Surface alignment is timed separately;
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
and visualization read this artifact. The pipeline does not require
an external segmentation tool.

`SEGMENTATION_TASKS` in `config.py` selects the model outputs to use:
`["thyroid"]`, `["nodule"]`, or `["thyroid", "nodule"]` (current default).
Task selection does not read checkpoint `class_name_mappings`. Saved labels
always use 0 for background, 1 for thyroid and 2 for nodules, including when
only nodules are selected. Nodules win overlaps regardless of task list order.
The H5 `tasks` attribute records the configured list. Rerun segmentation,
segmentation reconstruction and surface extraction to refresh existing results.

Multiclass segmentation uses `voxel_nearest` reconstruction. Averaging methods
are reserved for intensity because averaging class IDs would create invalid
labels.

`VOXEL_SIZE` in `config.py` specifies reconstruction spacing in XYZ order, in mm.
Set it to `None` to use each case's GT spacing for both intensity and segmentation
reconstruction, for all configured intensity methods. The pipeline reads only
`voxel_size` metadata from `data/ground_truth/<case>/labels.npz`, falling back to
`volume.npz` if the labels file is absent, and converts its ZYX spacing to XYZ.
Missing GT or missing, nonfinite, nonpositive or incorrectly shaped spacing raises
an explicit error. A manually configured tuple continues to work without GT.
This selects voxel spacing; array dimensions and origin still come from the sweep.
Rerun reconstruction and surface extraction to apply a new spacing to saved data.

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
Postprocessing expects background 0, thyroid 1 and nodules 2. For thyroid,
the pipeline closes small gaps with a spherical 3D footprint, retains the
largest 26-connected component, and applies Gaussian smoothing followed by
thresholding. Another largest-component selection removes fragments split off
by smoothing. For nodules, each original 26-connected component is smoothed
independently and thresholded, without closing or largest-component selection.
The resulting nodule masks are combined; nearby results can touch or merge.
If smoothing would erase a thyroid mask or an individual nodule completely,
its mask before smoothing is retained. Nodules take priority over thyroid where
the processed masks overlap. Reports count the final masks after this priority
is applied. An empty input produces an empty output and an empty class report.

Segmentation postprocessing uses CuPy and `cupyx.scipy.ndimage` on the
current CUDA device (GPU 0 by default). It does not change the active device;
`DEVICE` continues to configure reconstruction.
Input validation, crop preparation, finding nodule component bounding boxes,
and assembly of the saved NumPy volume run on CPU. Connected-component labeling,
morphology and Gaussian smoothing run on GPU. Final masks are downloaded with
synchronization, so the stage timer includes completed GPU work and transfers.
The console and class report identify the actual GPU;
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
`True`), `SEGMENTATION_CLOSING_RADIUS_MM` (`0.0` mm),
`SEGMENTATION_SMOOTHING_SIGMA_MM` (`1.0` mm), and
`SEGMENTATION_SMOOTHING_THRESHOLD` (`0.5`). A zero radius or sigma skips that
operation. Closing can join nearby fragments, but its radius is not an exact
maximum gap distance: the result also depends on fragment geometry. Increase
the radius cautiously if real parts remain disconnected. Spacing is handled
in physical units, including anisotropic voxels.

Thyroid uses `SEGMENTATION_CLOSING_RADIUS_MM` and
`SEGMENTATION_SMOOTHING_SIGMA_MM` (currently `0.0` and `1.0` mm in config).
Nodules use their own `SEGMENTATION_NODULE_SMOOTHING_SIGMA_MM` (initial value
`0.4` mm; zero disables their smoothing). Both classes use the shared
`SEGMENTATION_SMOOTHING_THRESHOLD`. The nodule sigma is saved in the NPZ
processing metadata, along with each class's actual smoothing sigma and counts.

Postprocessing changes the saved labels used by surfaces and viewing.
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

Before either viewer initializes Qt on Linux, the pipeline sets the current
user-owned `XDG_RUNTIME_DIR` permissions to `0700`, as required by Qt, to avoid
the `QStandardPaths: wrong permissions on runtime directory` warning. If the
variable is unset, an existing `/run/user/<uid>` directory is used. Missing,
foreign-owned or symlink directories are left unchanged.

If activating a moved virtual environment still loads packages from
`~/.local`, check that `VIRTUAL_ENV` in `.venv/bin/activate` points to the
current project directory. Reactivate it after correcting the path, or use
`.venv/bin/python` directly. Use `.venv/bin/python -m pip` for package installation.

## Surface alignment with GT

`src/registration/align_surfaces.py` provides rigid alignment for visualization and evaluation. It reads the
existing GT and predicted `class_1_surface.npz` meshes and attempts a rigid
alignment using area-weighted surface sampling, 24 PCA axis orientations,
unchanged and translation-only starts, and coarse/fine bidirectional ICP.
It uses NumPy and SciPy already in the project. It estimates only rotation and
translation, without scaling, reflection, deformation or volume resizing.
This is a geometric alignment attempt: symmetric or different shapes can
produce an incorrect anatomical match even when the distance is small.

Run it without rerunning preparation, segmentation or reconstruction:

```bash
.venv/bin/python -m src.registration.align_surfaces --case 003_01
```

The alignment viewer always fits thyroid (class 1). Optional controls are
`--class-id 1` (only 1 is accepted), `--model unet`,
`--method voxel_nearest`, `--samples 8000`, and `--overlap 0.9`.
Shared defaults for the pipeline, standalone alignment, and both evaluations
are in the surface alignment section of `config.py`: `ALIGN_SAMPLES` (8000 per
surface), `ALIGN_COARSE_SAMPLES` (1500), `ALIGN_COARSE_ITERATIONS` (60),
`ALIGN_FINE_ITERATIONS` (150), `ALIGN_FINE_CANDIDATES` (4), `ALIGN_OVERLAP` (0.9),
`ALIGN_RANDOM_SEED` (0).
The sample/overlap CLI flags override their config defaults for that run.
Higher sample counts can resolve smaller surface details but increase runtime;
more iterations only help if ICP has not already converged. Convergence
tolerances remain internal to ICP.
Overlap is the nearest correspondence fraction retained **in each direction**
during ICP and candidate scoring; it is not an estimate of acquisition coverage.
`--overlap 1` uses all correspondences. For a very incomplete reconstruction,
the symmetric objective and PCA initialization may still fail.
`--no-view` computes and saves the alignment without opening Napari.

The script prints the before/after symmetric mean nearest-neighbour distance
between sampled surfaces in mm as a diagnostic, not a validated evaluation
score. It saves the full aligned mesh and `transform_xyz` in
`data/surfaces/aligned/<case>/<model>/<method>/class_1_surface.npz`.
The homogeneous matrix maps the original predicted mesh's XYZ coordinates
to the GT mesh's XYZ coordinates. Both inputs retain their original files.
Existing meshes use local physical coordinates, so this estimates shape
alignment rather than recovering DICOM/world registration.

Napari displays every available class from both GT and predicted surface folders.
Thyroid GT is green and the aligned reconstruction is magenta. All GT nodules
are yellow; all reconstructed nodules are blue. Each class has one surface layer
per source, so separate nodules share the same color. Nodules are opaque and
thyroid remains translucent. A class present only in GT or only in prediction
is still displayed. Every predicted class receives the same rigid transform
computed from thyroid; nodules are never aligned independently in this viewer.
Before-alignment reconstruction layers are initially hidden (thyroid cyan,
nodules red); toggle them in the layer list to compare. Mesh XYZ coordinates
are converted to Napari ZYX. Only the fitted thyroid mesh is saved in the
aligned folder; other classes are transformed for display from original meshes.

In `main.py`, `RUN_ALIGN_SURFACES = True` performs this step after surface
extraction, using `VIS_CASE` or the first selected case and the first selected
segmentation model. It opens the alignment viewer instead of the ordinary
`RUN_NAPARI_VIS` viewer. Set it to `False` to return to ordinary visualization.

Ground truth is prepared only from separately supplied DICOM Volume and
Segmentation files; it is never generated by a model.

## Evaluation after alignment

The two independent switches immediately after `RUN_SIMPLE_VIS` in `main.py` are:

```python
RUN_EVAL_SURFACE = True
RUN_EVAL_VOLUME = True
```

Both stages run **after all visualization windows close**, for every selected
case and segmentation model. Each evaluator always calls
`src.registration.align_surfaces.compute_case_alignment` on the original meshes. It does not
read or reuse a previously saved alignment, transform or metrics report,
including the result from visualization or the other evaluator.
`RUN_ALIGN_SURFACES` controls the alignment viewer and is not required for metrics.
Class selection is `EVAL_CLASS_ID` in `config.py` (default 1).

Run either evaluator independently, without GUI or reconstruction:

```bash
.venv/bin/python -m src.evaluation.evaluate_surface --case 003_01
.venv/bin/python -m src.evaluation.evaluate_volume --case 003_01
```

`src/evaluation/evaluate_surface.py` evaluates **every vertex of both meshes**,
without random sampling or trimming. VTK finds the exact nearest point on the
opposite mesh's complete triangle surface, rather than its vertices.
Every triangle contributes one third of its area to each of its three vertices;
these weights prevent densely tessellated areas from dominating mean distances,
percentiles and Surface Dice. Reports include evaluated vertex and triangle
counts. VTK is an explicit requirement in `requirements.txt`.
The reported metrics are both directional mean distances, area-weighted ASSD
and RMSE, HD95 (maximum of the two area-weighted directional percentiles),
maximum vertex-to-mesh distance (`vertex_hausdorff_mm`), area-weighted Surface
Dice and surface area errors. All mesh vertices are used, but the statistics
remain a discrete approximation to integrals over continuous triangle interiors;
the maximum is over vertices, not a continuous Hausdorff maximum.
`EVAL_SURFACE_SAMPLES` and the evaluation `--samples` option have been removed.
`EVAL_SURFACE_TOLERANCE_MM` or `--tolerance-mm` controls Surface Dice tolerance.
Alignment still uses its own sampled point clouds and trimmed ICP; evaluation
always runs on the full meshes after the fresh transform is computed.

`src/evaluation/evaluate_volume.py` applies the newly computed transform to the original
segmentation mask, using the inverse XYZ-to-ZYX mapping and each volume's
physical spacing. It uses nearest-neighbour sampling onto the GT lattice,
extended to the union of both objects so that prediction outside GT is still
counted as false positives. Background outside the GT array is zero.
Foreground bounding boxes and slabs avoid allocating a full transformed
volume. `EVAL_VOLUME_CHUNK_DEPTH` or `--chunk-depth` controls slab depth.
It reports Dice, IoU, precision, recall, TP/FP/FN counts and physical FP/FN
volumes. Native physical volume errors use each mask's own voxel volume;
resampled volume and the change introduced by resampling are reported separately.

Both CLIs also accept `--class-id`, `--model`, `--method`,
`--alignment-samples` and `--overlap`. Reports include the fresh transform and
parameter definitions and are saved as:

```text
reports/evaluation/surface/<case>/<model>/<method>/class_1_metrics.json
reports/evaluation/volume/<case>/<model>/<method>/class_1_metrics.json
```

These scores assess shape agreement after rigid registration. Coordinates use
the local `index*spacing` convention of existing mesh extraction. Applying
the NPZ world origins in addition to that transform would misalign the masks.
