# Pipeline script map

## End-to-end flow

```text
data/raw/<case>/*.dcm
    │
    ▼
src/data_loading/prepare_dicom.py
    │
    ▼
data/prepared/<case>/sweep.h5
    ├──► src/segmentation/runner.py
    │        └──► data/segmented/<case>/<model>/labels.h5
    │
    ├──► src/unsupervised_segmentation/runner.py
    │        └──► runs/unsupervised/<case>/<run>/
    │                 │
    │                 ▼
    │    src/data_loading/unsupervised_output_loader.py
    │        └──► data/segmented/<case>/<model>/labels.h5
    │
    └──► src/reconstruction/runner.py
             └──► data/reconstructed/intensity/<case>/<method>/volume.npz

data/segmented/<case>/<model>/labels.h5
    │
    ▼
src/reconstruction/runner.py
    │
    ▼
data/reconstructed/segmentation/<case>/<model>/<method>/labels.npz
    ├──► src/isosurface_extraction/extract.py
    ├──► src/evaluation/segmentation_metrics.py
    └──► src/visualization/
```

## Entry point

`main.py` contains case selection and all `RUN_*` switches. It imports a stage
only when that stage is enabled. `config.py` contains paths and algorithm
parameters.

## Modules

### `src/data_loading/`

- `prepare_dicom.py` converts the sweep DICOM to the canonical prepared H5 and
  writes optional ground-truth volumes.
- `sweep.py` reads and writes the shared sweep H5 contract.
- `h5_utils.py` contains lower-level H5 inspection helpers.
- `unsupervised_output_loader.py` imports PNG predictions from a selected or
  latest unsupervised run into the canonical segmentation H5 contract.

### `src/segmentation/`

- `runner.py` loads the configured checkpoint and predicts a label image for
  every prepared sweep frame.
- `checkpoint.py` validates the checkpoint contract and reconstructs the model.
- `model.py` defines the segmentation network.
- `common.py` contains preprocessing and mask resizing.

### `src/unsupervised_segmentation/`

`runner.py` prepares the external dataset, generates configuration files,
runs the external pipeline, and stores each run below
`runs/unsupervised/<case>/`.

### `src/reconstruction/`

`runner.py` reconstructs either the prepared intensity sweep or a selected
segmentation artifact. Intensity may use any configured reconstruction method;
multiclass segmentation is restricted to `voxel_nearest`.

### `src/isosurface_extraction/`

`extract.py` extracts per-class surfaces exclusively from ground-truth and
reconstructed segmentation labels. Intensity reconstructions are not accepted.

### `src/evaluation/`

- `reconstruction_with_gt.py` compares intensity reconstruction with GT.
- `segmentation_metrics.py` evaluates reconstructed labels against GT labels.
- `no_reference.py` evaluates intensity reconstruction without a reference
  volume.

Reports are stored below `reports/evaluation/` without being treated as input
data.

### `src/visualization/`

- `napari_viewer.py` provides the interactive 3D view.
- `simple_viewer.py` shows central slices with Matplotlib.
- `common.py` loads reconstructed NPZ volumes.

### `src/utils/`

- `paths.py` is the single source of truth for artifact locations.
- `logging.py` prints stage inputs and outputs.

## Canonical paths

```text
data/prepared/<case>/sweep.h5
data/segmented/<case>/<model>/labels.h5
data/reconstructed/intensity/<case>/<method>/volume.npz
data/reconstructed/segmentation/<case>/<model>/<method>/labels.npz
data/ground_truth/<case>/{volume,labels}.npz
data/surfaces/ground_truth/<case>/
data/surfaces/predicted/<case>/<model>/<method>/
runs/unsupervised/<case>/<run>/
reports/evaluation/<evaluation_type>/<case>/
```
