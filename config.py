from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "prepared"
SEGMENTED_DIR = DATA_DIR / "segmented"
RECONSTRUCTED_DIR = DATA_DIR / "reconstructed"
RECONSTRUCTED_INTENSITY_DIR = RECONSTRUCTED_DIR / "intensity"
RECONSTRUCTED_SEGMENTATION_DIR = RECONSTRUCTED_DIR / "segmentation"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
SURFACES_DIR = DATA_DIR / "surfaces"
GROUND_TRUTH_SURFACES_DIR = SURFACES_DIR / "ground_truth"
PREDICTED_SURFACES_DIR = SURFACES_DIR / "predicted"
EVALUATION_DIR = ROOT / "reports" / "evaluation"

SEGMENTATION_MODELS_DIR = ROOT / "models" / "segmentation"


# DICOM preparation
VERBOSE = True
CONVERT_IMAGE_TO_FLOAT32 = True
IMAGE_DATASET_NAME = "img"


# Segmentation
SEGMENTATION_MODEL_NAME = "unet"
SEGMENTATION_CHECKPOINT = SEGMENTATION_MODELS_DIR / "best-epoch_008=066.ckpt"
# Select either task or both. Output labels: thyroid=1, nodule=2 (nodule wins overlaps).
SEGMENTATION_TASKS = ["thyroid", "nodule"]
SEGMENTATION_IMAGE_KEY = None
SEGMENTATION_DEVICE = None

# Reconstruction
INTENSITY_RECONSTRUCTION_METHODS = ["voxel_nearest"]

# Multiclass segmentation must use nearest-neighbour reconstruction because
# averaging methods would produce invalid class IDs.
SEGMENTATION_RECONSTRUCTION_METHOD = "voxel_nearest"
# Maximum perpendicular voxel-to-frame distance in mm; None disables the limit.
# Initial value to tune against acquisition spacing. Applies to all methods.
RECONSTRUCTION_MAX_PLANE_DISTANCE_MM = 1.0
# XYZ spacing in mm; None reads voxel_size from this case's GT (labels, then volume).
VOXEL_SIZE = (0.2, 0.2, 0.2)
PIXEL_SPACING = None
BACKGROUND_THRESHOLD = 0.0
CENTER = True
CHUNK_PIXELS = None
IMAGE_KEY = None
DEVICE = "cuda"
N_NEAREST = 4
SPLINE_ORDER = 3
DISTANCE_EPS = 1e-6
DISTANCE_POWER = 1.0

VNN_N_NEAREST = 1
VNN_MEAN_N_NEAREST = 4

DWR_N_NEAREST = 4
DWR_DISTANCE_POWER = 2.0
DWR_DISTANCE_EPS = 1e-6
DWR_INTERP = "bilinear"


# Postprocessing of reconstructed segmentation only (distances in mm).
SEGMENTATION_POSTPROCESS_ENABLED = True
SEGMENTATION_SMOOTHING_THRESHOLD = 0.5
# Class 1 (thyroid): closing, largest component and smoothing.
SEGMENTATION_CLOSING_RADIUS_MM = 0.0
SEGMENTATION_SMOOTHING_SIGMA_MM = 1.0
# Class 2 (nodules): Gaussian smoothing of each component, keeping all nodules.
SEGMENTATION_NODULE_SMOOTHING_SIGMA_MM = 0.4


# Isosurfaces
ISO_MIN_VOXELS = 20
ISO_STEP_SIZE = 1
ISO_SAVE_OBJ = True
ISO_CLASSES = None


# Rigid surface alignment (shared by the viewer and both evaluations).
ALIGN_SAMPLES = 10000  # Points sampled per surface for PCA and fine ICP.
ALIGN_COARSE_SAMPLES = 3000  # Per surface; capped at ALIGN_SAMPLES.
ALIGN_COARSE_ITERATIONS = 60  # Maximum iterations per initial orientation.
ALIGN_FINE_ITERATIONS = 150  # Maximum iterations per shortlisted candidate.
ALIGN_FINE_CANDIDATES = 4  # Best coarse candidates refined at full sample density.
ALIGN_OVERLAP = 0.9  # Nearest pair fraction retained separately in each direction.
ALIGN_RANDOM_SEED = 0  # Fixed seed makes surface sampling repeatable.


# Visualization
# Keep Napari focused on segmentation volumes and surfaces to reduce memory use.
SHOW_RECONSTRUCTION = False
SHOW_GROUND_TRUTH_VOLUME = False
SHOW_SEGMENTATION_LABELS = True
SHOW_SURFACES = True
NAPARI_NDISPLAY = 3


# Evaluation after freshly computed rigid surface alignment (local XYZ mm).
EVAL_CLASS_ID = 1
EVAL_SURFACE_TOLERANCE_MM = 1.0
EVAL_VOLUME_CHUNK_DEPTH = 16
