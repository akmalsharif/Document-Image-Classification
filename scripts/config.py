"""Shared configuration for ConvNeXt document classification."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
MODEL_DIR = ROOT_DIR / "models"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
SUBMISSION_DIR = ROOT_DIR / "submissions"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
BASELINE_CHECKPOINT_PATH = CHECKPOINT_DIR / "convnext_tiny_best.pth"
CHECKPOINT_PATH = CHECKPOINT_DIR / "convnext_tiny_tuned_best.pth"
HISTORY_PATH = ROOT_DIR / "reports" / "convnext_tuned_history.csv"
SUBMISSION_PATH = SUBMISSION_DIR / "submission_convnext_tuned.csv"

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 1
EPOCHS = 4
VALIDATION_SIZE = 0.25
BACKBONE_LEARNING_RATE = 3e-5
CLASSIFIER_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 3e-2
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 0.002
DEVICE = "cuda"


