import os
import torch

# Dataset Paths
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

# Image Settings
IMAGE_SIZE = 224

# Training Settings
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

# DataLoader
NUM_WORKERS = 0

# Random Seed
SEED = 42

# Model Settings
NUM_CLASSES = 2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output
# os.makedirs("outputs", exist_ok=True)
# # MODEL_SAVE_PATH = "outputs/latentguard_best.pth"
# MODEL_SAVE_PATH = "outputs/latentguard_robust.pth"

# Output Directories
OUTPUT_DIR = "outputs"

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Best Model Path
MODEL_SAVE_PATH = os.path.join(
    OUTPUT_DIR,
    "latentguard_20epoch_best.pth"
)


# Latest Checkpoint Path
LATEST_CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "latest_checkpoint.pth"
)

