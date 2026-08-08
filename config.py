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
EPOCHS = 2
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
os.makedirs("outputs", exist_ok=True)
# MODEL_SAVE_PATH = "outputs/latentguard_best.pth"
MODEL_SAVE_PATH = "outputs/latentguard_robust.pth"