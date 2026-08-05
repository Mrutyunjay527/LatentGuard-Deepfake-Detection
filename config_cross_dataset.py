import torch

# CIFAKE test dataset
TEST_DIR = "data/cifake_test/test"

# Model
MODEL_SAVE_PATH = "outputs/latentguard_best.pth"

# Image Settings
IMAGE_SIZE = 224
BATCH_SIZE = 8

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")