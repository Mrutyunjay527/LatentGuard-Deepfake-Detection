import torch
from torch.utils.data import DataLoader

from config import *

from src.dataset import ImageDataset
from src.transforms import train_transform

from models.image_encoder import ImageEncoder
from models.vae_encoder import VAEEncoder
from models.latent_consistency import LatentConsistency
from models.fusion import FusionMLP
from models.classifier import BinaryClassifier

from config import DEVICE

print("Using Device:", DEVICE)

print("=" * 50)
print("LATENTGUARD SANITY CHECK")
print("=" * 50)

# Dataset
dataset = ImageDataset(
    TRAIN_DIR,
    transform=train_transform
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

images, labels = next(iter(loader))

print("Input Images :", images.shape)
print("Labels       :", labels.shape)

images = images.to(DEVICE)

# Image Encoder
image_encoder = ImageEncoder().to(DEVICE)

image_feature = image_encoder(images)

print("Image Encoder Output :", image_feature.shape)

# VAE Encoder
vae_encoder = VAEEncoder().to(DEVICE)

latent_feature = vae_encoder(images)

print("VAE Output           :", latent_feature.shape)


# Consistency
consistency = LatentConsistency().to(DEVICE)

consistency_feature = consistency(
    image_feature,
    latent_feature
)

print("Consistency Output   :", consistency_feature.shape)


# Fusion
fusion = FusionMLP().to(DEVICE)

fusion_feature = fusion(consistency_feature)

print("Fusion Output        :", fusion_feature.shape)


# Classifier
classifier = BinaryClassifier().to(DEVICE)

prediction = classifier(fusion_feature)

print("Classifier Output    :", prediction.shape)

print("=" * 50)
print("SANITY CHECK PASSED")
print("=" * 50)