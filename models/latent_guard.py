import torch
import torch.nn as nn

from models.latent_consistency import LatentConsistency
from models.fusion import FusionMLP
from .classifier import BinaryClassifier

class LatentGuard(nn.Module):
    def __init__(self, image_encoder, vae_encoder):
        super().__init__()

        # Frozen encoders (provided by teammates)
        self.image_encoder = image_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.vae_encoder = vae_encoder
        for param in self.vae_encoder.parameters():
            param.requires_grad = False

        # Our modules
        self.consistency = LatentConsistency()
        self.fusion = FusionMLP()
        self.classifier = BinaryClassifier()

    def forward(self, images):

        # Extract semantic features
        image_feature = self.image_encoder(images)
        assert image_feature.ndim == 2, \
            f"Expected 2D tensor, got {image_feature.shape}"

        assert image_feature.shape[1] == 768, \
            f"Image encoder output should be 768, got {image_feature.shape}"
        
        # Extract latent features
        latent_feature = self.vae_encoder(images)
        assert latent_feature.ndim == 2, \
            f"Expected 2D tensor, got {latent_feature.shape}"

        assert latent_feature.shape[1] == 768, \
            f"VAE encoder output should be 768, got {latent_feature.shape}"

        # Build consistency representation
        consistency_feature = self.consistency(
            image_feature,
            latent_feature
        )

        # Fuse features
        fused_feature = self.fusion(
            consistency_feature
        )

        # Final prediction
        output = self.classifier(
            fused_feature
        )

        return output   