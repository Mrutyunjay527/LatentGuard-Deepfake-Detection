import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class VAEEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Load Stable Diffusion VAE
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        )

        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False

        # Pool latent feature map
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Project to 768 dimensions
        self.projection = nn.Linear(4, 768)

    def forward(self, images):
        with torch.no_grad():
            latent = self.vae.encode(images).latent_dist.sample()

        pooled = self.pool(latent)

        pooled = pooled.view(pooled.size(0), -1)

        features = self.projection(pooled)
        assert features.ndim == 2, \
            f"Expected 2D tensor, got {features.shape}"

        assert features.shape[1] == 768, \
            f"Expected 768 features, got {features.shape}"

        return features