import torch
import torch.nn as nn
import torch.nn.functional as fn


class LatentConsistency(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_feature, latent_feature):

        # schematic similarity
        cosine = fn.cosine_similarity(
            image_feature,
            latent_feature,
            dim=1
        ).unsqueeze(1)

         #feature discrepancy
        difference = torch.abs(
            image_feature - latent_feature
        )

        # feature interaction
        product = image_feature * latent_feature

        # final consistency representation
        consistency_feature = torch.cat(
            [
                image_feature,
                latent_feature,
                difference,
                product,
                cosine
            ],
            dim=1
        )

        return consistency_feature