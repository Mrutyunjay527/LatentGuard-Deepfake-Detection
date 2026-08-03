import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


class ImageEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base"
        )

        self.model = AutoModel.from_pretrained(
            "facebook/dinov2-base"
        )

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):

        with torch.no_grad():
            outputs = self.model(pixel_values=images)

        cls_embedding = outputs.last_hidden_state[:, 0]

        assert cls_embedding.ndim == 2, \
            f"Expected 2D tensor, got {cls_embedding.shape}"

        assert cls_embedding.shape[1] == 768, \
            f"Expected embedding size 768, got {cls_embedding.shape}"

        return cls_embedding