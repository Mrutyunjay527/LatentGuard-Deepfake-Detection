import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2)

        )

    def forward(self, x):
        assert x.shape[1] == 256, \
               f"Expected 256 features, got {x.shape}"

        return self.classifier(x)