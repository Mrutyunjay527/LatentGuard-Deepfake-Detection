import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(3073, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()

        )

    def forward(self, x):
        assert x.shape[1] == 3073, \
              f"Expected 3073 features, got {x.shape}"

        return self.network(x)