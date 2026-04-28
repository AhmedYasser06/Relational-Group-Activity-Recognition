"""
B1  Baseline (No Relations)

Stage 1: Fine-tune ResNet50, each person  2048-d feature.
Stage 2: Project each person to 128-d via a shared dense layer,
         split into 2 teams, max-pool each team to (1, 128),
         concatenate  256-d  classifier.
"""

import torch
import torch.nn as nn
from .model_utils import PersonClassifier, freeze, collate_fn


class GroupClassifier(nn.Module):
    def __init__(self, person_model, num_classes):
        super().__init__()

        self.resnet50 = person_model.resnet50
        freeze(self.resnet50)

        self.dense = nn.Linear(2048, 128)
        self.pool = nn.AdaptiveMaxPool2d((1, 128))

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        b, bb, c, h, w = x.shape
        x = x.view(b * bb, c, h, w)
        x = self.resnet50(x).view(b, bb, -1)   # (b, bb, 2048)
        x = self.dense(x)                       # (b, bb, 128)

        team1 = self.pool(x[:, :6, :])          # (b, 1, 128)
        team2 = self.pool(x[:, 6:, :])          # (b, 1, 128)

        x = torch.cat([team1, team2], dim=1)    # (b, 2, 128)
        x = x.view(b, -1)                       # (b, 256)
        return self.fc(x)
