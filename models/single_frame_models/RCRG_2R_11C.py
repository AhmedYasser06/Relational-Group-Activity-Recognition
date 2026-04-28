"""
RCRG-2R-11C

Two relational layers, both operating on 1 clique (all 12 players).
Outputs are concatenated per player  max-pool per team classifier.
"""

import torch
import torch.nn as nn
import itertools
from .model_utils import PersonClassifier, freeze, collate_fn
from .relational_layer import RelationalLayer


class GroupActivityClassifier(nn.Module):
    def __init__(self, person_model, num_classes, device):
        super().__init__()

        self.device = device
        self.resnet50 = person_model.resnet50
        freeze(self.resnet50)

        self.r1 = RelationalLayer(input_size=2048, hidden_size=512,  output_size=128)
        self.r2 = RelationalLayer(input_size=2048, hidden_size=1024, output_size=256)

        self.pool = nn.AdaptiveMaxPool2d((1, 384))

        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _full_clique(self, n):
        pairs = list(itertools.permutations(range(n), 2))
        return torch.tensor(pairs, dtype=torch.long).t().to(self.device)

    def forward(self, x):
        b, bb, c, h, w = x.shape
        x = x.view(b * bb, c, h, w)
        x = self.resnet50(x).view(b, bb, -1)       # (b, bb, 2048)

        edge_index = self._full_clique(bb)

        x1 = self.r1(x, edge_index)                # (b, bb, 128)
        x2 = self.r2(x, edge_index)                # (b, bb, 256)
        x = torch.cat([x1, x2], dim=2)             # (b, bb, 384)

        team1 = self.pool(x[:, :6, :])             # (b, 1, 384)
        team2 = self.pool(x[:, 6:, :])             # (b, 1, 384)

        x = torch.cat([team1, team2], dim=1).view(b, -1)   # (b, 768)
        return self.fc(x)
