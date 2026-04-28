"""
RCRG-2R-21C

R1: 2 cliques — one per team (intra-team relations).
R2: 1 clique  — all 12 players (cross-team relations).
Player outputs are concatenated → max-pool per team → classifier.
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

        self.r1 = RelationalLayer(input_size=2048, hidden_size=512, output_size=128)
        self.r2 = RelationalLayer(input_size=2048, hidden_size=512, output_size=128)

        self.pool = nn.AdaptiveMaxPool2d((1, 256))

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _clique(self, n):
        pairs = list(itertools.permutations(range(n), 2))
        return torch.tensor(pairs, dtype=torch.long).t().to(self.device)

    def forward(self, x):
        b, bb, c, h, w = x.shape
        x = x.view(b * bb, c, h, w)
        x = self.resnet50(x).view(b, bb, -1)       # (b, bb, 2048)

        # R1: intra-team (2 separate cliques of 6)
        edge_6 = self._clique(6)
        x1_t1 = self.r1(x[:, :6, :], edge_6)       # (b, 6, 128)
        x1_t2 = self.r1(x[:, 6:, :], edge_6)       # (b, 6, 128)
        x_r1 = torch.cat([x1_t1, x1_t2], dim=1)    # (b, 12, 128)

        # R2: all-pairs (1 clique of 12)
        edge_12 = self._clique(bb)
        x_r2 = self.r2(x, edge_12)                  # (b, 12, 128)

        x = torch.cat([x_r1, x_r2], dim=2)          # (b, 12, 256)

        team1 = self.pool(x[:, :6, :])              # (b, 1, 256)
        team2 = self.pool(x[:, 6:, :])              # (b, 1, 256)

        x = torch.cat([team1, team2], dim=1).view(b, -1)   # (b, 512)
        return self.fc(x)
