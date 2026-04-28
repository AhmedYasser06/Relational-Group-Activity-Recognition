"""
RCRG-1R-1C

ResNet50 backbone  single relational layer (R1) over 1 clique of all 12 players
 max-pool each team  concat  classifier.
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
        self.pool = nn.AdaptiveMaxPool2d((1, 128))

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def _full_clique(self, n):
        pairs = list(itertools.permutations(range(n), 2))
        return torch.tensor(pairs, dtype=torch.long).t().to(self.device)

    def forward(self, x):
        b, bb, c, h, w = x.shape
        x = x.view(b * bb, c, h, w)
        x = self.resnet50(x).view(b, bb, -1)       # (b, bb, 2048)

        edge_index = self._full_clique(bb)
        x = self.r1(x, edge_index)                  # (b, bb, 128)

        team1 = self.pool(x[:, :6, :])              # (b, 1, 128)
        team2 = self.pool(x[:, 6:, :])              # (b, 1, 128)

        x = torch.cat([team1, team2], dim=1).view(b, -1)   # (b, 256)
        return self.fc(x)
