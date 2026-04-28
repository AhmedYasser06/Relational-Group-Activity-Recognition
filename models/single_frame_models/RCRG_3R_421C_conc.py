"""
RCRG-3R-421C-conc

Same graph structure as RCRG-3R-421C (4/2/1 clique sizes),
but uses concatenation pooling  all player features flattened  classifier.
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

        self.r1 = RelationalLayer(input_size=2048, hidden_size=256,  output_size=128)
        self.r2 = RelationalLayer(input_size=2048, hidden_size=512,  output_size=256)
        self.r3 = RelationalLayer(input_size=2048, hidden_size=1024, output_size=512)

        self.fc = nn.Sequential(
            nn.Linear(12 * 896, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _clique(self, n):
        pairs = list(itertools.permutations(range(n), 2))
        return torch.tensor(pairs, dtype=torch.long).t().to(self.device)

    def forward(self, x):
        b, bb, c, h, w = x.shape
        x = x.view(b * bb, c, h, w)
        x = self.resnet50(x).view(b, bb, -1)       # (b, 12, 2048)

        # R1: 4 cliques of 3
        edge_3 = self._clique(3)
        x_r1 = torch.cat([
            self.r1(x[:,  :3, :], edge_3),
            self.r1(x[:, 3:6, :], edge_3),
            self.r1(x[:, 6:9, :], edge_3),
            self.r1(x[:, 9:,  :], edge_3),
        ], dim=1)                                   # (b, 12, 128)

        # R2: 2 cliques of 6
        edge_6 = self._clique(6)
        x_r2 = torch.cat([
            self.r2(x[:, :6, :], edge_6),
            self.r2(x[:, 6:, :], edge_6),
        ], dim=1)                                   # (b, 12, 256)

        # R3: 1 clique of 12
        edge_12 = self._clique(bb)
        x_r3 = self.r3(x, edge_12)                 # (b, 12, 512)

        x = torch.cat([x_r1, x_r2, x_r3], dim=2)  # (b, 12, 896)
        x = x.view(b, -1)                          # (b, 12*896)
        return self.fc(x)
