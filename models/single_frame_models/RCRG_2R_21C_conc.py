"""
RCRG-2R-21C-conc

Same graph structure as RCRG-2R-21C (R1 intra-team, R2 all-pairs),
but uses concatenation pooling instead of max-pooling.
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

        self.fc = nn.Sequential(
            nn.Linear(12 * 256, 1024),
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
        x = self.resnet50(x).view(b, bb, -1)       # (b, bb, 2048)

        # R1: intra-team
        edge_6 = self._clique(6)
        x1_t1 = self.r1(x[:, :6, :], edge_6)
        x1_t2 = self.r1(x[:, 6:, :], edge_6)
        x_r1 = torch.cat([x1_t1, x1_t2], dim=1)   # (b, 12, 128)

        # R2: all-pairs
        edge_12 = self._clique(bb)
        x_r2 = self.r2(x, edge_12)                 # (b, 12, 128)

        x = torch.cat([x_r1, x_r2], dim=2)         # (b, 12, 256)
        x = x.view(b, -1)                          # (b, 12*256)
        return self.fc(x)
