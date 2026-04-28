"""
RCRG-2R-11C-conc-temporal-V1

LSTM  Relational layers order: temporal context is built first, then spatial relations.
  1. ResNet50 extracts per-frame features: (b*bb, seq, 2048)
  2. LSTM runs per-person over time  (b*bb, 1024) per person
  3. Two relational layers (R1, R2) on the 1024-d person representations
  4. Concat R1 + R2 per player  flatten  classifier
"""

import torch
import torch.nn as nn
import itertools
from .model_utils import PersonClassifier, freeze, collate_fn_seq
from .relational_unit import RelationalUnit
from .temporal_eval import run_eval, run_eval_TTA
from utils import load_config


class GroupActivityClassifier(nn.Module):
    def __init__(self, person_model, num_classes, device):
        super().__init__()

        self.device = device
        self.resnet50 = person_model.resnet50
        freeze(self.resnet50)

        self.lstm = nn.LSTM(input_size=2048, hidden_size=1024, batch_first=True)

        # relational layers operate on 1024-d (LSTM output)
        self.r1 = RelationalUnit(in_channels=1024, out_channels=128, hidden_dim=512)
        self.r2 = RelationalUnit(in_channels=1024, out_channels=256, hidden_dim=512)

        self.fc = nn.Sequential(
            nn.Linear(12 * 384, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _full_clique(self, n):
        pairs = list(itertools.permutations(range(n), 2))
        return torch.tensor(pairs, dtype=torch.long).t().to(self.device)

    def forward(self, x):
        b, bb, seq, c, h, w = x.shape

        x = x.view(b * bb * seq, c, h, w)
        x = self.resnet50(x).view(b * bb, seq, -1)   # (b*bb, seq, 2048)

        x, _ = self.lstm(x)                           # (b*bb, seq, 1024)
        x = x[:, -1, :].view(b, bb, -1)              # (b, bb, 1024)

        edge_index = self._full_clique(bb)
        x1 = self.r1(x, edge_index)                  # (b, bb, 128)
        x2 = self.r2(x, edge_index)                  # (b, bb, 256)
        x = torch.cat([x1, x2], dim=2)               # (b, bb, 384)

        return self.fc(x.view(b, -1))
