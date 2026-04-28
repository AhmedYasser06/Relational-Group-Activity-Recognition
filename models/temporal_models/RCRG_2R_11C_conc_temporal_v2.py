"""
RCRG-2R-11C-conc-temporal-V2

Relational layers LSTM order: spatial relations are built first, then temporal context.
  1. ResNet50 extracts per-frame features
  2. Two relational layers (R1, R2) run per-frame on 2048-d features
  3. LSTM runs per-person over time on the 384-d relational output
  4. Last hidden state per player  flatten  classifier
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

        # relational layers operate on 2048-d (ResNet output)
        self.r1 = RelationalUnit(in_channels=2048, out_channels=128, hidden_dim=512)
        self.r2 = RelationalUnit(in_channels=2048, out_channels=256, hidden_dim=512)

        self.lstm = nn.LSTM(input_size=384, hidden_size=384, batch_first=True)

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
        x = self.resnet50(x).view(b * seq, bb, -1)   # (b*seq, bb, 2048)

        edge_index = self._full_clique(bb)
        x1 = self.r1(x, edge_index)                  # (b*seq, bb, 128)
        x2 = self.r2(x, edge_index)                  # (b*seq, bb, 256)
        x = torch.cat([x1, x2], dim=2)               # (b*seq, bb, 384)

        x = x.view(b * bb, seq, -1)                  # (b*bb, seq, 384)
        x, _ = self.lstm(x)                           # (b*bb, seq, 384)

        return self.fc(x[:, -1, :].view(b, -1))       # (b, bb*384)
