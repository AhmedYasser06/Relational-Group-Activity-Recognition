"""
B1-temporal

Temporal baseline  no relational reasoning.
Per-frame: pool each team independently  concat feed to LSTM over time.
Final representation = concat(last spatial pooled, last LSTM hidden)  classifier.
Used as a control to measure how much relational layers actually help.
"""

import torch
import torch.nn as nn
from .model_utils import PersonClassifier, freeze, collate_fn_seq
from .temporal_eval import run_eval, run_eval_TTA
from utils import load_config


class GroupActivityClassifier(nn.Module):
    def __init__(self, person_model, num_classes):
        super().__init__()

        self.resnet50 = person_model.resnet50
        freeze(self.resnet50)

        self.pool = nn.AdaptiveMaxPool2d((1, 1024))

        self.lstm = nn.LSTM(input_size=2048, hidden_size=1024, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        b, bb, seq, c, h, w = x.shape

        x = x.view(b * bb * seq, c, h, w)
        x = self.resnet50(x).view(b * seq, bb, -1)   # (b*seq, bb, 2048)

        team1 = self.pool(x[:, :6, :])               # (b*seq, 1, 1024)
        team2 = self.pool(x[:, 6:, :])               # (b*seq, 1, 1024)

        x_spatial = torch.cat([team1, team2], dim=1).view(b, seq, -1)  # (b, seq, 2048)

        x_temporal, _ = self.lstm(x_spatial)         # (b, seq, 1024)

        x = torch.cat([x_spatial[:, -1, :], x_temporal[:, -1, :]], dim=1)  # (b, 3072)
        return self.fc(x)
