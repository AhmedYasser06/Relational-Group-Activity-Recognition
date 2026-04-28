"""
RCRG-2R-11C-conc-temporal

Uses attention-based relational layers (RelationalUnit) instead of MLP ones.
Both layers use 1 clique (all 12 players).
An LSTM processes the temporal dimension after relational reasoning.
Final player representation = proj(last_relational) + lstm(last_hidden).
"""

import torch
import torch.nn as nn
import itertools
from .model_utils import PersonClassifier, freeze, collate_fn_seq
from .relational_attention import RelationalUnit


class GroupActivityClassifier(nn.Module):
    def __init__(self, person_model, num_classes, device):
        super().__init__()

        self.device = device
        self.resnet50 = person_model.resnet50
        freeze(self.resnet50)

        self.r1 = RelationalUnit(in_channels=2048, out_channels=2048, num_heads=4, dropout_rate=0.5)
        self.r2 = RelationalUnit(in_channels=2048, out_channels=2048, num_heads=4, dropout_rate=0.5)

        self.proj = nn.Linear(2048, 512)
        self.ln_proj = nn.LayerNorm(512)
        self.ln_lstm = nn.LayerNorm(512)

        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(12 * 512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _full_clique(self, n):
        pairs = list(itertools.permutations(range(n), 2))
        return torch.tensor(pairs, dtype=torch.long).t().to(self.device)

    def forward(self, x):
        b, bb, seq, c, h, w = x.shape

        x = x.view(b * bb * seq, c, h, w)
        x = self.resnet50(x).view(b * seq, bb, -1)  # (b*seq, bb, 2048)

        edge_index = self._full_clique(bb)

        x = self.r1(x, edge_index)                  # (b*seq, bb, 2048)
        x = self.r2(x, edge_index)                  # (b*seq, bb, 2048)

        x = x.view(b * bb, seq, -1)                 # (b*bb, seq, 2048)
        x_lstm, _ = self.lstm(x)                    # (b*bb, seq, 512)
        x_lstm = self.ln_lstm(x_lstm[:, -1, :])     # (b*bb, 512)  last frame

        x_proj = self.ln_proj(self.proj(x[:, -1, :]))  # (b*bb, 512)

        x = (x_proj + x_lstm).view(b, -1)           # (b, bb*512)
        return self.fc(x)
