import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class RelationalLayer(MessagePassing):
    """
    MLP-based relational layer using message passing.
    For each edge (i - j), it computes a message from [x_i || x_j],
    then sums all incoming messages at each node.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(aggr='add')

        self.mlp = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

    def update(self, aggr_out):
        return aggr_out
