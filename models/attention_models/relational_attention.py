import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class RelationalUnit(MessagePassing):
    """
    Attention-based relational unit, structured like a Transformer block.
    Multi-head attention over graph edges + FFN, with residual connections and LayerNorm.
    """
    def __init__(self, in_channels, out_channels, num_heads=8, hidden_size=1024, dropout_rate=0.5):
        super().__init__(aggr='add')

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** 0.5

        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels * 2, in_channels)

        self.ln1 = nn.LayerNorm(in_channels)
        self.drop1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_channels),
        )

        self.ln2 = nn.LayerNorm(out_channels)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        # attention + residual
        x_att = self.propagate(edge_index, x=x)
        x_att = self.ln1(x + self.drop1(x_att))

        # FFN + residual
        out = self.ln2(x_att + self.drop2(self.ffn(x_att)))
        return out

    def message(self, x_i, x_j, index, ptr, size_i):
        b, edges, _ = x_i.shape

        def split_heads(t):
            return t.view(b, edges, self.num_heads, self.head_dim).transpose(1, 2)
            # (b, num_heads, edges, head_dim)

        q = split_heads(self.q(x_i))
        k = split_heads(self.k(x_j))
        v = split_heads(self.v(torch.cat([x_i, x_j], dim=-1)))

        # attention scores
        attn = (q * k).sum(dim=-1) / self.scale              # (b, num_heads, edges)
        attn = softmax(attn, index, ptr, num_nodes=size_i, dim=-1)

        return (attn.unsqueeze(-1) * v).view(b, edges, self.num_heads * self.head_dim)

    def update(self, aggr_out):
        return aggr_out
