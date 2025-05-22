import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool

class ProteinGraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, heads=8, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout))
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)
