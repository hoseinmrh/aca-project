import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from base import Abstract_GNN, graph_readout

class GAT(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, readout, concat, num_heads, **kwargs):
        super().__init__(num_nodes, f1, f2, readout)
        self.conv1 = GATv2Conv(num_nodes, f1, heads=num_heads, concat=concat)
        m = num_heads if concat else 1
        self.conv2 = GATv2Conv(f1 * m, f2, heads=1)
        last_dim = 2 if readout == 'meanmax' else 1
        self.mlp = nn.Linear(f2 * last_dim, f2)
        self._reset_parameters()

    def forward(self, data, edge_index, batch):
        x = data
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = graph_readout(x, self.readout, batch)
        x = self.mlp(x)
        return x
