import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from base import Abstract_GNN, graph_readout

class GIN(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, readout, **kwargs):
        super().__init__(num_nodes, f1, f2, readout)
        self.conv1 = GINConv(
            Sequential(Linear(num_nodes, f1), BatchNorm1d(f1), ReLU(),
                       Linear(f1, f1), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(f1, f1), BatchNorm1d(f1), ReLU(),
                       Linear(f1, f1), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(f1, f2), BatchNorm1d(f2), ReLU(),
                       Linear(f2, f2), ReLU()))
        last_dim = 2 if readout == 'meanmax' else 1
        self.last = Linear(f2 * last_dim, f2)
        self._reset_parameters()

    def forward(self, data, edge_index, batch):
        x = data
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = graph_readout(x, self.readout, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.last(x)
        return x
