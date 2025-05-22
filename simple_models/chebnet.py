import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, EdgeConv
from base import Abstract_GNN, graph_readout

class ChebC(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, k, readout):
        super().__init__(num_nodes, f1, f2, readout)
        self.conv1 = ChebConv(num_nodes, f1, k)
        self.conv2 = ChebConv(f1, f1, k)
        self.conv3 = ChebConv(f1, f1, k)
        self.readout = readout
        last_dim = 2 if readout == 'meanmax' else 1
        self.fc = nn.Linear(f1 * last_dim, f2)

    def forward(self, data, edge_index, batch):
        x = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = graph_readout(x, self.readout, batch)
        x = self.fc(x)
        return x

class ChebEdge(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, k, readout):
        super().__init__(num_nodes, f1, f2, readout)
        self.conv1 = ChebConv(num_nodes, f1, k)
        self.conv2 = ChebConv(f1, f1, k)
        self.conv3 = ChebConv(f1, f1, k)
        self.edgeconv1 = EdgeConv(nn.Sequential(nn.Linear(f1*2, f1*2), nn.ReLU(), nn.Linear(f1*2, f1)))
        self.edgeconv2 = EdgeConv(nn.Sequential(nn.Linear(f1*2, f1*2), nn.ReLU(), nn.Linear(f1*2, f1)))
        last_dim = 2 if readout == 'meanmax' else 1
        self.fc = nn.Linear(f1 * last_dim, f2)

    def forward(self, data, edge_index, batch):
        x = data
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.edgeconv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.edgeconv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = graph_readout(x, self.readout, batch)
        x = self.fc(x)
        return x
