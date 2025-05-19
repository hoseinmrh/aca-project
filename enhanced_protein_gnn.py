import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, EdgeConv, global_mean_pool, global_max_pool, PairNorm
from base import Abstract_GNN, graph_readout

class EnhancedProteinGNN(Abstract_GNN):
    def __init__(self, num_nodes, hidden_dim, num_classes, k=5, readout='meanmax', dropout=0.3, **kwargs):
        """
        An enhanced GNN specifically designed for protein classification, building on
        what worked well in the ChebNet + EdgeConv model.
        
        Args:
            num_nodes: Number of node features
            hidden_dim: Dimension of hidden layers
            num_classes: Number of output classes
            k: Filter size for ChebConv
            readout: Graph readout method ('mean', 'sum', or 'meanmax')
            dropout: Dropout rate
        """
        super().__init__(num_nodes, hidden_dim, num_classes, readout)
        
        # First layer - node feature embedding
        self.node_embed = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # ChebConv layer 1 - captures spectral features 
        self.cheb1 = ChebConv(hidden_dim, hidden_dim, K=k)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # EdgeConv layer - captures local neighborhood structure
        self.edge_conv = EdgeConv(nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        ))
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # ChebConv layer 2 - further spectral feature refinement
        self.cheb2 = ChebConv(hidden_dim, hidden_dim, K=k)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # PairNorm helps with oversmoothing in deep GNNs
        self.pairnorm = PairNorm(scale=1.0)
        
        # Final classification with multi-level features
        last_dim = 2 if readout == 'meanmax' else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * last_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
        self._reset_parameters()
    
    def forward(self, data, edge_index, batch):
        # Ensure all inputs are on the same device
        device = data.device
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        # Initial node embedding
        x = self.node_embed(data)
        
        # Store original features for residual connection
        identity = x
        
        # First ChebConv
        x = self.cheb1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Add residual connection
        x = x + identity
        
        # Apply EdgeConv to capture local structure
        edge_features = self.edge_conv(x, edge_index)
        edge_features = self.bn2(edge_features)
        edge_features = F.relu(edge_features)
        
        # Combine with previous features
        x = x + edge_features
        x = self.pairnorm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second ChebConv for additional spectral refinement
        x = self.cheb2(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Graph readout - aggregate node features to graph level
        x = graph_readout(x, self.readout, batch)
        
        # Classification
        x = self.classifier(x)
        
        return x
