import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

def graph_readout(x, method, batch):
    if method == 'mean':
        return global_mean_pool(x, batch)
    elif method == 'meanmax':
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        return torch.cat((x_mean, x_max), dim=1)
    elif method == 'sum':
        return global_add_pool(x, batch)
    else:
        raise ValueError('Undefined readout operation')

class Abstract_GNN(torch.nn.Module):
    """
    An Abstract class for all GNN models
    Subclasses should implement the following:
    - forward()
    - predict()
    """
    def __init__(self, num_nodes, f1, f2, readout):
        super(Abstract_GNN, self).__init__()
        self.readout = readout

    def _reset_parameters(self):
        # Make parameter initialization deterministic when seed is set
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, data, edge_index, batch):
        raise NotImplementedError
