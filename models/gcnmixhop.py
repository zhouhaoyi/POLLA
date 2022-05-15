import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MixHopDenseLayer(nn.Module):
    def __init__(self, num_features, adj_pows, dim_per_pow):
        super(MixHopDenseLayer, self).__init__()
        self.num_features = num_features
        self.adj_pows = adj_pows
        self.dim_per_pow = dim_per_pow
        self.total_dim = 0
        self.linears = torch.nn.ModuleList()
        for dim in dim_per_pow:
            self.linears.append(nn.Linear(num_features, dim))
            self.total_dim += dim

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()

    def adj_pow_x(self, x, adj, p):
        # x [B, L, N, D]
        for _ in range(p):
            x = torch.einsum('nlvc,vw->nlwc',(x,adj))
        return x

    def forward(self, x, adj):
        output_list = []
        for p, linear in zip(self.adj_pows, self.linears):
            output = self.adj_pow_x(x, adj, p)
            output = linear(output)
            output_list.append(output)
        
        return torch.cat(output_list, dim=-1)

class MixHopDense(nn.Module):
    def __init__(self, c_in, c_out, dropout, layer1_pows, layer2_pows, device='cpu'):
        super(MixHopDense, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dropout = dropout
        layer_pows = [layer1_pows, layer2_pows]

        shapes = [c_in] + [sum(layer1_pows), sum(layer2_pows)]

        self.mixhops = nn.ModuleList(
            [
                MixHopDenseLayer(shapes[layer], [0, 1, 2, 3], layer_pows[layer])
                for layer in range(len(layer_pows))
            ]
        )
        self.fc = nn.Linear(shapes[-1], c_out)

    def forward(self, x, adj):
        for mixhop in self.mixhops:
            x = F.relu(mixhop(x, adj))
            x = F.dropout(x, p=self.dropout)
        x = self.fc(x)
        return x