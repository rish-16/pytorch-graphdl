import torch
from torch_geometric import nn
from gcn.layers import GCNLayer

class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.g1 = GCNLayer()
        self.g2 = GCNLayer()

    def forward(self, x, adj):
        pass