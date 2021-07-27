import torch
from torch import nn
import torch.nn.functional as F
from layers import GCNLayer

class GCN(nn.Module):
    def __init__(self, in_feat, n_classes):
        super().__init__()
        self.g1 = GCNLayer(in_feat, 512)
        self.g2 = GCNLayer(512, 256)
        self.g3 = GCNLayer(256, n_classes)

    def forward(self, x, edge_idx):
        x = F.relu(self.g1(x, edge_idx))
        x = F.relu(self.g2(x, edge_idx))
        out = F.softmax(self.g3(x, edge_idx), 1)
        
        return out