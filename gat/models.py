import torch
from torch import nn
import torch.nn.functional as F
from layers import GATLayer
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_feat, n_classes):
        super().__init__()
        self.g1 = GATLayer(in_feat, 512)
        self.g2 = GATLayer(512, 256)
        self.g3 = GATLayer(256, n_classes)

    def forward(self, x, edge_idx):
        x = F.relu(self.g1(x, edge_idx))
        x = F.relu(self.g2(x, edge_idx))
        out = F.softmax(self.g3(x, edge_idx), 1)
        
        return out
    
class GATFormal(nn.Module):
    def __init__(self, in_feat, n_classes):
        super().__init__()
        self.g1 = GATConv(in_channels=in_feat, out_channels=512, heads=1)
        self.g2 = GATConv(in_channels=512, out_channels=256, heads=1)
        self.g3 = GATConv(in_channels=256, out_channels=n_classes, heads=1, concat=False)
        
    def forward(self, x, edge_idx):
        x = F.relu(self.g1(x, edge_idx))
        x = F.relu(self.g2(x, edge_idx))
        out = F.softmax(self.g3(x, edge_idx), 1)
        
        return out