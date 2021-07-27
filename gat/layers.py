import torch_geometric as tg
from torch_geometric import nn
from torch_geometric.utils import add_self_loops, degree

class GATLayer(nn.MessagePassing):
    def __init__(self):
        super(GATLayer, self).__init__(aggr='add')
        
    def forward(self, x, edge_idx):
        pass
    
    def message(self, x_i):
        pass