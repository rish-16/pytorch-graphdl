import torch
from torch_geometric import nn
from torch_geometric.utils import add_self_loops, degree

class GCNLayer(nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__(aggr='add')
        self.l1 = torch.nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_idx):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # add self-loop to adjacency list
        x = self.lin(x) # project each node feature vector
        
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype) # normalise using the degree of in and out nodes
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] # the denominator in graph conv layer update rule
        
        return self.propagate(edge_idx, x=x, norm=norm)
    
    def message(self, x_i, norm):
        return norm.view(-1, 1) * x_i # GCN update rule