import torch
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric import nn
from torch_geometric.utils import add_self_loops, degree

class Attention(torch.nn.Module):
    def __init__(self, features, attn_dim):
        super(Attention, self).__init__()
        self.to_q = torch.nn.Linear(features, attn_dim)
        self.to_k = torch.nn.Linear(features, attn_dim)
        self.to_v = torch.nn.Linear(features, attn_dim)
        self.project = torch.nn.Linear(attn_dim, features)
        
    def forward(self, x):
        Q = self.to_q(x)
        K = self.to_k(x)
        V = self.to_v(x)
        
        dots = torch.bmm(Q, K.permute(0, 2, 1))
        attn = F.softmax(dots, 0)
        
        out = torch.bmm(attn, V)
        out = self.project(out)
        
        return out

class GATLayer(nn.MessagePassing):
    def __init__(self, in_channels, out_channels, attn_dim=64):
        super(GATLayer, self).__init__(aggr='add')
        self.attn = Attention(in_channels, attn_dim)
        self.l1 = torch.nn.Linear(in_channels, out_channels)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.proj_back = torch.nn.Linear(2 * out_channels, out_channels)
        
        self.a = torch.nn.Parameter(torch.zeros(2 * out_channels, 1))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
    def forward(self, x, edge_index):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # add self-loop to adjacency list        
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        proj_i = self.l1(x_i)
        proj_j = self.l1(x_j)
        cat = torch.cat([proj_i, proj_j], dim=1)
        tmp = self.a.T * cat
        out = F.softmax(self.leaky_relu(tmp), 1)
        out = self.proj_back(out)
        
        return out