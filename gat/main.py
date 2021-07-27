import torch.nn.functional as F
from torch.optim import Adam

import torch_geometric as tg
from torch_geometric.nn import MessagePassing
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_self_loops, degree

import matplotlib.pyplot as plt