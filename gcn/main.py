import numpy as np
import networkx as nx
import torch_geometric as tg

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

data = tg.datasets.CoraFull("./")

print (data.num_classes)
print (data.num_node_features)