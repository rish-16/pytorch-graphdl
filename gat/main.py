import torch
import torch.nn.functional as F
from torch.optim import Adam

import torch_geometric as tg
from torch_geometric.nn import MessagePassing
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_self_loops, degree

import matplotlib.pyplot as plt
from models import GAT, GATFormal

transform = T.Compose([
    T.AddTrainValTestMask('train_rest', num_val=500, num_test=500),
    T.TargetIndegree(),
])
dataset = tg.datasets.Planetoid(root="./dataset/", name="Cora", split="public", transform=transform)

data = dataset[0]
train_mask = data.train_mask
test_mask = data.test_mask
features = dataset.data.x
edge_idx = dataset.data.edge_index
labels = dataset.data.y

model = GATFormal(dataset.num_node_features, dataset.num_classes)
optimizer = Adam(model.parameters(), lr=0.01)

def get_test_acc(model):
    model.eval()
    correct = 0
    total = 0
    
    _, pred = model(features, edge_idx).max(dim=1)
    correct = int(pred[data.test_mask].eq(labels[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
        
    return acc
        
accs = []
epochs = 200

for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad()
    pred = model(features, edge_idx)
    
    loss = F.nll_loss(pred[data.train_mask], labels[data.train_mask])
    loss.backward()
    optimizer.step()
        
    acc = get_test_acc(model)
    accs.append(acc)
    
    if epoch % 10 == 0:
        print('Accuracy: {:.4f} | Epoch: {}'.format(acc, epoch))
        
plt.plot(range(epochs), accs, color="red")
plt.show()