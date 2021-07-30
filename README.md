# pytorch-graphdl
PyTorch Implementation of popular Graph Deep Learning (GDL) architectures.

## What is this?
This is a sincere effort to learn more about Graph Deep Learning and the various networks within. This is partly motivated by a few factors:

- I wish to enter GDL as a research focus area (and apply it in RL)
- I am inspired by the GDL researchers I follow on Twitter
- I am trying to apply GDL to biomedical/bioinformatics applications (so practice makes perfect!)
- I want to learn `torch_geometric` (and if time allows, `dgl` and `spektral`)

## Building Graph Layers
`torch_geometric` requires two methods to be filled in when building custom modules using the `nn.MessagePassing` class. The `forward` method takes in node feature vectors at that stage and performs operations on it. The `message` method prepares the message to be sent between nodes. The `propagate` method is called from within `forward` with custom parameters and arguments accepted by the `message` method.

## Aggregation Methods
`torch_geometric` allows for the following aggregation methods:

- `add`: adds all the node feature vectors
- `max`: gets the maximum node feature vector
- `mean`: gets the mean of all node feature vectors

## Tasks
In GDL, there are two types of tasks: `transductive` and `inductive`

- **Transductive Learning**: training and testing nodes co-exist during training; model captures feature vectors of testing nodes but not the labels
- **Inductive Learning**: testing nodes are completely disjoint from training nodes; model is unaware of testing nodes during training

## Models
This repository contains the following model implementations:

- Graph Convolutional Networks (GCN) [[`gcn`](https://github.com/rish-16/pytorch-graphdl/tree/main/gcn)]
- Graph Attention Networks (GAT) [[`gat`](https://github.com/rish-16/pytorch-graphdl/tree/main/gat)]
- DeepWalk [[`gat`](https://github.com/rish-16/pytorch-graphdl/tree/main/deepwalk)]

## TODO
- [ ] GraphSage
- [ ] Graph Transformer Networks (GTN)

## License
[MIT](https://github.com/rish-16/pytorch-graphdl/blob/main/LICENSE)