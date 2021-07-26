# pytorch-graphdl
PyTorch Implementation of popular Graph Neural Networks.

## What is this?
This is a sincere effort to learn more about Graph Deep Learning and the various networks within. This is partly motivated by a few factors:

- I wish to enter GDL as a research focus area
- I am inspired by the GDL researchers I follow on Twitter
- I am trying to apply GDL to biomedical/bioinformatics applications (so practice makes perfect!)
- I want to learn `torch_geometric` (and if time allows, `dgl` and `spektral`)

## Building Graph Layers
`torch_geometric` requires two methods to be filled in when building custom modules using the `nn.MessagePassing` class. The `forward` method takes in node feature vectors at that stage and performs operations on it. The `message` method prepares the message to be sent between nodes. The `propagate` method is called from within `forward` with custom parameters and arguments accepted by the `message` method.

## Aggregation Methods
`torch_geometric` allows for the following aggregation methods:

- `add`: adds all the node feature vectors
- `max`: gets the maximum node feature vector
- `min`: gets the minimum node feature vector
- `mean`: gets the mean of all node feature vectors

## Models
This repository contains the following model implementations:

- Graph Convolutional Networks (GCN) [[`gcn`](https://github.com/rish-16/pytorch-graphdl/tree/main/gcn)]

## TODO
- [ ] Graph Attention Networks
- [ ] Graph Transformer

## License
[MIT](https://github.com/rish-16/pytorch-graphdl/blob/main/LICENSE)