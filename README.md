# pytorch-graphdl
PyTorch Implementation of popular Graph Neural Networks

## Building Graph Layers
`torch_geometric` requires two methods to be filled in when building custom modules using the `nn.MessagePassing` class. The `forward` method takes in node feature vectors at that stage and performs operations on it. The `message` method prepares the message to be sent between nodes. The `propagate` method is called from within `forward` with custom parameters and arguments accepted by the `message` method.

## Aggregation Methods
`torch_geometric` allows for the following aggregation methods:

- `add`: adds all the node feature vectors
- `max`: gets the maximum node feature vector
- `min`: gets the minimum node feature vector
- `mean`: gets the mean of all node feature vectors