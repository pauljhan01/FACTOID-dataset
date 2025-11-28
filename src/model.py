import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric.nn
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv
from constants import *


class GatClassification(nn.Module):
    # nfeat -> number of features in each node
    # nhid -> number of output features of the GAT
    # nhead-> Number of attention heads
    # window -> window window length.
    # nfeat -> in_channels (default: 768)
    # nhid_graph -> hidden_channels (default: 256)
    # nhid -> hidden_channel of second layer (default: 128) -> resized to 2 classes for binary classification
    
    def __init__(self, nfeat, nhid_graph, nhid, nclass, dropout, nheads, gnn_name='gat'):
        super(GatClassification, self).__init__()
        self.gnn_name = gnn_name
        self.nhid_graph = nhid_graph
        self.gnn = torch_geometric.nn.GATv2Conv(nfeat, nhid_graph, heads=nheads, negative_slope=0.2, concat=False, dropout=dropout)
       
        self.linear_pass = nn.Linear(nhid_graph, nhid)
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(nhid, nclass)


    def forward(self, graph, fts, time_steps, adj=None):
        # graph-> list of graphs, 1 per timestep. Each graph in edge list format.
        # fts-> (window,Number of nodes, number of features in each node)
        # Please ensure all nodes are present in each graph. We can feed a zero vector for nodes that do not have any features on a particular time-step.
        y_full = []
        
        for i in range(time_steps):
            x = fts[i]
            G = graph[i]
            y = F.leaky_relu(self.gnn(x, G), 0.2)
            y_full.append(y.reshape(1, x.shape[0], self.nhid_graph))
        y = torch.cat(y_full)
    
        output = self.linear_pass(y)
        output = self.dropout(F.leaky_relu(output.reshape((fts.shape[1], -1)), 0.2))
     
        output = self.linear(output)
        return F.log_softmax(output, dim=1)

class GatV2Classification(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, nclass=2, num_layers=2, dropout=0.5):
        super(GatV2Classification, self).__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(self.out_channels, nclass)

        self.layers = nn.ModuleList()
        self.layers.append(GATv2Conv(in_channels, hidden_channels))

        for _ in range(self.num_layers - 2):
            self.layers.append(GATv2Conv(hidden_channels, hidden_channels))

        self.layers.append(GATv2Conv(hidden_channels, out_channels))

    def forward(self, graph, fts, time_steps, adj=None):
        y_full = []

        for i in range(time_steps):
            x = fts[i]
            G = graph[i]
            y = F.leaky_relu(self.layers[0](x, G), 0.2)

            for j, layer in enumerate(self.layers[1:-1]):
                y = F.leaky_relu(layer(y, G), 0.2)

            y_full.append(y.reshape(1, x.shape[0], self.hidden_channels))
        y = torch.cat(y_full)

        y = y[0]

        y = F.leaky_relu(self.layers[-1](y, G), 0.2)
        y = self.dropout(y)
        y = self.linear(y)
        output = F.log_softmax(y, dim=1).reshape((fts.shape[1], -1))
        return output

class GraphSageClassification(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nclass=2, num_layers=2, dropout=0.5):
        super(GraphSageClassification, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(self.out_channels, nclass)

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))

        for _ in range(self.num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))

        self.layers.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))

    def forward(self, graph, fts, time_steps, adj=None):
        y_full = []

        for i in range(time_steps):
            x = fts[i]
            G = graph[i]
            y = F.leaky_relu(self.layers[0](x, G), 0.2)

            for j, layer in enumerate(self.layers[1:-1]):
                y = F.leaky_relu(layer(y, G), 0.2)

            y_full.append(y.reshape(1, x.shape[0], self.hidden_channels))
        y = torch.cat(y_full)

        y = F.leaky_relu(self.layers[-1](y, G), 0.2)
        y = self.linear(y)
        output = F.log_softmax(y, dim=1).reshape((fts.shape[1], -1))
        return output


            

