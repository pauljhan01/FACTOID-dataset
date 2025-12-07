import torch
from model import GatV2Classification
from constants import DEVICE

# Fake graph with 10 nodes
edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9],
                           [1,2,3,4,5,6,7,8,9,0]], dtype=torch.long).to(DEVICE)

fts = torch.randn(5, 10, 768).to(DEVICE)
graphs = [edge_index for _ in range(5)]

model = GatV2Classification(
    in_channels=768,
    hidden_channels=256,
    out_channels=128,
    num_heads=1,
    nclass=2,
    num_layers=4
).to(DEVICE)

output = model(graphs, fts, time_steps=5)
print("MODEL OUTPUT SHAPE:", output.shape)
