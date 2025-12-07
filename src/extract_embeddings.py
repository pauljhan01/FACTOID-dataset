import torch
import pickle as pkl
import gzip
import os
from model import GraphSageClassification, GatV2Classification
from experiments import get_data
from constants import DEVICE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Load samples ---
train_samples, test_samples, val_samples = get_data()

sample = test_samples[0]          # use test set
fts = sample.features.to(DEVICE)
graphs = [g.to(DEVICE) for g in sample.graph_data]
time_steps = sample.window
labels = sample.labels.cpu().numpy()

# --- Load trained model ---
model_path = "../results/checkpoints/GatV2Classificationlayers_4_best_model.tar"
checkpoint = torch.load(model_path, map_location=DEVICE)
model = GatV2Classification(768, 256, 128, num_layers=4)
model.load_state_dict(checkpoint['state_dict'])
model.to(DEVICE)
model.eval()

# --- Extract hidden embeddings ---
with torch.no_grad():
    embeds = model.get_hidden(graphs, fts, time_steps).cpu().numpy()

print("Embedding shape:", embeds.shape)   # (num_users, hidden_dim)

# --- PCA ---
pca = PCA(n_components=2)
proj = pca.fit_transform(embeds)

# --- Plot ---
plt.figure(figsize=(7,5))
plt.scatter(proj[:,0], proj[:,1], c=labels, cmap="coolwarm", alpha=0.7)
plt.colorbar(label="Label (0=real, 1=fake)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of User Embeddings")
plt.savefig("../results/pca_plot.png", dpi=300)

print("Saved PCA plot → ../results/pca_plot.png")
