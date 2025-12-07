import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model import GatV2Classification
from experiments import get_data
from constants import DEVICE

# ---------------------------
# Load samples
# ---------------------------
train_samples, test_samples, val_samples = get_data()
sample = test_samples[0]

fts = sample.features.to(DEVICE)        # [T, N, 768]
graphs = [g.to(DEVICE) for g in sample.graph_data]
time_steps = sample.window
labels = sample.labels.cpu().numpy()

# ---------------------------
# Limit to 1000 users
# ---------------------------
MAX_USERS = 1000
fts = fts[:, :MAX_USERS, :]             # keep all 16 windows
labels = labels[:MAX_USERS]

# ---------------------------
# Load GAT model
# ---------------------------
model_path = "../results/checkpoints/GatV2Classificationlayers_4_best_model.tar"
checkpoint = torch.load(model_path, map_location=DEVICE)

model = GatV2Classification(768, 256, 128, num_layers=4)
model.load_state_dict(checkpoint["state_dict"])
model.to(DEVICE)
model.eval()

# ---------------------------
# Extract embeddings fast
# ---------------------------
with torch.no_grad():
    embeds = model.get_hidden(graphs, fts, time_steps)
    embeds = embeds[:MAX_USERS].cpu().numpy()

print("Embedding shape:", embeds.shape)

# ---------------------------
# GPU PCA (FAST)
# ---------------------------
pca = cuPCA(n_components=2)
proj = pca.fit_transform(embeds)

# ---------------------------
# Plot results
# ---------------------------
plt.figure(figsize=(7,5))
plt.scatter(proj[:,0], proj[:,1], c=labels, cmap="coolwarm", alpha=0.7)
plt.colorbar()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Hidden Embeddings (1000 users, GPU PCA)")
plt.savefig("../results/pca_plot_1000.png", dpi=300)

print("Saved PCA plot → ../results/pca_plot_1000.png")
