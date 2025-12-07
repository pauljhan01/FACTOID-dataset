import torch
import pickle as pkl
import gzip
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from experiments import get_data
from constants import DEVICE

# --- Load samples ---
train_samples, test_samples, val_samples = get_data()

sample = test_samples[0]          # use test set
fts = sample.features.to(DEVICE)
labels = sample.labels.cpu().numpy()

# --- Extract raw BERT embeddings ---
# Average over time dimension to get per-user embedding
raw_embeds = fts.mean(dim=0).cpu().numpy()  # shape: (num_users, 768)

print("Raw BERT embedding shape:", raw_embeds.shape)
print(f"Number of users: {len(labels)}")
print(f"Spreaders: {(labels == 1).sum()}, Non-spreaders: {(labels == 0).sum()}")

# --- PCA ---
pca = PCA(n_components=2)
proj = pca.fit_transform(raw_embeds)

print(f"\nPCA explained variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

# --- Plot ---
plt.figure(figsize=(7,5))
colors = ['#3498db', '#e74c3c']  # Blue for non-spreaders, Red for spreaders
for label in [0, 1]:
    mask = labels == label
    label_name = 'Non-spreader' if label == 0 else 'Misinformation Spreader'
    plt.scatter(proj[mask, 0], proj[mask, 1], 
                c=colors[label], label=label_name, 
                alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

plt.colorbar(label="Label (0=non-spreader, 1=spreader)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("PCA of Raw BERT Embeddings\n(Linguistic Features Only)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs("../results/visualizations", exist_ok=True)
plt.savefig("../results/visualizations/pca_raw_bert.png", dpi=300, bbox_inches='tight')
print("\nSaved PCA plot → ../results/visualizations/pca_raw_bert.png")

# --- Save embeddings for later comparison ---
output = {
    'embeddings': raw_embeds,
    'pca_2d': proj,
    'labels': labels,
    'explained_variance': pca.explained_variance_ratio_
}
pkl.dump(output, open('../results/visualizations/raw_bert_embeddings.pkl', 'wb'))
print("Saved embeddings → ../results/visualizations/raw_bert_embeddings.pkl")
