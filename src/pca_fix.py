import torch
import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import torch.nn.functional as F

print("=" * 70)
print("PCA ANALYSIS: Fixed Temporal Aggregation (Wrapper)")
print("=" * 70)

from model1 import GraphSageClassification, GatV2Classification
from experiments import get_data
from constants import DEVICE

# ============================================================================
# WRAPPER FUNCTION: Extract embeddings with averaging
# ============================================================================

def extract_gnn_embeddings_averaged(model, graphs, fts, time_steps):
    """
    Extract GNN embeddings by AVERAGING across all time steps
    (instead of just using the last month)
    """
    model.eval()
    y_full = []
    
    with torch.no_grad():
        for i in range(time_steps):
            x = fts[i]
            G = graphs[i]
            
            # Forward pass through GNN layers
            y = F.leaky_relu(model.layers[0](x, G), 0.2)
            for layer in model.layers[1:]:
                y = F.leaky_relu(layer(y, G), 0.2)
            
            y_full.append(y.unsqueeze(0))
        
        y = torch.cat(y_full, dim=0)  # Shape: (time_steps, n_users, hidden_dim)
        
        # AVERAGE across time instead of taking last
        y_averaged = y.mean(dim=0)  # Shape: (n_users, hidden_dim)
    
    return y_averaged.cpu().numpy()

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/5] Loading data...")
train_samples, test_samples, val_samples = get_data()
sample = test_samples[0]

fts = sample.features.to(DEVICE)
graphs = [g.to(DEVICE) for g in sample.graph_data]
time_steps = sample.window
labels = sample.labels.cpu().numpy()

print(f"  Time steps: {time_steps}")
print(f"  Total users: {len(labels)}")

# ============================================================================
# Subsample to 1000 users
# ============================================================================
n_users = 1000
print(f"\n[2/5] Subsampling to {n_users} users...")

if len(labels) > n_users:
    np.random.seed(42)
    indices = np.random.choice(len(labels), n_users, replace=False)
    indices = np.sort(indices)
    
    labels = labels[indices]
    fts = fts[:, indices, :]
    
    # Remap graph edges
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
    new_graphs = []
    
    for graph in graphs:
        src = graph[0].cpu().numpy()
        dst = graph[1].cpu().numpy()
        
        mask = np.isin(src, indices) & np.isin(dst, indices)
        src_new = np.array([index_map[s] for s in src[mask]])
        dst_new = np.array([index_map[d] for d in dst[mask]])
        
        new_graphs.append(torch.tensor([src_new, dst_new], dtype=torch.long).to(DEVICE))
    
    graphs = new_graphs

print(f"  Final users: {len(labels)}")
print(f"  Spreaders: {(labels==1).sum()}, Non-spreaders: {(labels==0).sum()}")

# ============================================================================
# Extract Embeddings
# ============================================================================
print("\n[3/5] Extracting embeddings...")

# Raw BERT
raw_embeddings = fts.mean(dim=0).cpu().numpy()
print(f"  ✓ Raw BERT: {raw_embeddings.shape}")

# Find checkpoints
checkpoint_dir = "../results/checkpoints"

def detect_layers(path):
    ckpt = torch.load(path, map_location='cpu',weights_only=False)
    layer_keys = [k for k in ckpt['state_dict'].keys() if 'layers.' in k and '.weight' in k]
    if layer_keys:
        nums = [int(k.split('layers.')[1].split('.')[0]) for k in layer_keys]
        return max(nums) + 1
    return 5  # default

# Load GAT
gat_files = [f for f in os.listdir(checkpoint_dir) if "GatV2" in f and f.endswith(".tar")]
gat_embeddings = None

if gat_files:
    gat_file = sorted(gat_files)[5]
    gat_path = os.path.join(checkpoint_dir, gat_file)
    num_layers = detect_layers(gat_path)
    
    print(f"\n  Loading GAT ({num_layers} layers): {gat_file}")
    
    checkpoint = torch.load(gat_path, map_location=DEVICE,weights_only=False)
    model_gat = GatV2Classification(768, 256, 128, num_layers=num_layers)
    model_gat.load_state_dict(checkpoint['state_dict'])
    model_gat.to(DEVICE)
    
    # Use wrapper function (averages across time)
    gat_embeddings = extract_gnn_embeddings_averaged(model_gat, graphs, fts, time_steps)
    print(f"  ✓ GAT embeddings: {gat_embeddings.shape}")
    
    del model_gat
    torch.cuda.empty_cache()

# Load GraphSAGE
sage_files = [f for f in os.listdir(checkpoint_dir) if "GraphSage" in f and f.endswith(".tar")]
sage_embeddings = None

if sage_files:
    sage_file = sorted(sage_files)[0]
    sage_path = os.path.join(checkpoint_dir, sage_file)
    num_layers = detect_layers(sage_path)
    
    print(f"\n  Loading GraphSAGE ({num_layers} layers): {sage_file}")
    
    checkpoint = torch.load(sage_path, map_location=DEVICE,weights_only=False)
    model_sage = GraphSageClassification(768, 256, 128, num_layers=num_layers)
    model_sage.load_state_dict(checkpoint['state_dict'])
    model_sage.to(DEVICE)
    
    # Use wrapper function (averages across time)
    sage_embeddings = extract_gnn_embeddings_averaged(model_sage, graphs, fts, time_steps)
    print(f"  ✓ GraphSAGE embeddings: {sage_embeddings.shape}")
    
    del model_sage
    torch.cuda.empty_cache()

# ============================================================================
# PCA
# ============================================================================
print("\n[4/5] Running PCA...")

pca_raw = PCA(n_components=2)
raw_2d = pca_raw.fit_transform(raw_embeddings)

gat_2d = None
if gat_embeddings is not None:
    pca_gat = PCA(n_components=2)
    gat_2d = pca_gat.fit_transform(gat_embeddings)


sage_2d = None
if sage_embeddings is not None:
    pca_sage = PCA(n_components=2)
    sage_2d = pca_sage.fit_transform(sage_embeddings)

def get_top_pca_features(pca, feature_type, top_k=10):
    print(f"\nTop {top_k} contributing features for {feature_type}")
    for i, pc in enumerate(pca.components_[:2]):  # PC1 and PC2
        top_idx = np.argsort(np.abs(pc))[::-1][:top_k]
        print(f"  PC{i+1} top features (indices): {top_idx}")
        print(f"  PC{i+1} loadings: {pc[top_idx]}")

# Plot the feature
def plot_pca_feature_importance(pca, name, top_k=10):
    for i, pc in enumerate(pca.components_[:2]):
        top_idx = np.argsort(np.abs(pc))[::-1][:top_k]
        plt.figure(figsize=(8, 4))
        plt.bar(range(top_k), pc[top_idx])
        plt.xticks(range(top_k), top_idx)
        plt.title(f'{name} - PC{i+1} Top {top_k} Features')
        plt.ylabel('Component Weight')
        plt.xlabel('Feature Index')
        plt.tight_layout()
        plt.savefig(f'../results/pca_fixed/{name}_PC{i+1}_features.png')
        plt.close()

# ============================================================================
# Metrics
# ============================================================================
print("\n  Computing metrics...")

def calc_metrics(emb_2d, labels, name):
    sil = silhouette_score(emb_2d, labels)
    
    c1 = emb_2d[labels == 1].mean(axis=0)
    c0 = emb_2d[labels == 0].mean(axis=0)
    dist = np.linalg.norm(c1 - c0)
    
    print(f"  {name}: Silhouette={sil:.4f}, Distance={dist:.4f}")
    
    return {'silhouette': float(sil), 'centroid_distance': float(dist)}

metrics = {}
metrics['raw_bert'] = calc_metrics(raw_2d, labels, "Raw BERT")

if gat_2d is not None:
    metrics['gat'] = calc_metrics(gat_2d, labels, "GAT (Averaged)")

if sage_2d is not None:
    metrics['sage'] = calc_metrics(sage_2d, labels, "GraphSAGE (Averaged)")

# ============================================================================
# Visualization
# ============================================================================
print("\n[5/5] Creating visualization...")

output_dir = "../results/pca_fixed"
os.makedirs(output_dir, exist_ok=True)

n_plots = 1 + (gat_2d is not None) + (sage_2d is not None)
fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
if n_plots == 1:
    axes = [axes]

colors = ['#4ECDC4', '#FF6B6B']
labels_txt = ['Non-spreader', 'Misinformation Spreader']

idx = -1

# Raw BERT
for lbl in [0, 1]:
    mask = labels == lbl
    axes[idx].scatter(raw_2d[mask, 0], raw_2d[mask, 1],
                     c=colors[lbl], label=labels_txt[lbl],
                     alpha=0.6, s=60, edgecolors='black', linewidth=0.3)

axes[idx].set_title('Raw BERT Embeddings\n(Linguistic Features)', fontsize=14, fontweight='bold')
axes[idx].set_xlabel('PC1', fontsize=11)
axes[idx].set_ylabel('PC2', fontsize=11)
axes[idx].legend(loc='best', fontsize=10)
axes[idx].grid(True, alpha=0.2)
axes[idx].text(0.03, 0.97, 
               f"Silhouette: {metrics['raw_bert']['silhouette']:.3f}\nDist: {metrics['raw_bert']['centroid_distance']:.2f}",
               transform=axes[idx].transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
idx += 1

# GAT
if gat_2d is not None:
    for lbl in [0, 1]:
        mask = labels == lbl
        axes[idx].scatter(gat_2d[mask, 0], gat_2d[mask, 1],
                         c=colors[lbl], label=labels_txt[lbl],
                         alpha=0.6, s=60, edgecolors='black', linewidth=0.3)
    
    axes[idx].set_title('GAT Embeddings (Time-Averaged)\n+ Graph Attention', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('PC1', fontsize=11)
    axes[idx].set_ylabel('PC2', fontsize=11)
    axes[idx].legend(loc='best', fontsize=10)
    axes[idx].grid(True, alpha=0.2)
    axes[idx].text(0.03, 0.97,
                   f"Silhouette: {metrics['gat']['silhouette']:.3f}\nDist: {metrics['gat']['centroid_distance']:.2f}",
                   transform=axes[idx].transAxes, fontsize=10, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    idx += 1

# GraphSAGE
if sage_2d is not None:
    for lbl in [0, 1]:
        mask = labels == lbl
        axes[idx].scatter(sage_2d[mask, 0], sage_2d[mask, 1],
                         c=colors[lbl], label=labels_txt[lbl],
                         alpha=0.6, s=60, edgecolors='black', linewidth=0.3)
    
    axes[idx].set_title('GraphSAGE Embeddings (Time-Averaged)\n+ Neighborhood Aggregation', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('PC1', fontsize=11)
    axes[idx].set_ylabel('PC2', fontsize=11)
    axes[idx].legend(loc='best', fontsize=10)
    axes[idx].grid(True, alpha=0.2)
    axes[idx].text(0.03, 0.97,
                   f"Silhouette: {metrics['sage']['silhouette']:.3f}\nDist: {metrics['sage']['centroid_distance']:.2f}",
                   transform=axes[idx].transAxes, fontsize=10, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

plt.suptitle('PCA Comparison (Time-Averaged Embeddings)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/pca_comparison_averaged.png", dpi=300, bbox_inches='tight')
print(f"\n  ✓ Saved: {output_dir}/pca_comparison_averaged.png")

# Save data
import json
json.dump(metrics, open(f"{output_dir}/metrics_averaged.json", 'w'), indent=2)
pkl.dump({'raw_2d': raw_2d, 'gat_2d': gat_2d, 'sage_2d': sage_2d, 'labels': labels, 'metrics': metrics},
         open(f"{output_dir}/embeddings_averaged.pkl", 'wb'))

print(f"  ✓ Saved: {output_dir}/metrics_averaged.json")
print(f"  ✓ Saved: {output_dir}/embeddings_averaged.pkl")

# extract the most important features
if raw_2d is not None:
    get_top_pca_features(pca_raw, "Raw BERT")

if gat_2d is not None:
    get_top_pca_features(pca_gat, "GAT")

if sage_2d is not None:
    get_top_pca_features(pca_sage, "GraphSAGE")

plot_pca_feature_importance(pca_raw, "Raw_BERT")
plot_pca_feature_importance(pca_gat, "GAT")
plot_pca_feature_importance(pca_sage, "GraphSAGE")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY (Time-Averaged Aggregation)")
print("=" * 70)
print(f"\nSilhouette Scores:")
print(f"  Raw BERT:  {metrics['raw_bert']['silhouette']:.4f}")
if 'gat' in metrics:
    print(f"  GAT:       {metrics['gat']['silhouette']:.4f}  (Δ={metrics['gat']['silhouette']-metrics['raw_bert']['silhouette']:+.4f})")
if 'sage' in metrics:
    print(f"  GraphSAGE: {metrics['sage']['silhouette']:.4f}  (Δ={metrics['sage']['silhouette']-metrics['raw_bert']['silhouette']:+.4f})")

print("\n" + "=" * 70)
print("✓ DONE! Check results/pca_fixed/")
print("=" * 70)
