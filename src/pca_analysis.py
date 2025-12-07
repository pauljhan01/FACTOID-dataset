import torch
import pickle as pkl
import gzip
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib
import torch.serialization
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
matplotlib.use('Agg')

print("=" * 70)
print("PCA ANALYSIS: Raw BERT vs GNN Embeddings")
print("=" * 70)

# Check GPU
print(f"\nGPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

from model import GraphSageClassification, GatV2Classification
from experiments import get_data
from constants import DEVICE

print(f"Using device: {DEVICE}")

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[Step 1/6] Loading test data...")
train_samples, test_samples, val_samples = get_data()
sample = test_samples[0]

print(f"  Total users: {sample.features.shape[1]}")
print(f"  Time steps: {sample.window}")
print(f"  Features: {sample.features.shape[2]}")

# ============================================================================
# STEP 2: Subsample to 1000 users
# ============================================================================
n_users = 1000
print(f"\n[Step 2/6] Subsampling to {n_users} users...")

fts = sample.features.to(DEVICE)
graphs = [g.to(DEVICE) for g in sample.graph_data]
time_steps = sample.window
labels = sample.labels.cpu().numpy()

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
print(f"  Spreaders: {(labels == 1).sum()} ({100*labels.sum()/len(labels):.1f}%)")
print(f"  Non-spreaders: {(labels == 0).sum()} ({100*(1-labels.mean()):.1f}%)")

# ============================================================================
# STEP 3: Extract Raw BERT Embeddings
# ============================================================================
print("\n[Step 3/6] Extracting raw BERT embeddings...")
raw_embeddings = fts.mean(dim=0).cpu().numpy()
print(f"  Shape: {raw_embeddings.shape}")

# ============================================================================
# STEP 4: Load Models and Extract GNN Embeddings
# ============================================================================
print("\n[Step 4/6] Loading GNN models and extracting embeddings...")

checkpoint_dir = "../results/checkpoints"
available = os.listdir(checkpoint_dir)

gat_files = [f for f in available if f.startswith("GatV2") and f.endswith(".tar")]
sage_files = [f for f in available if f.startswith("GraphSage") and f.endswith(".tar")]

print(f"\n  Found {len(gat_files)} GAT checkpoint(s)")
print(f"  Found {len(sage_files)} GraphSAGE checkpoint(s)")

def auto_detect_layers(checkpoint_path):
    """Auto-detect number of layers from checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
    state_dict = ckpt['state_dict']
    
    # Find all keys with 'layers.N'
    layer_keys = [k for k in state_dict.keys() if 'layers.' in k and '.weight' in k]
    
    if layer_keys:
        # Extract layer numbers
        layer_nums = []
        for k in layer_keys:
            try:
                num = int(k.split('layers.')[1].split('.')[0])
                layer_nums.append(num)
            except:
                pass
        
        if layer_nums:
            num_layers = max(layer_nums) + 1
            return num_layers
    
    # Fallback: count parameters
    for key in state_dict.keys():
        if 'layers' in key:
            print(f"    Sample key: {key}")
    
    return 4  # default fallback

gat_embeddings = None
sage_embeddings = None
gat_layers_used = None
sage_layers_used = None

# --- Load GAT ---
if gat_files:
    # Use the first GAT checkpoint
    gat_file = sorted(gat_files)[7]
    gat_path = os.path.join(checkpoint_dir, gat_file)
    
    print(f"\n  Loading GAT: {gat_file}")
    print(f"  Auto-detecting architecture...")
    
    num_layers = auto_detect_layers(gat_path)
    print(f"  Detected: {num_layers} layers")
    
    try:
        checkpoint = torch.load(gat_path, map_location=DEVICE,weights_only=False)
        model_gat = GatV2Classification(768, 256, 128, num_layers=num_layers)
        model_gat.load_state_dict(checkpoint['state_dict'])
        model_gat.to(DEVICE)
        model_gat.eval()
        
        print(f"  ✓ Model loaded successfully")
        print(f"  Running forward pass on GPU...")
        
        with torch.no_grad():
            gat_embeddings = model_gat.get_hidden(graphs, fts, time_steps).cpu().numpy()
        
        print(f"  ✓ GAT embeddings: {gat_embeddings.shape}")
        gat_layers_used = num_layers
        
        del model_gat
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ✗ Error loading GAT: {e}")
        print(f"  Trying different layer counts...")
        
        # Try common layer counts
        for try_layers in [2, 4, 8, 16]:
            try:
                print(f"    Trying {try_layers} layers...")
                checkpoint = torch.load(gat_path, map_location=DEVICE,weights_only=False)
                model_gat = GatV2Classification(768, 256, 128, num_layers=try_layers)
                model_gat.load_state_dict(checkpoint['state_dict'])
                model_gat.to(DEVICE)
                model_gat.eval()
                
                with torch.no_grad():
                    gat_embeddings = model_gat.get_hidden(graphs, fts, time_steps).cpu().numpy()
                
                print(f"    ✓ Success with {try_layers} layers!")
                gat_layers_used = try_layers
                
                del model_gat
                torch.cuda.empty_cache()
                break
                
            except:
                continue

# --- Load GraphSAGE ---
if sage_files:
    sage_file = sorted(sage_files)[0]
    sage_path = os.path.join(checkpoint_dir, sage_file)
    
    print(f"\n  Loading GraphSAGE: {sage_file}")
    print(f"  Auto-detecting architecture...")
    
    num_layers = auto_detect_layers(sage_path)
    print(f"  Detected: {num_layers} layers")
    
    try:
        checkpoint = torch.load(sage_path, map_location=DEVICE,weights_only=False)
        model_sage = GraphSageClassification(768, 256, 128, num_layers=num_layers)
        model_sage.load_state_dict(checkpoint['state_dict'])
        model_sage.to(DEVICE)
        model_sage.eval()
        
        print(f"  ✓ Model loaded successfully")
        print(f"  Running forward pass on GPU...")
        
        with torch.no_grad():
            sage_embeddings = model_sage.get_hidden(graphs, fts, time_steps).cpu().numpy()
        
        print(f"  ✓ GraphSAGE embeddings: {sage_embeddings.shape}")
        sage_layers_used = num_layers
        
        del model_sage
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ✗ Error loading GraphSAGE: {e}")
        print(f"  Trying different layer counts...")
        
        # Try common layer counts
        for try_layers in [2, 4, 8, 16]:
            try:
                print(f"    Trying {try_layers} layers...")
                checkpoint = torch.load(sage_path, map_location=DEVICE)
                model_sage = GraphSageClassification(768, 256, 128, num_layers=try_layers)
                model_sage.load_state_dict(checkpoint['state_dict'])
                model_sage.to(DEVICE)
                model_sage.eval()
                
                with torch.no_grad():
                    sage_embeddings = model_sage.get_hidden(graphs, fts, time_steps).cpu().numpy()
                
                print(f"    ✓ Success with {try_layers} layers!")
                sage_layers_used = try_layers
                
                del model_sage
                torch.cuda.empty_cache()
                break
                
            except:
                continue

if torch.cuda.is_available():
    print(f"\n  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

# ============================================================================
# STEP 5: PCA
# ============================================================================
print("\n[Step 5/6] Running PCA...")

pca_raw = PCA(n_components=2)
raw_2d = pca_raw.fit_transform(raw_embeddings)
print(f"  Raw BERT variance: {pca_raw.explained_variance_ratio_[0]:.3f}, {pca_raw.explained_variance_ratio_[1]:.3f}")

gat_2d = None
if gat_embeddings is not None:
    pca_gat = PCA(n_components=2)
    gat_2d = pca_gat.fit_transform(gat_embeddings)
    print(f"  GAT variance: {pca_gat.explained_variance_ratio_[0]:.3f}, {pca_gat.explained_variance_ratio_[1]:.3f}")

sage_2d = None
if sage_embeddings is not None:
    pca_sage = PCA(n_components=2)
    sage_2d = pca_sage.fit_transform(sage_embeddings)
    print(f"  GraphSAGE variance: {pca_sage.explained_variance_ratio_[0]:.3f}, {pca_sage.explained_variance_ratio_[1]:.3f}")

# ============================================================================
# STEP 6: Metrics and Visualization
# ============================================================================
print("\n[Step 6/6] Computing metrics and creating plots...")

def calc_metrics(emb_2d, labels, name):
    sil = silhouette_score(emb_2d, labels)
    
    c1 = emb_2d[labels == 1].mean(axis=0)
    c0 = emb_2d[labels == 0].mean(axis=0)
    dist = np.linalg.norm(c1 - c0)
    
    print(f"\n  {name}:")
    print(f"    Silhouette: {sil:.4f}")
    print(f"    Centroid Distance: {dist:.4f}")
    
    return {'silhouette': float(sil), 'centroid_distance': float(dist)}

metrics = {}
metrics['raw_bert'] = calc_metrics(raw_2d, labels, "Raw BERT")

if gat_2d is not None:
    metrics['gat'] = calc_metrics(gat_2d, labels, f"GAT ({gat_layers_used}L)")

if sage_2d is not None:
    metrics['sage'] = calc_metrics(sage_2d, labels, f"GraphSAGE ({sage_layers_used}L)")

# Plot
print("\n  Creating visualization...")

n_plots = 1 + (gat_2d is not None) + (sage_2d is not None)
fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
if n_plots == 1:
    axes = [axes]

colors = ['#4ECDC4', '#FF6B6B']
labels_txt = ['Non-spreader', 'Misinformation Spreader']

idx = 0

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
    
    axes[idx].set_title(f'GAT ({gat_layers_used} layers)\n+ Graph Attention', fontsize=14, fontweight='bold')
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
    
    axes[idx].set_title(f'GraphSAGE ({sage_layers_used} layers)\n+ Neighborhood Aggregation', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('PC1', fontsize=11)
    axes[idx].set_ylabel('PC2', fontsize=11)
    axes[idx].legend(loc='best', fontsize=10)
    axes[idx].grid(True, alpha=0.2)
    axes[idx].text(0.03, 0.97,
                   f"Silhouette: {metrics['sage']['silhouette']:.3f}\nDist: {metrics['sage']['centroid_distance']:.2f}",
                   transform=axes[idx].transAxes, fontsize=10, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

plt.suptitle(f'PCA Embedding Comparison ({len(labels)} users)', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save
output_dir = "../results/pca"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/pca_comparison.png", dpi=300, bbox_inches='tight')
print(f"\n  ✓ Saved: {output_dir}/pca_comparison.png")

# Save data
pkl.dump({
    'raw_2d': raw_2d, 'gat_2d': gat_2d, 'sage_2d': sage_2d,
    'labels': labels, 'metrics': metrics,
    'gat_layers': gat_layers_used, 'sage_layers': sage_layers_used
}, open(f"{output_dir}/embeddings.pkl", 'wb'))

import json
json.dump(metrics, open(f"{output_dir}/metrics.json", 'w'), indent=2)

print(f"  ✓ Saved: {output_dir}/embeddings.pkl")
print(f"  ✓ Saved: {output_dir}/metrics.json")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nUsers: {len(labels)} (Spreaders: {(labels==1).sum()}, Non-spreaders: {(labels==0).sum()})")
print("\nSilhouette Scores:")
print(f"  Raw BERT:  {metrics['raw_bert']['silhouette']:.4f}")
if 'gat' in metrics:
    print(f"  GAT ({gat_layers_used}L):   {metrics['gat']['silhouette']:.4f}  (Δ={metrics['gat']['silhouette']-metrics['raw_bert']['silhouette']:+.4f})")
if 'sage' in metrics:
    print(f"  SAGE ({sage_layers_used}L):  {metrics['sage']['silhouette']:.4f}  (Δ={metrics['sage']['silhouette']-metrics['raw_bert']['silhouette']:+.4f})")

print("\n" + "=" * 70)
print("✓ DONE!")
print("=" * 70)
