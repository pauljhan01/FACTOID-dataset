import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

from model import GraphSageClassification, GatV2Classification
from experiments import get_data
from constants import DEVICE

# ============================================================================
# Load Data
# ============================================================================
print("\n[1/5] Loading data...")
train_samples, test_samples, val_samples = get_data()
sample = test_samples[0]

fts = sample.features.to(DEVICE)
graphs = [g.to(DEVICE) for g in sample.graph_data]
time_steps = sample.window
labels = sample.labels.to(DEVICE)

# Subsample
n_users = 500  # Smaller for faster computation
np.random.seed(42)
if fts.shape[1] > n_users:
    indices = np.random.choice(fts.shape[1], n_users, replace=False)
    indices = np.sort(indices)
    
    labels = labels[indices]
    fts = fts[:, indices, :]
    
    # Remap graphs
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

print(f"  Users: {len(labels)}, Spreaders: {(labels==1).sum().item()}")

# ============================================================================
# Method 1: Integrated Gradients (Input Feature Attribution)
# ============================================================================
print("\n[2/5] Computing Integrated Gradients...")

def integrated_gradients(model, graphs, fts, time_steps, labels, target_class=1, steps=50):
    """
    Compute feature importance using Integrated Gradients
    Returns: importance scores for each of the 768 BERT dimensions
    """
    model.eval()
    
    # Baseline: zeros
    baseline = torch.zeros_like(fts)
    
    # Interpolate between baseline and actual input
    alphas = torch.linspace(0, 1, steps).to(DEVICE)
    
    gradients = []
    
    for alpha in alphas:
        # Interpolated input
        interpolated_fts = baseline + alpha * (fts - baseline)
        interpolated_fts.requires_grad = True
        
        # Forward pass
        y_full = []
        for i in range(time_steps):
            x = interpolated_fts[i]
            G = graphs[i]
            
            y = torch.nn.functional.leaky_relu(model.layers[0](x, G), 0.2)
            for layer in model.layers[1:]:
                y = torch.nn.functional.leaky_relu(layer(y, G), 0.2)
            y_full.append(y.unsqueeze(0))
        
        y = torch.cat(y_full, dim=0).mean(dim=0)  # Average across time
        
        # Final classification
        output = model.linear_pass(y)
        output = torch.nn.functional.leaky_relu(output.reshape(fts.shape[1], -1), 0.2)
        output = model.linear(output)
        
        # Get gradients for target class
        class_score = output[:, target_class].sum()
        class_score.backward()
        
        gradients.append(interpolated_fts.grad.clone())
        
        # Zero gradients
        model.zero_grad()
        interpolated_fts.grad.zero_()
    
    # Average gradients
    avg_gradients = torch.stack(gradients).mean(dim=0)
    
    # Integrated gradients = (input - baseline) * avg_gradients
    integrated_grads = (fts - baseline) * avg_gradients
    
    # Aggregate: average across time and users
    importance = integrated_grads.abs().mean(dim=[0, 1]).cpu().numpy()  # Shape: (768,)
    
    return importance

# Load model
checkpoint_dir = "../results/checkpoints"

def detect_layers(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)  # ✅ FIXED
    layer_keys = [k for k in ckpt['state_dict'].keys() if 'layers.' in k and '.weight' in k]
    if layer_keys:
        nums = [int(k.split('layers.')[1].split('.')[0]) for k in layer_keys]
        return max(nums) + 1
    return 5

# Choose model (GraphSAGE or GAT)
model_files = [f for f in os.listdir(checkpoint_dir) if "GraphSage" in f and f.endswith(".tar")]
if not model_files:
    model_files = [f for f in os.listdir(checkpoint_dir) if "GatV2" in f and f.endswith(".tar")]

if model_files:
    model_file = sorted(model_files)[0]
    model_path = os.path.join(checkpoint_dir, model_file)
    num_layers = detect_layers(model_path)
    
    print(f"  Loading: {model_file} ({num_layers} layers)")
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)  # ✅ FIXED
    
    if "GraphSage" in model_file:
        model = GraphSageClassification(768, 256, 128, num_layers=num_layers)
    else:
        model = GatV2Classification(768, 256, 128, num_layers=num_layers)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"  Computing feature importance (this may take a few minutes)...")
    importance_scores = integrated_gradients(model, graphs, fts, time_steps, labels, steps=20)
    
    print(f"  ✓ Feature importance computed: {importance_scores.shape}")

# ============================================================================
# Method 2: PCA to find most informative dimensions
# ============================================================================
print("\n[3/5] Finding most informative dimensions via PCA...")

raw_embeddings = fts.mean(dim=0).cpu().numpy()  # (n_users, 768)
labels_cpu = labels.cpu().numpy()

# PCA
pca = PCA(n_components=768)
pca.fit(raw_embeddings)

# Top components by explained variance
top_components = np.argsort(pca.explained_variance_ratio_)[::-1][:20]
top_variance = pca.explained_variance_ratio_[top_components]

print(f"  Top 20 components explain {top_variance.sum():.2%} of variance")

# ============================================================================
# Method 3: Correlation with Label
# ============================================================================
print("\n[4/5] Computing feature-label correlation...")

# Correlation between each feature dimension and the label
correlations = np.zeros(768)
for dim in range(768):
    correlations[dim] = np.corrcoef(raw_embeddings[:, dim], labels_cpu)[0, 1]

top_corr_dims = np.argsort(np.abs(correlations))[::-1][:20]

print(f"  Top correlated dimensions: {top_corr_dims[:10]}")
print(f"  Correlations: {correlations[top_corr_dims[:10]]}")

# ============================================================================
# Visualizations
# ============================================================================
print("\n[5/5] Creating visualizations...")

output_dir = "../results/feature_importance"
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Integrated Gradients
top_ig = np.argsort(importance_scores)[::-1][:30]
axes[0, 0].barh(range(30), importance_scores[top_ig][::-1], color='steelblue')
axes[0, 0].set_yticks(range(30))
axes[0, 0].set_yticklabels(top_ig[::-1], fontsize=8)
axes[0, 0].set_xlabel('Importance Score (Integrated Gradients)', fontsize=11)
axes[0, 0].set_ylabel('BERT Dimension', fontsize=11)
axes[0, 0].set_title('Top 30 Features by Integrated Gradients\n(How much each BERT dimension affects predictions)', 
                     fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Plot 2: PCA Variance
axes[0, 1].bar(range(20), top_variance, color='coral')
axes[0, 1].set_xlabel('Principal Component', fontsize=11)
axes[0, 1].set_ylabel('Explained Variance Ratio', fontsize=11)
axes[0, 1].set_title('Top 20 Principal Components\n(Which dimensions capture most variation)', 
                     fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Correlation with Label
top_pos_corr = top_corr_dims[:15]
axes[1, 0].barh(range(15), correlations[top_pos_corr][::-1], color='green', alpha=0.7)
axes[1, 0].set_yticks(range(15))
axes[1, 0].set_yticklabels(top_pos_corr[::-1], fontsize=8)
axes[1, 0].set_xlabel('Correlation with Label', fontsize=11)
axes[1, 0].set_ylabel('BERT Dimension', fontsize=11)
axes[1, 0].set_title('Top 15 Features by Label Correlation\n(Which dimensions best separate spreaders vs non-spreaders)', 
                     fontsize=12, fontweight='bold')
axes[1, 0].axvline(0, color='black', linewidth=0.8)
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Plot 4: Distribution of importance scores
axes[1, 1].hist(importance_scores, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(importance_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {importance_scores.mean():.4f}')
axes[1, 1].set_xlabel('Importance Score', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title('Distribution of Feature Importance\n(Most features have low importance)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Feature Importance Analysis: Which BERT Dimensions Matter Most?', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir}/feature_importance.png")

# ============================================================================
# Save Results
# ============================================================================
import json

results = {
    'integrated_gradients': {
        'scores': importance_scores.tolist(),
        'top_30_dims': top_ig[:30].tolist(),
        'top_30_scores': importance_scores[top_ig[:30]].tolist()
    },
    'pca_variance': {
        'top_20_components': top_components.tolist(),
        'explained_variance': top_variance.tolist(),
        'cumulative_variance': float(top_variance.sum())
    },
    'label_correlation': {
        'correlations': correlations.tolist(),
        'top_15_dims': top_corr_dims[:15].tolist(),
        'top_15_correlations': correlations[top_corr_dims[:15]].tolist()
    }
}

json.dump(results, open(f"{output_dir}/feature_importance.json", 'w'), indent=2)
print(f"✓ Saved: {output_dir}/feature_importance.json")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Most Important Features")
print("=" * 70)

print("\nTop 10 Features (Integrated Gradients):")
for i, dim in enumerate(top_ig[:10]):
    print(f"  {i+1}. Dimension {dim}: {importance_scores[dim]:.6f}")

print("\nTop 10 Features (Label Correlation):")
for i, dim in enumerate(top_corr_dims[:10]):
    print(f"  {i+1}. Dimension {dim}: {correlations[dim]:.4f}")

print("\nKey Insights:")
print(f"  • {(importance_scores > importance_scores.mean()).sum()} / 768 dimensions are above-average importance")
print(f"  • Top 20 PCA components explain {top_variance.sum():.1%} of variance")
print(f"  • Top 10 correlated features have |r| > {np.abs(correlations[top_corr_dims[9]]):.3f}")

print("\n" + "=" * 70)
print("✓ DONE!")
print("=" * 70)
