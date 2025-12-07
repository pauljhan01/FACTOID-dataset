import torch
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import pandas as pd
import os

print("=" * 70)
print("TEMPORAL ANALYSIS: User Embedding Drift Over Time")
print("=" * 70)

from experiments import get_data
from constants import DEVICE

# ============================================================================
# STEP 1: Load All Time Windows
# ============================================================================
print("\n[Step 1/5] Loading all time windows...")

train_samples, test_samples, val_samples = get_data()
sample = test_samples[0]

# Features: (time_steps, n_users, 768)
fts = sample.features.to(DEVICE)
labels = sample.labels.cpu().numpy()
time_steps = sample.window

print(f"  Time windows: {time_steps}")
print(f"  Users: {fts.shape[1]}")
print(f"  Feature dim: {fts.shape[2]}")
print(f"  Spreaders: {(labels==1).sum()}")
print(f"  Non-spreaders: {(labels==0).sum()}")

# Convert to numpy for analysis
embeddings_temporal = fts.cpu().numpy()  # Shape: (16, N, 768)

# ============================================================================
# STEP 2: Compute Temporal Drift Metrics
# ============================================================================
print("\n[Step 2/5] Computing temporal drift metrics...")

N = embeddings_temporal.shape[1]

# 2.1: Month-to-month cosine similarity (per user)
print("\n  Computing month-to-month similarity...")
user_drift = np.zeros((N, time_steps - 1))  # Drift between consecutive months

for user_idx in range(N):
    for t in range(time_steps - 1):
        emb_t = embeddings_temporal[t, user_idx]
        emb_t1 = embeddings_temporal[t + 1, user_idx]
        
        # Cosine similarity (1 = no change, 0 = orthogonal, -1 = opposite)
        similarity = np.dot(emb_t, emb_t1) / (np.linalg.norm(emb_t) * np.linalg.norm(emb_t1) + 1e-10)
        user_drift[user_idx, t] = similarity

# Average drift per user across all months
avg_drift_per_user = user_drift.mean(axis=1)  # Shape: (N,)
print(f"  ✓ Average drift per user computed")

# 2.2: Total drift (distance from month 0 to month 15)
print("\n  Computing total drift (first to last month)...")
total_drift = np.zeros(N)

for user_idx in range(N):
    emb_first = embeddings_temporal[0, user_idx]
    emb_last = embeddings_temporal[-1, user_idx]
    
    # Euclidean distance
    total_drift[user_idx] = euclidean(emb_first, emb_last)

print(f"  ✓ Total drift computed")

# 2.3: Identify most/least drifting users
most_drifting_indices = np.argsort(total_drift)[-10:][::-1]  # Top 10 drifters
least_drifting_indices = np.argsort(total_drift)[:10]  # Bottom 10

print(f"\n  Most drifting users (top 10):")
for idx in most_drifting_indices:
    print(f"    User {idx}: drift={total_drift[idx]:.3f}, label={'Spreader' if labels[idx]==1 else 'Non-spreader'}")

print(f"\n  Least drifting users (bottom 10):")
for idx in least_drifting_indices:
    print(f"    User {idx}: drift={total_drift[idx]:.3f}, label={'Spreader' if labels[idx]==1 else 'Non-spreader'}")

# ============================================================================
# STEP 3: Activity Analysis (Embedding Magnitude as Proxy)
# ============================================================================
print("\n[Step 3/5] Analyzing user activity over time...")

# Compute L2 norm of embeddings as activity proxy
activity = np.linalg.norm(embeddings_temporal, axis=2)  # Shape: (16, N)

# Average activity per month
activity_per_month_spreaders = activity[:, labels == 1].mean(axis=1)
activity_per_month_nonspreaders = activity[:, labels == 0].mean(axis=1)

months = ['Jan20', 'Feb20', 'Mar20', 'Apr20', 'May20', 'Jun20', 
          'Jul20', 'Aug20', 'Sep20', 'Oct20', 'Nov20', 'Dec20',
          'Jan21', 'Feb21', 'Mar21', 'Apr21']

print(f"\n  Activity by month (avg embedding magnitude):")
for t, month in enumerate(months):
    print(f"    {month}: Spreaders={activity_per_month_spreaders[t]:.2f}, Non-spreaders={activity_per_month_nonspreaders[t]:.2f}")

# ============================================================================
# STEP 4: Cluster Transitions (PCA per month)
# ============================================================================
print("\n[Step 4/5] Computing cluster transitions...")

# Run PCA for each month
pca_per_month = []
for t in range(time_steps):
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings_temporal[t])
    pca_per_month.append(emb_2d)

pca_per_month = np.array(pca_per_month)  # Shape: (16, N, 2)

# Compute cluster "center of mass" for each class per month
spreader_centers = []
nonspreader_centers = []

for t in range(time_steps):
    spreader_mask = labels == 1
    nonspreader_mask = labels == 0
    
    spreader_centers.append(pca_per_month[t, spreader_mask].mean(axis=0))
    nonspreader_centers.append(pca_per_month[t, nonspreader_mask].mean(axis=0))

spreader_centers = np.array(spreader_centers)  # Shape: (16, 2)
nonspreader_centers = np.array(nonspreader_centers)

# Distance between cluster centers over time
cluster_separation = np.linalg.norm(spreader_centers - nonspreader_centers, axis=1)

print(f"\n  Cluster separation over time:")
for t, month in enumerate(months):
    print(f"    {month}: {cluster_separation[t]:.3f}")

# ============================================================================
# STEP 5: Visualizations
# ============================================================================
print("\n[Step 5/5] Creating visualizations...")

output_dir = "../results/temporal_analysis"
os.makedirs(output_dir, exist_ok=True)

# --- Plot 1: Activity Over Time ---
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

ax.plot(range(time_steps), activity_per_month_spreaders, 
        marker='o', linewidth=2, markersize=6, label='Misinformation Spreaders', color='#FF6B6B')
ax.plot(range(time_steps), activity_per_month_nonspreaders, 
        marker='s', linewidth=2, markersize=6, label='Non-spreaders', color='#4ECDC4')

ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Avg Embedding Magnitude (Activity Proxy)', fontsize=12, fontweight='bold')
ax.set_title('User Activity Over Time (Jan 2020 - Apr 2021)', fontsize=14, fontweight='bold')
ax.set_xticks(range(time_steps))
ax.set_xticklabels(months, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/activity_over_time.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/activity_over_time.png")
plt.close()

# --- Plot 2: Drift Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of total drift
axes[0].hist(total_drift[labels == 0], bins=30, alpha=0.6, label='Non-spreaders', color='#4ECDC4')
axes[0].hist(total_drift[labels == 1], bins=30, alpha=0.6, label='Spreaders', color='#FF6B6B')
axes[0].set_xlabel('Total Drift (Euclidean Distance)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of Total Drift\n(Jan 2020 → Apr 2021)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
data_boxplot = [total_drift[labels == 0], total_drift[labels == 1]]
bp = axes[1].boxplot(data_boxplot, labels=['Non-spreaders', 'Spreaders'], patch_artist=True)
bp['boxes'][0].set_facecolor('#4ECDC4')
bp['boxes'][1].set_facecolor('#FF6B6B')
axes[1].set_ylabel('Total Drift', fontsize=11)
axes[1].set_title('Drift Comparison by User Type', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{output_dir}/drift_distribution.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/drift_distribution.png")
plt.close()

# --- Plot 3: Month-to-Month Similarity Heatmap ---
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Average similarity across all users for each month pair
similarity_matrix = np.zeros((time_steps, time_steps))

for t1 in range(time_steps):
    for t2 in range(time_steps):
        if t1 == t2:
            similarity_matrix[t1, t2] = 1.0
        else:
            emb_t1 = embeddings_temporal[t1]  # Shape: (N, 768)
            emb_t2 = embeddings_temporal[t2]
            
            # Compute pairwise cosine similarity and average
            sim = cosine_similarity(emb_t1, emb_t2)
            similarity_matrix[t1, t2] = np.diag(sim).mean()  # Diagonal = same user across time

sns.heatmap(similarity_matrix, annot=False, cmap='RdYlGn', vmin=0.5, vmax=1.0,
            xticklabels=months, yticklabels=months, cbar_kws={'label': 'Cosine Similarity'}, ax=ax)
ax.set_title('Temporal Embedding Similarity\n(User Consistency Across Months)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Month', fontsize=11)

plt.tight_layout()
plt.savefig(f"{output_dir}/temporal_similarity_heatmap.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/temporal_similarity_heatmap.png")
plt.close()

# --- Plot 4: Cluster Separation Over Time ---
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

ax.plot(range(time_steps), cluster_separation, 
        marker='o', linewidth=2.5, markersize=7, color='#9B59B6')
ax.fill_between(range(time_steps), 0, cluster_separation, alpha=0.2, color='#9B59B6')

ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Cluster Separation (PCA Space)', fontsize=12, fontweight='bold')
ax.set_title('How Separable Are Spreaders vs Non-spreaders Over Time?', fontsize=14, fontweight='bold')
ax.set_xticks(range(time_steps))
ax.set_xticklabels(months, rotation=45, ha='right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/cluster_separation_over_time.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/cluster_separation_over_time.png")
plt.close()

# --- Plot 5: Individual User Trajectories (Top 5 Drifters) ---
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Use PCA projection of month 0 as reference
pca_reference = PCA(n_components=2)
ref_projection = pca_reference.fit(embeddings_temporal[0])

# Project all months onto the same PCA space
trajectories_2d = []
for t in range(time_steps):
    proj = ref_projection.transform(embeddings_temporal[t])
    trajectories_2d.append(proj)

trajectories_2d = np.array(trajectories_2d)  # Shape: (16, N, 2)

# Plot trajectories for top 5 drifting users
colors_traj = plt.cm.tab10(np.linspace(0, 1, 5))

for i, user_idx in enumerate(most_drifting_indices[:5]):
    trajectory = trajectories_2d[:, user_idx, :]  # Shape: (16, 2)
    
    label_type = 'Spreader' if labels[user_idx] == 1 else 'Non-spreader'
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
            marker='o', linewidth=2, markersize=5, alpha=0.7,
            color=colors_traj[i], label=f'User {user_idx} ({label_type})')
    
    # Mark start and end
    ax.scatter(trajectory[0, 0], trajectory[0, 1], 
              s=150, marker='*', color=colors_traj[i], edgecolors='black', linewidth=1.5, zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], 
              s=150, marker='X', color=colors_traj[i], edgecolors='black', linewidth=1.5, zorder=5)

ax.set_xlabel('PCA Component 1', fontsize=11)
ax.set_ylabel('PCA Component 2', fontsize=11)
ax.set_title('User Embedding Trajectories (Top 5 Drifters)\n★ = Jan 2020, ✖ = Apr 2021', 
             fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/user_trajectories.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/user_trajectories.png")
plt.close()

# ============================================================================
# Save All Metrics
# ============================================================================
print("\n  Saving metrics...")

metrics = {
    'user_drift': user_drift.tolist(),
    'avg_drift_per_user': avg_drift_per_user.tolist(),
    'total_drift': total_drift.tolist(),
    'most_drifting_users': most_drifting_indices.tolist(),
    'least_drifting_users': least_drifting_indices.tolist(),
    'activity_spreaders': activity_per_month_spreaders.tolist(),
    'activity_nonspreaders': activity_per_month_nonspreaders.tolist(),
    'cluster_separation': cluster_separation.tolist(),
    'months': months
}

import json
json.dump(metrics, open(f"{output_dir}/temporal_metrics.json", 'w'), indent=2)
print(f"  ✓ Saved: {output_dir}/temporal_metrics.json")

# Save detailed data
pkl.dump({
    'embeddings_temporal': embeddings_temporal,
    'labels': labels,
    'pca_per_month': pca_per_month,
    'trajectories_2d': trajectories_2d,
    'user_drift': user_drift,
    'total_drift': total_drift,
    'similarity_matrix': similarity_matrix
}, open(f"{output_dir}/temporal_data.pkl", 'wb'))
print(f"  ✓ Saved: {output_dir}/temporal_data.pkl")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nDrift Statistics:")
print(f"  Avg drift (spreaders):     {total_drift[labels==1].mean():.3f} ± {total_drift[labels==1].std():.3f}")
print(f"  Avg drift (non-spreaders): {total_drift[labels==0].mean():.3f} ± {total_drift[labels==0].std():.3f}")

from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(total_drift[labels==1], total_drift[labels==0])
print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")

if p_val < 0.05:
    print(f"  ✓ Significant difference in drift between groups (p < 0.05)")
else:
    print(f"  ✗ No significant difference in drift (p >= 0.05)")

print(f"\nActivity Trends:")
activity_increase_spreaders = activity_per_month_spreaders[-1] - activity_per_month_spreaders[0]
activity_increase_nonspreaders = activity_per_month_nonspreaders[-1] - activity_per_month_nonspreaders[0]
print(f"  Spreaders change:     {activity_increase_spreaders:+.2f}")
print(f"  Non-spreaders change: {activity_increase_nonspreaders:+.2f}")

print(f"\nCluster Separation:")
print(f"  Max separation: {cluster_separation.max():.3f} (Month {cluster_separation.argmax()}: {months[cluster_separation.argmax()]})")
print(f"  Min separation: {cluster_separation.min():.3f} (Month {cluster_separation.argmin()}: {months[cluster_separation.argmin()]})")

print("\n" + "=" * 70)
print("✓ TEMPORAL ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\nResults saved to: {output_dir}/")
