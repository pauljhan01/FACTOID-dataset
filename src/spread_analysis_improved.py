import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import deque
import os

print("=" * 70)
print("IMPROVED SPREAD ANALYSIS WITH CLEAR VISUALIZATIONS")
print("=" * 70)

from model import GraphSageClassification, GatV2Classification
from experiments import get_data
from constants import DEVICE

# ============================================================================
# Load Data and Model
# ============================================================================
print("\n[1/5] Loading data and model...")

train_samples, test_samples, val_samples = get_data()
sample = test_samples[0]

fts = sample.features.to(DEVICE)
graphs = [g.to(DEVICE) for g in sample.graph_data]
time_steps = sample.window
labels = sample.labels.cpu().numpy()

checkpoint_dir = "../results/checkpoints"

def detect_layers(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    layer_keys = [k for k in ckpt['state_dict'].keys() if 'layers.' in k and '.weight' in k]
    if layer_keys:
        nums = [int(k.split('layers.')[1].split('.')[0]) for k in layer_keys]
        return max(nums) + 1
    return 5

model_files = [f for f in os.listdir(checkpoint_dir) if "GraphSage" in f and f.endswith(".tar")]
if not model_files:
    model_files = [f for f in os.listdir(checkpoint_dir) if "GatV2" in f and f.endswith(".tar")]

model_file = sorted(model_files)[0]
model_path = os.path.join(checkpoint_dir, model_file)
num_layers = detect_layers(model_path)

checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

if "GraphSage" in model_file:
    model = GraphSageClassification(768, 256, 128, num_layers=num_layers)
    model_name = "GraphSAGE"
else:
    model = GatV2Classification(768, 256, 128, num_layers=num_layers)
    model_name = "GAT"

model.load_state_dict(checkpoint['state_dict'])
model.to(DEVICE)
model.eval()

print(f"  ✓ {model_name} loaded ({num_layers} layers)")

# Get predictions
with torch.no_grad():
    output = model(graphs, fts, time_steps, adj=None)
    probabilities = torch.softmax(output, dim=1)
    predictions = output.argmax(dim=1).cpu().numpy()
    spreader_confidence = probabilities[:, 1].cpu().numpy()

# ============================================================================
# Build Network
# ============================================================================
print("\n[2/5] Building network...")

last_graph = graphs[-1]
src = last_graph[0].cpu().numpy()
dst = last_graph[1].cpu().numpy()

G = nx.Graph()
G.add_edges_from(zip(src, dst))
degrees = dict(G.degree())

n_nodes = len(labels)
print(f"  Network: {n_nodes} nodes, {G.number_of_edges()} edges")

# ============================================================================
# Select Seeds: High-risk spreaders identified by model
# ============================================================================
print("\n[3/5] Selecting high-risk seed nodes...")

high_risk_spreaders = []
for node_id in range(n_nodes):
    if node_id not in G.nodes():
        continue
    
    if (labels[node_id] == 1 and 
        spreader_confidence[node_id] > 0.7 and 
        degrees[node_id] > 5):
        
        high_risk_spreaders.append({
            'node_id': node_id,
            'confidence': spreader_confidence[node_id],
            'degree': degrees[node_id],
            'risk': spreader_confidence[node_id] * degrees[node_id]
        })

high_risk_spreaders = sorted(high_risk_spreaders, key=lambda x: x['risk'], reverse=True)
n_seeds = min(5, len(high_risk_spreaders))
seed_nodes = [s['node_id'] for s in high_risk_spreaders[:n_seeds]]

print(f"  Selected {n_seeds} high-risk seeds")

# ============================================================================
# Simulation Functions
# ============================================================================

def simulate_spread(G, seeds, infection_prob, max_hops=4):
    """Basic BFS spread simulation"""
    infected = set(seeds)
    infected_per_hop = {i: set() for i in range(max_hops + 1)}
    infected_per_hop[0] = set(seeds)
    
    queue = deque([(node, 0) for node in seeds])
    visited = set(seeds)
    
    while queue:
        current, hop = queue.popleft()
        if hop >= max_hops:
            continue
        
        for neighbor in G.neighbors(current):
            if neighbor not in infected and np.random.random() < infection_prob:
                infected.add(neighbor)
                infected_per_hop[hop + 1].add(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))
    
    cumulative = []
    total = 0
    for h in range(max_hops + 1):
        total += len(infected_per_hop[h])
        cumulative.append(total)
    
    return infected_per_hop, cumulative, infected

# ============================================================================
# Run Multiple Simulations
# ============================================================================
print("\n[4/5] Running spread simulations...")

n_simulations = 20
max_hops = 4

# Storage
misinfo_results = []
normal_results = []
random_results = []

# Random seed selection for baseline
random_seed_nodes = np.random.choice(list(G.nodes()), size=n_seeds, replace=False).tolist()

for sim in range(n_simulations):
    np.random.seed(sim)
    
    # High infection rate (misinformation)
    _, cumulative_misinfo, _ = simulate_spread(G, seed_nodes, infection_prob=0.7, max_hops=max_hops)
    misinfo_results.append(cumulative_misinfo)
    
    # Lower infection rate (normal info)
    _, cumulative_normal, _ = simulate_spread(G, seed_nodes, infection_prob=0.3, max_hops=max_hops)
    normal_results.append(cumulative_normal)
    
    # Random seeds baseline
    _, cumulative_random, _ = simulate_spread(G, random_seed_nodes, infection_prob=0.5, max_hops=max_hops)
    random_results.append(cumulative_random)

# Calculate statistics
misinfo_mean = np.mean(misinfo_results, axis=0)
misinfo_std = np.std(misinfo_results, axis=0)

normal_mean = np.mean(normal_results, axis=0)
normal_std = np.std(normal_results, axis=0)

random_mean = np.mean(random_results, axis=0)
random_std = np.std(random_results, axis=0)

print(f"\n  Results (averaged over {n_simulations} simulations):")
print(f"    Misinformation: {misinfo_mean[-1]:.0f} ± {misinfo_std[-1]:.0f} users")
print(f"    Normal info: {normal_mean[-1]:.0f} ± {normal_std[-1]:.0f} users")
print(f"    Random baseline: {random_mean[-1]:.0f} ± {random_std[-1]:.0f} users")

# ============================================================================
# Create Meaningful Visualizations
# ============================================================================
print("\n[5/5] Creating visualizations...")

output_dir = "../results/spread_improved"
os.makedirs(output_dir, exist_ok=True)

# Define clear, distinguishable colors
COLOR_MISINFO = '#E74C3C'      # Red
COLOR_NORMAL = '#3498DB'       # Blue  
COLOR_RANDOM = '#95A5A6'       # Gray
COLOR_SEED = '#F39C12'         # Orange

hops = np.arange(max_hops + 1)

# ============================================================================
# PLOT 1: Cumulative Reach with Confidence Intervals
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Plot means with confidence intervals
ax.plot(hops, misinfo_mean, marker='o', linewidth=3.5, markersize=10, 
        color=COLOR_MISINFO, label='Misinformation (p=0.7)', zorder=3)
ax.fill_between(hops, misinfo_mean - misinfo_std, misinfo_mean + misinfo_std, 
                alpha=0.2, color=COLOR_MISINFO, zorder=1)

ax.plot(hops, normal_mean, marker='s', linewidth=3.5, markersize=10, 
        color=COLOR_NORMAL, label='Normal Information (p=0.3)', zorder=3)
ax.fill_between(hops, normal_mean - normal_std, normal_mean + normal_std, 
                alpha=0.2, color=COLOR_NORMAL, zorder=1)

ax.plot(hops, random_mean, marker='^', linewidth=3, markersize=9, 
        color=COLOR_RANDOM, label='Random Baseline (p=0.5)', linestyle='--', zorder=2)
ax.fill_between(hops, random_mean - random_std, random_mean + random_std, 
                alpha=0.15, color=COLOR_RANDOM, zorder=1)

# Annotations
for h in hops:
    ax.text(h, misinfo_mean[h] + 8, f'{int(misinfo_mean[h])}', 
           ha='center', fontsize=10, fontweight='bold', color=COLOR_MISINFO)
    ax.text(h, normal_mean[h] - 12, f'{int(normal_mean[h])}', 
           ha='center', fontsize=10, fontweight='bold', color=COLOR_NORMAL)

ax.set_xlabel('Network Distance (Hops from Seed)', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Users Reached', fontsize=14, fontweight='bold')
ax.set_title(f'{model_name}-Identified Spreaders: Misinformation Cascades Rapidly\n'
            f'{n_seeds} high-risk seeds reach {int(misinfo_mean[-1])} users '
            f'({100*misinfo_mean[-1]/n_nodes:.1f}% of network) in {max_hops} hops', 
            fontsize=15, fontweight='bold', pad=15)

ax.set_xticks(hops)
ax.set_xticklabels([f'Seed\n({n_seeds})'] + [f'Hop {i}' for i in range(1, max_hops + 1)])
ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f"{output_dir}/cumulative_spread.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/cumulative_spread.png")
plt.close()

# ============================================================================
# PLOT 2: Growth Rate Analysis
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: New infections per hop
new_infections_misinfo = np.diff(np.concatenate([[0], misinfo_mean]))
new_infections_normal = np.diff(np.concatenate([[0], normal_mean]))
new_infections_random = np.diff(np.concatenate([[0], random_mean]))

x = np.arange(len(hops))
width = 0.25

bars1 = axes[0].bar(x - width, new_infections_misinfo, width, 
                    label='Misinformation', color=COLOR_MISINFO, 
                    edgecolor='black', linewidth=1.5, alpha=0.9)
bars2 = axes[0].bar(x, new_infections_normal, width, 
                    label='Normal Information', color=COLOR_NORMAL,
                    edgecolor='black', linewidth=1.5, alpha=0.9)
bars3 = axes[0].bar(x + width, new_infections_random, width, 
                    label='Random Baseline', color=COLOR_RANDOM,
                    edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{int(height)}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')

axes[0].set_xlabel('Network Hops', fontsize=13, fontweight='bold')
axes[0].set_ylabel('New Users Infected', fontsize=13, fontweight='bold')
axes[0].set_title('Growth Rate: New Infections at Each Layer\n'
                 'Misinformation peaks early (Hop 1-2)', 
                 fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Seed'] + [f'Hop {i}' for i in range(1, max_hops + 1)])
axes[0].legend(fontsize=11, framealpha=0.95)
axes[0].grid(True, alpha=0.3, axis='y')

# Right: Percentage of network reached
pct_misinfo = (misinfo_mean / n_nodes) * 100
pct_normal = (normal_mean / n_nodes) * 100
pct_random = (random_mean / n_nodes) * 100

axes[1].plot(hops, pct_misinfo, marker='o', linewidth=3.5, markersize=10,
            color=COLOR_MISINFO, label='Misinformation')
axes[1].plot(hops, pct_normal, marker='s', linewidth=3.5, markersize=10,
            color=COLOR_NORMAL, label='Normal Information')
axes[1].plot(hops, pct_random, marker='^', linewidth=3, markersize=9,
            color=COLOR_RANDOM, label='Random Baseline', linestyle='--')

axes[1].set_xlabel('Network Hops', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Percentage of Network Reached (%)', fontsize=13, fontweight='bold')
axes[1].set_title('Network Penetration Over Time\n'
                 f'Misinformation reaches {pct_misinfo[-1]:.1f}% vs {pct_normal[-1]:.1f}% (normal)', 
                 fontsize=13, fontweight='bold')
axes[1].set_xticks(hops)
axes[1].set_xticklabels([f'Seed\n({n_seeds})'] + [f'Hop {i}' for i in range(1, max_hops + 1)])
axes[1].legend(fontsize=11, framealpha=0.95)
axes[1].grid(True, alpha=0.3)

# Add horizontal line at 10%, 20%
for pct in [10, 20]:
    axes[1].axhline(pct, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    axes[1].text(max_hops + 0.1, pct, f'{pct}%', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(f"{output_dir}/growth_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/growth_analysis.png")
plt.close()

# ============================================================================
# PLOT 3: Amplification Factor
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

amplification_misinfo = misinfo_mean / n_seeds
amplification_normal = normal_mean / n_seeds
amplification_random = random_mean / n_seeds

ax.plot(hops, amplification_misinfo, marker='o', linewidth=3.5, markersize=11,
       color=COLOR_MISINFO, label='Misinformation')
ax.plot(hops, amplification_normal, marker='s', linewidth=3.5, markersize=11,
       color=COLOR_NORMAL, label='Normal Information')
ax.plot(hops, amplification_random, marker='^', linewidth=3, markersize=10,
       color=COLOR_RANDOM, label='Random Baseline', linestyle='--')

# Fill area
ax.fill_between(hops, amplification_misinfo, alpha=0.15, color=COLOR_MISINFO)
ax.fill_between(hops, amplification_normal, alpha=0.15, color=COLOR_NORMAL)

ax.set_xlabel('Network Hops', fontsize=14, fontweight='bold')
ax.set_ylabel('Amplification Factor (Reach / Seeds)', fontsize=14, fontweight='bold')
ax.set_title(f'Viral Amplification: How Fast Does It Spread?\n'
            f'{n_seeds} seeds → {int(misinfo_mean[-1])} users = '
            f'{amplification_misinfo[-1]:.1f}× amplification in {max_hops} hops', 
            fontsize=14, fontweight='bold', pad=15)

ax.set_xticks(hops)
ax.set_xticklabels(['Start'] + [f'Hop {i}' for i in range(1, max_hops + 1)])
ax.legend(fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3)

# Annotate final values
for y, color, label in [(amplification_misinfo[-1], COLOR_MISINFO, 'Misinfo'),
                         (amplification_normal[-1], COLOR_NORMAL, 'Normal')]:
    ax.text(max_hops + 0.05, y, f'{y:.1f}×', 
           fontsize=11, fontweight='bold', color=color, va='center')

plt.tight_layout()
plt.savefig(f"{output_dir}/amplification_factor.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/amplification_factor.png")
plt.close()

# ============================================================================
# PLOT 4: Comparison Table Visualization
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.axis('off')

# Create comparison data
comparison_data = []
for hop in hops:
    comparison_data.append([
        f"Hop {hop}" if hop > 0 else "Seed",
        f"{int(misinfo_mean[hop])} ({pct_misinfo[hop]:.1f}%)",
        f"{int(normal_mean[hop])} ({pct_normal[hop]:.1f}%)",
        f"{int(random_mean[hop])} ({pct_random[hop]:.1f}%)",
        f"{amplification_misinfo[hop]:.1f}×"
    ])

table = ax.table(cellText=comparison_data,
                colLabels=['Layer', 'Misinformation', 'Normal Info', 'Random', 'Amplification'],
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.25, 0.25, 0.2, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#34495E')
    cell.set_text_props(weight='bold', color='white', fontsize=12)

# Style rows
for i in range(1, len(comparison_data) + 1):
    for j in range(5):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ECF0F1')
        else:
            cell.set_facecolor('white')
        
        # Highlight misinformation column
        if j == 1:
            cell.set_text_props(weight='bold', color=COLOR_MISINFO)
        elif j == 2:
            cell.set_text_props(weight='bold', color=COLOR_NORMAL)

ax.set_title(f'{model_name} Spread Analysis: Quantitative Comparison\n'
            f'Starting from {n_seeds} high-risk spreaders identified by the model\n'
            f'(Averaged over {n_simulations} simulations)', 
            fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f"{output_dir}/comparison_table.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/comparison_table.png")
plt.close()

# ============================================================================
# Save Summary Statistics
# ============================================================================
import json

summary = {
    'model': model_name,
    'num_layers': num_layers,
    'num_seeds': n_seeds,
    'num_simulations': n_simulations,
    'results': {
        'misinformation': {
            'final_reach': int(misinfo_mean[-1]),
            'percentage': float(pct_misinfo[-1]),
            'amplification': float(amplification_misinfo[-1]),
            'per_hop': misinfo_mean.tolist()
        },
        'normal_info': {
            'final_reach': int(normal_mean[-1]),
            'percentage': float(pct_normal[-1]),
            'amplification': float(amplification_normal[-1]),
            'per_hop': normal_mean.tolist()
        },
        'speedup_factor': float(misinfo_mean[-1] / max(normal_mean[-1], 1))
    }
}

json.dump(summary, open(f"{output_dir}/summary_stats.json", 'w'), indent=2)
print(f"✓ Saved: {output_dir}/summary_stats.json")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n{model_name} ({num_layers} layers) identified {n_seeds} high-risk spreaders:")
print(f"  After {max_hops} hops:")
print(f"    Misinformation:    {int(misinfo_mean[-1])} users ({pct_misinfo[-1]:.1f}%) - {amplification_misinfo[-1]:.1f}× amplification")
print(f"    Normal info:       {int(normal_mean[-1])} users ({pct_normal[-1]:.1f}%) - {amplification_normal[-1]:.1f}× amplification")
print(f"    Speedup factor:    {misinfo_mean[-1]/max(normal_mean[-1],1):.2f}× faster than normal information")

print(f"\n  Why {num_layers} layers matter:")
print(f"    • Each GNN layer captures 1 hop of neighborhood aggregation")
print(f"    • {num_layers} layers can detect cascade patterns up to {num_layers} hops")
print(f"    • Critical for identifying spreaders who amplify across multiple degrees of separation")

print("\n" + "=" * 70)
print("✓ ANALYSIS COMPLETE!")
print("=" * 70)
