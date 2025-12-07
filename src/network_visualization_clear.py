import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import os

print("=" * 70)
print("CLEAR NETWORK VISUALIZATION: Hop-by-Hop Spread")
print("=" * 70)

from model import GraphSageClassification, GatV2Classification
from experiments import get_data
from constants import DEVICE

# ============================================================================
# Load Data and Model
# ============================================================================
print("\n[1/4] Loading data and model...")

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

# Get predictions
with torch.no_grad():
    output = model(graphs, fts, time_steps, adj=None)
    probabilities = torch.softmax(output, dim=1)
    spreader_confidence = probabilities[:, 1].cpu().numpy()

print(f"  ✓ {model_name} loaded")

# ============================================================================
# Build Network
# ============================================================================
print("\n[2/4] Building network graph...")

last_graph = graphs[-1]
src = last_graph[0].cpu().numpy()
dst = last_graph[1].cpu().numpy()

G = nx.Graph()
G.add_edges_from(zip(src, dst))
degrees = dict(G.degree())

n_nodes = len(labels)
print(f"  Network: {n_nodes} nodes, {G.number_of_edges()} edges")

# ============================================================================
# Select High-Risk Seeds
# ============================================================================
print("\n[3/4] Selecting seed nodes...")

high_risk = []
for node_id in range(n_nodes):
    if node_id not in G.nodes():
        continue
    if (labels[node_id] == 1 and 
        spreader_confidence[node_id] > 0.7 and 
        degrees[node_id] > 5):
        high_risk.append({
            'node_id': node_id,
            'risk': spreader_confidence[node_id] * degrees[node_id]
        })

high_risk = sorted(high_risk, key=lambda x: x['risk'], reverse=True)
n_seeds = min(3, len(high_risk))  # Use 3 seeds for clearer visualization
seed_nodes = [s['node_id'] for s in high_risk[:n_seeds]]

print(f"  Selected {n_seeds} seeds: {seed_nodes}")

# ============================================================================
# Simulate Spread and Track Hops
# ============================================================================
print("\n[4/4] Running spread simulation...")

def simulate_with_hop_tracking(G, seeds, max_hops=4):
    """Track which hop each node was infected at"""
    hop_assignment = {}
    for seed in seeds:
        hop_assignment[seed] = 0
    
    queue = deque([(node, 0) for node in seeds])
    visited = set(seeds)
    
    while queue:
        current, hop = queue.popleft()
        if hop >= max_hops:
            continue
        
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                if np.random.random() < 0.7:  # Infection probability
                    hop_assignment[neighbor] = hop + 1
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))
    
    return hop_assignment

np.random.seed(42)
hop_assignment = simulate_with_hop_tracking(G, seed_nodes, max_hops=4)

# Count nodes per hop
hop_counts = {i: 0 for i in range(5)}
for node, hop in hop_assignment.items():
    hop_counts[hop] += 1

print(f"\n  Spread results:")
for hop in range(5):
    print(f"    Hop {hop}: {hop_counts[hop]} users")
print(f"    Total reached: {len(hop_assignment)} users")

# ============================================================================
# Create Clear Visualization
# ============================================================================
print("\n  Creating network visualization...")

output_dir = "../results/network_viz_clear"
os.makedirs(output_dir, exist_ok=True)

# Build subgraph for visualization
infected_nodes = list(hop_assignment.keys())
subgraph_nodes = set(infected_nodes)

# Add some uninfected neighbors for context (up to 50)
uninfected_neighbors = set()
for node in infected_nodes[:20]:  # Sample from infected
    for neighbor in G.neighbors(node):
        if neighbor not in subgraph_nodes:
            uninfected_neighbors.add(neighbor)
            if len(uninfected_neighbors) >= 50:
                break
    if len(uninfected_neighbors) >= 50:
        break

subgraph_nodes.update(uninfected_neighbors)
subgraph = G.subgraph(list(subgraph_nodes))

# ============================================================================
# PLOT: Network with Clear Colors
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(16, 14))

# Layout
pos = nx.spring_layout(subgraph, k=1.5, iterations=50, seed=42)

# Define VERY distinguishable colors
COLORS = {
    0: '#FF0000',    # Bright Red - Seeds
    1: '#0000FF',    # Bright Blue - Hop 1
    2: '#00FF00',    # Bright Green - Hop 2
    3: '#FF00FF',    # Magenta - Hop 3
    4: '#FFA500',    # Orange - Hop 4
    -1: '#D3D3D3'    # Light Gray - Not infected
}

# Assign colors and sizes
node_colors = []
node_sizes = []
node_borders = []

for node in subgraph.nodes():
    hop = hop_assignment.get(node, -1)  # -1 if not infected
    node_colors.append(COLORS[hop])
    
    # Larger nodes for infected users
    if hop >= 0:
        base_size = 300 if hop == 0 else 200  # Seeds larger
        node_sizes.append(base_size + degrees.get(node, 1) * 10)
        node_borders.append('black')
    else:
        node_sizes.append(80)
        node_borders.append('gray')

# Draw edges first
nx.draw_networkx_edges(subgraph, pos, alpha=0.1, width=0.5, ax=ax)

# Draw nodes
nx.draw_networkx_nodes(subgraph, pos,
                       node_color=node_colors,
                       node_size=node_sizes,
                       edgecolors=node_borders,
                       linewidths=2,
                       alpha=0.95,
                       ax=ax)

# Add seed labels
for seed in seed_nodes:
    if seed in pos:
        ax.text(pos[seed][0], pos[seed][1], '★', 
               fontsize=20, ha='center', va='center', 
               color='yellow', weight='bold')

# Create legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS[0], edgecolor='black', linewidth=2, 
          label=f'Seeds (n={hop_counts[0]}) ★'),
    Patch(facecolor=COLORS[1], edgecolor='black', linewidth=2, 
          label=f'Hop 1 (n={hop_counts[1]})'),
    Patch(facecolor=COLORS[2], edgecolor='black', linewidth=2, 
          label=f'Hop 2 (n={hop_counts[2]})'),
    Patch(facecolor=COLORS[3], edgecolor='black', linewidth=2, 
          label=f'Hop 3 (n={hop_counts[3]})'),
    Patch(facecolor=COLORS[4], edgecolor='black', linewidth=2, 
          label=f'Hop 4 (n={hop_counts[4]})'),
    Patch(facecolor=COLORS[-1], edgecolor='gray', linewidth=1, 
          label=f'Not Infected')
]

ax.legend(handles=legend_elements, loc='upper left', 
         fontsize=13, framealpha=0.95, edgecolor='black', fancybox=True)

ax.set_title(f'{model_name}-Guided Misinformation Spread\n'
            f'{n_seeds} High-Risk Seeds → {len(hop_assignment)} Users Infected in 4 Hops '
            f'({100*len(hop_assignment)/n_nodes:.1f}% of Network)',
            fontsize=16, fontweight='bold', pad=20)

ax.axis('off')
plt.tight_layout()
plt.savefig(f"{output_dir}/network_spread_clear.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir}/network_spread_clear.png")
plt.close()

# ============================================================================
# BONUS: Side-by-Side Hop Visualization
# ============================================================================
print("\n  Creating hop-by-hop breakdown...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for hop_idx in range(5):
    ax = axes[hop_idx]
    
    # Show only nodes up to this hop
    nodes_visible = [n for n, h in hop_assignment.items() if h <= hop_idx]
    subgraph_hop = G.subgraph(nodes_visible)
    
    if len(nodes_visible) > 0:
        # Use same layout for consistency
        pos_hop = {n: pos[n] for n in nodes_visible if n in pos}
        
        # Colors
        colors_hop = [COLORS[hop_assignment[n]] for n in nodes_visible if n in pos]
        sizes_hop = [300 if hop_assignment[n] == 0 else 200 for n in nodes_visible if n in pos]
        
        # Draw
        nx.draw_networkx_edges(subgraph_hop, pos_hop, alpha=0.15, width=0.5, ax=ax)
        nx.draw_networkx_nodes(subgraph_hop, pos_hop,
                              nodelist=[n for n in nodes_visible if n in pos],
                              node_color=colors_hop,
                              node_size=sizes_hop,
                              edgecolors='black',
                              linewidths=1.5,
                              alpha=0.9,
                              ax=ax)
    
    ax.set_title(f'After Hop {hop_idx}\n{sum(1 for h in hop_assignment.values() if h <= hop_idx)} users infected',
                fontsize=13, fontweight='bold')
    ax.axis('off')

# Remove extra subplot
axes[5].axis('off')

plt.suptitle(f'{model_name} Spread Animation (Step-by-Step)\nRed→Blue→Green→Magenta→Orange',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/hop_by_hop_progression.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/hop_by_hop_progression.png")
plt.close()

# ============================================================================
# BONUS: Radial Layout (Shows Hops Clearly)
# ============================================================================
print("\n  Creating radial visualization...")

fig, ax = plt.subplots(1, 1, figsize=(14, 14))

# Create radial layout based on hop distance
pos_radial = {}
for node, hop in hop_assignment.items():
    # Place nodes in concentric circles based on hop
    angle = np.random.uniform(0, 2 * np.pi)
    radius = hop * 2 + np.random.uniform(-0.3, 0.3)  # Add jitter
    pos_radial[node] = (radius * np.cos(angle), radius * np.sin(angle))

# Add uninfected nodes at outer ring
for node in uninfected_neighbors:
    if node in subgraph.nodes():
        angle = np.random.uniform(0, 2 * np.pi)
        radius = 10
        pos_radial[node] = (radius * np.cos(angle), radius * np.sin(angle))

# Colors and sizes
node_colors_radial = []
node_sizes_radial = []

for node in subgraph.nodes():
    if node in pos_radial:
        hop = hop_assignment.get(node, -1)
        node_colors_radial.append(COLORS[hop])
        node_sizes_radial.append(400 if hop == 0 else (250 if hop >= 0 else 100))

# Draw
nx.draw_networkx_edges(subgraph, pos_radial, alpha=0.08, width=0.4, ax=ax)
nx.draw_networkx_nodes(subgraph, pos_radial,
                       node_color=node_colors_radial,
                       node_size=node_sizes_radial,
                       edgecolors='black',
                       linewidths=1.5,
                       alpha=0.95,
                       ax=ax)

# Draw concentric circles to show hop boundaries
for hop in range(5):
    circle = plt.Circle((0, 0), hop * 2, fill=False, 
                       edgecolor='gray', linestyle='--', 
                       linewidth=1.5, alpha=0.4)
    ax.add_patch(circle)
    
    # Label the circle
    ax.text(0, hop * 2 + 0.3, f'Hop {hop}', 
           ha='center', fontsize=11, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.legend(handles=legend_elements, loc='upper right', 
         fontsize=12, framealpha=0.95)

ax.set_title(f'Radial View: Misinformation Spreading Outward\n'
            f'{n_seeds} seeds (center) → {len(hop_assignment)} users in {max(hop_assignment.values())} hops',
            fontsize=15, fontweight='bold', pad=20)

ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.axis('off')

plt.tight_layout()
plt.savefig(f"{output_dir}/radial_spread_view.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/radial_spread_view.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nSpread from {n_seeds} seeds:")
cumulative = 0
for hop in range(5):
    cumulative += hop_counts[hop]
    print(f"  After Hop {hop}: {hop_counts[hop]:3d} new | {cumulative:3d} total ({100*cumulative/n_nodes:.1f}%)")

print(f"\nVisualization color key:")
print(f"  🔴 Red    = Seeds (Hop 0)")
print(f"  🔵 Blue   = Hop 1")
print(f"  🟢 Green  = Hop 2")
print(f"  🟣 Magenta = Hop 3")
print(f"  🟠 Orange = Hop 4")
print(f"  ⚪ Gray   = Not infected")

print("\n" + "=" * 70)
print("✓ VISUALIZATION COMPLETE!")
print("=" * 70)
