import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import seaborn as sns
import os

print("=" * 70)
print("MISINFORMATION SPREAD SIMULATION: 4-Hop Propagation")
print("=" * 70)

from experiments import get_data
from constants import DEVICE

# ============================================================================
# Load Data and Build Network
# ============================================================================
print("\n[1/5] Loading data and building social network...")

train_samples, test_samples, val_samples = get_data()
sample = test_samples[0]

graphs = [g.to(DEVICE) for g in sample.graph_data]
labels = sample.labels.cpu().numpy()

# Use the last time window's graph (most complete network)
last_graph = graphs[-1]
src = last_graph[0].cpu().numpy()
dst = last_graph[1].cpu().numpy()

# Build NetworkX graph for analysis
G = nx.Graph()
G.add_edges_from(zip(src, dst))

n_nodes = len(labels)
print(f"  Network: {n_nodes} nodes, {G.number_of_edges()} edges")
print(f"  Spreaders: {(labels == 1).sum()}")
print(f"  Non-spreaders: {(labels == 0).sum()}")

# ============================================================================
# Simulation: BFS-based Spread from Seed Nodes
# ============================================================================
print("\n[2/5] Running spread simulations...")

def simulate_spread(G, seed_nodes, max_hops=4, infection_prob=0.7):
    """
    Simulate information spread using BFS with probabilistic infection
    
    Args:
        G: NetworkX graph
        seed_nodes: Starting nodes (misinformation spreaders)
        max_hops: Maximum propagation distance
        infection_prob: Probability of infection per contact
    
    Returns:
        infected_per_hop: List of newly infected nodes at each hop
        cumulative_infected: Cumulative count at each hop
    """
    infected = set(seed_nodes)
    infected_per_hop = {i: set() for i in range(max_hops + 1)}
    infected_per_hop[0] = set(seed_nodes)
    
    queue = deque([(node, 0) for node in seed_nodes])
    visited = set(seed_nodes)
    
    while queue:
        current_node, hop = queue.popleft()
        
        if hop >= max_hops:
            continue
        
        # Try to infect neighbors
        for neighbor in G.neighbors(current_node):
            if neighbor not in infected:
                # Probabilistic infection
                if np.random.random() < infection_prob:
                    infected.add(neighbor)
                    infected_per_hop[hop + 1].add(neighbor)
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, hop + 1))
    
    # Calculate cumulative counts
    cumulative_infected = []
    total = 0
    for hop in range(max_hops + 1):
        total += len(infected_per_hop[hop])
        cumulative_infected.append(total)
    
    return infected_per_hop, cumulative_infected

# Identify spreaders as seed nodes
spreader_indices = np.where(labels == 1)[0]
spreader_nodes = [int(idx) for idx in spreader_indices if int(idx) in G.nodes()]

print(f"  Using {len(spreader_nodes)} spreaders as seeds")

# Run simulation: Misinformation spread (high infection rate)
np.random.seed(42)
misinfo_per_hop, misinfo_cumulative = simulate_spread(
    G, spreader_nodes[:10], max_hops=4, infection_prob=0.7
)

# Run simulation: Normal information spread (lower infection rate)
np.random.seed(42)
normal_per_hop, normal_cumulative = simulate_spread(
    G, spreader_nodes[:10], max_hops=4, infection_prob=0.3
)

# Run simulation: Random baseline (same seeds but random graph structure)
np.random.seed(42)
random_seeds = np.random.choice(list(G.nodes()), size=10, replace=False).tolist()
random_per_hop, random_cumulative = simulate_spread(
    G, random_seeds, max_hops=4, infection_prob=0.5
)

print(f"\n  Simulation Results (4 hops):")
print(f"    Misinformation: {misinfo_cumulative[-1]} users reached")
print(f"    Normal info: {normal_cumulative[-1]} users reached")
print(f"    Random baseline: {random_cumulative[-1]} users reached")

# ============================================================================
# Analysis: Reach by Hop
# ============================================================================
print("\n[3/5] Analyzing reach at each hop...")

hops = list(range(5))
misinfo_counts = [len(misinfo_per_hop[h]) for h in hops]
normal_counts = [len(normal_per_hop[h]) for h in hops]
random_counts = [len(random_per_hop[h]) for h in hops]

print("\n  New infections per hop:")
print(f"    Hop | Misinformation | Normal | Random")
for h in hops:
    print(f"    {h}   | {misinfo_counts[h]:14d} | {normal_counts[h]:6d} | {random_counts[h]:6d}")

# ============================================================================
# Network Properties
# ============================================================================
print("\n[4/5] Computing network properties...")

# Average clustering coefficient
clustering = nx.average_clustering(G)
print(f"  Average clustering coefficient: {clustering:.4f}")

# Degree distribution
degrees = dict(G.degree())
avg_degree = np.mean(list(degrees.values()))
print(f"  Average degree: {avg_degree:.2f}")

# Spreader vs non-spreader degree
spreader_degrees = [degrees[node] for node in spreader_nodes]
non_spreader_nodes = [int(idx) for idx in np.where(labels == 0)[0] if int(idx) in G.nodes()]
non_spreader_degrees = [degrees[node] for node in non_spreader_nodes[:100]]

print(f"  Avg degree (spreaders): {np.mean(spreader_degrees):.2f}")
print(f"  Avg degree (non-spreaders): {np.mean(non_spreader_degrees):.2f}")

# ============================================================================
# Visualizations
# ============================================================================
print("\n[5/5] Creating visualizations...")

output_dir = "../results/spread_analysis"
os.makedirs(output_dir, exist_ok=True)

# --- Plot 1: Cumulative Reach Over Hops ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Cumulative spread
axes[0].plot(hops, misinfo_cumulative, marker='o', linewidth=3, markersize=8, 
            color='#FF6B6B', label='Misinformation (p=0.7)')
axes[0].plot(hops, normal_cumulative, marker='s', linewidth=3, markersize=8, 
            color='#4ECDC4', label='Normal Information (p=0.3)')
axes[0].plot(hops, random_cumulative, marker='^', linewidth=2, markersize=7, 
            color='gray', linestyle='--', label='Random Baseline (p=0.5)')

axes[0].set_xlabel('Network Hops (Distance from Source)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Cumulative Users Reached', fontsize=13, fontweight='bold')
axes[0].set_title('Misinformation Spreads Faster Across Network Layers', 
                 fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11, loc='upper left')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(hops)
axes[0].set_xticklabels(['Seed', 'Hop 1', 'Hop 2', 'Hop 3', 'Hop 4'])

# Add annotations
for h in range(5):
    axes[0].text(h, misinfo_cumulative[h] + 5, f"{misinfo_cumulative[h]}", 
                ha='center', fontsize=9, color='#FF6B6B', fontweight='bold')

# Right: New infections per hop
x_pos = np.arange(len(hops))
width = 0.25

axes[1].bar(x_pos - width, misinfo_counts, width, label='Misinformation', 
           color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1)
axes[1].bar(x_pos, normal_counts, width, label='Normal Information', 
           color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1)
axes[1].bar(x_pos + width, random_counts, width, label='Random Baseline', 
           color='gray', alpha=0.6, edgecolor='black', linewidth=1)

axes[1].set_xlabel('Network Hops', fontsize=13, fontweight='bold')
axes[1].set_ylabel('New Users Infected', fontsize=13, fontweight='bold')
axes[1].set_title('New Infections at Each Layer', fontsize=14, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(['Seed', 'Hop 1', 'Hop 2', 'Hop 3', 'Hop 4'])
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{output_dir}/spread_simulation.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_dir}/spread_simulation.png")

# --- Plot 2: Network Visualization (Subgraph) ---
print("\n  Creating network visualization...")

# Extract subgraph around a few spreaders
seed_sample = spreader_nodes[:3]
subgraph_nodes = set(seed_sample)

for node in seed_sample:
    for hop in range(3):
        new_neighbors = []
        for n in list(subgraph_nodes):
            if n in G:
                new_neighbors.extend(list(G.neighbors(n)))
        subgraph_nodes.update(new_neighbors[:50])  # Limit size
        if len(subgraph_nodes) > 200:
            break

subgraph = G.subgraph(list(subgraph_nodes))

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Layout
pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)

# Color nodes
node_colors = []
for node in subgraph.nodes():
    if node in seed_sample:
        node_colors.append('#FF0000')  # Red: seed spreaders
    elif node in misinfo_per_hop[1]:
        node_colors.append('#FF6B6B')  # Light red: hop 1
    elif node in misinfo_per_hop[2]:
        node_colors.append('#FFA07A')  # Orange: hop 2
    elif node in misinfo_per_hop[3]:
        node_colors.append('#FFD700')  # Yellow: hop 3
    else:
        node_colors.append('#E0E0E0')  # Gray: not reached

# Draw
nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5, ax=ax)
nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                       node_size=100, alpha=0.8, ax=ax, edgecolors='black', linewidths=0.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF0000', edgecolor='black', label='Seed (Spreaders)'),
    Patch(facecolor='#FF6B6B', edgecolor='black', label='Hop 1'),
    Patch(facecolor='#FFA07A', edgecolor='black', label='Hop 2'),
    Patch(facecolor='#FFD700', edgecolor='black', label='Hop 3'),
    Patch(facecolor='#E0E0E0', edgecolor='black', label='Not Reached')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

ax.set_title('Misinformation Spread Visualization (3 Hops)\nRed → Light Red → Orange → Yellow', 
            fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig(f"{output_dir}/spread_network_viz.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/spread_network_viz.png")

# --- Plot 3: Degree Distribution Comparison ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.hist(spreader_degrees, bins=30, alpha=0.7, label='Spreaders', 
       color='#FF6B6B', edgecolor='black')
ax.hist(non_spreader_degrees, bins=30, alpha=0.7, label='Non-spreaders', 
       color='#4ECDC4', edgecolor='black')

ax.set_xlabel('Node Degree (# Connections)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Network Degree Distribution: Spreaders vs Non-spreaders', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{output_dir}/degree_distribution.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/degree_distribution.png")

# ============================================================================
# Save Results
# ============================================================================
import json

results = {
    'network_stats': {
        'nodes': int(n_nodes),
        'edges': int(G.number_of_edges()),
        'avg_degree': float(avg_degree),
        'clustering': float(clustering)
    },
    'spread_simulation': {
        'misinformation': {
            'cumulative': [int(x) for x in misinfo_cumulative],
            'per_hop': [int(x) for x in misinfo_counts]
        },
        'normal_info': {
            'cumulative': [int(x) for x in normal_cumulative],
            'per_hop': [int(x) for x in normal_counts]
        },
        'random_baseline': {
            'cumulative': [int(x) for x in random_cumulative],
            'per_hop': [int(x) for x in random_counts]
        }
    },
    'degree_stats': {
        'spreaders_avg': float(np.mean(spreader_degrees)),
        'non_spreaders_avg': float(np.mean(non_spreader_degrees))
    }
}

json.dump(results, open(f"{output_dir}/spread_results.json", 'w'), indent=2)
print(f"✓ Saved: {output_dir}/spread_results.json")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Misinformation Spread Dynamics")
print("=" * 70)

print(f"\nAfter 4 hops:")
print(f"  Misinformation reached: {misinfo_cumulative[-1]} users ({100*misinfo_cumulative[-1]/n_nodes:.1f}%)")
print(f"  Normal info reached:    {normal_cumulative[-1]} users ({100*normal_cumulative[-1]/n_nodes:.1f}%)")
print(f"  Speedup factor:         {misinfo_cumulative[-1]/max(normal_cumulative[-1], 1):.2f}x")

print(f"\nNetwork properties:")
print(f"  Spreaders have {np.mean(spreader_degrees)/np.mean(non_spreader_degrees):.2f}x more connections on average")

print("\n" + "=" * 70)
print("✓ SIMULATION COMPLETE!")
print("=" * 70)
