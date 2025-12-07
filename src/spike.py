import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments import get_data
from constants import DEVICE

print("\nLoading data...")
train_samples, test_samples, val_samples = get_data()
sample = test_samples[0]

# fts: [T, N, D]
fts = sample.features.cpu().numpy()
labels = sample.labels.cpu().numpy()

T, N, D = fts.shape
print(f"Shape: T={T}, N={N}, D={D}")

# ---------------------------------------------------------
# 1. Activity Spike per Month
# ---------------------------------------------------------
activity_per_month = []

for t in range(T):
    month_features = fts[t]                 # [N, 768]
    active_mask = np.linalg.norm(month_features, axis=1) > 0  # user has activity
    
    spreaders_active = ((labels == 1) & active_mask).sum()
    nonspreaders_active = ((labels == 0) & active_mask).sum()
    
    activity_per_month.append((spreaders_active, nonspreaders_active))

print("\nSpreaders & Non-spreaders active per month:")
for t, (s, ns) in enumerate(activity_per_month):
    print(f"Month {t}: Spreaders={s}, Nonspreaders={ns}")

# Plot activity
months = np.arange(T)
spread = [s for s, ns in activity_per_month]
nonspread = [ns for s, ns in activity_per_month]

plt.figure(figsize=(8,5))
plt.plot(months, spread, marker='o', label='Spreaders')
plt.plot(months, nonspread, marker='s', label='Non-spreaders')
plt.title("User Activity Over Time")
plt.xlabel("Month Index")
plt.ylabel("Active Users")
plt.legend()
plt.grid(True)
plt.savefig("../results/pca/activity_spike.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 2. Embedding Drift Over Time (Cosine Similarity)
# ---------------------------------------------------------
def cosine(a, b, eps=1e-8):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

monthly_vecs = fts.mean(axis=1)    # [T, 768]

cosine_sim = []
for t in range(1, T):
    cos = cosine(monthly_vecs[t], monthly_vecs[t-1])
    cosine_sim.append(cos)

print("\nMonth-to-Month Embedding Drift (cosine similarity):")
for t, c in enumerate(cosine_sim, start=1):
    print(f"Month {t-1} → {t}: CosSim={c:.4f}")

plt.figure(figsize=(8,5))
plt.plot(range(1, T), cosine_sim, marker='o')
plt.title("Embedding Drift Over Time")
plt.xlabel("Month Transition")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.savefig("../results/pca/embedding_drift.png", dpi=200)
plt.close()

# ---------------------------------------------------------
# 3. Users who changed the most over time
# ---------------------------------------------------------
user_volatility = np.zeros(N)

for t in range(1, T):
    diff = np.linalg.norm(fts[t] - fts[t-1], axis=1)   # per-user drift
    user_volatility += diff

# rank users by volatility
top_users = user_volatility.argsort()[::-1][:10]

print("\nTop 10 Most Changing Users:")
for idx in top_users:
    print(f"User {idx} | Label={labels[idx]} | Volatility={user_volatility[idx]:.4f}")

# Plot volatility distribution
plt.figure(figsize=(8,5))
plt.hist(user_volatility, bins=40, color='purple', alpha=0.7)
plt.title("Distribution of User Volatility Over Time")
plt.xlabel("Volatility (L2 Drift Across Months)")
plt.ylabel("Count of Users")
plt.grid(True)
plt.savefig("../results/pca/user_volatility.png", dpi=200)
plt.close()

print("\nSaved:")
print(" - activity_spike.png")
print(" - embedding_drift.png")
print(" - user_volatility.png")
print("\n✓ DONE")

