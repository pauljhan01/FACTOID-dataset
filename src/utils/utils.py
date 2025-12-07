import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pickle
import gzip

# ------------------------------------------------------------------
# Utility for converting yes/no strings to bool
# ------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

# ------------------------------------------------------------------
# Convert JSON-serializable values
# ------------------------------------------------------------------
def convert_value(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return [convert_value(v) for v in value]
    return str(value)

# ------------------------------------------------------------------
# Focal class-balanced loss (CB Loss)
# ------------------------------------------------------------------
def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    weights = torch.tensor(weights, dtype=torch.float).cuda()

    labels_one_hot = F.one_hot(labels, no_of_classes).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1).unsqueeze(1)

    if loss_type == "softmax":
        pred = F.log_softmax(logits, dim=1)
        loss = -weights * labels_one_hot * pred
        loss = loss.sum(1)
    else:
        raise ValueError("Only softmax CB loss is implemented.")

    return loss.mean()

# ------------------------------------------------------------------
# Accuracy metric
# ------------------------------------------------------------------
def accuracy(output, labels):
    preds = output.argmax(dim=1)
    return (preds == labels).float().mean()

# ------------------------------------------------------------------
# Pretty print of precision, recall, F1
# ------------------------------------------------------------------
def print_metrics(preds, labels):
    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)

    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {acc:.4f}")

    # Return in same order as original code expected
    return [p, r, f1, f1, acc]

# ------------------------------------------------------------------
# Save training checkpoints
# ------------------------------------------------------------------
def save_checkpoint(state, checkpoint=".", name="checkpoint.tar"):
    path = os.path.join(checkpoint, name)
    torch.save(state, path)
    print(f"Checkpoint saved → {path}")

# ------------------------------------------------------------------
# Read user IDs
# ------------------------------------------------------------------
def read_userids(filename, root):
    ids = []
    with open(os.path.join(root, filename), "r") as f:
        for line in f:
            ids.append(line.strip())
    return ids
