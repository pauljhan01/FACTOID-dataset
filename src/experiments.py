import os
import time
import datetime
import json
import gzip
import pickle as pkl
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

from constants import DEVICE
from model import GraphSageClassification, GatV2Classification
from utils.utils import CB_loss, accuracy, print_metrics, save_checkpoint

# ============================================================
# GLOBAL CONFIG
# ============================================================

# Set to True when debugging or doing quick PCA-style tests
# Set to False for full training runs
TEST_MODE = False   # <-- flip this when you want "small / fast" mode

# ============================================================
# ARG PARSER
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--max_epochs", dest="max_epochs", default=50, type=int)
parser.add_argument("--sample_dir", dest="sample_dir",
                    default="../data/bert_embeddings/", type=str)
parser.add_argument("--checkpoint_dir", dest="checkpoint_dir",
                    default="../results/checkpoints/", type=str)
parser.add_argument("--learning_rate", dest="learning_rate", default=5e-5, type=float)
parser.add_argument("--weight_decay", dest="weight_decay", default=1e-3, type=float)
parser.add_argument("--patience", dest="patience", default=10, type=int)
parser.add_argument("--run_id", dest="run_id", default='no_id_given')
parser.add_argument("--result_dir", dest="result_dir",
                    default="../results/", type=str)
parser.add_argument("--nheads", dest="nheads", default=4, type=int)
parser.add_argument("--dropout", dest="dropout", default=0.2, type=float)
parser.add_argument("--model_seed", dest="model_seed", type=int, default=1234)
parser.add_argument("--load_best_model", dest="load_best_model",
                    type=bool, default=False)
parser.add_argument("--lazy_loading", dest="lazy_loading",
                    type=bool, default=False)

args = parser.parse_args()

max_epochs = args.max_epochs
checkpoint_dir = args.checkpoint_dir
lr = args.learning_rate
weight_decay = args.weight_decay
early_stopping_patience = args.patience
load_best_model = args.load_best_model

in_channel = 768
hidden_channel = 256
out_channel = 128

# ============================================================
# LOSS + DATA HELPERS
# ============================================================

def loss_fn(output, targets, samples_per_cls, no_of_classes=2):
    beta = 0.9999
    gamma = 2.0
    loss_type = "softmax"
    return CB_loss(targets, output, samples_per_cls,
                   no_of_classes, loss_type, beta, gamma)


def get_samples_per_class(labels):
    return torch.bincount(labels).tolist()


def generate_set_of_data(path, n):
    """
    Load dataset samples from disk.

    When TEST_MODE is True, we only load a very small number of samples
    to make runs fast (e.g., for debugging or PCA scripts).
    """
    data = []
    n_samples = min(2, n) if TEST_MODE else n

    for i in range(n_samples):
        sample_path = join(path, f"sample_{i}.data")
        data.append(pkl.load(gzip.open(sample_path, "rb")))
    return data


def get_data():
    """
    Load train/test/val sets according to args.sample_dir.
    This is called both from experiments.py and from other scripts
    (e.g., extract_embeddings).
    """
    descriptor_path = os.path.join(args.sample_dir, "dataset_descriptor.json")
    descriptor = json.load(open(descriptor_path, "r"))

    train_path = os.path.join(args.sample_dir, "train_samples/")
    test_path = os.path.join(args.sample_dir, "test_samples/")
    val_path = os.path.join(args.sample_dir, "val_samples/")

    train_set = generate_set_of_data(train_path, descriptor["n_train_samples"])
    test_set = generate_set_of_data(test_path, descriptor["n_test_samples"])
    val_set = generate_set_of_data(val_path, descriptor["n_val_samples"])

    return train_set, test_set, val_set


# Load data once here for training / evaluation.
# When imported (e.g. by extract_embeddings), TEST_MODE usually True → cheap.
train_samples, test_samples, val_samples = get_data()

# ============================================================
# TRAINING / EVAL
# ============================================================

def train_model(model, optimizer):
    """
    Train for ONE epoch over train_samples, evaluate on val/test, and
    return metrics dict.
    """
    model.train()

    acc_train = []
    losses_train = []

    # -------- TRAIN LOOP --------
    for sample in train_samples:
        if args.lazy_loading:
            train_sample = pkl.load(gzip.open(sample, "rb"))
        else:
            train_sample = sample

        train_features = train_sample.features.to(DEVICE)
        train_label = train_sample.labels.to(DEVICE)
        all_graphs = [graph.to(DEVICE) for graph in train_sample.graph_data]
        time_steps = train_sample.window

        optimizer.zero_grad()
        output = model(all_graphs, train_features, time_steps)
        loss_train = loss_fn(output, train_label, get_samples_per_class(train_label))

        acc_train.append(accuracy(output, train_label).detach().cpu().numpy())
        loss_train.backward()
        optimizer.step()
        losses_train.append(loss_train.detach().cpu().numpy())

    # -------- EVAL (VAL + TEST) --------
    model.eval()
    accuracy_val = []
    accuracy_test = []
    losses_val = []
    losses_test = []

    with torch.no_grad():
        # Validation
        for sample in val_samples:
            if args.lazy_loading:
                val_sample = pkl.load(gzip.open(sample, "rb"))
            else:
                val_sample = sample

            val_features = val_sample.features.to(DEVICE)
            val_label = val_sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in val_sample.graph_data]
            time_steps = val_sample.window

            output = model(all_graphs, val_features, time_steps)
            loss_val = F.nll_loss(output, val_label)
            accuracy_val.append(accuracy(output, val_label).detach().cpu().numpy())
            losses_val.append(loss_val.detach().cpu().numpy())

        # Test
        for sample in test_samples:
            if args.lazy_loading:
                test_sample = pkl.load(gzip.open(sample, "rb"))
            else:
                test_sample = sample

            test_features = test_sample.features.to(DEVICE)
            test_label = test_sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in test_sample.graph_data]
            time_steps = test_sample.window

            output = model(all_graphs, test_features, time_steps)
            loss_test = F.nll_loss(output, test_label)
            accuracy_test.append(accuracy(output, test_label).detach().cpu().numpy())
            losses_test.append(loss_test.detach().cpu().numpy())

    metrics = {
        "train_acc": np.mean(acc_train),
        "val_acc": np.mean(accuracy_val),
        "test_acc": np.mean(accuracy_test),
        "train_loss": float(np.mean(losses_train)),
        "val_loss": float(np.mean(losses_val)),
        "test_loss": float(np.mean(losses_test)),
    }

    print(metrics)
    return metrics


def find_best_model(model):
    """
    Train a model with early stopping and save the best checkpoint.
    """
    TIMESTAMP = str(datetime.datetime.now()).replace(" ", "_").replace(".", ":")
    name = model.__class__.__name__
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    accs = {}
    current_epoch = 0
    top_val_acc = 0.0
    no_improvement_epochs = 0
    best_res = None

    for i in range(max_epochs):
        current_epoch = i
        metrics = train_model(model, optimizer=optimizer)
        accs[i] = metrics

        if metrics["val_acc"] >= top_val_acc:
            top_val_acc = metrics["val_acc"]
            no_improvement_epochs = 0
            best_res = metrics
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered")
            print("Best val results:")
            print(best_res)
            break

    # Name: GatV2Classificationlayers_4_best_model.tar  etc
    ckpt_name = name + f"layers_{model.num_layers}_best_model.tar"
    save_checkpoint(
        {
            "epoch": current_epoch,
            "state_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "metrics": best_res,
        },
        checkpoint=checkpoint_dir,
        name=ckpt_name,
    )

    return model, name


def eval_best_model(model):
    """
    Evaluate a given model on test_samples and return print_metrics output.
    """
    accuracy_test = []
    losses_test = []
    test_metrics = None

    model.eval()
    with torch.no_grad():
        for sample in test_samples:
            if args.lazy_loading:
                test_sample = pkl.load(gzip.open(sample, "rb"))
            else:
                test_sample = sample

            test_features = test_sample.features.to(DEVICE)
            test_label = test_sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in test_sample.graph_data]
            time_steps = test_sample.window

            output = model(all_graphs, test_features, time_steps)
            loss_test = F.nll_loss(output, test_label)
            accuracy_test.append(accuracy(output, test_label).detach().cpu().numpy())
            losses_test.append(loss_test.detach().cpu().numpy())

            gold = output.max(1)[1].type_as(test_label).detach().cpu().numpy()
            test_metrics = print_metrics(gold, test_label.cpu().numpy())

    return test_metrics


# ============================================================
# MAIN SCRIPT LOGIC
# ============================================================

def main():
    os.makedirs(os.path.join(args.result_dir, "plots"), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -------------------------
    # 1) LOAD & EVAL CHECKPOINTS ONLY
    # -------------------------
    if load_best_model:
        gat_models = []
        graphsage_models = []
        gat_f1_score = []
        graphsage_f1_score = []

        # We only consider layers 2..7 based on your prior work
        for i in range(2, 8):
            # GraphSAGE
            graphsage_model_dir = os.path.join(
                checkpoint_dir,
                f"GraphSageClassificationlayers_{i}_best_model.tar",
            )
            graphsage_checkpoint = torch.load(
                graphsage_model_dir, map_location=DEVICE, weights_only=False
            )
            graphsage_model = GraphSageClassification(
                in_channels=in_channel,
                hidden_channels=hidden_channel,
                out_channels=out_channel,
                nclass=2,
                num_layers=i,
            )
            graphsage_model.load_state_dict(graphsage_checkpoint["state_dict"])
            graphsage_model.to(DEVICE)
            graphsage_models.append(graphsage_model)

            # GATv2
            gatmodel_dir = os.path.join(
                checkpoint_dir,
                f"GatV2Classificationlayers_{i}_best_model.tar",
            )
            gatmodel_checkpoint = torch.load(
                gatmodel_dir, map_location=DEVICE, weights_only=False
            )
            gat_model = GatV2Classification(
                in_channels=in_channel,
                hidden_channels=hidden_channel,
                out_channels=out_channel,
                nclass=2,
                num_layers=i,
            )
            gat_model.load_state_dict(gatmodel_checkpoint["state_dict"])
            gat_model.to(DEVICE)
            gat_models.append(gat_model)

        # Evaluate and collect F1 (index 3 from print_metrics)
        for i in range(len(gat_models)):
            metrics = eval_best_model(model=graphsage_models[i])
            graphsage_f1_score.append(metrics[3])

            metrics = eval_best_model(model=gat_models[i])
            gat_f1_score.append(metrics[3])

        # Plot F1 vs layers
        layers = list(range(2, 8))
        plt.figure()
        plt.plot(layers, graphsage_f1_score, marker="o", label="GraphSAGE F1 Score")
        plt.plot(layers, gat_f1_score, marker="o", label="GATv2 F1 Score")
        plt.title("Number of Model Layers vs. F1 Score")
        plt.xlabel("Number of Layers")
        plt.ylabel("F1 Score (weighted)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(args.result_dir, "plots", "f1_score.png")
        plt.savefig(out_path)
        print(f"Saved F1 vs layers plot → {out_path}")
        return

    # -------------------------
    # 2) TRAIN NEW MODELS
    # -------------------------
    gat_models = []
    graphsage_models = []
    gat_f1_score = []
    graphsage_f1_score = []

    # Build models for layers 2..7
    for i in range(2, 8):
        gat_models.append(
            GatV2Classification(
                in_channels=in_channel,
                hidden_channels=hidden_channel,
                out_channels=out_channel,
                nclass=2,
                num_layers=i,
                dropout=args.dropout,
            ).to(DEVICE)
        )
        graphsage_models.append(
            GraphSageClassification(
                in_channels=in_channel,
                hidden_channels=hidden_channel,
                out_channels=out_channel,
                nclass=2,
                num_layers=i,
                dropout=args.dropout,
            ).to(DEVICE)
        )

    # Train + save checkpoints
    for model in gat_models + graphsage_models:
        find_best_model(model)

    # Evaluate best models and plot
    for i in range(len(gat_models)):
        metrics = eval_best_model(model=graphsage_models[i])
        graphsage_f1_score.append(metrics[3])

        metrics = eval_best_model(model=gat_models[i])
        gat_f1_score.append(metrics[3])

    layers = list(range(2, 8))
    plt.figure()
    plt.plot(layers, graphsage_f1_score, marker="o", label="GraphSAGE F1 Score")
    plt.plot(layers, gat_f1_score, marker="o", label="GATv2 F1 Score")
    plt.title("Number of Model Layers vs. F1 Score")
    plt.xlabel("Number of Layers")
    plt.ylabel("F1 Score (weighted)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(args.result_dir, "plots", "f1_score.png")
    plt.savefig(out_path)
    print(f"Saved F1 vs layers plot → {out_path}")


if __name__ == "__main__":
    main()
