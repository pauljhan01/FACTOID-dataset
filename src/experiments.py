import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import os

import time
import gzip
import numpy as np
import torch.optim as optim
from dataset.reddit_user_dataset import RedditUserDataset

from argparse import ArgumentParser
import datetime
import time
from os.path import  join
import json
from tqdm import tqdm
from utils.metrics import *
from utils.utils import *
from utils.loss_fct import *
from utils.train_utils import save_checkpoint
from model import GatClassification, GatV2Classification, GraphSageClassification
from constants import *

import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--max_epochs", dest="max_epochs", default=300, type=int)
parser.add_argument("--sample_dir", dest="sample_dir", type=str, required=True)
parser.add_argument("--checkpoint_dir", dest="checkpoint_dir", type=str, required=True)
parser.add_argument("--learning_rate", dest="learning_rate", default=5e-6, type=float)
parser.add_argument("--weight_decay", dest="weight_decay", default=1e-3, type=float)
parser.add_argument("--patience", dest="patience", default=10, type=int)
parser.add_argument("--run_id", dest="run_id", default='no_id_given')
parser.add_argument("--result_dir", dest="result_dir", default="data/results", type=str)
parser.add_argument("--nheads", dest="nheads", default=8, type=int)
parser.add_argument("--dropout", dest="dropout", default=0.2, type=float)
parser.add_argument("--model_seed" , dest="model_seed", type=int, default=1234)
parser.add_argument("--load_best_model", dest="load_best_model", type=bool, default=False)
parser.add_argument("--lazy_loading", dest="lazy_loading", type=bool, default=False)

args = parser.parse_args()
max_epochs = args.max_epochs
dest = args.sample_dir
checkpoint_dir = args.checkpoint_dir
lr = args.learning_rate
weight_decay = args.weight_decay
early_stopping_patience = args.patience
load_best_model = args.load_best_model
in_channel = 768
hidden_channel = 256
out_channel = 128

def loss_fn(output, targets, samples_per_cls, no_of_classes=2):
    beta = 0.9999
    gamma = 2.0
    loss_type = "softmax"

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)

def get_samples_per_class(labels):
    return torch.bincount(labels).tolist()

def generate_set_of_data(path, n):
    data = []
    for i in range(n):
        sample_path = join(path, 'sample_' + str(i) + '.data')
        data.append(pkl.load(gzip.open(sample_path, 'rb')))
        # data.append(join(path, 'sample_' + str(i) + '.data'))
    return data

def get_data():
    descriptor = json.load(open(os.path.join(args.sample_dir, 'dataset_descriptor.json'), 'r'))
    train_path = os.path.join(args.sample_dir, 'train_samples/')
    test_path = os.path.join(args.sample_dir, 'test_samples/')
    val_path = os.path.join(args.sample_dir, 'val_samples/')
    train_set = generate_set_of_data(train_path, descriptor['n_train_samples'])
    test_set = generate_set_of_data(test_path, descriptor['n_test_samples'])
    val_set = generate_set_of_data(val_path, descriptor['n_val_samples'])

    return train_set, test_set, val_set

train_samples, test_samples, val_samples = get_data()

def train_model(model, optimizer):
    t = time.time()
    model.train()
    #optimizer.zero_grad() #TODO: This is not inside the samples loop. Are gradients supposed to be accumulated? I think it is a bug.

    acc_train = []
    accuracy_val = []
    accuracy_test = []
    losses_train = []
    losses_val = []
    losses_test = []
    
    for sample in train_samples:  #
        print(sample)
        train_sample = None
        if args.lazy_loading:
            train_sample = pkl.load(gzip.open(sample, 'rb'))
        else:
            train_sample = sample
        start = time.time()
        optimizer.zero_grad()
        train_features = train_sample.features.to(DEVICE)
        train_label = train_sample.labels.to(DEVICE)
        all_graphs = [graph.to(DEVICE) for graph in train_sample.graph_data]
        time_steps = train_sample.window
        # adj = train_sample.adj
        end = time.time()
        compute = end - start
        start = time.time()
        # output = model(all_graphs, train_features, time_steps, adj)
        # output = model(all_graphs, train_features, time_steps)
        output = model(all_graphs, train_features, time_steps)
        #loss_train = F.nll_loss(output, train_label)
        loss_train = loss_fn(output, train_label,  get_samples_per_class(train_label))
        acc_train.append(accuracy(output, train_label).detach().cpu().numpy())
        loss_train.backward()
        optimizer.step()
        end = time.time()
        losses_train.append(np.mean(loss_train.detach().cpu().numpy()))

    model.eval()
    with torch.no_grad():
        for sample in val_samples:
            val_sample = None
            if args.lazy_loading:
                val_sample = pkl.load(gzip.open(sample, 'rb'))
            else:
                val_sample = sample
            val_features = val_sample.features.to(DEVICE)
            val_label = val_sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in val_sample.graph_data]
            time_steps = val_sample.window
            # adj = val_sample.adj

            output = model(all_graphs, val_features, time_steps)
            #loss_val = F.nll_loss(output, val_label)
            loss_val = loss_fn(output, val_label,  get_samples_per_class(val_label))
            accuracy_val.append(accuracy(output, val_label).detach().cpu().numpy())
            losses_val.append(np.mean(loss_val.detach().cpu().numpy()))

    
        for sample in test_samples:
            test_sample = None
            if args.lazy_loading:
                test_sample = pkl.load(gzip.open(sample, 'rb'))
            else:
                test_sample = sample
            test_features = test_sample.features.to(DEVICE)
            test_label = test_sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in test_sample.graph_data]
            time_steps = test_sample.window
            # adj = test_sample.adj

            output = model(all_graphs, test_features, time_steps)
            #loss_test = F.nll_loss(output, test_label)
            loss_test = loss_fn(output, test_label,  get_samples_per_class(test_label))
            accuracy_test.append(accuracy(output, test_label).detach().cpu().numpy())
            losses_test.append(np.mean(loss_test.detach().cpu().numpy()))

    metrics = {'train_acc': np.mean(np.array(acc_train)),
               'val_acc': np.mean(np.array(accuracy_val)),
               'test_acc': np.mean(np.array(accuracy_test)),
               'train_loss': float(np.mean(np.array(losses_train))),
               'val_loss': float(np.mean(np.array(losses_val))),
               'test_loss': float(np.mean(np.array(losses_test)))}

    print(metrics)
    return metrics

def find_best_model(model):
    TIMESTAMP = str(datetime.datetime.now()).replace(" ", "_").replace(".", ":")
    name = model.__class__.__name__
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    accs = {}
    current_epoch = 0
    top_val_acc = 0
    no_improvement_epochs = 0
    for i in range(max_epochs):
        current_epoch = i
        metrics = train_model(model, optimizer=optimizer)
        accs[i] = metrics
        if metrics['val_acc'] >= top_val_acc:
            top_val_acc = metrics['val_acc']
            no_improvement_epochs = 0
            best_res = metrics
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= early_stopping_patience:
            print('Early stopping triggered')
            print('Best val results:')
            print(best_res)
            break

    save_checkpoint({
    'epoch': current_epoch,
    'state_dict': model.state_dict(),
    'optim_dict': optimizer.state_dict(),
    'metrics': metrics}, 
    checkpoint=checkpoint_dir, name=name + f"layers_{i}_best_model.tar" 
    )

    return model, name

def eval_best_model(model):
    accuracy_test = []
    losses_test = []
    with torch.no_grad():
        for sample in test_samples:
            test_sample = None
            if args.lazy_loading:
                test_sample = pkl.load(gzip.open(sample, 'rb'))
            else:
                test_sample = sample
            test_features = test_sample.features.to(DEVICE)
            test_label = test_sample.labels.to(DEVICE)
            all_graphs = [graph.to(DEVICE) for graph in test_sample.graph_data]
            time_steps = test_sample.window
            # adj = test_sample.adj

            output = model(all_graphs, test_features, time_steps)
            loss_test = F.nll_loss(output, test_label)
            accuracy_test.append(accuracy(output, test_label).detach().cpu().numpy())
            losses_test.append(np.mean(loss_test.detach().cpu().numpy()))
            gold = output.max(1)[1].type_as(test_label).detach().cpu().numpy()
            test_metrics = print_metrics(gold, test_label.cpu().numpy())

    return test_metrics

if load_best_model:
    gat_models = []
    graphsage_models = []
    gat_f1_score = []
    graphsage_f1_score = []

    for i in range(8):
        graphsage_model_dir = os.path.join(checkpoint_dir, f"GraphSageClassification_{i}_best_model.tar")
        graphsage_checkpoint = torch.load(graphsage_model_dir, map_location=DEVICE, weights_only=False)
        graphsage_model = GraphSageClassification(in_channels=in_channel, hidden_channels=hidden_channel, out_channels=out_channel, num_layers=i)
        graphsage_model.load_state_dict(graphsage_checkpoint['state_dict'])
        graphsage_model.to(DEVICE)
        graphsage_models.append(graphsage_model)

        gatmodel_dir = os.path.join(checkpoint_dir, f"GatV2Classification_{i}_best_model.tar")
        gatmodel_checkpoint = torch.load(gatmodel_dir, map_location=DEVICE, weights_only=False)
        gat_model = GatV2Classification(in_channels=in_channel, hidden_channels=hidden_channel, out_channels=out_channel, num_layers=i)
        gat_model.load_state_dict(gatmodel_checkpoint['state_dict'])
        gat_model.to(DEVICE)
        gat_models.append(gat_model)
    
    for i in range(len(gat_models)):
        metrics = eval_best_model(model=graphsage_models[i])
        # 3 = F1 weighted score
        graphsage_f1_score.append(metrics[3])

        metrics = eval_best_model(model=gat_models[i])
        gat_f1_score.append(metrics[3])

    plt.plot(graphsage_f1_score, label="GraphSAGE F1 Score per layer")
    plt.plot(gat_f1_score, label="GAT F1 Score per layer")
    plt.title("Number of Model Layers vs. F1 Score")
    plt.xlabel("Number of Layers")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig("~/work/f1_score.png")

else:
    gat_models = []
    graphsage_models = []
    gat_f1_score = []
    graphsage_f1_score = []

    for i in range(8):    
        gat_models.append(GatV2Classification(in_channels=in_channel, hidden_channels=hidden_channel, out_channels=out_channel, num_layers=i).to(DEVICE))
        graphsage_models.append(GraphSageClassification(in_channels=in_channel, hidden_channels=hidden_channel, out_channels=out_channel, num_layers=i).to(DEVICE))

    for model in gat_models:
        find_best_model(model)
        metrics = eval_best_model(model)
        gat_f1_score.append(metrics[3])

    for model in graphsage_models:
        find_best_model(model)
        metrics = eval_best_model(model)
        graphsage_f1_score.append(metrics[3])

    plt.plot(graphsage_f1_score, label="GraphSAGE F1 Score per layer")
    plt.plot(gat_f1_score, label="GAT F1 Score per layer")
    plt.title("Number of Model Layers vs. F1 Score")
    plt.xlabel("Number of Layers")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig("~/work/f1_score.png")










    
