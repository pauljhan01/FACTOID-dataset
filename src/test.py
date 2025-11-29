import os
import json
import pickle as pkl
import gzip
from os.path import join

sample_dir = "../data/bert_embeddings"
train_path = os.path.join(sample_dir, "train_samples/")

def generate_set_of_data(path, n):
    data = []
    for i in range(n):
        sample_path = join(path, "sample_" + str(i) + '.data')
        data.append(pkl.load(gzip.open(sample_path, 'rb')))
    return data

def get_data():
    descriptor = json.load(open(os.path.join(sample_dir, 'dataset_descriptor.json'), 'r'))
    train_path = os.path.join(sample_dir, 'train_samples/')
    test_path = os.path.join(sample_dir, 'test_samples/')
    val_path = os.path.join(sample_dir, 'val_samples/')
    train_set = generate_set_of_data(train_path, descriptor['n_train_samples'])
    test_set = generate_set_of_data(test_path, descriptor['n_test_samples'])
    val_set = generate_set_of_data(val_path, descriptor['n_val_samples'])

    return train_set, test_set, val_set

get_data()