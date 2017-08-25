import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py

from datetime import datetime
import sys
import os
from multiprocessing import Pool
from functools import partial

seq_len = int(sys.argv[1])

idx_time = 0
idx_close = 1

def load_raw_data():
    gold_raw_data = np.loadtxt("data/XAUUSD.csv", delimiter=" ")
    gold_raw_data = gold_raw_data[:, [0, -2]]
    gold_raw_data = remove_duplicate(gold_raw_data)

    return gold_raw_data

def remove_duplicate(data):
    
    _, idx = np.unique(data[:, 0], return_index=True)
    data = data[idx]
   
    return data

def make_data_and_label(gold_raw_data):
    data = []
    label = []

    for i in range(seq_len, gold_raw_data.shape[0]):
        start = i - seq_len
        end = i

        if(gold_raw_data[end, idx_time] - gold_raw_data[start, idx_time] != seq_len * 60.0):#check data continuity
            continue

        if(gold_raw_data[end, idx_time] - gold_raw_data[end-1, idx_time] != 60.0):#check label continuity
            continue

        data.append(gold_raw_data[start:end, idx_close])
        label.append(gold_raw_data[end, idx_close])

    data = np.array(data)
    data = np.expand_dims(data, 2)
    label = np.array(label)

    return data, label

def unison_shuffled_copies(data, label):
    assert (data.shape[0] == label.shape[0]) 

    p = np.random.permutation(data.shape[0])

    return data[p], label[p]

def normalization(train_data, train_label, test_data, test_label):
    max_val = np.amax(train_data)
    max_val /= 10.0

    train_data /= max_val
    train_label /= max_val
    test_data /= max_val
    test_label /= max_val

    return train_data, train_label, test_data, test_label

def write_HDF5(data, label, name):
    path = "data/%s.hdf5" % name
    dataset = h5py.File(path)
    dataset.create_dataset("data", data=data)
    dataset.create_dataset("label", data=label)
    dataset.close()

def main():
    gold_raw_data = load_raw_data()

    #make train and test dataset
    data, label = make_data_and_label(gold_raw_data)
    data, label = unison_shuffled_copies(data, label)

    offset = int(0.9 * gold_raw_data.shape[0])
    train_data, test_data = data[:offset], data[offset:]
    train_label, test_label = label[:offset], label[offset:] 
    
    #normalize
    train_data, train_label, test_data, test_label = normalization(train_data, train_label, test_data, test_label)
    
    #write to hdf5 file
    write_HDF5(train_data, train_label, "trainset")
    write_HDF5(test_data, test_label, "testset")

main()   
    
