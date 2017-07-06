import numpy as np
import math
import h5py

train_data = []
train_label = []
test_data = []
test_label = []
raw_data = np.loadtxt("data/GOLD1.csv", dtype=np.str, delimiter=",")
data = raw_data[:, 2:-1].astype(np.float64)

offset = int(math.ceil(data.shape[0] * 0.9))
seq_len = 3 

#normalization
max_val = np.amax(data) / 10.0
data /= max_val
label = data[:, -1]

def compute_SMA():
    SMAs = np.ndarray(data.shape[0])
    for i in range(seq_len):
        end = i + 1
        SMAs[i] = np.mean(data[:end, -1])
    for i in range(seq_len, data.shape[0]):
        start = i-seq_len
        SMAs[i] = np.mean(data[start:i, -1])
    SMAs = np.reshape(SMAs, [-1, 1])

    return SMAs

def compute_EMA():
    EMA_w = 2 / (seq_len + 1)
    EMAs = np.ndarray(data.shape[0])

    for i in range(data.shape[0]):
        if(i <= 0):
            EMAs[i] = data[0, -1]
            continue

        EMAs[i] = data[i, -1] * EMA_w + EMAs[i -1] * (1 - EMA_w)
    EMAs = np.reshape(EMAs, [-1, 1])

    return EMAs

def merge_features(features, start, end):
    merged = None

    for feature in features:
        if(merged is None):
            merged = feature[start:end]
        else:
            merged = np.hstack([merged, feature[start:end]])
    
    return merged

def create_training_data(features):
    global train_data, train_label

    for i in range(seq_len, offset):
        start = i-seq_len
        row_data = merge_features(features, start, i) 
        train_data.append(row_data)
        train_label.append(label[i])

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    print(train_data.shape, train_label.shape)

def create_testing_data(features):
    global test_data, test_label

    for i in range(offset+seq_len, data.shape[0]):
        start = i-seq_len
        row_data = merge_features(features, start, i) 
        test_data.append(row_data)
        test_label.append(label[i])

    test_data = np.array(test_data)
    test_label = np.array(test_label)
    print(test_data.shape, test_label.shape)

def main():
    SMAs = compute_SMA()
    EMAs = compute_EMA()

    features = [data, SMAs, EMAs]

    create_training_data(features)
    create_testing_data(features)

    trainset = h5py.File("data/trainset.hdf5")
    train_xs = trainset.create_dataset("data", data=train_data)
    train_ys = trainset.create_dataset("label", data=train_label)

    testset = h5py.File("data/testset.hdf5")
    test_xs = testset.create_dataset("data", data=test_data)
    test_ys = testset.create_dataset("label", data=test_label)

    trainset.close()
    testset.close()

main()
