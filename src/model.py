import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import h5py

learning_rate = 1e-2

def lstm_cell(hiddens):
    return rnn.BasicLSTMCell(hiddens) 

def forward(x, seq_len, isTraining):
    weights1 = tf.Variable(tf.random_normal([256, 16]))
    biases1 = tf.Variable(tf.random_normal([16]))
    weights2 = tf.Variable(tf.random_normal([16, 1]))
    biases2 = tf.Variable(tf.random_normal([1]))

    x = tf.unstack(x, seq_len, 1)

    layers = []
    n_hiddens = np.array([256, 256, 256, 256])
    for i in range(n_hiddens.shape[0]):
        cell = lstm_cell(n_hiddens[i])
        if(isTraining):
            cell = rnn.DropoutWrapper(cell, input_keep_prob=0.8,
                                            output_keep_prob=0.8)
        layers.append(cell)

    multi_lstm_cell = rnn.MultiRNNCell(layers)
    outputs, states = rnn.static_rnn(multi_lstm_cell, x, dtype=tf.float32)

    fc1 = tf.matmul(outputs[-1], weights1) + biases1
    fc2 = tf.matmul(fc1, weights2) + biases2

    return fc2 

def loss(logits, labels):
    return tf.reduce_mean(tf.squared_difference(logits, labels))

def training(loss):
    tf.summary.scalar("train_loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def accuracy(logits, labels):
    correct_pred = tf.equal(logits, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


