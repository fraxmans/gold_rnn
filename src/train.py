import time

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import h5py

import model

prefix = "gold_forex/5min"
with h5py.File("data/trainset.hdf5") as f:
    train_data = f["data"][()].astype(np.float32)
    train_label = f["label"][()]
train_label = np.reshape(train_label, [-1, 1])

with h5py.File("data/testset.hdf5") as f:
    test_data = f["data"][()].astype(np.float32)
    test_label = f["label"][()]
test_label = np.reshape(test_label, [-1, 1])

features = train_data.shape[2] 
seq_len = train_data.shape[1]
logdir = "log/raw_shuffled_2lstm/%dmin" % seq_len
batch_size = 256 
batch_num = train_data.shape[0] // batch_size + 1
epoch_num = 3000
steps = batch_num * epoch_num + 1

display_step = 1e3 

def unison_shuffled_copies(data, labels):
    assert data.shape[0] == labels.shape[0]
    p = np.random.permutation(labels.shape[0])

    return data[p], labels[p]

def fill_feed_dict(step, dict_type):
    if(dict_type == "train"):
        start = (step * batch_size) % (train_data.shape[0] - batch_size)
        feed_dict = {
            batch_data: train_data[start:(start+batch_size), :, :],
            batch_label: train_label[start:(start+batch_size), :],
            isTraining: True
        }
    else:
        start = (step * batch_size) % (test_data.shape[0] - batch_size)
        feed_dict = {
            batch_data: test_data[start:(start+batch_size), :, :],
            batch_label: test_label[start:(start+batch_size), :],
            isTraining: False
        }

    return feed_dict

def save_model(step, saver, sess):
    path = "/data/gold/price_sma_ema/%depoch.ckpt" % (step/batch_num)
    save_path = saver.save(sess, path)
    print("[%s] Model saved in file: %s" % (time.asctime(), save_path))

def testing(sess, batch_data, batch_label, loss_op):
    loss_val = 0
    
    for step in range(test_data.shape[0]//batch_size):
        feed_dict = fill_feed_dict(step, "test")
        loss_val += sess.run(loss_op, feed_dict=feed_dict)

    loss_val /= (test_data.shape[0]//batch_size)

    return loss_val

with tf.Graph().as_default():
    #placeholder
    batch_data = tf.placeholder(tf.float32, shape=(batch_size, seq_len, features))
    batch_label = tf.placeholder(tf.float32, shape=(batch_size, 1))
    isTraining = tf.placeholder(dtype=tf.bool)
    test_loss = tf.Variable(0, dtype=tf.float32)

    #operation
    logits_op = model.forward(batch_data, seq_len, isTraining)
    loss_op = model.loss(logits_op, batch_label)
    train_op = model.training(loss_op)

    #summary
    train_loss_scalar = tf.summary.scalar("train_loss", loss_op)
    train_summary_op = tf.summary.merge([train_loss_scalar])

    test_loss_scalar = tf.summary.scalar("test_loss", test_loss)
    test_summary_op = tf.summary.merge([test_loss_scalar])

    sess = tf.Session()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=0)

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    sess.run(init)

    start_time = time.time()
    for step in range(steps):
        feed_dict = fill_feed_dict(step, "train")
        _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)

        if(step % display_step == 0):
            print("[%s] [TRAIN] At step %d\tloss: %f" % (time.asctime(), step, loss))
            summary = sess.run(train_summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
        
        if(step % batch_num == 0):
            train_data, train_label = unison_shuffled_copies(train_data, train_label)
            loss = testing(sess, batch_data, batch_label, loss_op)
            _, summary = sess.run([test_loss.assign(loss), test_summary_op])
            summary_writer.add_summary(summary, step)
        
            print("[%s] [TEST] At epoch %d\tloss: %f" % (time.asctime(), step/batch_num, loss))
            #save_model(step, saver, sess)
           
    print("--- %s --- seconds" % (time.time() - start_time))
