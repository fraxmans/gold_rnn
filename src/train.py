import time

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import h5py

import model

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
logdir = "log/normalized/%dmin" % seq_len
batch_size = 256 
batch_num = train_data.shape[0] // batch_size + 1
epoch_num = 30000
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
            batch_label: train_label[start:(start+batch_size), :]
        }
    else:
        start = (step * batch_size) % (test_data.shape[0] - batch_size)
        feed_dict = {
            test_batch_data: test_data[start:(start+batch_size), :, :],
            test_batch_label: test_label[start:(start+batch_size), :]
        }

    return feed_dict

def save_model(step, saver, sess):
    path = "/data/gold/price_sma_ema/%depoch.ckpt" % (step/batch_num)
    save_path = saver.save(sess, path)
    print("[%s] Model saved in file: %s" % (time.asctime(), save_path))

def testing(sess):
    test_loss = 0
    test_batch_data = tf.placeholder(tf.float32, shape=(batch_size, seq_len, features))
    test_batch_label = tf.placeholder(tf.float32, shape=(batch_size, 1))
    
    logits = model.forward(test_batch_data, seq_len, False)
    loss = model.loss(logits, test_batch_label)

    for step in range(test_data.shape[0]//batch_size):
        feed_dict = fill_feed_dict(step, "test")
        test_loss += sess.run(loss, feed_dict=feed_dict)

    tf.summary.scalar("test_loss", test_loss)    
    return test_loss

with tf.Graph().as_default():
    batch_data = tf.placeholder(tf.float32, shape=(batch_size, seq_len, features))
    batch_label = tf.placeholder(tf.float32, shape=(batch_size, 1))

    logits = model.forward(batch_data, seq_len, True)
    loss = model.loss(logits, batch_label)
    train_op = model.training(loss)
    accuracy = model.accuracy(logits, batch_label)

    tf.summary.scalar("loss", loss)
    summary_op = tf.summary.merge_all()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=0)

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    sess.run(init)

    start_time = time.time()
    for step in range(steps):
        feed_dict = fill_feed_dict(step, "train")
        _, train_loss = sess.run([train_op, loss], feed_dict=feed_dict)

        if(step % display_step == 0):
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print("[%s] [TRAIN] At step %d\tloss: %f\taccuracy: %.2f%%" % (time.asctime(), step, train_loss, train_accuracy))
            summary = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary, step/batch_num)
        
        if(step % batch_num == 0):
            train_data, train_label = unison_shuffled_copies(train_data, train_label)        
            #test_loss = testing(sess)
            #print("[%s] [TEST] At epoch %d\tloss: %f" % (time.asctime(), step/batch_num, test_loss))
            #save_model(step, saver, sess)
           
    print("--- %s --- seconds" % (time.time() - start_time))
