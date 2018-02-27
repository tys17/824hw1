
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt



def cnn_model_fn(features, mode='train'):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == 'train'))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),

        "logits": logits
    }
    return predictions



def loadMNIST():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return train_data, train_labels, eval_data, eval_labels



if __name__ == '__main__':
    learning_rate = 0.001
    batch_size = 100
    display_step = 100
    maxIter = 20000
    shuffle_every_epoch = True

    train_data, train_labels, eval_data, eval_labels = loadMNIST()

    x = tf.placeholder(tf.float32, [batch_size, 784])
    y = tf.placeholder(tf.float32, [batch_size])
    #mode = tf.placeholder(tf.string, 'train')
    mode = 'train'



    prediction = cnn_model_fn(x, mode=mode)




    onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=10)
    loss = tf.identity(tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=prediction["logits"]), name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    train_num = train_data.shape[0]
    batch_num = int(np.ceil(train_num / batch_size))
    train_idx = np.arange(0, train_num)
    phases = ['train', 'test']
    dataAll = {}
    dataAll['train'] = [train_data, train_labels]
    dataAll['test'] = [eval_data, eval_labels]
    idx = {phase: np.arange(0, dataAll[phase][0].shape[0]) for phase in phases}

    iter = 0
    with tf.Session() as sess:
        sess.run(init)
        while True:
            for phase in phases:
                data = dataAll[phase][0].copy()
                dataNum = data.shape[0]
                label = dataAll[phase][1].copy()
                if phase == 'train':
                    if shuffle_every_epoch:
                        np.random.shuffle(idx[phase])
                        data = data[idx[phase], :]
                        label = label[idx[phase]]
                batch_start = 0
                for batch_i in range(batch_num):
                    batch_end = min([batch_start + batch_size, dataNum])
                    data_batch = data[batch_start:batch_end, :]
                    label_batch = label[batch_start:batch_end]
                    #prediction = cnn_model_fn(data_batch, mode=phase)
                    #_, l = sess.run([optimizer, loss], feed_dict={x: data_batch, y: label_batch, mode: phase})
                    if phase == 'train':
                        train_op.run(feed_dict={x: data_batch, y: label_batch})
                    l = sess.run(loss, feed_dict={x: data_batch, y: label_batch})
                    batch_start = batch_end
                    if phase == 'train':
                        if iter % display_step == 0:
                            print(l)

                        iter += 1
                    if iter >= maxIter:
                        break
                if iter >= maxIter:
                    break
            if iter >= maxIter:
                break