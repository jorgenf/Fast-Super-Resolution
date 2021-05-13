from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist

""" hyperparameters """
# mnist
num_classes = 10
num_features = 28*28

# training
learning_rate = 0.001
training_steps = 1000
batch_size = 256
display_step = 50

""" dataset """
# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# y_train = np.array(y_train, dtype=np.int32)

# preprocess (normalize and reshape)
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.

# use tf.data to efficiently feed the model shuffled data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

""" create model """
# create weight-vector of size [784, 10]
W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")

# create bias-vector of shape [10]
b = tf.Variable(tf.zeros([num_classes]), name="bias")

# calculate x*w + b and normalize to [0, 1]
def logistic_regression(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)

# cross-entropy loss function
def cross_entropy(y_pred, y_true):
    # create one-hot vector
    y_true = tf.one_hot(y_true, depth=num_classes)

    # clip values to avoid potential log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)

    #compute cross-entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# how to measure accuracy
def accuracy(y_pred, y_true):
    # compares prediction with label
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# using SDG optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# model training
def run_optimization(x, y):
    # tensorflow magic
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)
    
    # compute gradient
    gradients = g.gradient(loss, [W, b])

    # use gradients to update weights and bias
    optimizer.apply_gradients(zip(gradients, [W, b]))

""" run model """
# iterate through the tf.data object for duration of training
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # optimize
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print(f"Step: {step}\tLoss: {loss:.2f}\tAccuracy: {acc:.2f}")

""" test model """
pred = logistic_regression(x_test)
print(f"Test accuracy: {accuracy(pred, y_test):.2f}")