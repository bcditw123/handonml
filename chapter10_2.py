# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 12:45:20 2018

@author: hvlpr
"""

import tensorflow as tf
import numpy as np


n_inputs = 28**2
n_hidden1 = 300
n_hidden2 = 100

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.float32, shape(None), name='y')

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='weights') # Randomly initialize weight matrix 
        # for each pair (input, neurion)
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(X, W) + b
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z

with tf.name_scope('dnn'):
    hidden1 = neuron_layer(X, n_hidden1, name='hidden2', activation='relu')
    hidden2 = neuron_layer(hidden1, n_hidden2, name='hidden2', activation='relu')
    logits = neuron_layer(hidden2, n_outputs, 'output')



        
        