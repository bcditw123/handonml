# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 23:27:04 2018

@author: hvlpr
"""

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

######################## Linear regression ##########################

# Step-by-step 
# 1. Fetching data
# 2. Split to data and target
# 3. Assign to tensorflow node
# 4. Calculate using graph

housing = fetch_california_housing()
m, n = housing.data.shape

# Data and target
X_np = housing.data
y_np = housing.target.reshape(-1, 1)
# Data plus bias
X_np = np.c_[np.ones((m, 1)), X_np]
# Assign to graph
X = tf.constant(X_np, dtype=tf.float32, name='X')
y = tf.constant(y_np, dtype=tf.float32, name='y')
X_transpose = tf.transpose(X)


############################# Normal equation ##############################

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(X_transpose, X)), X_transpose), y)

with tf.Session() as sess:
    result = theta.eval()
    
    
######################## Gradient descent ##########################
    
theta_np = np.random.randn(n + 1, 1)
learning_rate = 0.01
theta = tf.Variable(theta_np, dtype=tf.float32, name='theta')

# Thera = theta - learning_rate*gradienty
y_pred = tf.matmul(X, theta)
error = y_pred - y
gradient = 2/m*tf.matmul(X_transpose, error)

init = tf.global_variables_initializer()
n_iterates = 10
training_ops = tf.assign(theta, theta - learning_rate*gradient)

with tf.Session() as sess:
    init.run()
    for i in range(0, n_iterates):
        sess.run(training_ops)
        print(gradient.eval())
    # print(theta.eval())
    
    
#################### Gradient descent with auto diff ########################
learning_rate = 0.01
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1, 1), name='theta')
y_pred = tf.matmul(X, theta)
error = y_pred - y
mse = tf.reduce_mean(tf.square(error))
gradient = tf.gradients(mse, [theta])[0]

training_ops = tf.assign(theta, theta - learning_rate*gradient)

n_iterates = 5

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    for i in range(0, n_iterates):
        sess.run(training_ops)
    print(theta.eval())        
    
