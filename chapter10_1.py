# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 10:42:34 2018

@author: hvlpr

"""

import matplotlib 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def get_mnist_data():
    mnist = fetch_mldata('MNIST Original')
    X = mnist['data']
    y = mnist['target']
    return random_split(X, y, ratio=0.2)


def random_split(X, y=None, ratio=0.2):
    train_size = int(ratio*X.shape[0])
    indices = np.random.permutation(int(X.shape[0]))
    if y is None:
        return X[indices[0: train_size]], X[indices[train_size:]]
    else:
        return X[indices[0: train_size]], X[indices[train_size:]],\
               y[indices[0: train_size]].astype(np.int32), y[indices[train_size:]].astype(np.int32)
               
               
def print_image(data):
    data_image = data.reshape(28, 28)
    plt.imshow(data_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.show()
   
   
def dnn_predict_to_array(y):
    y_pred = [d['class_ids'] for d in list(y)]
    return np.array([d[-1] for d in y_pred])
    

scaler = StandardScaler()
X_train, X_test, y_train, y_test = get_mnist_data()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
feature_columns = [tf.feature_column.numeric_column("X_train", shape=[1, 784])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[1000, 500], n_classes=10, feature_columns=feature_columns)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'X_train': X_train},
    y=y_train,
    num_epochs=1,
    shuffle=False
)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'X_train': X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False
)

dnn_clf.train(input_fn=test_input_fn, steps=10000000)
scores = dnn_clf.evaluate(input_fn=test_input_fn)["accuracy"]

y_pred = dnn_predict_to_array(dnn_clf.predict(input_fn=test_input_fn))

score = accuracy_score(y_test, y_pred)
