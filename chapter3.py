from sklearn.linear_model import SGDClassifier
from sklearn.datasets import mldata
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

mnist = mldata.fetch_mldata('MNIST Original')


def shuffle_split(data, labels, test_ratio=0.2):
    indices = np.random.permutation(data.shape[0])
    test_size = int(test_ratio*data.shape[0])
    return data[indices[test_size:]], labels[indices[test_size:]], \
           data[indices[: test_size]], labels[indices[: test_size]]


def print_image(data):
    data_image = data.reshape(28, 28)
    plt.imshow(data_image, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.show()


def evaluate_model(data, labels):
    classifier = SGDClassifier(random_state=42)
    labels_predict = cross_val_predict(classifier, data, labels == 5, cv=3)
    return confusion_matrix(labels, labels_predict)


def multinomial_evaluate_model(data, labels):
    classifier = SGDClassifier(random_state=42)
    labels_predict = cross_val_predict(classifier, data, labels, cv=3)
    return confusion_matrix(labels, labels_predict)


X_train, y_train, X_test, y_test = shuffle_split(mnist['data'], mnist['target'])

matrix = evaluate_model(X_train, y_train)
multinomial_matrix = multinomial_evaluate_model(X_train, y_train)

knn_classifier = KNeighborsClassifier()

y_train_multi = np.c_[y_train >= 7, y_train % 2 == 1]

knn_classifier.fit(X_train, y_train_multi)

knn_classifier.predict()





