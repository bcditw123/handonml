from sklearn.linear_model import SGDClassifier
from sklearn.datasets import mldata
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


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


X_train, y_train, X_test, y_test = shuffle_split(mnist['data'], mnist['target'])
knn_classifier = KNeighborsClassifier()

param_grid = [{'n_neighbors': list(range(3, 10)), 'weights': ['uniform', 'distance']}]

grid_search = GridSearchCV(knn_classifier, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

