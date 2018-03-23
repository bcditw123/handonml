import os
import numpy as np
import pandas as pd
from six.moves import urllib
import tarfile
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


class MyLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelBinarizer()

    def fit(self, X, y=None):
        self.encoder.fit(data, X, y)
        return self

    def transform(self, X, y=None):
        self.encoder.transform(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        return self.encoder.fit_transform(X)


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return X.select_dtypes(include=self.dtype)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)


def fetching_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(housing_path)
    data_tgz.close()


def get_data_from_url(url=os.path.join(HOUSING_PATH, 'housing.csv')):
    return pd.read_csv(url)


def get_data():
    tgz_url = os.path.join(HOUSING_PATH, 'housing.csv')
    if not os.path.exists(tgz_url):
        fetching_data()
    return get_data_from_url()


def add_stratified_attr(data):
    data['income_cat'] = np.ceil(data['median_income'] / 1.5)
    data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)


def remove_stratified_attr(data):
    data.drop(['income_cat'], axis=1, inplace=True)


def stratified_split(data):
    add_stratified_attr(data)
    splitter = StratifiedShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    for train_indices, test_indices in splitter.split(data, data['income_cat']):
        train_set_s = data.loc[train_indices]
        test_set_s = data.loc[test_indices]
    remove_stratified_attr(data)
    remove_stratified_attr(train_set_s)
    remove_stratified_attr(test_set_s)
    return train_set_s, test_set_s


def data_labels_split(data):
    attr_list = data.columns.tolist()
    attr_list.remove('median_house_value')
    return data[attr_list], data['median_house_value']


def data_transform(data):
    numeric_dtype = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    nonnumeric_dtype = ['object']
    numeric_pipeline = Pipeline([('selector', DataFrameSelector(numeric_dtype)), ('imputer', Imputer(strategy='mean')),
                                 ('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('selector', DataFrameSelector(nonnumeric_dtype)), ('binarizer', MyLabelBinarizer())])
    transformer = FeatureUnion(transformer_list=[('num_pipeline', numeric_pipeline), ('cat_pipeline', cat_pipeline)])
    return transformer.fit_transform(data)


def model_evaluate(data, label):
    list_gress = [('Linear Regression', LinearRegression()),
                  ('Decision Tree Regression', DecisionTreeRegressor()),
                  ('Random Forest Regression', RandomForestRegressor()),
                  ('Suport Vector Machine', SVR())]

    for name, gressor in list_gress:
        scores = cross_val_score(gressor, data, label, scoring="neg_mean_squared_error", cv=10)
        print(name, end=': ')
        print(np.sqrt(-scores.mean()))


dataset = get_data()
train_set, test_set = stratified_split(dataset)
train_set_data, train_set_labels = data_labels_split(train_set)
train_set_data = data_transform(train_set_data)
model_evaluate(train_set_data, train_set_labels)

