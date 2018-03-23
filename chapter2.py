# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:39:10 2018

@author: hvlpr
"""

import os
import numpy as np
import pandas as pd
import tarfile
import matplotlib.pyplot as plt
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from six.moves import urllib
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)


class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.fit_model = None
        
        
    def fit(self, X, y=None):
        dat_copy = X.copy()
        np.c_[dat_copy, dat_copy[:, population_ix] / dat_copy[:, household_ix]]
        if self.add_bedrooms_per_room:
                np.c_[dat_copy, dat_copy[:, bedrooms_ix]/ dat_copy[:, rooms_ix]]
        self.fit_model = dat_copy
    
    
    def transform(self, X, y=None):
        if self.fit_model is not None:
            return self.fit_model
        else:
            return None
    
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    

def load_housing_data(housing_path=HOUSING_PATH):
    return pd.read_csv(os.path.join(housing_path, 'housing.csv'))
    
    
def train_test_split_auto(df, test_ratio):
    random_indices = np.random.permutation(len(df))
    test_size = int(test_ratio*len(df))
    test_indices = random_indices[:test_size]
    train_indices = random_indices[test_size:]
    return df.iloc[test_indices], df.iloc[train_indices]
    
    
def is_in_test(dat, hashf, test_ratio):
    return hashf(np.int64(dat)).digest()[-1] < 256 * test_ratio
    
    
def train_test_split_by_id(data, id_index, test_ratio):
    ids = data[id_index]
    in_test_set = ids.apply(lambda id_: is_in_test(id_, hashlib.md5, test_ratio))
    return data[in_test_set], data[~in_test_set]         
    

def full_prepare_data(data):
    attributes_list = data.columns.tolist()
    num_attributes = attributes_list[:-1]
    cat_attributes = attributes_list[-1]
    num_pipeline = Pipeline([('selector', DataFrameSelector(num_attributes)),
                         ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombinedAttributeAdder()),
    ('std_scaler', StandardScaler())])

    cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attributes)),
                          ('encoder', CustomLabelBinarizer())])
                         
    full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline),
                                               ('cat_pipeline', cat_pipeline)])
    return full_pipeline.fit_transform(data)
    
def add_stritified_attr(data):
    data['income_cat'] = np.ceil(data['median_income'] / 1.5)
# print(housing['median_income'].value_counts())
# housing['income_cat'].hist(bins=50)
    data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)
    
    
def remove_stritified_attr(data):
    data.drop(['income_cat'], axis=1, inplace=True)
    
    
def stratified_split(data):
    add_stritified_attr(data)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data['income_cat']):
        test_set = data.loc[test_index]
        train_set = data.loc[train_index]
    remove_stritified_attr(test_set)
    remove_stritified_attr(train_set)
    return test_set, train_set


fetch_housing_data()
housing = load_housing_data()
# Split by id based on index of the row
#housing = housing.reset_index()
# Split based on a hash function with some value in a row
#housing['id'] = housing['longitude']*1000 + housing['latitude']

# Some test hash function
# train_set_2, test_set_2 = train_test_split_by_id(housing, 'index', 0.2)
# train_set_3, test_set_3 = train_test_split_by_id(housing, 'id', 0.2)
# train_set_4, test_set_4 = train_test_split(housing, test_size=0.2, random_state=42)



# income_cat.hist(bins=50)
# plt.show()

strat_test_set, strat_train_set = stratified_split(housing)

housing = strat_train_set.copy()

# housing.plot(kind='scatter', x='longitude', y='latitude')
# housing.plot(kind='scatter', x='longitude', y='latitudetware revenue will top
#, alpha=0.1)

# Plot data with x is longitude and y is latitude radius is the median
# population and color is median housing price

# housing.plot(kind='scatter', x='longitude', y='latitude',
  #            s=housing['population']/100, label='population',
   #          c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
            
# plt.legend()

# Calculate the standard correlation coefficient of the data with numpy.corr()
# corr_matrix = housing.corr()
# Plot correlation between data
attributes=['median_house_value', 'median_income', 'total_rooms',
            'housing_median_age']
# scatter_matrix(housing[attributes], figsize=(12,8))
# Data cleaning before apply machine learning algorithm

housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

housing_test = strat_test_set.drop('median_house_value', axis=1)
housing_test_labels = strat_test_set['median_house_value'].copy()

housing_test = full_prepare_data(housing_test)
# housing.dropna(subset=['total_bedrooms']) Drop all rows with null value in 
# total_bedrooms
# housing.drop('total_bedrooms', axis=1) Drop total_bedrooms column
# Fill all rows missing value with the median
# median = housing['total_bedrooms'].median()
# housing['total_bedrooms'].fillna(median)

# imputer = Imputer(strategy='median')
# housing_num = housing.drop('ocean_proximity', axis=1)
# imputer.fit(housing_num)

# X= imputer.transform(housing_num)
# housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# encoder = LabelEncoder()
# housing_cat = housing['ocean_proximity']
# housing_cat_encoded = encoder.fit_transform(housing_cat)

# one_hot_encoder = OneHotEncoder()
# label_binarizer = LabelBinarizer()
# housing_cat_encoded = housing_cat_encoded.reshape(-1, 1)
# housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat_encoded)

# housing_cat_1hot_2 = label_binarizer.fit_transform(housing_cat)

# attr_adder = CombinedAttributeAdder(add_bedrooms_per_room=True)
# housing_extra = attr_adder.fit_transform(strat_train_set.values)

# num_pipeline = Pipeline([('imputer', Imputer(strategy='median')),
# ('attribs_adder', CombinedAttributeAdder()),('scaler', StandardScaler())])

data_prepared = full_prepare_data(housing)

# Start linear regeression

lin_gress = LinearRegression()
lin_gress.fit(data_prepared, housing_labels)

housing_test_predict = lin_gress.predict(housing_test)

lin_mse = mean_squared_error(housing_test_labels, housing_test_predict)
lin_rmse = np.sqrt(lin_mse)


################### Regress with decision tree algorithm ########

tree_gress = DecisionTreeRegressor()
tree_gress.fit(data_prepared, housing_labels)
housing_test_tree_predict = tree_gress.predict(housing_test)

tree_rmse = np.sqrt(mean_squared_error(housing_test_tree_predict, housing_test_labels))

############### Using cross validation regressor #####################

lin_scores = cross_val_score(lin_gress, data_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

tree_scores = cross_val_score(tree_gress, data_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)


print(np.sqrt(-lin_scores.mean()))

print(np.sqrt(-tree_scores.mean()))

forest_gress = RandomForestRegressor()

forest_scores = cross_val_score(forest_gress, data_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

print(np.sqrt(-forest_scores.mean()))
