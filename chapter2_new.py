import os
import numpy as np
import pandas as pd
from six.moves import urllib
import tarfile


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


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


def str


dataset = get_data()
