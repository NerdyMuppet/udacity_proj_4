import pandas as pd
import numpy as np
import os
import sys
import pytest
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.append(os.path.dirname(PAR_DIR))
from starter.train_model import proc_all_data, split_data

@pytest.fixture
def data():
    try:
        data = pd.read_csv("starter/data/clean_census.csv")
        print("starter/data/clean_census.csv")
        return data
    except FileNotFoundError as err:
        print(f"{err}, using test data instead of full data" )
        data = pd.read_csv(
            #os.path.join(os.path.normpath(os.path.join(cwd, os.pardir)),
            "starter/data/test_clean_census.csv"#)
            )
        return data

@pytest.fixture
def np_data(data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    train, test = split_data(data)
    X_train, y_train, X_test, y_test, _, _ = proc_all_data(train, test, cat_features)
    return X_train, y_train, X_test, y_test

def test_null_values(data):
    assert data.shape == data.dropna().shape

def test_is_numpy(np_data):
    assert isinstance(np_data[0], np.ndarray)
    assert isinstance(np_data[1], np.ndarray)
    assert isinstance(np_data[2], np.ndarray)
    assert isinstance(np_data[3], np.ndarray)

def test_mean(np_data):
    assert np_data[0].mean() + 50 > np_data[2].mean()
    assert np_data[0].mean() - 50 < np_data[2].mean()