# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import os
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Add code to load in the data.
def load_data(load_dir="starter/data/census.csv"):
    #script is started from root directory -->
    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd,load_dir))
    data.columns = data.columns.str.replace(' ','')
    return data


# Optional enhancement, use K-fold cross validation instead of a train-test split.
def split_data(data):
    train, test = train_test_split(data, test_size=0.20)
    return train, test

def proc_all_data(train, test):
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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    
    return X_train, y_train, X_test, y_test

def save_model(model, name, pth="starter/model/"):
    try:
        save_pth = os.path.join(pth, name)
        pickle.dump(model, open(save_pth, 'wb'))
        print(f"Model saved successfully under {save_pth}")
    except OSError as err:
        print(f"Model was not saved with the following error: {err}")

    # # some time later...

    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # # Train and save a model.

def go():
    data = load_data()
    train, test = split_data(data)
    X_train, y_train, X_test, y_test = proc_all_data(train, test)
    model = train_model(X_train, y_train)
    save_model(model, "RF_model")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

go()
