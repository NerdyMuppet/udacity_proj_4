# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
import pickle
import os
# SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# ML_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "ml"))
# sys.path.append(os.path.dirname(ML_DIR))
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Add code to load in the data.
def get_cat_features():
    cat_feat = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_feat

def load_data(load_dir):
    #script is started from root directory -->
    try:
        cwd = os.getcwd()
        data = pd.read_csv(os.path.join(cwd,load_dir))
    except FileNotFoundError as err:
        print(f"{err}, using test data instead of full data" )
        data = pd.read_csv(
            os.path.join(os.path.normpath(os.path.join(cwd, os.pardir)),
            "starter/data/test_clean_census.csv")
            )
    return data

def save_clean_data(data, save_dir):
    try:
        data.to_csv(save_dir)
        print(f"cleaned data saved successfully to {save_dir}")
    except OSError as err:
        print(f"could not save the cleaned data: {err}")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
def split_data(data):
    train, test = train_test_split(data, test_size=0.20)
    return train, test

def proc_all_data(train, test, cat_features):

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    
    return X_train, y_train, X_test, y_test, encoder, lb

def save_model(model, save_pth):
    try:
        pickle.dump(model, open(save_pth, 'wb'))
        print(f"Model saved successfully under {save_pth}")
    except OSError as err:
        print(f"Model was not saved with the following error: {err}")

def load_model(pth):
    try:
        model = pickle.load(open(pth, 'rb'))
        print(f"Model was loaded succesfully from {pth}")
        return model
    except OSError as err:
        print(f"Model could not be loaded with the following error: {err}")
        return None

def save_encoder(encoder, save_pth):
    try:
        pickle.dump(encoder, open(save_pth, 'wb'))
        print(f"Encoder saved successfully under {save_pth}")
    except OSError as err:
        print(f"Encoder was not saved with the following error: {err}")

def load_encoder(pth):
    try:
        encoder = pickle.load(open(pth, 'rb'))
        print(f"Encoder was loaded succesfully from {pth}")
        return encoder
    except OSError as err:
        print(f"Encoder could not be loaded with the following error: {err}")
        return None

def create_slice_inference_on_column(dat, model, column, cat_features):
    vals = dat[column].unique().tolist()
    encoder = load_encoder(os.path.normcase("starter/model/encoder"))
    lb = load_encoder(os.path.normcase("starter/model/lb"))
    scores = np.zeros((len(vals), 3))
    for i, val in enumerate(vals):
        dat_slice = dat.loc[dat[column] == val]
        X_slice, y_slice, _, _ = process_data(
            dat_slice, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        preds_slice= inference(model, X_slice)
        scores[i] = compute_model_metrics(y_slice, preds_slice)
    
    return list(zip(vals, scores))

def write_slicing_resu(resu):
    file = open("slice_output.txt","w")
    for i in resu:
        file.writelines(f"Category {i[0]} has precision: {i[1][0]}, recall: {i[1][1]}, fbeta: {i[1][2]}\n")
    file.close()
        

def go():
    data = load_data(os.path.normcase("starter/data/census.csv"))
    data.columns = data.columns.str.replace(' ','')
    data.replace(' ', '', regex=True)
    save_clean_data(data, os.path.normcase("starter/data/clean_census.csv"))
    
    cat_features =  get_cat_features()

    train, test = split_data(data)
    X_train, y_train, X_test, y_test, encoder, lb = proc_all_data(train, test, cat_features)

    model = train_model(X_train, y_train)

    # save_model(model, os.path.normcase("starter/model/RF_model"))
    # save_encoder(encoder, os.path.normcase("starter/model/encoder"))
    # save_encoder(lb, os.path.normcase("starter/model/lb"))
    # model = load_model(os.path.normcase("starter/model/RF_model"))
    # encoder = load_encoder(os.path.normcase("starter/model/encoder"))
    # slb = load_encoder(os.path.normcase("starter/model/lb"))

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    resu = create_slice_inference_on_column(data, model, "education", cat_features)
    write_slicing_resu(resu)

go()