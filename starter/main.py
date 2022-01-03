import os
import numpy as np
import pandas as pd
import sys
from inspect import getsourcefile
from fastapi import FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ML_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "starter/starter/"))
sys.path.append(os.path.dirname(ML_DIR))
from train_model import load_model, load_encoder, get_cat_features, process_data, inference

# Declare the data object with its components and their type.
class SingleQuery(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 39, 
                "workclass": "State-gov",
                "fnlgt": 77516, 
                "education": "Bachelors", 
                "education_num": 13, 
                "marital_status": "Never-married", 
                "occupation": "Adm-clerical", 
                "relationship": "Not-in-family",
                "race":  "White",
                "sex":  "Male", 
                "capital_gain": 2174, 
                "capital_loss": 0, 
                "hours_per_week": 40, 
                "native_country": "United-States"
            }
        }

app = FastAPI()

# This allows sending of data (our TaggedItem) via POST to the API.
@app.get("/")
async def say_hello():
    return {"Hello! Welcome to the query app of an AI model of the udacity project 4!"}

@app.post("/infer")
async def infer(user_data: SingleQuery):
    full_path = os.path.abspath(getsourcefile(lambda:0))
    #path, filename = os.path.split(full_path)
    root_dir = os.path.abspath(os.path.join(full_path, os.pardir, os.pardir))
    #os.path.join(root_dir, 'starter/model')
    model = load_model(os.path.join(root_dir, 'starter/model/RF_model'))
    encoder = load_encoder(os.path.join(root_dir, 'starter/model/encoder'))
    lb = load_encoder(os.path.join(root_dir, 'starter/model/lb'))

    array = np.array([[
                     user_data.age,
                     user_data.workclass,
                     user_data.fnlgt,
                     user_data.education,
                     user_data.education_num,
                     user_data.marital_status,
                     user_data.occupation,
                     user_data.relationship,
                     user_data.race,
                     user_data.sex,
                     user_data.capital_gain,
                     user_data.capital_loss,
                     user_data.hours_per_week,
                     user_data.native_country
                     ]])

    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"
    ])

    X, _, _, _ = process_data(
        df_temp, categorical_features=get_cat_features(), label=None, training=False, encoder=encoder, lb=lb
    )

    pred = inference(model, X)

    pred = lb.inverse_transform(pred)[0]
    return {"prediction": pred}

# age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, salary
# 39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
# 50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
# 38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
# 53, Private, 234721, 11th, 7, Married-civ-spouse, Handlers-cleaners, Husband, Black, Male, 0, 0, 40, United-States, <=50K
# 28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 40, Cuba, <=50K
# 37, Private, 284582, Masters, 14, Married-civ-spouse, Exec-managerial, Wife, White, Female, 0, 0, 40, United-States, <=50K
# 49, Private, 160187, 9th, 5, Married-spouse-absent, Other-service, Not-in-family, Black, Female, 0, 0, 16, Jamaica, <=50K
# 52, Self-emp-not-inc, 209642, HS-grad, 9, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 45, United-States, >50K