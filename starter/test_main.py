from fastapi.testclient import TestClient
import requests
import json


from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()[0] == "Hello! Welcome to the query app of an AI model of the udacity project 4!"

def test_less_than():
    r = client.post("/infer", json= {
            "age": 28,
            "workclass": "Private", 
            "fnlgt": 338409,  
            "education": "Bachelors",  
            "education_num": 13,  
            "marital_status": "Married-civ-spouse",  
            "occupation": "Prof-specialty", 
            "relationship": "Wife",
            "race":  "Black",
            "sex":  "Female", 
            "capital_gain": 0, 
            "capital_loss": 0, 
            "hours_per_week": 40, 
            "native_country": "Cuba"
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": " <=50K"}

#Federal-gov,337895, Bachelors,13, Married-civ-spouse, Prof-specialty, Husband, Black, Male,0,0,40, United-States, >50K
# 41, Private,220531, Prof-school,15, Married-civ-spouse, Prof-specialty, Husband, White, Male,0,0,60, United-States
# 53          State-gov  229465      Doctorate             16   Married-civ-spouse   Prof-specialty         Husband   White     Male             0             0              50   United-States
#43, Self-emp-not-inc,292175, Masters,14, Divorced, Exec-managerial, Unmarried, White, Female,0,0,45, United-States, >50K
#32            Private  209538        Masters             14   Married-civ-spouse    Exec-managerial      Husband   White     Male             0             0              55   United-States
def test_more_than():
    r = client.post("/infer", json={
            "age": 50, 
            "workclass": "Private",
            "fnlgt": 209538, 
            "education": "Masters", 
            "education_num": 14, 
            "marital_status": "Married-civ-spouse", 
            "occupation": "Exec-managerial", 
            "relationship": "Husband",
            "race":  "White",
            "sex":  "Male", 
            "capital_gain": 3645743, 
            "capital_loss": 0, 
            "hours_per_week": 50, 
            "native_country": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": " >50K"}

#42, Private,116632, Doctorate,16, Married-civ-spouse, Prof-specialty, Husband, White, Male,0,0,45, United-States

# data = {
#             "age": 28,
#             "workclass": "Private", 
#             "fnlgt": 338409,  
#             "education": "Bachelors",  
#             "education_num": 13,  
#             "marital_status": "Married-civ-spouse",  
#             "occupation": "Prof-specialty", 
#             "relationship": "Wife",
#             "race":  "Black",
#             "sex":  "Female", 
#             "capital_gain": 0, 
#             "capital_loss": 0, 
#             "hours_per_week": 40, 
#             "native_country": "Cuba"
#     }

# r = requests.post("http://127.0.0.1:8000/infer", data=json.dumps(data))

# print(r.json())
# def test_get_path_query():
#     r = client.get("/items/42?count=5")
#     assert r.status_code == 200
#     assert r.json() == {"fetch": "Fetched 5 of 42"}


# def test_get_malformed():
#     r = client.get("/items")
#     assert r.status_code != 200