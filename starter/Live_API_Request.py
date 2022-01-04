import requests
import json

data = {
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
    }

response = requests.post(
    "https://udacity-project-four.herokuapp.com/infer",
    data=json.dumps(data))
print(response)
if response.status_code == 200:
    if response.json()['prediction'] == ' <=50K':
        print('predicted salary: <=50K')
    else:
        print('predicted salary: >50K')
else:
    print('prediction process failed!')