from dataLoad import load_pickle
from utils import convert_params
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from dotenv import load_dotenv
import joblib
import neptune

import os
import yaml

def main(file_code):
    load_dotenv()
    
    project_name = os.getenv("PROJECT_NAME")
    api_token = os.getenv("API_TOKEN")
    
    run = neptune.init_run(project=project_name, api_token=api_token)
    
    train_data = load_pickle(f"Data/train_{file_code}.pickle")
    test_data = load_pickle(f"Data/test_{file_code}.pickle")
    # vectorizer = load_pickle(f"Data/vectorizer_{file_code}.pickle")
    
    X_train = train_data["x"]
    y_train = train_data["y"]
    X_test = test_data["x"]
    y_test = test_data["y"]
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_mapping = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "SVC": SVC,
        "MultinomialNB": MultinomialNB,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
    }
    
    choosen_model = config["choose_model"]
    print(f"Using model: {choosen_model}")
    
    method = [x for x in config["method"] if x["name"] == choosen_model][0]
    
    model_name = method["name"]
    model_version = config["model_version"]
    model_config = convert_params(method["params"])
    
    model_namespace = f"models/{model_name}/{model_version}"
    
    modelClass = model_mapping[model_name]
    model = modelClass(**model_config)
    
    for param_name, param_value in model_config.items():
        run[f"{model_namespace}/parameters/{param_name}"] = param_value

    model.fit(X_train, y_train)
    
    model_file_name = f"{model_name}_{model_version}.joblib"
    joblib.dump(model, model_file_name)
    
    run[f"{model_namespace}/artifacts"].upload(model_file_name)
    
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    run[f"{model_namespace}/classification_report"] = report
    print(report)
    
if __name__ == '__main__':
    file_code = "20240825_204934"
    main(file_code)