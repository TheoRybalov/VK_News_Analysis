import pickle
import os
from data_preprocessing.feature_utils import get_path_in_project

def load_model(path="models/svc_model.pkl"):
    full_path = get_path_in_project(path)
    with open(full_path, "rb") as f:
        return pickle.load(f)
    
def save_model(model, path="models/svc_sentiment.pkl"):
    full_path = get_path_in_project(path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as f:
        pickle.dump(model, f)