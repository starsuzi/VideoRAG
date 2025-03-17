import os
import json 
import pickle


def get_base_data(file_path: str = '', is_synthetic: bool = False):
    file_path = file_path if file_path else (
        f'./datasets/qa/{"synthetic" if is_synthetic else "wikihow"}.json'
    )
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_features(file_path: str):
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    return features


def save_features(features: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(features, f)


def load_results(file_path: str):
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results


def save_results(results: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
