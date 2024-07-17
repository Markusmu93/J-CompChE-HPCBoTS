import os
import pickle

def load_histories(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        return {}

def save_histories(histories_dict, file_path):
    existing_histories = load_histories(file_path)
    for key, value in histories_dict.items():
        if key in existing_histories:
            existing_histories[key].extend(value)
        else:
            existing_histories[key] = value
    with open(file_path, 'wb') as file:
        pickle.dump(existing_histories, file)
