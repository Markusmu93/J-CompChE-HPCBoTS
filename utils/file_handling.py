import os
import pickle
import re

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

def extract_case_and_num(filename):
    # Regular expression to match the pattern 'case_[number]_num_[number]'
    match = re.search(r'case_(\d+)_num_(\d+)', filename)
    
    if match:
        # Extract matched numbers
        case_num = int(match.group(1))
        num = int(match.group(2))
        return case_num, num
    else:
        raise ValueError("Filename doesn't match the expected pattern!")
# import os
# import pickle

# def load_histories(file_path):
#     if os.path.exists(file_path):
#         with open(file_path, 'rb') as file:
#             return pickle.load(file)
#     else:
#         return {}

# def save_histories(histories_dict, file_path):
#     existing_histories = load_histories(file_path)
#     for key, value in histories_dict.items():
#         if key in existing_histories:
#             existing_histories[key].extend(value)
#         else:
#             existing_histories[key] = value
#     with open(file_path, 'wb') as file:
#         pickle.dump(existing_histories, file)

# def extract_case_and_num(filename):
#     import re
#     match = re.search(r'gmm_case_(\d+)_num_(\d+)', filename)
#     if match:
#         return int(match.group(1)), int(match.group(2))
#     else:
#         return None, None
