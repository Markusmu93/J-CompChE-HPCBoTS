# import numpy as np
# import pickle
# from scipy.stats import multivariate_normal

# class GaussianMixtureModel:
#     def __init__(self, filename):
#         with open(filename, "rb") as file:
#             data = pickle.load(file)
#         self.n_components = data['n_components']
#         self.means = data['means']
#         self.covariances = data['covariances']
#         self.weights = data['weights']

#     def pdf(self, x):
#         pdf = np.sum([self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k]) for k in range(self.n_components)], axis=0)
#         return pdf

# def gmm_objective_function_array(X, gmm):
#     return np.array([gmm.pdf(x) for x in X])

# def gmm_objective_function_dict(gmm, **kwargs):
#     X = np.array([kwargs[key] for key in sorted(kwargs.keys())]).reshape(1, -1)
#     y = gmm.pdf(X[0])
#     return y




# import numpy as np
# import pickle
# from scipy.stats import multivariate_normal

# class GaussianMixtureModel:
#     def __init__(self, filename):
#         with open(filename, "rb") as file:
#             data = self._load_pickle(file)
#         self.n_components = data['n_components']
#         self.means = data['means']
#         self.covariances = data['covariances']
#         self.weights = data['weights']

#     def _load_pickle(self, file):
#         def rename_module(old_module, new_module):
#             class RenameUnpickler(pickle.Unpickler):
#                 def find_class(self, module, name):
#                     if module == old_module:
#                         module = new_module
#                     return super().find_class(module, name)
#             return RenameUnpickler(file).load()

#         return rename_module('src_gmm', 'objective_functions')

#     def pdf(self, x):
#         pdf = np.sum([self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k]) for k in range(self.n_components)], axis=0)
#         return pdf

# def gmm_objective_function_array(X, gmm):
#     return np.array([gmm.pdf(x) for x in X])

# def gmm_objective_function_dict(gmm, **kwargs):
#     X = np.array([kwargs[key] for key in sorted(kwargs.keys())]).reshape(1, -1)
#     y = gmm.pdf(X[0])
#     return y



# import numpy as np
# import pickle
# from scipy.stats import multivariate_normal

# class CustomUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == "src_gmm":
#             module = "objective_functions.src_gmm"
#         return super().find_class(module, name)

# class GaussianMixtureModel:
#     def __init__(self, filename):
#         with open(filename, "rb") as file:
#             data = CustomUnpickler(file).load()
#         self.n_components = data['n_components']
#         self.means = data['means']
#         self.covariances = data['covariances']
#         self.weights = data['weights']

#     def pdf(self, x):
#         pdf = np.sum([self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k]) for k in range(self.n_components)], axis=0)
#         return pdf

# def gmm_objective_function_array(X, gmm):
#     return np.array([gmm.pdf(x) for x in X])

# def gmm_objective_function_dict(gmm, **kwargs):
#     X = np.array([kwargs[key] for key in sorted(kwargs.keys())]).reshape(1, -1)
#     y = gmm.pdf(X[0])
#     return y


# import sys
# import os
# import numpy as np
# import pickle
# from scipy.stats import multivariate_normal

# # Add the src_gmm directory to the system path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './src_gmm')))

# class GaussianMixtureModel:
#     def __init__(self, filename):
#         with open(filename, "rb") as file:
#             data = pickle.load(file)
#         self.n_components = data['n_components']
#         self.means = data['means']
#         self.covariances = data['covariances']
#         self.weights = data['weights']

#     def pdf(self, x):
#         pdf = np.sum([self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k]) for k in range(self.n_components)], axis=0)
#         return pdf

# def gmm_objective_function_array(X, gmm):
#     return np.array([gmm.pdf(x) for x in X])

# def gmm_objective_function_dict(gmm, **kwargs):
#     X = np.array([kwargs[key] for key in sorted(kwargs.keys())]).reshape(1, -1)
#     y = gmm.pdf(X[0])
#     return y


# class CustomUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         # Update the module path to match the new location
#         if module == "src_gmm":
#             module = "objective_functions.src_gmm"
#         return super().find_class(module, name)

# class GaussianMixtureModel:
#     def __init__(self, filename):
#         with open(filename, "rb") as file:
#             data = CustomUnpickler(file).load()
#         self.n_components = data.n_components
#         self.means = data.means
#         self.covariances = data.covariances
#         self.weights = data.weights

#     def pdf(self, x):
#         pdf = np.sum([self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k]) for k in range(self.n_components)], axis=0)
#         return pdf

# def gmm_objective_function_array(X, gmm):
#     return np.array([gmm.pdf(x) for x in X])

# def gmm_objective_function_dict(gmm, **kwargs):
#     X = np.array([kwargs[key] for key in sorted(kwargs.keys())]).reshape(1, -1)
#     y = gmm.pdf(X[0])
#     return y

import json
import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components, means, covariances, weights):
        self.n_components = n_components
        self.means = np.array(means)
        self.covariances = [np.array(cov) for cov in covariances]
        self.weights = np.array(weights)

    def pdf(self, x):
        pdf = np.sum([self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k]) for k in range(self.n_components)], axis=0)
        return pdf

def load_gmm_from_json(json_filename):
    with open(json_filename, 'r') as f:
        gmm_data = json.load(f)

    return GaussianMixtureModel(
        n_components=gmm_data['n_components'],
        means=gmm_data['means'],
        covariances=gmm_data['covariances'],
        weights=gmm_data['weights']
    )

def gmm_objective_function_array(X, gmm):
    return np.array([gmm.pdf(x) for x in X])

def gmm_objective_function_dict(gmm, **kwargs):
    X = np.array([kwargs[key] for key in sorted(kwargs.keys())]).reshape(1, -1)
    y = gmm.pdf(X[0])
    return y