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

def gmm_objective_function_scalar(X, gmm):
    return np.array([gmm.pdf(x) for x in X])[0]

def gmm_objective_function_dict(gmm, **kwargs):
    X = np.array([kwargs[key] for key in sorted(kwargs.keys())]).reshape(1, -1)
    y = gmm.pdf(X[0])
    return y
