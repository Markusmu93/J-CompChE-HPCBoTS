import pickle
import numpy as np

class GaussianProcessModel:
    def __init__(self, filename):
        with open(filename, "rb") as file:
            self.gp = pickle.load(file)

    def predict(self, x):
        return self.gp.predict(x.reshape(1, -1))

def gp_objective_function_array(X, gp_model):
    X = np.atleast_2d(X)
    return -np.array([gp_model.predict(x)[0] for x in X])

def gp_objective_function_scalar(X, gp_model):
    X = np.atleast_2d(X)
    return -np.array([gp_model.predict(x)[0] for x in X])[0]
