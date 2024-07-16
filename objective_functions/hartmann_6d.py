import numpy as np

alpha = np.array([1.0, 1.2, 3.0, 3.2])
A = np.array([
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14]
])
P = 1e-4 * np.array([
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381]
])

def hartmann_6d(X):
    X = np.atleast_2d(X)
    result = np.zeros(X.shape[0])
    for i in range(4):
        result -= alpha[i] * np.exp(-np.sum(A[i] * (X - P[i])**2, axis=1))
    return -1 * np.array([np.squeeze(result)])

def hartmann_6d_for_bayesian(**kwargs):
    X = np.array([kwargs[key] for key in sorted(kwargs.keys())]).reshape(1, -1)
    values = np.zeros(X.shape[0])
    for i in range(4):
        values -= alpha[i] * np.exp(-np.sum(A[i] * (X - P[i])**2, axis=1))
    return -1 * np.squeeze(values)
