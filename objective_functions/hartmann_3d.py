# import all necessary libraries here
import numpy as np

def hartmann_3d(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [381, 5743, 8828]])
    X = np.atleast_2d(x)  # Ensure X is 2D
    outer = np.zeros(X.shape[0])
    for i in range(4):
        inner = np.zeros(X.shape[0])
        for j in range(3):
            inner += A[i, j] * (X[:, j] - P[i, j]) ** 2
        outer += alpha[i] * np.exp(-inner)
    return -outer  # Negative for maximization