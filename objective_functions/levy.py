import numpy as np

# Define the Levy function for 6D that matches the output structure of the Ackley function
def levy_6d(X):
    """Levy function modified to accept both 1D and 2D array input."""
    X = np.atleast_2d(X)  # Ensure X is 2D for consistent numpy operations
    
    w = 1 + (X - 1) / 4
    term1 = np.sin(np.pi * w[:, 0]) ** 2
    term3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    term2 = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1)
    
    result = term1 + term2 + term3
    
    return -1 * np.array([np.squeeze(result)])  # Return a scalar if a single value, else array