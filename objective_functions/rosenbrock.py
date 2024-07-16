# import all necessary libraries
import numpy as np

def rosenbrock_3d(x, a=1, b=100):
    """
    Computes the Rosenbrock function in 3 dimensions.
    The function is generally defined by:
        f(x, y, z) = (a-x1)^2 + b*(x2-x1^2)^2 + (a-x2)^2 + b*(x3-x2^2)^2
    where typically a = 1 and b = 100.

    Parameters:
    - x : array_like, The input variables array where x = [x1, x2, x3].
    - a : float, The constant term for the (a-x)^2 part (default: 1).
    - b : float, The constant term for the b*(y-x^2)^2 part (default: 100).

    Returns:
    - float, The Rosenbrock function evaluated at the point x.
    """
    X = np.atleast_2d(x)  # Ensure X is 2D
    # if X.shape[1] != 3:
    #     raise ValueError("Rosenbrock 3D function input must be a 3-dimensional vector.")
    
    # Calculate the Rosenbrock function for each row in X (each input vector)
    sum_terms = (a - X[:, 0])**2 + b * (X[:, 1] - X[:, 0]**2)**2 + (a - X[:, 1])**2 + b * (X[:, 2] - X[:, 1]**2)**2
    return - sum_terms

# # Run script
# def main():
#     x = [0.5, 0.5, 1, 1]
#     print(rosenbrock_3d(x))

# if __name__ == "__main__":
#     main()
