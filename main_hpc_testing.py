from hpc_BO.hcp_TS import HierarchicalPCBO
from parallel_BO.gp_ucb_pe import GPUCB_PE

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.learning.gaussian_process.kernels import Matern
from skopt.learning import GaussianProcessRegressor

# Defining the objective function
def objective_function(x):
    X = np.atleast_2d(x)
    return -np.sum((X-2)**2, axis=1) # Quadratic function x^* centered at 2,2, ... d times

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
    if X.shape[1] != 3:
        raise ValueError("Rosenbrock 3D function input must be a 3-dimensional vector.")
    
    # Calculate the Rosenbrock function for each row in X (each input vector)
    sum_terms = (a - X[:, 0])**2 + b * (X[:, 1] - X[:, 0]**2)**2 + (a - X[:, 1])**2 + b * (X[:, 2] - X[:, 1]**2)**2
    return - sum_terms

# Rosenbrock function adapted for skopt
space = [
    Real(-2.048, 2.048, name='x1'),
    Real(-2.048, 2.048, name='x2'),
    Real(-2.048, 2.048, name='x3')
]

@use_named_args(space)
def rosenbrock_3d_skopt(x1, x2, x3):
    a = 1
    b = 100
    return -((a - x1)**2 + b * (x2 - x1**2)**2 + (a - x2)**2 + b * (x3 - x2**2)**2)


def styblinski_tang(x):
    """
    Evaluate the Styblinski-Tang function at a point x.
    The function is commonly used as a benchmark function for testing optimization algorithms.
    
    Parameters:
    - x : array_like, Input array, should be a one-dimensional vector of size n.
    
    Returns:
    - float, The function value at x.
    """
    X = np.atleast_2d(x)  # Ensure X is 2D for consistent output format
    term = np.power(X, 4) - 16 * np.square(X) + 5 * X
    return -0.5 * np.sum(term, axis=1)  # The sum here is along rows if X is truly 2D

def run_hpc_ts(index, bounds, T, L, K):
    optimizer = HierarchicalPCBO(rosenbrock_3d, bounds, T, L, K, random_state=index)
    optimizer.optimize('ucb')
    return optimizer.y_train.flatten()  # Assuming y_train stores the observations

def run_gp_ucb_pe(index, bounds, T, K_GP):
    optimizer = GPUCB_PE(bounds, rosenbrock_3d, random_state=index)
    optimizer.optimize(T-1, K_GP, n_initial_points=8)
    return optimizer.Y_train.flatten()  # Assuming Y_train stores the observations

def random_strategy(bounds, T, random_state=None):
    """
    Randomly sample points from the design space and evaluate the objective function.
    """
    np.random.seed(random_state)
    bounds = np.array(bounds)
    results = np.zeros(T)
    for t in range(T):
        # Generate a random point within the bounds
        random_point = np.random.uniform(bounds[:, 0], bounds[:, 1])
        # Evaluate the objective function
        results[t] = rosenbrock_3d(random_point)
    return results

def run_bayesian_optimization(index, space, T):
    # Define the GP with Matérn kernel
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
    result = gp_minimize(
        rosenbrock_3d_skopt, 
        space,
        acq_func="EI",
        n_calls=T,
        n_random_starts=10,
        base_estimator=gp,
        random_state=index
    )
    return result.func_vals 

def main():
    N = 3  # Number of runs
    bounds = [[-2.048, 2.048], [-2.048, 2.048], [-2.048, 2.048]]
    T = 15  # Number of iterations
    L = 3
    K = [1, 2, 4]
    K_GP = 8

    # Arrays to store results
    y_results_hpc_ts = np.zeros((N, T * np.prod(K)))
    y_results_gp_ucb_pe = np.zeros((N, T * K_GP))
    y_results_random = np.zeros((N, T * K_GP))

    # Arrays to store single sequential BO
    y_results_bayes_opt = np.zeros((N, T))


    with ProcessPoolExecutor(max_workers=6) as executor:
        # Run HPC-TS and GP-UCB-PE
        futures_hpc_ts = [executor.submit(run_hpc_ts, i, bounds, T, L, K) for i in range(N)]
        futures_gp_ucb_pe = [executor.submit(run_gp_ucb_pe, i, bounds, T, K_GP) for i in range(N)]
        futures_random = [executor.submit(random_strategy, bounds, T * K_GP, i) for i in range(N)]
        # Run Bayesian Optimization
        futures_bayes_opt = [executor.submit(run_bayesian_optimization, i, space, T) for i in range(N)]


        for i, future in enumerate(futures_hpc_ts):
            y_results_hpc_ts[i, :] = future.result()
        for i, future in enumerate(futures_gp_ucb_pe):
            y_results_gp_ucb_pe[i, :] = future.result()
        for i, future in enumerate(futures_random):
            y_results_random[i, :] = future.result()
        for i, future in enumerate(futures_bayes_opt):
            y_results_bayes_opt[i, :] = future.result()

    # Save the results to a CSV file
    np.savetxt("bayes_opt_results_rosenbrock_3d_test.csv", y_results_bayes_opt, delimiter=",")
    np.savetxt("random_results_rosenbrock_3d_test.csv", y_results_random, delimiter=",")
    np.savetxt("gp_ucb_pe_results_rosenbrock_3d_test.csv", y_results_gp_ucb_pe, delimiter=",")
    np.savetxt("hpc_ts_results_rosenbrock_3d_random_state_test.csv", y_results_hpc_ts, delimiter=",")

if __name__ == "__main__":
    main()