import sys
sys.path.append('../../PC-BO')
from src import BO_PC_basic_direct_gpucb, BO_TS_constrained_direct, BO_PC_Nested_direct_gpucb
import numpy as np
import pandas as pd
import sklearn.gaussian_process.kernels as sklearnGP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from scipy.optimize import differential_evolution, minimize, direct
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import time
import copy
import numpy as np
import pickle
import pandas as pd
import concurrent.futures
from functools import partial
from utils import random_strategy, bayesian_optimization_strategy


# Define the constants for the Hartmann 6D function
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
    """6D Hartmann function modified to accept both 1D and 2D array input."""
    X = np.atleast_2d(X)  # Ensure X is 2D for consistent numpy operations
    
    # Calculate the Hartmann value for each row/sample
    result = np.zeros(X.shape[0])
    for i in range(4):
        result -= alpha[i] * np.exp(-np.sum(A[i] * (X - P[i])**2, axis=1))
    
    return -1 * np.array([np.squeeze(result)])  # Return a scalar if a single value, else array

def hartmann_6d_for_bayesian(**kwargs):
    """Objective function for 6D Hartmann that accepts keyword arguments."""
    # Extract the variables from kwargs and create a list in the order x1 to x6
    X = np.array([kwargs[key] for key in sorted(kwargs.keys())]).reshape(1, -1)

    # Calculate Rosenbrock value for each row/sample
    values = np.zeros(X.shape[0])
    
    for i in range(4):
        values -= alpha[i] * np.exp(-np.sum(A[i] * (X - P[i])**2, axis=1))
    
    # Use the previously defined hartmann_6d function
    return -1 * np.squeeze(values)


# Define the run_high_dimensional_experiment function
def run_high_dimensional_experiment(args):
    domain, n_itier, experiments_per_obj_f, list_of_methods, batch_size = args

    # Redefine the domain for bayes_opt package class 
    bayesian_domain = {d['name']: d['domain'] for d in domain}

    # Initialize the histories dictionary
    histories_dict = {}

    for optimization_strategy, label in list_of_methods:
        histories = []  # Initialize the list of histories for the current strategy

        # Count the number of unconstrained dimensions
        num_unconstrained = sum(1 for d in domain if d['type'] == 'unconstrained')
        print(f"Started {label} for domain configuration: {num_unconstrained}")
        for j in range(experiments_per_obj_f):
                if label == 'Bayesian':
                    history = optimization_strategy(hartmann_6d_for_bayesian, bayesian_domain, n_itier, i_random=j)
                elif label == 'Random': 
                    history = optimization_strategy(hartmann_6d, domain, n_itier)
                elif label in ['TS-Constrained_UCB', 'TS-Constrained_EI', 'TS-Constrained_PI', 'PC-Basic-UCB', 'PC-Basic-GPUCB']:
                    optimizer = optimization_strategy(objective_function=hartmann_6d, domain=domain, B=batch_size, n_init=1, kernel=None)  # Replace None with your objective function
                    optimizer.optimize(n_itier)
                    history = optimizer.history['y']  # Assuming 'history' attribute stores the optimization history    
                elif label in ['PC-Nested-GPUCB', 'PC-Nested-UCB']:
                    optimizer = optimization_strategy(objective_function=hartmann_6d, domain=domain, B=batch_size, n_init=1, kernel=None, X_initial=None, y_initial=None)  # Replace None with your objective function
                    optimizer.optimize(n_itier)
                    history = optimizer.history_inner['y']
                histories.append(history)
                histories_dict[label] = histories
                print(f'Completed {label} of objective  number: gp_experimental')
        # Save the histories dictionary to a file
        with open(f'./results/histories_dict_function_experimental_{label}_hartmann6d.pkl', 'wb') as f:
                pickle.dump(histories_dict, f)

# Main execution
if __name__ == "__main__":
    # Setup
    num_processes = 6
    n_itier = 76
    experiments_per_obj_f = 10

    batch_size = 4

    # Partial functions for Bayesian Optimization methods
    pc_BO_TS_UCB = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='UCB')
    pc_BO_TS_EI = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='EI')
    pc_BO_TS_PI = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='PI')
    pc_BO_nested_ucb = partial(BO_PC_Nested_direct_gpucb.pc_BO_nested, acquisition_type='UCB')
    pc_BO_nested_gpucb = partial(BO_PC_Nested_direct_gpucb.pc_BO_nested, acquisition_type='GP-UCB')
    pc_BO_basic_UCB = partial(BO_PC_basic_direct_gpucb.pc_BO_basic, acquisition_type='UCB')
    pc_BO_basic_GPUCB = partial(BO_PC_basic_direct_gpucb.pc_BO_basic, acquisition_type='GP-UCB')


    # List of methods to test
    list_of_methods = [
        # (pc_BO_TS_UCB, 'TS-Constrained_UCB'), 
        # (pc_BO_TS_EI, 'TS-Constrained_EI'), 
        # (pc_BO_TS_PI, 'TS-Constrained_PI'),
        # (pc_BO_basic_UCB, 'PC-Basic-UCB'),
        # (pc_BO_basic_GPUCB, 'PC-Basic-GPUCB'),
        # (pc_BO_nested_ucb, 'PC-Nested-UCB')
        # (pc_BO_nested_gpucb, 'PC-Nested-GPUCB'),
        (random_strategy, 'Random'),
        (bayesian_optimization_strategy, 'Bayesian')
    ]


    # Define the domain configurations
    domain = [
        {'name': 'x1', 'type': 'constrained', 'domain': (0,1)},
        {'name': 'x2', 'type': 'constrained', 'domain': (0,1)},
        {'name': 'x3', 'type': 'constrained', 'domain': (0,1)},
        {'name': 'x4', 'type': 'unconstrained', 'domain': (0,1)}, 
        {'name': 'x5', 'type': 'unconstrained', 'domain': (0,1)}, 
        {'name': 'x6', 'type': 'unconstrained', 'domain': (0,1)}
    ]
    
    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Use the executor to run the experiments in parallel
        # The executor.map function takes the function to execute and an iterable of arguments to pass to it
        # In this case, the function is run_experiment and the arguments are the filenames of the gmms which should be optimized and the number of iterations 
        executor.map(run_high_dimensional_experiment, [(domain, n_itier, experiments_per_obj_f, [method], batch_size) for method in list_of_methods])

    # for method in list_of_methods:
    #     run_high_dimensional_experiment((domain, n_itier, experiments_per_obj_f, [method], batch_size))
