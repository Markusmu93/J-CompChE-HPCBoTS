import sys
sys.path.append('../../PC-BO')
from src import BO_TS_constrained_direct, BO_PC_Nested_direct_gpucb, BO_PC_basic_direct_gpucb
import numpy as np
import pandas as pd
import sklearn.gaussian_process.kernels as sklearnGP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
#from GPyOpt.methods import BayesianOptimization

# import cProfile
# import pstats
from line_profiler import LineProfiler


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


import os

# Define a function to check for the existence of the file and load it
def load_histories(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        return {}

# Define a function to update and save the histories dictionary
def save_histories(histories_dict, file_path):
    existing_histories = load_histories(file_path)
    
    # Update the existing dictionary with the new results
    for key, value in histories_dict.items():
        if key in existing_histories:
            existing_histories[key].extend(value)  # Assuming the values are lists and you want to append
        else:
            existing_histories[key] = value
    
    # Save the updated dictionary
    with open(file_path, 'wb') as file:
        pickle.dump(existing_histories, file)

def rosenbrock_4d_modified(X):
    """4D Rosenbrock function modified to accept both 1D and 2D array input."""
    # Ensure input is a numpy array
    X = np.asarray(X)
    
    # If 1D input, reshape to 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Calculate Rosenbrock value for each row/sample
    values = [sum(100.0 * (x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0 for i in range(3)) for x in X]
    
    return -1 * np.array(values)

def rosenbrock_4d_for_bayesian(**kwargs):
    """Objective function for 4D Rosenbrock that accepts keyword arguments"""
    # Convert the keyword arguments to a numpy array
    X = np.array([kwargs[key] for key in sorted(kwargs.keys())]).reshape(1, -1)
    
    # Calculate Rosenbrock value for each row/sample
    values = [sum(100.0 * (x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0 for i in range(3)) for x in X]
    
    return -1 * values[0]  # Extract the scalar prediction

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
                history = optimization_strategy(rosenbrock_4d_for_bayesian, bayesian_domain, n_itier, i_random=j)
            elif label == 'Random': 
                history = optimization_strategy(rosenbrock_4d_modified, domain, n_itier)
            elif label in ['TS-Constrained_UCB', 'TS-Constrained_EI', 'TS-Constrained_PI', 'PC-Basic-GPUCB', 'PC-Basic-UCB']:
                optimizer = optimization_strategy(objective_function=rosenbrock_4d_modified, domain=domain, B=batch_size, n_init=1, kernel=None)  # Replace None with your objective function
                optimizer.optimize(n_itier)
                history = optimizer.history['y']  # Assuming 'history' attribute stores the optimization history    
            elif label in ['PC-Nested-GPUCB', 'PC-Nested-UCB']:
                optimizer = optimization_strategy(objective_function=rosenbrock_4d_modified, domain=domain, B=batch_size, n_init=1, kernel=None, X_initial=None, y_initial=None)  # Replace None with your objective function
                optimizer.optimize(n_itier)
                history = optimizer.history_inner['y']

            # Include additional cases for other optimization methods as needed

            histories.append(history)

        histories_dict[label] = histories
        print(f"Completed {label} for domain configuration: {num_unconstrained}")

    # Save the histories dictionary to a file
    #with open(f'./results/histories_dict_num_unconstrained_{num_unconstrained}.pkl', 'wb') as f:
    # with open(f'./results_gpucb/histories_dict_num_unconstrained_{num_unconstrained}.pkl', 'wb') as f: 
    #     pickle.dump(histories_dict, f)
    results_file_path = f'./results_gpucb/histories_dict_num_unconstrained_{num_unconstrained}.pkl'
    save_histories(histories_dict, results_file_path)


# Main execution
if __name__ == "__main__":
    # Setup
    num_processes = 3
    n_itier = 75
    experiments_per_obj_f = 10
    batch_size=4

    # # Define the partial functions for the methods with different acquisition types
    # pc_BO_TS_UCB = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='UCB')
    # pc_BO_TS_EI = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='EI')
    # pc_BO_TS_PI = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='PI')

    # list_of_methods = [
    #     # (pc_BO_TS_UCB, 'TS-Constrained_UCB'), 
    #     # (pc_BO_TS_EI, 'TS-Constrained_EI'), 
    #     # (pc_BO_TS_PI,'TS-Constrained_PI') , 
    #     # (BO_PC_basic_direct.pc_BO_basic, 'PC-Basic'), 
    #     # (BO_PC_Nested_direct.pc_BO_nested, 'PC-Nested'), 
    #     (random_strategy, 'Random'), 
    #     (bayesian_optimization_strategy, 'Bayesian')]

    #list_of_methods = [(pc_BO_TS_EI, 'TS-Constrained_EI')]

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
        (pc_BO_basic_GPUCB, 'PC-Basic-GPUCB'),
        # (pc_BO_nested_ucb, 'PC-Nested-UCB'),
        (pc_BO_nested_gpucb, 'PC-Nested-GPUCB')
        # (random_strategy, 'Random'),
        # (bayesian_optimization_strategy, 'Bayesian')
    ]


    # Define the domain configurations
    base_domain = [
        {'name': 'x1', 'type': 'unconstrained', 'domain': (-2, 2)},
        {'name': 'x2', 'type': 'unconstrained', 'domain': (-2, 2)},
        {'name': 'x3', 'type': 'unconstrained', 'domain': (-2, 2)},
        {'name': 'x4', 'type': 'unconstrained', 'domain': (-2, 2)}
    ]


    domain_configs = []
    for i in range(1, 4):
        domain = copy.deepcopy(base_domain)
        for j in range(i):
            domain[j]['type'] = 'constrained'
        domain_configs.append(domain)

    # # Parallel execution
    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     executor.map(
    #         run_high_dimensional_experiment,
    #         [(domain, n_itier, experiments_per_obj_f, list_of_methods) for domain in domain_configs]
    #     )
    
    for domain in domain_configs:
        run_high_dimensional_experiment((domain, n_itier, experiments_per_obj_f, list_of_methods, batch_size))