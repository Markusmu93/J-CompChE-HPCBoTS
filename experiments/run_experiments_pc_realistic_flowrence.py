import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure 'PC-BO' is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../PC-BO')))

from src import BO_PC_basic_direct_gpucb, BO_TS_constrained_direct, BO_PC_Nested_direct_gpucb
import numpy as np
import concurrent.futures
from functools import partial
from utils.file_handling import load_histories, save_histories
from objective_functions.realistic_flowrence_gp_model import GaussianProcessModel, gp_objective_function_array, gp_objective_function_scalar
from optimization_methods.random_strategy import random_strategy_X
from optimization_methods.bayesian_optimization_general import bayesian_optimization_strategy_X


# Ensure 'PC-BO' is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../PC-BO')))

def run_experiment(args):
    filename, n_itier, experiments_per_obj_f, method = args

    gp_model = GaussianProcessModel(filename)

    optimization_strategy, label = method
    print(f'We are starting {label}')

    domain = [{'name': 'x1', 'type': 'constrained', 'domain': (0, 55)},
              {'name': 'x2', 'type': 'unconstrained', 'domain': (520, 590)}]

    bayesian_domain = {d['name']: d['domain'] for d in domain}

    histories_dict_y = {}
    histories_dict_X = {}

    histories_y = []
    histories_X = []
    for j in range(experiments_per_obj_f):
        print(f'experiment number: {j}')
        if label == 'Random':
            history_y, history_X = optimization_strategy(partial(gp_objective_function_array, gp_model=gp_model), domain, n_itier, state_seed=j)
        elif label == 'Bayesian':
            history_y, history_X = optimization_strategy(partial(gp_objective_function_scalar, gp_model=gp_model), bayesian_domain, n_itier, random_state=j)
        elif label in ['PC-Nested-UCB', 'PC-Nested-GPUCB']:
            optimizer = optimization_strategy(objective_function=partial(gp_objective_function_array, gp_model=gp_model), domain=domain, B=4, n_init=1, state_seed=j, kernel=None, X_initial=None, y_initial=None)
            optimizer.optimize(n_itier)
            history_y = optimizer.history_inner['y']
            history_X = optimizer.history_inner['X']
        else:
            optimizer = optimization_strategy(objective_function=partial(gp_objective_function_array, gp_model=gp_model), domain=domain, B=4, n_init=1, state_seed=j, kernel=None, X_initial=None, y_initial=None)
            optimizer.optimize(n_itier)
            history_y = optimizer.history['y']
            history_X = optimizer.history['X']
        histories_y.append(history_y)
        histories_X.append(history_X)

    histories_dict_y[label] = histories_y
    histories_dict_X[label] = histories_X
    print(f'Completed {label} of objective number: gp_experimental')

    y_file_path = f'../results_pcBO/results_realistic_flowrence/histories_dict_y_function_case_{1}_num_{1}_experimental_{label}_initial_points.pkl'
    # x_file_path = f'../investigation_results_experimental_validation_run_initial_points_equal/histories_dict_X_function_case_{1}_num_{1}_experimental_{label}_initial_points.pkl'
    save_histories(histories_dict_y, y_file_path)
    # save_histories(histories_dict_X, x_file_path)

if __name__ == "__main__":
    num_processes = 6
    n_iterations = 5
    experiments_per_obj_f = 3


    # Partial functions for Bayesian Optimization methods
    pc_BO_TS_UCB = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='UCB')
    pc_BO_TS_EI = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='EI')
    pc_BO_TS_PI = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='PI')
    pc_BO_nested_ucb = partial(BO_PC_Nested_direct_gpucb.pc_BO_nested, acquisition_type='UCB')
    pc_BO_nested_gpucb = partial(BO_PC_Nested_direct_gpucb.pc_BO_nested, acquisition_type='GP-UCB')
    pc_BO_basic_UCB = partial(BO_PC_basic_direct_gpucb.pc_BO_basic, acquisition_type='UCB')
    pc_BO_basic_GPUCB = partial(BO_PC_basic_direct_gpucb.pc_BO_basic, acquisition_type='GP-UCB')


    list_of_methods = [
        # (pc_BO_TS_UCB, 'TS-Constrained_UCB'), 
        # (pc_BO_TS_EI, 'TS-Constrained_EI'), 
        # (pc_BO_TS_PI, 'TS-Constrained_PI'),
        # (pc_BO_basic_UCB, 'PC-Basic-UCB'),
        # (pc_BO_basic_GPUCB, 'PC-Basic-GPUCB'),
        # (pc_BO_nested_ucb, 'PC-Nested-UCB'), 
        # (pc_BO_nested_gpucb, 'PC-Nested-GPUCB'),
        (random_strategy_X, 'Random'),
        (bayesian_optimization_strategy_X, 'Bayesian')
    ]

    filename = '../objective_functions/pickled_flowrence_gp/gp_experiment.pkl'
    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     executor.map(run_experiment, [(filename, n_iterations, experiments_per_obj_f, method) for method in methods])
    # Sequential execution for debugging
    for method in list_of_methods:
        run_experiment((filename, n_iterations, experiments_per_obj_f, method))
