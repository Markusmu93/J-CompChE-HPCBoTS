import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure 'PC-BO' is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../PC-BO')))

from src import BO_PC_basic_direct_gpucb, BO_TS_constrained_direct, BO_PC_Nested_direct_gpucb
import numpy as np
import copy
import concurrent.futures
from functools import partial
from utils.file_handling import load_histories, save_histories
from objective_functions.rosenbrock import rosenbrock_nd
from optimization_methods.random_strategy import random_strategy
from optimization_methods.bayesian_optimization_general import bayesian_optimization_strategy


def run_high_dimensional_experiment(args):
    domain, n_iterations, experiments_per_obj_f, list_of_methods, batch_size = args
    bayesian_domain = {d['name']: d['domain'] for d in domain}
    histories_dict = {}

    for optimization_strategy, label in list_of_methods:
        histories = []
        num_unconstrained = sum(1 for d in domain if d['type'] == 'unconstrained')
        print(f"Started {label} for domain configuration: {num_unconstrained}")

        for j in range(experiments_per_obj_f):
            if label == 'Bayesian':
                history = optimization_strategy(rosenbrock_nd, bayesian_domain, n_iterations, random_state=j)
            elif label == 'Random': 
                history = optimization_strategy(rosenbrock_nd, domain, n_iterations)
            else:
                optimizer = optimization_strategy(objective_function=rosenbrock_nd, domain=domain, B=batch_size, state_seed=j, n_init=1, kernel=None)
                optimizer.optimize(n_iterations)
                if label in ['TS-Constrained_UCB', 'TS-Constrained_EI', 'TS-Constrained_PI', 'PC-Basic-GPUCB', 'PC-Basic-UCB']:
                    history = optimizer.history['y']
                else:
                    history = optimizer.history_inner['y']

            histories.append(history)
        histories_dict[label] = histories
        print(f"Completed {label} for domain configuration: {num_unconstrained}")

    results_file_path = f'../results_pcBO/results_rosenbrock4d/histories_dict_num_unconstrained_{num_unconstrained}.pkl'
    save_histories(histories_dict, results_file_path)

if __name__ == "__main__":
    num_processes = 3
    n_iterations = 10
    experiments_per_obj_f = 3
    batch_size = 4

    pc_BO_TS_UCB = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='UCB')
    pc_BO_TS_EI = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='EI')
    pc_BO_TS_PI = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='PI')
    pc_BO_nested_ucb = partial(BO_PC_Nested_direct_gpucb.pc_BO_nested, acquisition_type='UCB')
    pc_BO_nested_gpucb = partial(BO_PC_Nested_direct_gpucb.pc_BO_nested, acquisition_type='GP-UCB')
    pc_BO_basic_UCB = partial(BO_PC_basic_direct_gpucb.pc_BO_basic, acquisition_type='UCB')
    pc_BO_basic_GPUCB = partial(BO_PC_basic_direct_gpucb.pc_BO_basic, acquisition_type='GP-UCB')

    list_of_methods = [
        (pc_BO_TS_UCB, 'TS-Constrained_UCB'), 
        (pc_BO_TS_EI, 'TS-Constrained_EI'), 
        (pc_BO_TS_PI, 'TS-Constrained_PI'),
        (pc_BO_basic_UCB, 'PC-Basic-UCB'),
        (pc_BO_basic_GPUCB, 'PC-Basic-GPUCB'),
        (pc_BO_nested_ucb, 'PC-Nested-UCB'),
        (pc_BO_nested_gpucb, 'PC-Nested-GPUCB'),
        (random_strategy, 'Random'),
        (bayesian_optimization_strategy, 'Bayesian')
    ]

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

    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     executor.map(run_high_dimensional_experiment, [(domain, n_iterations, experiments_per_obj_f, list_of_methods, batch_size) for domain in domain_configs])

    # Uncomment for sequential execution to debug and ensure correctness
    
        run_high_dimensional_experiment((domain, n_iterations, experiments_per_obj_f, list_of_methods, batch_size))
