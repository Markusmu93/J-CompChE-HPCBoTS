import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure 'PC-BO' is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../PC-BO')))

from src import BO_PC_basic_direct_gpucb, BO_TS_constrained_direct, BO_PC_Nested_direct_gpucb
import numpy as np
import concurrent.futures
import pickle
from functools import partial
from objective_functions.levy import levy_6d
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
                history = optimization_strategy(levy_6d, bayesian_domain, n_iterations, random_state=j)
            elif label == 'Random':
                history = optimization_strategy(levy_6d, domain, n_iterations)
            else:
                optimizer = optimization_strategy(objective_function=levy_6d, domain=domain, B=batch_size, state_seed=j, n_init=1, kernel=None)
                optimizer.optimize(n_iterations)
                if label in ['TS-Constrained_UCB', 'TS-Constrained_EI', 'TS-Constrained_PI', 'PC-Basic-UCB', 'PC-Basic-GPUCB']:
                    history = optimizer.history['y']
                else:
                    history = optimizer.history_inner['y']
            histories.append(history)
        histories_dict[label] = histories
        print(f'Completed {label} of objective number: gp_experimental')
        with open(f'../results_pcBO/results_levy6d/histories_dict_function_experimental_{label}_levy6d.pkl', 'wb') as f:
            pickle.dump(histories_dict, f)

if __name__ == "__main__":
    num_processes = 7
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
        (pc_BO_nested_gpucb, 'PC-Nested-GPUCB')
        # (random_strategy, 'Random'),
        # (bayesian_optimization_strategy, 'Bayesian')
    ]

    domain = [
        {'name': 'x1', 'type': 'constrained', 'domain': (-5, 5)},
        {'name': 'x2', 'type': 'constrained', 'domain': (-5, 5)},
        {'name': 'x3', 'type': 'constrained', 'domain': (-5, 5)},
        {'name': 'x4', 'type': 'unconstrained', 'domain': (-5, 5)}, 
        {'name': 'x5', 'type': 'unconstrained', 'domain': (-5, 5)}, 
        {'name': 'x6', 'type': 'unconstrained', 'domain': (-5, 5)}
    ]

    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     executor.map(run_high_dimensional_experiment, [(domain, n_iterations, experiments_per_obj_f, [method], batch_size) for method in list_of_methods])
    
    # # Instead of concurrent just sequentially loop over each experiment to test that experiment is set up correctly, concurrent does not indicate errors
    for method in list_of_methods:
        run_high_dimensional_experiment((domain, n_iterations, experiments_per_obj_f, [method], batch_size))
