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
from utils.file_handling import load_histories, save_histories, extract_case_and_num
from objective_functions.gmm import load_gmm_from_json, gmm_objective_function_array, gmm_objective_function_scalar
from optimization_methods.random_strategy import random_strategy_X
from optimization_methods.bayesian_optimization_general import bayesian_optimization_strategy_X


# def run_high_dimensional_experiment(args):
#     domain, n_iterations, experiments_per_obj_f, list_of_methods, batch_size, filename = args

#     case_num, num = extract_case_and_num(filename)

#     gmm = GaussianMixtureModel(filename)

#     bayesian_domain = {d['name']: d['domain'] for d in domain}

#     histories_dict_y = {}
#     histories_dict_X = {}

#     for optimization_strategy, label in list_of_methods:
#         histories_y = []
#         histories_X = []

#         print(f"Started {label} for GMM")
#         for j in range(experiments_per_obj_f):
#             if label == 'Bayesian':
#                 history_y, history_X = optimization_strategy(partial(gmm_objective_function_dict, gmm=gmm), bayesian_domain, n_iterations, random_state=j)
#             elif label == 'Random':
#                 history_y, history_X = optimization_strategy(partial(gmm_objective_function_array, gmm=gmm), domain, n_iterations, state_seed=j)
#             else:
#                 optimizer = optimization_strategy(objective_function=partial(gmm_objective_function_array, gmm=gmm), domain=domain, B=batch_size, state_seed=j, n_init=1, kernel=None)
#                 optimizer.optimize(n_iterations)
#                 if label in ['TS-Constrained_UCB', 'TS-Constrained_EI', 'TS-Constrained_PI', 'PC-Basic-UCB', 'PC-Basic-GPUCB']:
#                     history_y = optimizer.history['y']
#                     history_X = optimizer.history['X']
#                 else:
#                     history_y = optimizer.history_inner['y']
#                     history_X = optimizer.history_inner['X']

#             histories_y.append(history_y)
#             histories_X.append(history_X)

#         histories_dict_y[label] = histories_y
#         histories_dict_X[label] = histories_X
#         print(f'Completed {label} of objective  number: gp_experimental')

#         y_file_path = f'../results_pcBO/results_gmms/histories_dict_y_function_experimental_{label}_gmm_case_{case_num}.pkl'
#         x_file_path = f'../results_pcBO/results_gmms/histories_dict_X_function_experimental_{label}_gmm_case_{case_num}.pkl'
        
#         save_histories(histories_dict_y, y_file_path)
#         save_histories(histories_dict_X, x_file_path)

# if __name__ == "__main__":
#     num_processes = 6
#     n_iterations = 10
#     experiments_per_obj_f = 3
#     batch_size = 4

#     pc_BO_TS_UCB = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='UCB')
#     pc_BO_TS_EI = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='EI')
#     pc_BO_TS_PI = partial(BO_TS_constrained_direct.pc_BO_TS, acquisition_type='PI')
#     pc_BO_nested_ucb = partial(BO_PC_Nested_direct_gpucb.pc_BO_nested, acquisition_type='UCB')
#     pc_BO_nested_gpucb = partial(BO_PC_Nested_direct_gpucb.pc_BO_nested, acquisition_type='GP-UCB')
#     pc_BO_basic_UCB = partial(BO_PC_basic_direct_gpucb.pc_BO_basic, acquisition_type='UCB')
#     pc_BO_basic_GPUCB = partial(BO_PC_basic_direct_gpucb.pc_BO_basic, acquisition_type='GP-UCB')

#     list_of_methods = [
#         (pc_BO_TS_UCB, 'TS-Constrained_UCB'), 
#         (pc_BO_TS_EI, 'TS-Constrained_EI'), 
#         (pc_BO_TS_PI, 'TS-Constrained_PI'),
#         (pc_BO_basic_UCB, 'PC-Basic-UCB'),
#         (pc_BO_basic_GPUCB, 'PC-Basic-GPUCB'),
#         (pc_BO_nested_ucb, 'PC-Nested-UCB'),
#         (pc_BO_nested_gpucb, 'PC-Nested-GPUCB'),
#         (random_strategy, 'Random'),
#         (bayesian_optimization_strategy, 'Bayesian')
#     ]

#     domain = [{'name': 'x1', 'type': 'constrained', 'domain': (-3, 3)}, 
#               {'name': 'x2', 'type': 'unconstrained', 'domain': (-3, 3)}]

#     # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
#     #     for i in range(1, 5):
#     #         filename = f'../objective_functions/pickled_gmms/gmm_case_{i}_num_1.pkl'
#     #         executor.map(run_high_dimensional_experiment, [(domain, n_iterations, experiments_per_obj_f, list_of_methods, batch_size, filename) for method in list_of_methods])

#     # Uncomment for sequential execution to debug and ensure correctness
#     for i in range(1, 5):
#         filename = f'../objective_functions/pickled_gmms/gmm_case_{i}_num_1.pkl'
#         for method in list_of_methods:
#             run_high_dimensional_experiment((domain, n_iterations, experiments_per_obj_f, [method], batch_size, filename))


def run_high_dimensional_experiment(args):
    domain, n_iterations, experiments_per_obj_f, list_of_methods, batch_size, json_filename = args

    case_num, num = extract_case_and_num(json_filename)

    gmm = load_gmm_from_json(json_filename)

    bayesian_domain = {d['name']: d['domain'] for d in domain}

    histories_dict_y = {}
    histories_dict_X = {}

    for optimization_strategy, label in list_of_methods:
        histories_y = []
        histories_X = []

        print(f"Started {label} for GMM")
        for j in range(experiments_per_obj_f):
            if label == 'Bayesian':
                history_y, history_X = optimization_strategy(partial(gmm_objective_function_scalar, gmm=gmm), bayesian_domain, n_iterations, random_state=j)
            elif label == 'Random':
                history_y, history_X = optimization_strategy(partial(gmm_objective_function_array, gmm=gmm), domain, n_iterations, state_seed=j)
            else:
                optimizer = optimization_strategy(objective_function=partial(gmm_objective_function_array, gmm=gmm), domain=domain, B=batch_size, state_seed=j, n_init=1, kernel=None)
                optimizer.optimize(n_iterations)
                if label in ['TS-Constrained_UCB', 'TS-Constrained_EI', 'TS-Constrained_PI', 'PC-Basic-UCB', 'PC-Basic-GPUCB']:
                    history_y = optimizer.history['y']
                    history_X = optimizer.history['X']
                else:
                    history_y = optimizer.history_inner['y']
                    history_X = optimizer.history_inner['X']

            histories_y.append(history_y)
            histories_X.append(history_X)

        histories_dict_y[label] = histories_y
        histories_dict_X[label] = histories_X
        print(f'Completed {label} of objective number: gp_experimental')

        y_file_path = f'../results_pcBO/results_gmms/histories_dict_y_function_experimental_{label}_gmm_case_{case_num}.pkl'
        x_file_path = f'../results_pcBO/results_gmms/histories_dict_X_function_experimental_{label}_gmm_case_{case_num}.pkl'
        
        save_histories(histories_dict_y, y_file_path)
        # save_histories(histories_dict_X, x_file_path)

if __name__ == "__main__":
    num_processes = 6
    n_iterations = 5
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
        (random_strategy_X, 'Random'),
        (bayesian_optimization_strategy_X, 'Bayesian')
    ]

    domain = [{'name': 'x1', 'type': 'constrained', 'domain': (-3, 3)}, 
              {'name': 'x2', 'type': 'unconstrained', 'domain': (-3, 3)}]

    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     for i in range(1, 5):
    #         #     #         filename = f'../objective_functions/pickled_gmms/gmm_case_{i}_num_1.pkl'
    #         json_filename = f'../objective_functions/json_gmms/gmm_case_{i}_num_1.json'
    #         executor.map(run_high_dimensional_experiment, [(domain, n_iterations, experiments_per_obj_f, list_of_methods, batch_size, json_filename) for method in list_of_methods])

    # Uncomment for sequential execution to debug and ensure correctness
    for i in range(1, 5):
        json_filename = f'../objective_functions/json_gmms/gmm_case_{i}_num_1.json'
        for method in list_of_methods:
            run_high_dimensional_experiment((domain, n_iterations, experiments_per_obj_f, [method], batch_size, json_filename))
