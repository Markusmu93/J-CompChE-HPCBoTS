import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

def random_strategy(objective_function, domain, n_iterations, random_state=None):
    np.random.seed(random_state)
    # bounds = np.array([domain[d['name']] for d in domain])
    bounds = np.array([d['domain'] for d in domain])
    results = np.zeros(n_iterations)
    for t in range(n_iterations):
        random_point = np.random.uniform(bounds[:, 0], bounds[:, 1])
        results[t] = objective_function(random_point)
    return results

