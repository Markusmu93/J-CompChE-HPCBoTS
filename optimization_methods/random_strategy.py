import sys
import os
from scipy.stats import uniform

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



# Define the random strategy
def random_strategy_X(objective_function, domain, n_iterations, state_seed):
    # Generate random points in the domain
    np.random.seed(state_seed)
    points = np.array([uniform.rvs(loc=d['domain'][0], scale=d['domain'][1]-d['domain'][0], size=n_iterations) for d in domain]).T
    
    # Evaluate the objective function at these points
    return [objective_function(point.reshape(1, -1)) for point in points], points