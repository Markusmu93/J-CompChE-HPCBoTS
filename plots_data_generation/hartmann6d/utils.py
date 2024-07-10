import sys
import os
import re
import numpy as np
import sklearn.gaussian_process.kernels as sklearnGP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from GPyOpt.methods import BayesianOptimization as BayesianOptimization_GPy
from skopt import gp_minimize
#from tqdm import tqdm
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from scipy.stats import uniform

from scipy.stats import multivariate_normal


#experiment.py file 
class GaussianMixtureModel:
    def __init__(self, n_components, means, covariances, weights):
        self.n_components = n_components
        self.means = means
        self.covariances = covariances
        self.weights = weights
        
    def pdf(self, x):
        # Calculate the PDF of the GMM at x
        pdf = np.sum([self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k]) for k in range(self.n_components)], axis=0)
        return pdf

# Define the random strategy
def random_strategy(objective_function, domain, n_iterations):
    # Generate random points in the domain
    points = np.array([uniform.rvs(loc=d['domain'][0], scale=d['domain'][1]-d['domain'][0], size=n_iterations) for d in domain]).T
    
    # Evaluate the objective function at these points
    return [objective_function(point.reshape(1, -1)) for point in points]

# Define the single sequential Bayesian optimization strategy
def bayesian_optimization_strategy(objective_function, domain, n_iterations, i_random):
    # Initialize the Bayesian optimizer
    optimizer = BayesianOptimization(f=objective_function, pbounds=domain, random_state=i_random)
    
    # Save the original stdout
    original_stdout = sys.stdout

    # Redirect stdout to a null device
    sys.stdout = open(os.devnull, 'w')

    # Run the optimization process with the 'ucb' acquisition function
    acquisition_function = UtilityFunction(kind="ucb")
    optimizer.maximize(init_points=1, n_iter=n_iterations-1)
    
    # Restore the original stdout
    sys.stdout = original_stdout

    # Return the history of function evaluations
    return [res['target'] for res in optimizer.res]


# Define the GPyOpt strategy
def gpyopt_strategy(objective_function, domain, n_iterations, i_random):
    # Define the problem
    problem = BayesianOptimization_GPy(
        f=objective_function, 
        domain=domain, 
        acquisition_type='UCB', 
        exact_feval=True, 
        maximize=True,
        initial_design_numdata=1,
        evaluator_type='sequential',
        random_state=i_random
    )
    
    # Run the optimization process
    problem.run_optimization(max_iter=n_iterations-1)
    
    # Return the history of function evaluations
    return problem.Y_best.tolist()

# Define the skopt strategy
def skopt_strategy(objective_function, domain, n_iterations, i_random):
    # Run the optimization process
    result = gp_minimize(
        func=objective_function, 
        dimensions=domain, 
        n_calls=n_iterations, 
        random_state=i_random
    )
    
    # Return the history of function evaluations
    return result.func_vals.tolist()

def extract_case_and_num(filename):
    # Regular expression to match the pattern 'case_[number]_num_[number]'
    match = re.search(r'case_(\d+)_num_(\d+)', filename)
    
    if match:
        # Extract matched numbers
        case_num = int(match.group(1))
        num = int(match.group(2))
        return case_num, num
    else:
        raise ValueError("Filename doesn't match the expected pattern!")
