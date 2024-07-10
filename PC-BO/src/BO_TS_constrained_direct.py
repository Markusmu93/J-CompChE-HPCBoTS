#import sys
# sys.path.append('/Users/mgrimm/AICAT_git_correct/BO-TS-CONSTRAINED')
# from src import BO_TS_constrained_direct
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


class pc_BO_TS:
    def __init__(self, objective_function, domain, B, state_seed, kernel=None, n_init=3, X_initial=None, y_initial=None, acquisition_type='UCB'):
        self.objective_function = objective_function
        # convert list to dictionary for easy access
        self.domain = {d['name']: d for d in domain}
        # get names of constrained and unconstrained variables
        self.constrained_vars = [name for name, var in self.domain.items() if var['type'] == 'constrained']
        self.unconstrained_vars = [name for name, var in self.domain.items() if var['type'] == 'unconstrained']
        self.B = B 
        # use Matern kernel by default
        self.kernel = kernel if kernel is not None else Matern(nu=2.5)
        # initialize Gaussian process model
        self.gp_model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            #normalize_y=False,
            normalize_y=True,
            n_restarts_optimizer=5)

        # Initialize seed
        self.state_seed = state_seed
        # attributes to store initial dataset
        self.X_initial = X_initial
        self.y_initial = y_initial
        # Initialize the number of initial points
        self.n_init = n_init
        # new attribute to store history
        self.history = {'X': [], 'y': [], 'gp_models': []}  # add 'gp_models' to the history
        # Define Aquisition function that should be used 
        self.acquisition_type = acquisition_type

    def initialize(self):
        # Check if the history is empty
        if not self.history['X'] and not self.history['y']:
            # If it is, check if an initial dataset was provided
            if self.X_initial is not None and self.y_initial is not None:
                # If so, use it as the initial dataset
                X = self.X_initial
                y = self.y_initial
            else:
                # If not, generate a random initial dataset
                #print('Quatsch')
                # current_time = time.time()  # Get the current time in seconds
                # milliseconds = round(current_time * 1000) % 1000
                np.random.seed(self.state_seed) # Set a different seed each time
                X = np.concatenate([np.random.uniform(*self.domain[var_name]['domain'], size=(self.n_init, 1)) for var_name in self.domain], axis=1)
                #print(f'first point: {X}')
                y = self.objective_function(X)
        else:
            # If the history is not empty, use the last point in the history as the initial dataset
            X = self.history['X'][-1]
            y = self.history['y'][-1]
        return X, y

    def _get_acquisition_function(self):
        if self.acquisition_type == 'UCB':
            return self.acquisition_function_UCB
        elif self.acquisition_type == 'EI':
            return self.acquisition_function_EI
        elif self.acquisition_type == 'PI':
            return self.acquisition_function_PI
        else:
            raise ValueError("Invalid acquisition type specified.")
    
    def save_state(self, iteration):
        # Save the pc_BO_TS object
        print(self.history['X'])
        print(self.history['y'])
        print(self.history['gp_models'])
        print('here we start showing length of X, y, gp_models')
        print(len(self.history['X']))
        print(len(self.history['y']))
        print(len(self.history['gp_models']))
        with open(f'pc_BO_TS_state_{iteration}.pkl', 'wb') as f:
            pickle.dump(self, f)

        # Save the history of observations to a CSV file
        df = pd.DataFrame(self.history)
        df.to_csv(f'pc_BO_TS_history_{iteration}.csv', index=False)

    def get_function_values(self, proposed_points):
        # Print all proposed points
        for i, point in enumerate(proposed_points):
            print(f'Point {i}: {point}')
        print('Please test the function at the above points.')

        # Initialize an empty array to store the function values
        y_next = np.empty(proposed_points.shape[0])
        # For each proposed point
        for i in range(proposed_points.shape[0]):
            # Prompt the user to enter the function value
            y_next[i] = float(input(f'Enter the function value for point {i}: '))
        return y_next

    def sample_GP_conditioned_on_fixed_vars(self, fixed_vars):
        # generate test inputs for fixed variables and varying the others
        X_test = self._generate_X_test(fixed_vars)
        # generate samples from the GP model
        np.random.seed(0)
        #! We need to update the number of samples: n_samples=B-1? 
        # samples = self.gp_model.sample_y(X_test, n_samples=3)
        samples = self.gp_model.sample_y(X_test, n_samples=self.B-1)
        return samples, X_test

    # modified function
    def _generate_X_test(self, fixed_vars_dict):
        """
        Generate the design space for GP predictions, considering the fixed variables passed as a dictionary.
        """
        # Extract fixed variables values from the dictionary
        fixed_vars = np.array([fixed_vars_dict[var_name] for var_name, var_info in self.domain.items() if var_info['type'] == 'constrained'])
        
        # Determine the number of fixed and varying dimensions
        #num_fixed_dims = fixed_vars.shape[0]
        #total_dims = len(self.domain)
        #num_varying_dims = total_dims - num_fixed_dims

        # Create a grid of values for each varying variable within its domain
        grids = np.meshgrid(*[np.linspace(var_info['domain'][0], var_info['domain'][1], 10) 
                            for var_name, var_info in self.domain.items() if var_info['type'] == 'unconstrained'])
        X_test_varying = np.hstack([grid.reshape(-1, 1) for grid in grids])

        # Repeat the fixed variables values for the entire design space
        fixed_vars_values = np.repeat(fixed_vars.reshape(1, -1), X_test_varying.shape[0], axis=0)

        # Concatenate the fixed and varying parts
        X_test = np.hstack([fixed_vars_values, X_test_varying])
        
        return X_test


    def _generate_full_X_test(self, n_points_per_dimension=25):
        # generate a grid of values for each variable
        grids = np.meshgrid(*[np.linspace(*self.domain[var_name]['domain'], num=n_points_per_dimension) for var_name in self.domain])
        full_X_test = np.hstack([grid.reshape(-1, 1) for grid in grids])
        return full_X_test

    def select_next_point(self):
        # Choose the acquisition function
        acq_function = self._get_acquisition_function()

        # Optimize the acquisition function using the direct algorithm
        x_1 = acq_function()

        # Determine the fixed variables based on the position in the array
        fixed_vars = {self.constrained_vars[i]: x_1[i] for i in range(len(self.constrained_vars))}
        proposed_points = np.array([x_1])

        # Sample function instances at the fixed variables and propose new points
        f_samples, X_test = self.sample_GP_conditioned_on_fixed_vars(fixed_vars)

        # Maximize the function samples to get the proposed point
        proposed_point_indices = np.argmax(f_samples, axis=0)
        for idx in proposed_point_indices:
            proposed_point_x_b = X_test[idx]
            # Concatenate the proposed point to the existing array
            proposed_points = np.vstack([proposed_points, proposed_point_x_b])

        return proposed_points


    def optimize(self, max_iter):
        # Optimization function for simulations, i.e. convergence analysis
        # start with initial observation
        X, y = self.initialize()
        print(f'starting point {X}')
        # Store initial points 
        self.history['X'].append(X)
        self.history['y'].append(y)

        for t in range(max_iter):
            # Update the GP model with the current observation
            self.gp_model.fit(X, y)

            # Add the current GP model to the history
            self.history['gp_models'].append(copy.deepcopy(self.gp_model))

            # Select the next point to sample the objective function at
            proposed_points = self.select_next_point()
            #print(f'Iteration {t} Method TS')

            # Sample the objective function at the chosen point
            y_next = np.array([self.objective_function(point.reshape(1, -1)) for point in proposed_points]).reshape(-1)

            # Add to the history
            self.history['X'].append(proposed_points)
            self.history['y'].append(y_next)

            # Update the history of observations
            X = np.vstack([X, proposed_points])
            y = np.concatenate([y, y_next], axis=0)
        # if self.state_seed==1:
        #     print(f'Those are the sampled data X for pc-BO-TS: {X}\n')

        # Return the best observed point and its corresponding output
        best_idx = np.argmax(y)
        return X[best_idx], y[best_idx]
        
    def optimize_iterative(self, max_iter):
        # Optimization procedure for the interactive case
        # start with initial observation
        X, y = self.initialize()

        # Store initial points
        self.history['X'].append(X)
        self.history['y'].append(y)

        for t in range(max_iter):
            # Update the GP model with the current observation
            self.gp_model.fit(X, y)

            # Add the current GP model to the history
            self.history['gp_models'].append(copy.deepcopy(self.gp_model))

            self.save_state(t)

            # Select the next point to sample the objective function at
            proposed_points = self.select_next_point()

            # Sample the objective function at the chosen point 
            # TODO: Make clear that reshape was added during optimization procedure 
            y_next = self.get_function_values(proposed_points).reshape(-1,1)

            # Add to the history
            self.history['X'].append(proposed_points)
            self.history['y'].append(y_next)

            # Update the history of observations
            X = np.vstack([X, proposed_points])
            y = np.concatenate([y, y_next], axis=0)

        # Return the best observed point and its corresponding output
        best_idx = np.argmax(y)
        return X[best_idx], y[best_idx]

    def acquisition_function_UCB(self, kappa=2.0):
        def neg_UCB(x):
            x = x.reshape(1, -1)  # reshape for the GP model's predict method
            mu, std = self.gp_model.predict(x, return_std=True)
            return -(mu + kappa * std)

        bounds = [(var_info['domain'][0], var_info['domain'][1]) for var_name, var_info in self.domain.items()]
        result = direct(neg_UCB, bounds, len_tol=1e-3)
        x_next = result.x
        return x_next

    def acquisition_function_EI(self):
        f_best = np.max(np.concatenate(self.history['y']))
        def neg_EI(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp_model.predict(x, return_std=True)
            with np.errstate(divide='warn'):
                Z = (mu - f_best) / sigma
                ei = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            return -ei

        bounds = [(var_info['domain'][0], var_info['domain'][1]) for var_name, var_info in self.domain.items()]
        result = direct(neg_EI, bounds)
        x_next = result.x
        return x_next

    def acquisition_function_PI(self):
        f_best = np.max(np.concatenate(self.history['y']))
        def neg_PI(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp_model.predict(x, return_std=True)
            with np.errstate(divide='warn'):
                Z = (mu - f_best) / sigma
                pi = norm.cdf(Z)
                pi[sigma == 0.0] = 0.0
            return -pi

        bounds = [(var_info['domain'][0], var_info['domain'][1]) for var_name, var_info in self.domain.items()]
        result = direct(neg_PI, bounds)
        x_next = result.x
        return x_next





# if __name__ == "__main__":
# #     with open('gp_experiment.pkl', 'rb') as f:
# #         gp_experiment = pickle.load(f)


# #     # Objective test function 
# #     def gp_experiment_test(X):
# #         """Test Function using Gaussian Process Experiment"""
# #         y = gp_experiment.predict(X)
# #         return y

# #     # Define Domain
# #     domain = [{'name': 'FIC_110_SP', 'type': 'constrained', 'domain':(0,55)},
# #             {'name': 'Reactor_Temperature_SP', 'type': 'unconstrained', 'domain':(520,590)}]
# #     # bo_test_gp_experiment = BO_TS_constrained.pc_BO_TS(objective_function=gp_experiment_test, domain=domain, B=4, n_init=1, kernel=kernel)
# #     bo_test_gp_experiment = pc_BO_TS(objective_function=gp_experiment_test, domain=domain, B=4, n_init=1, kernel=None, acquisition_type='EI')
# #     best_x, best_y = bo_test_gp_experiment.optimize(max_iter=15)

# #     # Define the bounds for the optimization as required by the differential_evolution function
# #     bounds = [domain_item['domain'] for domain_item in domain]

# #     # Transform the gp_experiment function to work with 1D arrays instead of 2D arrays
# #     def func(params):
# #         return -gp_experiment_test(params.reshape(1, -1))  # use -gp_experiment_test to maximize instead of minimize

# #     # Use differential evolution to find the global optimum
# #     result = differential_evolution(func, bounds)

# #     #print(f"Global maximum according to differential evolution: {result.fun}, at parameters: {result.x}")

# #     # Now compare with the results from the Bayesian optimization
# #     print(f"Best parameters according to Bayesian optimization: {best_x}")
# #     print(f"Best value according to Bayesian optimization: {best_y}")

#     ############################-------4D------#####################################
#     ########################################################################
#     ########################################################################

    # def rosenbrock_4d_modified(X):
    #     """4D Rosenbrock function modified to accept 2D array input."""
    #     # Ensure input is a numpy array
    #     X = np.asarray(X)
        
    #     # Calculate Rosenbrock value for each row/sample
    #     values = [sum(100.0 * (x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0 for i in range(3)) for x in X]
        
    #     return -1 * np.array(values)

    # def rosenbrock_4d_modified(X):
    #     """4D Rosenbrock function modified to accept both 1D and 2D array input."""
    #     # Ensure input is a numpy array
    #     X = np.asarray(X)
        
    #     # If 1D input, reshape to 2D
    #     if X.ndim == 1:
    #         X = X.reshape(1, -1)
        
    #     # Calculate Rosenbrock value for each row/sample
    #     values = [sum(100.0 * (x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0 for i in range(3)) for x in X]
        
    #     return -1 * np.array(values)

    # # Define Domain
    # domain = [{'name': 'x1', 'type': 'constrained', 'domain':(-2,2)},
    #         {'name': 'x2', 'type': 'unconstrained', 'domain':(-2,2)},
    #         {'name': 'x3', 'type': 'unconstrained', 'domain':(-2,2)},
    #         {'name': 'x4', 'type': 'unconstrained', 'domain':(-2,2)}]

    # bo_test_gp_experiment = pc_BO_TS(objective_function=rosenbrock_4d_modified, domain=domain, B=32, n_init=1, kernel=None, acquisition_type='UCB')

    # # # profiler = cProfile.Profile()
    # # # profiler.enable()
    # # # #bo_test_gp_experiment._generate_X_test()

    # # profiler = LineProfiler()
    # # profiler.add_function(pc_BO_TS.acquisition_function_UCB)


    # # profiler.enable()  # Start profiling
    # best_x, best_y = bo_test_gp_experiment.optimize(max_iter=10)
    # # profiler.disable()  # Stop profiling

    # # profiler.print_stats()
    # # # profiler.disable()
    # # # profiler.dump_stats("profile_optimization.prof")

    # # # stats = pstats.Stats("profile_optimization.prof")
    # # # stats.sort_stats(pstats.SortKey.TIME)
    # # # stats.print_stats()

    # # Define the bounds for the optimization as required by the differential_evolution function
    # bounds = [domain_item['domain'] for domain_item in domain]

    # def func(params):
    #     return -1 * rosenbrock_4d_modified(params.reshape(1, -1))  # use -gp_experiment_test to maximize instead of minimize

    # # Use differential evolution to find the global optimum
    # result = differential_evolution(func, bounds)
    # print(f"Global maximum according to differential evolution: {result.fun}, at parameters: {result.x}")

    # # Now compare with the results from the Bayesian optimization
    # print(f"Best parameters according to Bayesian optimization: {best_x}")
    # print(f"Best value according to Bayesian optimization: {best_y}")

#     ############################-------6D------#####################################
#     ########################################################################
#     ########################################################################
    
    # def rosenbrock_6d_modified(X):
    #     """6D Rosenbrock function modified to accept both 1D and 2D array input."""
    #     X = np.asarray(X)
    #     if X.ndim == 1:
    #         X = X.reshape(1, -1)
    #     values = [sum(100.0 * (x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0 for i in range(5)) for x in X]
    #     return -1 * np.array(values)


    # # Define Domain
    # domain = [{'name': 'x1', 'type': 'constrained', 'domain':(-2,2)},
    #         {'name': 'x2', 'type': 'constrained', 'domain':(-2,2)},
    #         {'name': 'x3', 'type': 'constrained', 'domain':(-2,2)},
    #         {'name': 'x4', 'type': 'unconstrained', 'domain':(-2,2)},
    #         {'name': 'x5', 'type': 'unconstrained', 'domain':(-2,2)},
    #         {'name': 'x6', 'type': 'unconstrained', 'domain':(-2,2)}]
    
    # bo_test_gp_experiment = pc_BO_TS(objective_function=rosenbrock_6d_modified, domain=domain, B=4, n_init=1, kernel=None, acquisition_type='UCB')
    
    # bounds = [domain_item['domain'] for domain_item in domain]

    # def func(params):
    #     return -1 * rosenbrock_6d_modified(params.reshape(1, -1))

    # result = differential_evolution(func, bounds)
    # print(f"Global maximum according to differential evolution: {result.fun}, at parameters: {result.x}")

    # best_x, best_y = bo_test_gp_experiment.optimize(max_iter=10)
    # print(f"Best parameters according to Bayesian optimization: {best_x}")
    # print(f"Best value according to Bayesian optimization: {best_y}")


#################################################### -- 8D -- ####################################################
##################################################################################################################
    
    # def rosenbrock_8d_modified(X):
    #     """8D Rosenbrock function modified to accept both 1D and 2D array input."""
    #     # Ensure input is a numpy array
    #     X = np.asarray(X)
        
    #     # If 1D input, reshape to 2D
    #     if X.ndim == 1:
    #         X = X.reshape(1, -1)
        
    #     # Calculate Rosenbrock value for each row/sample
    #     values = [sum(100.0 * (x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0 for i in range(7)) for x in X]
        
    #     return -1 * np.array(values)

    # # Define Domain for 8D
    # domain = [{'name': 'x1', 'type': 'constrained', 'domain':(-2,2)},
    #         {'name': 'x2', 'type': 'constrained', 'domain':(-2,2)},
    #         {'name': 'x3', 'type': 'constrained', 'domain':(-2,2)},
    #         {'name': 'x4', 'type': 'constrained', 'domain':(-2,2)},
    #         {'name': 'x5', 'type': 'unconstrained', 'domain':(-2,2)},
    #         {'name': 'x6', 'type': 'unconstrained', 'domain':(-2,2)},
    #         {'name': 'x7', 'type': 'unconstrained', 'domain':(-2,2)},
    #         {'name': 'x8', 'type': 'unconstrained', 'domain':(-2,2)}]

    # bo_test_gp_experiment = pc_BO_TS(objective_function=rosenbrock_8d_modified, domain=domain, acquisition_type="UCB", B=4, n_init=1, kernel=None)
    # best_x, best_y = bo_test_gp_experiment.optimize(max_iter=10)

    # # Define the bounds for the optimization as required by the differential_evolution function
    # bounds = [domain_item['domain'] for domain_item in domain]

    # def func(params):
    #     return -1 * rosenbrock_8d_modified(params.reshape(1, -1))

    # # Use differential evolution to find the global optimum
    # result = differential_evolution(func, bounds)

    # print(f"Global maximum according to differential evolution: {result.fun}, at parameters: {result.x}")
    # # Now compare with the results from the Bayesian optimization
    # print(f"Best parameters according to Bayesian optimization: {best_x}")
    # print(f"Best value according to Bayesian optimization: {best_y}")

