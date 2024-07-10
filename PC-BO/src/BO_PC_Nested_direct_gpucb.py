import numpy as np
import pandas as pd
import sklearn.gaussian_process.kernels as sklearnGP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
import matplotlib.pyplot as plt

import time
from scipy.optimize import differential_evolution, direct
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import pickle
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import copy
import numpy as np

class pc_BO_nested:
    # def __init__(self, objective_function, domain, B, kernel=None, n_init=3, X_initial=None, y_initial=None, gp_model_outer=None, gp_model_inner=None, visualize_virtual_gp=False, optimize_kernel=True):
    def __init__(self, objective_function, domain, B, state_seed=None, acquisition_type="UCB", delta=0.1, 
                 kernel=None, n_init=3, X_initial=None, y_initial=None, gp_model_outer=None, 
                 gp_model_inner=None, visualize_virtual_gp=False, optimize_kernel=True):
        # Initialize the objective function, domain, batch size, kernel, and number of initial points
        self.objective_function = objective_function
        self.domain = {d['name']: d for d in domain}
        self.constrained_vars = [name for name, var in self.domain.items() if var['type'] == 'constrained']
        self.unconstrained_vars = [name for name, var in self.domain.items() if var['type'] == 'unconstrained']
        self.B = B 
        self.kernel = kernel if kernel is not None else Matern(nu=2.5)
        
        # Initialize two GP models: one for the outer stage and one for the inner stage
        self.gp_model_outer = gp_model_outer if gp_model_outer is not None else GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5 if optimize_kernel else 0,
            optimizer='fmin_l_bfgs_b' if optimize_kernel else None)
        self.gp_model_inner = gp_model_inner if gp_model_inner is not None else GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5 if optimize_kernel else 0,
            optimizer='fmin_l_bfgs_b' if optimize_kernel else None)
            
        self.n_init = n_init
        # set the random seed
        self.state_seed = state_seed

        # Initialize initial dataset
        self.X_initial = X_initial
        self.y_initial = y_initial

        # Initialize separate histories for the outer and inner datasets
        self.history_outer = {'X': [], 'y': [], 'gp_models': []}
        self.history_inner = {'X': [], 'y': [], 'gp_models': []}

        # Initialize inner loop flag 
        self.visualize_virtual_gp_flag = visualize_virtual_gp
        # GP_UCB 
        self.acquisition_type = acquisition_type  
        self.delta = delta  



    
    def save_state(self, iteration):
        # Save the state of the pc_BO_nested object to a pickle file
        with open(f'pc_BO_nested_state_{iteration}.pkl', 'wb') as f:
            pickle.dump(self, f)

        # Save the history of observations to pickle files
        with open(f'pc_BO_nested_history_outer_{iteration}.pkl', 'wb') as f:
            pickle.dump(self.history_outer, f)

        with open(f'pc_BO_nested_history_inner_{iteration}.pkl', 'wb') as f:
            pickle.dump(self.history_inner, f)

    def create_virtual_gp(self):
        # Create a new GP model with the same kernel parameters as the original model
        virtual_gp_model = GaussianProcessRegressor(
            kernel=self.gp_model_inner.kernel_,
            alpha=1e-6,
            normalize_y=True,
            optimizer=None)
        return virtual_gp_model

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

    def initialize(self):
        # Check if the history is empty
        if not self.history_outer['X'] and not self.history_outer['y']:
            # If it is, check if an initial dataset was provided
            if self.X_initial is not None and self.y_initial is not None:
                # If so, use it as the initial dataset
                X = self.X_initial
                y = self.y_initial
            else:
                # If not, generate a random initial dataset
                # current_time = time.time()  # Get the current time in seconds
                # milliseconds = round(current_time * 1000) % 1000
                # np.random.seed(int(milliseconds))  # Set a different seed each time
                np.random.seed(self.state_seed)
                X = np.concatenate([np.random.uniform(*self.domain[var_name]['domain'], size=(self.n_init, 1)) for var_name in self.domain], axis=1)
                y = self.objective_function(X)
        else:
            # If the history is not empty, use the last point in the history as the initial dataset
            X = self.history_outer['X'][-1]
            y = self.history_outer['y'][-1]
        # Split the samples into constrained and unconstrained variables
        X_c = X[:, :len(self.constrained_vars)]
        X_uc = X[:, len(self.constrained_vars):]
        return X_c, y, X_uc, y

    def initialize_iterative(self):
        # Check if an initial dataset was provided
        if self.X_initial is not None and self.y_initial is not None:
            # If so, use it as the initial dataset
            X = self.X_initial
            y = self.y_initial
        else:
            # If not, generate a random initial dataset
            np.random.seed(self.state_seed)
            X = np.concatenate([np.random.uniform(*self.domain[var_name]['domain'], size=(self.n_init, 1)) for var_name in self.domain], axis=1)
            y = self.objective_function(X)
        
        # Split the samples into constrained and unconstrained variables
        X_c = X[:, :len(self.constrained_vars)]
        X_uc = X[:, len(self.constrained_vars):]

        # Find unique rows in X_c and their indices in the original array
        X_c_unique, indices = np.unique(X_c, axis=0, return_inverse=True)

        # For each unique row in X_c, find the maximum y value among the corresponding rows in y
        y_outer = np.array([np.max(y[indices == i]) for i in range(X_c_unique.shape[0])])

        return X_c_unique, y_outer, X_uc, y, X, y
    
    @staticmethod
    def GP_UCB(mu, sigma, t, d, v=1, delta=0.1):
        """
        GP-UCB acquisition function.
        
        :param mu: Mean of the GP prediction.
        :param sigma: Standard deviation of the GP prediction.
        :param t: Current iteration.
        :param d: Dimensionality of the input space.
        :param v: Hyperparameter v for GP-UCB.
        :param delta: Hyperparameter delta for GP-UCB.
        :return: GP-UCB acquisition value.
        """
        kappa = np.sqrt(v * (2 * np.log((t**(d/2. + 2))*(np.pi**2)/(3. * delta))))
        return mu + kappa * sigma


    def acquisition_function_UCB_inner(self, fixed_vars, kappa=2.0):
        """
        Acquisition function for the inner stage.
        
        :param fixed_vars: Fixed values for constrained variables.
        :param kappa: Exploration-exploitation trade-off parameter for UCB.
        :return: The unconstrained input that maximizes the acquisition function.
        """
        def neg_acquisition_function(x_uc):
            x_uc = x_uc.reshape(1, -1)
            fixed_vars_array = np.array([fixed_vars[var_name] for var_name in self.constrained_vars]).reshape(1, -1)
            x_full = np.hstack([fixed_vars_array, x_uc])
            mu, std = self.gp_model_inner.predict(x_full, return_std=True)
            mu = np.squeeze(mu)
            std = np.squeeze(std)

            if self.acquisition_type == "UCB":
                return -(mu + kappa * std).item()
            elif self.acquisition_type == "GP-UCB":
                t = len(self.history_outer['y'])
                d = x_uc.shape[1]
                return -self.GP_UCB(mu, std, t, d, delta=self.delta).item()

        # Bounds only for the unconstrained space
        bounds = self._get_bounds("inner")

        # Convert the fixed_vars dictionary to an ordered numpy array
        fixed_vars_array = np.array([fixed_vars[var_name] for var_name in self.constrained_vars]).reshape(1, -1)
        
        # Use direct to maximize the negative UCB
        result = direct(neg_acquisition_function, bounds, len_tol=1e-3)
        

        # Combine the fixed values of the constrained variables with the result to get the full point
        combined_result = np.hstack([fixed_vars_array, result.x.reshape(1,-1)])
        
        return combined_result

    def acquisition_function_UCB_outer(self, iteration, kappa=2.0):
        """
        Acquisition function for the outer stage.
        
        :param kappa: Exploration-exploitation trade-off parameter for UCB.
        :return: The constrained input that maximizes the acquisition function.
        """
        def neg_acquisition_function(x_c):
            x_c = x_c.reshape(1, -1)
            mu, std = self.gp_model_outer.predict(x_c, return_std=True)
            mu = np.squeeze(mu)
            std = np.squeeze(std)

            if self.acquisition_type == "UCB":
                return -(mu + kappa * std).item()
            elif self.acquisition_type == "GP-UCB":
                t = len(self.history_outer['y'])
                d = x_c.shape[1]
                return -self.GP_UCB(mu, std, t, d, delta=self.delta).item()

        # Bounds only for the constrained space
        bounds = self._get_bounds("outer")

        # Use direct to maximize the negative UCB
        result = direct(neg_acquisition_function, bounds, len_tol=1e-3)
        succeded = result.success
        print(f'Direct at iteration {iteration} succeded {succeded}')
        
        return result.x

    def _get_bounds(self, stage):
        if stage == "outer":
            return [(var_info['domain'][0], var_info['domain'][1]) for var_name, var_info in self.domain.items() if var_info['type'] == 'constrained']
        elif stage == "inner":
            return [(var_info['domain'][0], var_info['domain'][1]) for var_name, var_info in self.domain.items() if var_info['type'] == 'unconstrained']


    def _generate_X_test_constrained(self, n_points_per_dimension=100):
        """
        Generate a grid of values for each constrained variable within its domain.
        """
        # Create a grid of values for each constrained variable
        grids = np.meshgrid(*[np.linspace(*self.domain[var_name]['domain'], num=n_points_per_dimension) for var_name in self.constrained_vars])
        
        # Transform grids to a 2D array
        X_test = np.hstack([grid.reshape(-1, 1) for grid in grids])
        
        return X_test

    def _generate_X_test_unconstrained(self, fixed_vars, n_points_per_dimension=100):
        """
        Generate the design space for GP predictions, considering the fixed variables passed as a dictionary.
        """
        # Extract fixed variables values from the dictionary
        fixed_vars_values = np.array([fixed_vars[var_name] for var_name in self.constrained_vars]).reshape(1, -1)
        
        # Create a grid of values for each unconstrained variable within its domain
        grids = np.meshgrid(*[np.linspace(*self.domain[var_name]['domain'], num=n_points_per_dimension) for var_name in self.unconstrained_vars])
        
        # Transform grids to a 2D array
        X_test_varying = np.hstack([grid.reshape(-1, 1) for grid in grids])

        # Repeat the fixed variables values for the entire design space
        fixed_vars_repeated = np.repeat(fixed_vars_values, X_test_varying.shape[0], axis=0)
        
        # Concatenate the fixed and varying parts
        X_test = np.hstack([fixed_vars_repeated, X_test_varying])
        
        return X_test


    def select_next_points_inner(self, fixed_vars):
        # Select the first unconstrained variable using the acquisition function
        x_uc_t_0 = self.acquisition_function_UCB_inner(fixed_vars)

        proposed_points = np.array(x_uc_t_0)

        X_history_iter_t = np.concatenate(self.history_inner['X'], axis=0)
        y_history_iter_t = np.concatenate(self.history_inner['y'], axis=0)
        y_history_iter_t = y_history_iter_t.reshape(len(y_history_iter_t), 1)
        
        if self.visualize_virtual_gp_flag:
            self.visualize_inner_gp(fixed_vars, proposed_points)
        
        for b in range(2, self.B + 1):
            # "Halucinate" the observations at proposed points
            mu = self.gp_model_inner.predict(proposed_points).reshape(-1, 1)

            X_halu = np.vstack([X_history_iter_t, proposed_points])
            y_halu = np.vstack([y_history_iter_t, mu])

            # Create a new virtual GP model
            virtual_gp_model = self.create_virtual_gp()

            virtual_gp_model.fit(X_halu, y_halu)
            #print(virtual_gp_model.kernel_)

            if self.visualize_virtual_gp_flag:
                self.visualize_virtual_gp(virtual_gp_model, fixed_vars, b)

            x_uc_b = self.maximize_posterior_variance(fixed_vars, virtual_gp_model)
            #proposed_points = np.vstack([proposed_points, np.array([x_uc_b])])
            proposed_points = np.vstack([proposed_points, x_uc_b])
        return proposed_points


    def maximize_posterior_variance(self, fixed_vars, virtual_gp_model):
        """
        Find the point in the unconstrained space that maximizes the 
        posterior variance of the virtual GP model using DIRECT optimization.
        """
        
        def neg_variance(x_uc, *args):
            """
            Compute the negative predictive variance for a given point 
            in the unconstrained space.
            """
            x_uc = x_uc.reshape(1, -1)
            fixed_vars_array = np.array([fixed_vars[var_name] for var_name in self.constrained_vars]).reshape(1, -1)
            x_full = np.hstack([fixed_vars_array, x_uc])
            _, std = virtual_gp_model.predict(x_full, return_std=True)
            return -std.item()
        
        bounds = self._get_bounds("inner")
        
        # Utilize DIRECT method for optimization
        result = direct(neg_variance, bounds, maxiter=1000)
        
        # Extract the optimal point from the result
        optimal_x_uc = result.x
        
        # Combine the fixed values of the constrained variables with the optimal unconstrained values
        fixed_vars_array = np.array([fixed_vars[var_name] for var_name in self.constrained_vars]).reshape(1, -1)
        optimal_x_full = np.hstack([fixed_vars_array, optimal_x_uc.reshape(1, -1)])
        
        return optimal_x_full

    def select_next_batch(self, iteration):
        # Select the constrained variables in the outer stage
        x_c_t = self.acquisition_function_UCB_outer(iteration)
        fixed_vars = {self.constrained_vars[i]: x_c_t[i] for i in range(len(self.constrained_vars))}

        #print(f'x_c_t: {x_c_t}')

        # Select the unconstrained variables in the inner stage
        proposed_points = self.select_next_points_inner(fixed_vars)

        return proposed_points

    def optimize(self, max_iter):
        # Initialize the model with random samples
        X_c, y, X_uc, _ = self.initialize()

        #print(f'X_c: {X_c}')
        #print(f'X_uc: {X_uc}')
        
        # Store the initial points in the history
        self.history_outer['X'].append(X_c)
        self.history_outer['y'].append(y)
        self.history_inner['X'].append(np.hstack([X_c, X_uc]))
        self.history_inner['y'].append(y)
        
        # Iterate for the maximum number of iterations
        for t in range(max_iter):
            # Update the GP models with the current observations
            y_outer = np.concatenate(self.history_outer['y'], axis=0)
            self.gp_model_outer.fit(X_c, y_outer)
            X_inner = np.concatenate(self.history_inner['X'], axis=0)
            self.gp_model_inner.fit(X_inner, y)

            # Store the current GP models in the history
            self.history_outer['gp_models'].append(copy.deepcopy(self.gp_model_outer))
            self.history_inner['gp_models'].append(copy.deepcopy(self.gp_model_inner))
            
            # Select the next batch of points to sample the objective function at
            proposed_points = self.select_next_batch(t)
            print(f'Iteration {t} Method Nested')
            
            # Sample the objective function at the chosen points
            y_next = np.array([self.objective_function(point.reshape(1, -1)) for point in proposed_points]).reshape(-1)
            
            # Find the point that maximizes the function
            max_idx = np.argmax(y_next)
            max_point = proposed_points[max_idx]
            max_y = y_next[max_idx]

            # Add the proposed point that maximizes the function and its corresponding output to the outer history
            self.history_outer['X'].append(max_point[:len(self.constrained_vars)])
            self.history_outer['y'].append(np.array([max_y]))

            # Add all the proposed points and their corresponding outputs to the inner history
            self.history_inner['X'].append(proposed_points)
            self.history_inner['y'].append(y_next)
            
            # Update the history of observations
            X_c = np.vstack([X_c, max_point[:len(self.constrained_vars)]])
            y = np.concatenate([y, y_next])
            X_uc = np.vstack([X_uc, max_point[len(self.constrained_vars):]])
        
        # Find the best observed point and its corresponding output
        best_idx_outer = np.argmax(self.history_outer['y'])
        best_X_c = np.array(self.history_outer['X'][best_idx_outer]).reshape(1, -1)
        # Find the index of the best objective function value within the array in history_inner['y'][best_idx_outer]
        best_idx_inner = np.argmax(self.history_inner['y'][best_idx_outer])
        best_X_uc = np.array(self.history_inner['X'][best_idx_outer][best_idx_inner][len(self.constrained_vars):]).reshape(1, -1)
        xinn = self.history_inner['X']
        xout = self.history_outer['X']
        # if self.state_seed==1:
        #     print(f'inner points: {xinn}')
        #     print(f'outer points: {xout}')
        # X_final = np.concatenate([self.history_inner['X'], self.history_outer['X']], axis=1)
        # print(f'Method pc-BO-Nested-{self.acquisition_type}: {X_final}')

        return np.concatenate([best_X_c, best_X_uc], axis=1), self.history_outer['y'][best_idx_outer]

    def optimize_iterative_test(self, max_iter):
        # Initialize the model with random samples
        X_c, y_outer, X_uc, y, X, _ = self.initialize_iterative()

        # Store the initial points in the history
        self.history_outer['X'].append(X_c)
        
        # TODO: tranfrom this somehow into a helper function append_outer_y or something
        for y_value in y_outer:
            self.history_outer['y'].append(np.array([y_value]).reshape(-1, 1))

        self.history_inner['X'].append(X)
        self.history_inner['y'].append(y)

        for t in range(max_iter):
            # Update the GP models with the current observation
            y_outer = np.concatenate(self.history_outer['y'], axis=0)
            self.gp_model_outer.fit(X_c, y_outer)
            X_inner = np.concatenate(self.history_inner['X'], axis=0)
            self.gp_model_inner.fit(X_inner, y)

            # Store the current GP models in the history
            self.history_outer['gp_models'].append(copy.deepcopy(self.gp_model_outer))
            self.history_inner['gp_models'].append(copy.deepcopy(self.gp_model_inner))


            # Select the next batch of points to sample the objective function at
            proposed_points = self.select_next_batch()

            # Sample the objective function at the chosen points
            y_next = self.get_function_values(proposed_points).reshape(-1,1)

            # Find the point that maximizes the function
            max_idx = np.argmax(y_next)
            max_point = proposed_points[max_idx]
            max_y = y_next[max_idx]

            # Add the proposed point that maximizes the function and its corresponding output to the outer history
            self.history_outer['X'].append(max_point[:len(self.constrained_vars)])
            self.history_outer['y'].append(np.array([max_y]).reshape(-1, 1))

            # Add all the proposed points and their corresponding outputs to the inner history
            self.history_inner['X'].append(proposed_points)
            self.history_inner['y'].append(y_next)

            # Update the history of observations
            X_c = np.vstack([X_c, max_point[:len(self.constrained_vars)]])
            
            # print(f'{y}')
            # print(f'{y_next}')
            y = np.concatenate([y, y_next])
            X_uc = np.vstack([X_uc, max_point[len(self.constrained_vars):]])
            
            # Save the state at the end of each iteration
            self.save_state(t)


        # Find the best observed point and its corresponding output
        best_idx_outer = np.argmax(self.history_outer['y'])
        best_X_c = np.array(self.history_outer['X'][best_idx_outer]).reshape(1, -1)
        # Find the index of the best objective function value within the array in history_inner['y'][best_idx_outer]
        best_idx_inner = np.argmax(self.history_inner['y'][best_idx_outer])
        best_X_uc = np.array(self.history_inner['X'][best_idx_outer][best_idx_inner][len(self.constrained_vars):]).reshape(1, -1)

        return np.concatenate([best_X_c, best_X_uc], axis=1), self.history_outer['y'][best_idx_outer]
    

# if __name__ == "__main__":

    #################################################### -- 2D -- ####################################################
    ##################################################################################################################
    # with open('../methods_testing/pc-BO-nested/gp_experiment.pkl', 'rb') as f:
    #     gp_experiment = pickle.load(f)


    # # Objective test function 
    # def gp_experiment_test(X):
    #     """Test Function using Gaussian Process Experiment"""
    #     y = gp_experiment.predict(X)
    #     return y

    # # Define Domain
    # domain = [{'name': 'FIC_110_SP', 'type': 'constrained', 'domain':(0,55)},
    #         {'name': 'Reactor_Temperature_SP', 'type': 'unconstrained', 'domain':(520,590)}]
    # # bo_test_gp_experiment = BO_TS_constrained.pc_BO_TS(objective_function=gp_experiment_test, domain=domain, B=4, n_init=1, kernel=kernel)
    # bo_test_gp_experiment = pc_BO_nested(objective_function=gp_experiment_test, domain=domain, acquisition_type="GP-UCB", B=4, n_init=1, kernel=None)
    # best_x, best_y = bo_test_gp_experiment.optimize(max_iter=15)

    # # Define the bounds for the optimization as required by the differential_evolution function
    # bounds = [domain_item['domain'] for domain_item in domain]

    # # Transform the gp_experiment function to work with 1D arrays instead of 2D arrays
    # def func(params):
    #     return -gp_experiment_test(params.reshape(1, -1))  # use -gp_experiment_test to maximize instead of minimize

    # # Use differential evolution to find the global optimum
    # result = differential_evolution(func, bounds)

    # print(f"Global maximum according to differential evolution: {result.fun}, at parameters: {result.x}")

    # # Now compare with the results from the Bayesian optimization
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
    # domain = [{'name': f'x{i}', 'type': 'constrained' if i < 4 else 'unconstrained', 'domain': (-2, 2)} for i in range(1, 9)]

    # bo_test_gp_experiment = pc_BO_nested(objective_function=rosenbrock_8d_modified, domain=domain, acquisition_type="GP-UCB", B=4, n_init=1, kernel=None)
    # best_x, best_y = bo_test_gp_experiment.optimize(max_iter=30)

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
    
    # bo_test_gp_experiment = pc_BO_nested(objective_function=rosenbrock_6d_modified, domain=domain, B=4, n_init=1, kernel=None, acquisition_type='GP-UCB')
    
    # bounds = [domain_item['domain'] for domain_item in domain]

    # def func(params):
    #     return -1 * rosenbrock_6d_modified(params.reshape(1, -1))

    # result = differential_evolution(func, bounds)
    # print(f"Global maximum according to differential evolution: {result.fun}, at parameters: {result.x}")

    # best_x, best_y = bo_test_gp_experiment.optimize(max_iter=10)
    # print(f"Best parameters according to Bayesian optimization: {best_x}")
    # print(f"Best value according to Bayesian optimization: {best_y}")