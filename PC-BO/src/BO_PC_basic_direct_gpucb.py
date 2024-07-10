from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import copy
import time
import numpy as np
from scipy.optimize import differential_evolution, direct

class pc_BO_basic:
    def __init__(self, objective_function, domain, B, state_seed, kernel=None, n_init=3, X_initial=None, y_initial=None, acquisition_type='UCB'):
        # Initialize the objective function, domain, batch size, kernel, and number of initial points
        self.objective_function = objective_function
        self.domain = {d['name']: d for d in domain}
        self.constrained_vars = [name for name, var in self.domain.items() if var['type'] == 'constrained']
        self.unconstrained_vars = [name for name, var in self.domain.items() if var['type'] == 'unconstrained']
        self.B = B 
        self.kernel = kernel if kernel is not None else Matern(nu=2.5)
        self.gp_model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5)

        self.state_seed = state_seed
        self.X_initial = X_initial
        self.y_initial = y_initial
        self.n_init = n_init
        self.history = {'X': [], 'y': [], 'gp_models': []}
        # Define Aquisition function that should be used 
        self.acquisition_type = acquisition_type

    def initialize(self):
        # Check if an initial dataset was provided
        if self.X_initial is not None and self.y_initial is not None:
            # If so, use it as the initial dataset
            X = self.X_initial
            y = self.y_initial
        else:
            # # If not, generate a random initial dataset
            # current_time = time.time()  # Get the current time in seconds
            # milliseconds = round(current_time * 1000) % 1000
            np.random.seed(self.state_seed)
            X = np.concatenate([np.random.uniform(*self.domain[var_name]['domain'], size=(self.n_init, 1)) for var_name in self.domain], axis=1)
            y = self.objective_function(X)
        return X, y

    # def _get_acquisition_function(self):
    #     if self.acquisition_type == 'UCB':
    #         return self.acquisition_function_UCB
    #     elif self.acquisition_type == 'EI':
    #         return self.acquisition_function_EI
    #     elif self.acquisition_type == 'PI':
    #         return self.acquisition_function_PI
    #     else:
    #         raise ValueError("Invalid acquisition type specified.")

    # Overriding the _get_acquisition_function method to include GP-UCB
    def _get_acquisition_function(self):
        if self.acquisition_type == 'UCB':
            return self.acquisition_function_UCB
        elif self.acquisition_type == 'EI':
            return self.acquisition_function_EI
        elif self.acquisition_type == 'PI':
            return self.acquisition_function_PI
        elif self.acquisition_type == 'GP-UCB':
            return self.acquisition_function_GP_UCB
        else:
            raise ValueError("Invalid acquisition type specified.")

    # modified function
    def _generate_X_test(self, fixed_vars_dict):
        """
        Generate the design space for GP predictions, considering the fixed variables passed as a dictionary.
        """
        # Extract fixed variables values from the dictionary
        fixed_vars = np.array([fixed_vars_dict[var_name] for var_name, var_info in self.domain.items() if var_info['type'] == 'constrained'])

        # Create a grid of values for each varying variable within its domain
        grids = np.meshgrid(*[np.linspace(var_info['domain'][0], var_info['domain'][1], 10) 
                            for var_name, var_info in self.domain.items() if var_info['type'] == 'unconstrained'])
        X_test_varying = np.hstack([grid.reshape(-1, 1) for grid in grids])

        # Repeat the fixed variables values for the entire design space
        fixed_vars_values = np.repeat(fixed_vars.reshape(1, -1), X_test_varying.shape[0], axis=0)

        # Concatenate the fixed and varying parts
        X_test = np.hstack([fixed_vars_values, X_test_varying])
        
        return X_test
    
    def _generate_full_X_test(self, n_points_per_dimension=100):
        # Generate a grid of values for each variable
        grids = np.meshgrid(*[np.linspace(*self.domain[var_name]['domain'], num=n_points_per_dimension) for var_name in self.domain])
        full_X_test = np.hstack([grid.reshape(-1, 1) for grid in grids])
        return full_X_test

    def maximize_posterior_variance(self, fixed_vars, virtual_gp_model):
        # Generate test inputs for fixed variables and varying the others
        X_test = self._generate_X_test(fixed_vars)
        # Predict the standard deviation of the GP model
        _, std = virtual_gp_model.predict(X_test, return_std=True)
        # Select the input that maximizes the standard deviation
        x_next = X_test[np.argmax(std)]
        return x_next
    
    def select_next_batch(self):
        # Check if history is empty and if so, initialize
        if not self.history['X'] or not self.history['y']:
            X, y = self.initialize()
            self.history['X'].append(X)
            self.history['y'].append(y)
            
        # Choose the acquisition function
        acq_function = self._get_acquisition_function()

        # Optimize the acquisition function using the direct algorithm
        x_1 = acq_function()

        # Determine the fixed variables based on the position in the array
        fixed_vars = {self.constrained_vars[i]: x_1[i] for i in range(len(self.constrained_vars))}

        proposed_points = np.array([x_1])
        # Create a copy of the GP model to update with virtual observations
        virtual_gp_model = copy.deepcopy(self.gp_model)
        X_history_iter_t = np.concatenate(self.history['X'], axis=0)
        y_history_iter_t = np.concatenate(self.history['y'], axis=0)
        y_history_iter_t = y_history_iter_t.reshape(len(y_history_iter_t), 1)
        for b in range(2, self.B + 1):
            # "Halucinate" the observations at proposed points
            mu = self.gp_model.predict(proposed_points).reshape(-1, 1)

            X_halu = np.vstack([X_history_iter_t, proposed_points])
            y_halu = np.vstack([y_history_iter_t, mu])
            # Update the virtual GP model with the last proposed point
            virtual_gp_model.fit(X_halu, y_halu)

            # Use the virtual GP model with updated variance to maximize the posterior variance
            x_uc_b = self.maximize_posterior_variance(fixed_vars, virtual_gp_model)
            proposed_points = np.vstack([proposed_points, x_uc_b])
        return proposed_points


    def optimize(self, max_iter):
        # Initialize the model with random samples
        X, y = self.initialize()
        # Store the initial points in the history
        self.history['X'].append(X)
        self.history['y'].append(y)
        
        # Iterate for the maximum number of iterations
        for t in range(max_iter):
            # Update the GP model with the current observations
            self.gp_model.fit(X, y)
            # Store the current GP model in the history
            self.history['gp_models'].append(copy.deepcopy(self.gp_model))
            
            # Select the next batch of points to sample the objective function at
            proposed_points = self.select_next_batch()
            print(f'Iteration {t} Method Basic')
            # Sample the objective function at the chosen points
            y_next = np.array([self.objective_function(point.reshape(1, -1)) for point in proposed_points]).reshape(-1)
            
            # Add the proposed points and their corresponding outputs to the history
            self.history['X'].append(proposed_points)
            self.history['y'].append(y_next)
            
            # Update the history of observations
            X = np.vstack([X, proposed_points])
            y = np.concatenate([y, y_next], axis=0)
        
        # Find the best observed point and its corresponding output
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

    def acquisition_function_GP_UCB(self, v=1, delta=0.1):
        """
        Acquisition function for GP-UCB.
        
        :param v: Hyperparameter v for GP-UCB.
        :param delta: Hyperparameter delta for GP-UCB.
        :return: The input that maximizes the GP-UCB acquisition function.
        """
        def neg_acquisition_function(x):
            x = x.reshape(1, -1)  # reshape for the GP model's predict method
            mu, sigma = self.gp_model.predict(x, return_std=True)
            mu = np.squeeze(mu)
            sigma = np.squeeze(sigma)
            
            # Compute the GP-UCB value
            t = len(np.concatenate(self.history['y']))
            d = x.shape[1]
            return -self.GP_UCB(mu, sigma, t, d, v=v, delta=delta).item()

        bounds = [(var_info['domain'][0], var_info['domain'][1]) for var_name, var_info in self.domain.items()]
        
        # Use direct to maximize the negative GP-UCB
        result = direct(neg_acquisition_function, bounds, len_tol=1e-3)
        
        return result.x