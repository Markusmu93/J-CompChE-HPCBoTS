import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern 
from scipy.optimize import direct, Bounds
from scipy.stats import norm
import matplotlib.pyplot as plt

class HierarchicalPCBO:
    def __init__(self, function, bounds, T, L, K, random_state=None):
        """
        Initializes the HPC-BO optimizer.

        Parameters:
        - function : callable
            The objective function to optimize.
        - bounds : array-like of shape (n_features, 2)
            The domain (min, max) bounds for each dimension of the input space.
        - T : int
            Total number of iterations to run the optimization process.
        - L : int
            Number of levels in the hierarchy.
        - K : list of int
            Batch sizes for each level of the hierarchy.
        - random_state : int, optional
            Seed for the random number generator for reproducibility.
        """
        self.function = function
        self.bounds = np.array(bounds)
        self.T = T
        self.L = L
        self.K = K
        self.random_state = np.random.RandomState(random_state)
        
        # Initialize the Gaussian Process with an appropriate kernel
        kernel = Matern(nu=2.5)
        # self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6, random_state=random_state)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-6)
        # Data lists for storing samples and observations
        self.X_train = []
        self.y_train = []

    def initialize_uniform_x_t_0(self):
        """Initializes the optimizer with a random sample within the bounds."""
        # Sample a random initial point
        initial_point = self.random_state.uniform(self.bounds[:, 0], self.bounds[:, 1])
        # initial_response = self.function(initial_point)

        return initial_point

    def gp_predict(self, x):
        """ Return mean and standard deviation of GP prediction at x. """
        x = np.atleast_2d(x)
        mean, std = self.gp.predict(x, return_std=True)
        return mean.ravel(), std.ravel()

    def current_best(self):
        """ Returns the best observed value. """
        return np.max(self.y_train)

    def ucb(self, x, kappa=1.96):
        """ Upper Confidence Bound acquisition function. """
        mean, std = self.gp_predict(x)
        return mean + kappa * std

    def ei(self, x, xi=0.01):
        """ Expected Improvement acquisition function. """
        mean, std = self.gp_predict(x)
        improvement = mean - self.current_best() - xi
        Z = improvement / std

        return improvement * norm.cdf(Z) + std * norm.pdf(Z)

    def pi(self, x, xi=0.01):
        """ Probability of Improvement acquisition function. """
        mean, std = self.gp_predict(x)
        improvement = mean - self.current_best() - xi
        Z = improvement / std

        return norm.cdf(Z)

    def optimize_acquisition(self, choice):
        """ Optimizes the acquisition function specified. """
        # Example: Randomly select an acquisition function
        # choice = np.random.choice(['ucb', 'ei', 'pi'])
        if choice == 'ucb':
            acq_func = self.ucb
        elif choice == 'ei':
            acq_func = self.ei
        else:
            acq_func = self.pi

        # Optimization setup
        # Convert bounds to the format required by scipy.optimize.direct
        bounds = Bounds(self.bounds[:, 0], self.bounds[:, 1])

        # Define a wrapper for the acquisition function to handle the negative sign
        # DIRECT minimizes, so we negate the acquisition function to perform maximization
        def neg_acq_func(x):
            return -acq_func(x)

        # Use DIRECT algorithm to find the global minimum of the negated acquisition function
        result = direct(neg_acq_func, bounds, maxfun=1000, maxiter=1000)
        return result.x
        # Check if optimization was successful and return the best point found
        # if result.success:
        #     return result.x
        # else:
        #     raise RuntimeError("Optimization failed: " + result.message)

    def generate_new_sample(self, z, ell):
        # Define the bounds for this subspace based on the hierarchy level ell
        # and optimize the acquisition function within these bounds
        bounds_ell = self.define_bounds_for_level(z, ell)
        x0_ell = np.array([z[i] if i < ell else self.random_state.uniform(bounds_ell[i][0], bounds_ell[i][1]) for i in range(len(z))])
        return self.optimize_acquisition(x0_ell)
    
    def generate_grid(self, z, ell, num_points=10):
        """
        Generate a grid where the resulting matrix has z fixing the first 0:ell parameters,
        resulting in a matrix loosely represented by [z, X_grid].
        """
        # Adjust `ell` to be index-based for slicing
        variable_bounds = self.bounds[ell:]

        # Create a range for each domain
        domain_ranges = [np.linspace(domain[0], domain[1], num_points) for domain in variable_bounds]
        
        # Create a meshgrid for the variable parts
        variable_meshgrid = np.meshgrid(*domain_ranges, indexing='ij')
        
        # Flatten each dimension of the meshgrid to create a list of points
        variable_points = np.vstack([grid.flatten() for grid in variable_meshgrid]).T
        
        # Repeat the fixed vector z for the same number of times as the number of points in the variable meshgrid
        repeated_z = np.repeat(z[:ell][np.newaxis, :], variable_points.shape[0], axis=0)
        
        # Concatenate the fixed z parts with the variable parts
        full_grid = np.hstack((repeated_z, variable_points))
        
        return full_grid

    def sample_function_from_GP(self, z, ell):
        complete_grid = self.generate_grid(z, ell)
        # Assuming self.gp is properly trained and can provide a meaningful sample
        f_sample = self.gp.sample_y(complete_grid, random_state=None)
        return f_sample, complete_grid
        

    def define_bounds_for_level(self, z, ell):
        # This function would adjust the bounds based on the hierarchy level and the value of z
        return self.bounds  # Simplified return for placeholder purposes

    def visualize(self):
        # Generate grid
        x = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
        y = np.linspace(self.bounds[1, 0], self.bounds[1, 1], 100)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.ravel(), Y.ravel()]).T

        # True function
        Z_true = self.function(xy).reshape(X.shape)

        # GP mean
        Z_gp, _ = self.gp.predict(xy, return_std=True)
        Z_gp = Z_gp.reshape(X.shape)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.contourf(X, Y, Z_true, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title('True Function')
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c='red')

        plt.subplot(1, 2, 2)
        plt.contourf(X, Y, Z_gp, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title('GP Mean Estimate')
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c='red')

        plt.show()


    def optimize(self, choice):
        x_t_0 = self.initialize_uniform_x_t_0()  # Initializes with one random point and updates the GP
        # Initialize D to accumulate all points and their evaluations
        D = np.empty((0, len(self.bounds)))  # Assuming each row in D will hold a point
        y_all = np.empty((0,)) 
        for t in range(1, self.T + 1):
            # Find the new point for this iteration by optimizing the acquisition function
            # x_t_0 = self.optimize_acquisition(self.X_train[-1])
            # x_t_0 = self.optimize_acquisition(choice)
            X_t = {0: [x_t_0]}

            # Start the hierarchical batching
            for ell in range(1, self.L):
                X_t[ell] = [x_t_0]
                # print(f'At ell: {ell} we have X_t^ell:{X_t[ell]}')

                # First handle the root batch member to spawn the first set
                z = X_t[0][0]
                for k in range(1, self.K[ell]):
                    # print(f'z at {ell}: {z[0:ell]}')
                    # x_t_k = self.generate_new_sample(z, ell)
                    # X_t[ell].append(x_t_k)
                    f, grid = self.sample_function_from_GP(z, ell)
                    x_t_k = grid[np.argmax(f)]
                    X_t[ell].append(x_t_k) 

                for z in X_t[ell - 1][1:]:  # Skip the first since already handled
                    for k in range(0, self.K[ell]):
                        # print(f'z for other branch at ell {ell}: {z[0:ell]}')
                        # x_t_k = self.generate_new_sample(z, ell)
                        # X_t[ell].append(x_t_k)
                        f, grid = self.sample_function_from_GP(z, ell)
                        x_t_k = grid[np.argmax(f)]
                        X_t[ell].append(x_t_k) 
            
            all_points = np.vstack(X_t[self.L-1])
            all_evals = self.function(all_points)
            print(f'all_points shape: {all_points.shape}')
            print(f'all_evals shape: {all_evals.shape}')
            

            # Accumulate in D
            D = np.vstack([D, all_points])  # Stack new points vertically in D
            y_all = np.hstack([y_all, all_evals])  # Append new evaluations

            # Update training data and re-train GP
            self.X_train = D
            self.y_train = y_all
            self.gp.fit(self.X_train, self.y_train)
            print(f'X_train shape: {self.X_train.shape}')
            print(f'y_train shape: {self.y_train.shape}')
            x_t_0 = self.optimize_acquisition(choice)
            # Visualize
            # self.visualize()

        # After all iterations, return the best observed point
        best_index = np.argmax(self.y_train)
        return self.X_train[best_index], self.y_train[best_index]

