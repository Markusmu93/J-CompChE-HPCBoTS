import numpy as np
from scipy.optimize import direct, Bounds
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

class GPUCB_PE:
    def __init__(self, bounds, objective_function, kernel=None, alpha=1e-6, random_state=None):
        self.objective_function = objective_function
        self.kernel = kernel if kernel is not None else Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=5)
        self.bounds = np.array(bounds)  # List of tuples for each dimension [(low, high), ...]
        self.X_train = np.empty((0, len(self.bounds)))
        self.Y_train = np.empty((0, 1))

        self.random_state = np.random.RandomState(random_state)

    def ucb(self, x, kappa=2.0):
        """ Upper Confidence Bound acquisition function. """
        x = np.atleast_2d(x)  # Ensure x is 2D
        mean, std = self.gp.predict(x, return_std=True)
        return -(mean + kappa * std)

    def optimize_acquisition_function(self, acquisition_function):
        """ Optimize an acquisition function using scipy's direct function. """
        bounds = Bounds(self.bounds[:, 0], self.bounds[:, 1])
        result = direct(acquisition_function, bounds=bounds)
        return result.x
    
    def optimize_posterior_variance(self, posterior_variance):
        bounds = Bounds(self.bounds[:, 0], self.bounds[:, 1])
        result = direct(posterior_variance, bounds=bounds)
        return result.x

    def initialize(self, N):
        """ Randomly select N points from the design space, query the objective function,
            and update the GP model with these initial observations. """
        # Randomly generate N points within the bounds
        X_initial = self.random_state.uniform(self.bounds[:, 0], self.bounds[:, 1], (N, len(self.bounds)))
        #X_initial = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (N, len(self.bounds)))

        # Query the objective function at these points
        Y_initial = np.array([self.objective_function(x) for x in X_initial]).reshape(-1, 1)

        # Update the GP model with initial data
        self.gp.fit(X_initial, Y_initial)
        self.X_train = X_initial
        self.Y_train = Y_initial

    # def visualize(self):
    #     # Generate grid
    #     x = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100)
    #     y = np.linspace(self.bounds[1, 0], self.bounds[1, 1], 100)
    #     X, Y = np.meshgrid(x, y)
    #     xy = np.stack([X.ravel(), Y.ravel()]).T

    #     # True function
    #     Z_true = self.objective_function(xy).reshape(X.shape)

    #     # GP mean
    #     Z_gp, _ = self.gp.predict(xy, return_std=True)
    #     Z_gp = Z_gp.reshape(X.shape)

    #     # Plotting
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.contourf(X, Y, Z_true, levels=50, cmap='viridis')
    #     plt.colorbar()
    #     plt.title('True Function')
    #     plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c='red')

    #     plt.subplot(1, 2, 2)
    #     plt.contourf(X, Y, Z_gp, levels=50, cmap='viridis')
    #     plt.colorbar()
    #     plt.title('GP Mean Estimate')
    #     plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c='red')

    #     plt.show()

    def optimize(self, T, K, n_initial_points):
        self.initialize(n_initial_points)

        # Save the original optimizer setting
        original_optimizer = self.gp.optimizer
        # Visualize randomly sampled results 
        # self.visualize()
        for t in range(T):
            # Find the point with the highest UCB value
            xi = self.optimize_acquisition_function(self.ucb)
            Xt = [xi]


            # Predict the expected mean for the newly selected point
            mean_xi, _ = self.gp.predict(np.atleast_2d(xi), return_std=True)
            # Temporarily disable hyperparameter optimization
            self.gp.optimizer = None
            # Update the model with the selected point
            self.gp.fit(np.vstack([self.X_train, xi]), np.vstack([self.Y_train, mean_xi]))  # Dummy response

            for k in range(1, K):
                # Function to minimize for subsequent points
                def neg_conditional_sigma(x):
                    x = np.atleast_2d(x)
                    _, sigma = self.gp.predict(x, return_std=True)
                    return -sigma  # Negated for minimization

                xi = self.optimize_posterior_variance(neg_conditional_sigma)
                Xt.append(xi)
                
                # Update model with the selected point
                self.gp.fit(np.vstack([self.X_train, xi]), np.vstack([self.Y_train, [[0]]]))  # Dummy response
                # self.gp.X_train_ = np.vstack([self.gp.X_train_, xi])

            # Query the function at points in Xt
            Yt = np.array([self.objective_function(x) for x in Xt]).reshape(-1, 1)

            # Update the dataset with actual observations
            self.X_train = np.vstack([self.X_train, Xt])
            self.Y_train = np.vstack([self.Y_train, Yt])

            # Restore the optimizer for the actual update
            self.gp.optimizer = original_optimizer

            # Update the GP model with actual observations
            self.gp.fit(self.X_train, self.Y_train)
            
            # Visualize the GP model
            # self.visualize()
            print(self.X_train.shape)
         # After all iterations, return the best observed point
        best_index = np.argmax(self.Y_train)
        return self.X_train[best_index], self.Y_train[best_index]

        # return self.X_train, self.Y_train


# if __name__ == "__main__":
#     # Define the objective function
#     def objective_function(X):
#         X = np.atleast_2d(X) 
#         #print(X)
#         return -np.sum((X - 2)**2, axis=1)  # Quadratic function centered at 2,2

#     bounds = [[0, 5], [0, 5], [0, 5]]
#     model = GPUCB_PE(bounds, objective_function, random_state=42)
#     #beta = np.linspace(1, 2, 20)  # Example beta schedule
#     X_train, Y_train = model.optimize(T=7, K=4, n_initial_points=4)
#     print("Queried points:\n", model.X_train.shape)
#     print("Observations:\n", model.Y_train.shape)
