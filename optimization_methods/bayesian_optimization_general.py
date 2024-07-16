import numpy as np
from scipy.optimize import direct, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import Bounds
from scipy.stats import norm

class BayesianOptimization:
    def __init__(self, bounds, func, initial_points=None, state_seed=None, kernel=None, alpha=1e-10, kappa=2.576, xi=0.01):
        self.bounds = np.array(bounds)
        self.func = func
        self.xi = xi
        self.kappa = kappa
        self.state_seed = state_seed
        self.gpr = GaussianProcessRegressor(kernel=kernel if kernel else Matern(), alpha=alpha, n_restarts_optimizer=10, normalize_y=True)

        # Normalize initial points or generate a random initial point
        if initial_points is None:
            normalized_x = self._normalize(np.array([self._random_sample()]))
            self.X = normalized_x
            self.y = np.array([func(self._denormalize(normalized_x)[0])])
        else:
            self.X = self._normalize(np.array(initial_points))
            self.y = np.array([func(self._denormalize(x)) for x in initial_points])

        self.gpr.fit(self.X, self.y)

    def _random_sample(self):
        np.random.seed(self.state_seed)
        return np.array([np.random.uniform(low, high) for low, high in self.bounds])

    def _normalize(self, X):
        # Normalize X to [0, 1]
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def _denormalize(self, X):
        # Rescale X back to the original bounds
        return X * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

    def _ucb(self, x, *args):
        x = np.atleast_2d(x)
        mean, std = self.gpr.predict(x, return_std=True)
        return -(mean - self.kappa * std)[0]  # Minus for minimization in scipy.optimize.direct

    def _expected_improvement(self, x):
        """Calculate the expected improvement at point x."""
        x = np.atleast_2d(x)
        mean, std = self.gpr.predict(x, return_std=True)
        if std == 0:
            return 0
        else:
            improvement = mean - self.y.max() - self.xi
            Z = improvement / std
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            return -ei  # Minimize negative EI for maximization problem

    def optimize(self, n_iter=10, eps=1e-4, maxfun=None, maxiter=1000):
        for _ in range(n_iter):
            res = direct(self._expected_improvement, Bounds([0] * len(self.bounds), [1] * len(self.bounds)),
                         eps=eps, maxfun=maxfun, maxiter=maxiter, locally_biased=True)

            if res.success:
                new_x = res.x
            else:
                # Fallback to differential evolution if DIRECT fails
                res = differential_evolution(self._expected_improvement, bounds=self.bounds)
                print("DIRECT failed; used Differential Evolution instead.")
                new_x = res.x

            new_y = self.func(self._denormalize(new_x))

            # Update the model with new data point in normalized space
            self.X = np.vstack([self.X, new_x])
            self.y = np.append(self.y, new_y)
            self.gpr.fit(self.X, self.y)

    def best_point(self):
        idx_best = np.argmax(self.y)
        return self.X[idx_best], self.y[idx_best]

    def get_observations(self):
        return self.y

    def get_X(self):
        return self.X

def bayesian_optimization_strategy(objective_function, domain, n_iterations, random_state=None):
    bounds = [(low, high) for low, high in domain.values()]
    optimizer = BayesianOptimization(bounds=bounds, func=objective_function, state_seed=random_state)
    optimizer.optimize(n_iter=n_iterations)
    return optimizer.get_observations()
