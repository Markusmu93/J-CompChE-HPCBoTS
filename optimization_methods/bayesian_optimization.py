from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.learning.gaussian_process.kernels import Matern
from skopt.learning import GaussianProcessRegressor

space = [
    Real(-2.048, 2.048, name='x1'),
    Real(-2.048, 2.048, name='x2'),
    Real(-2.048, 2.048, name='x3')
]

@use_named_args(space)
def rosenbrock_3d_skopt(x1, x2, x3):
    a = 1
    b = 100
    return -((a - x1)**2 + b * (x2 - x1**2)**2 + (a - x2)**2 + b * (x3 - x2**2)**2)

def run_bayesian_optimization(index, space, T):
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
    result = gp_minimize(
        rosenbrock_3d_skopt, 
        space,
        acq_func="EI",
        n_calls=T,
        n_random_starts=10,
        base_estimator=gp,
        random_state=index
    )
    return result.func_vals
