import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from objective_functions.rosenbrock import rosenbrock_3d
from hpc_BO.hcp_TS import HierarchicalPCBO
from parallel_BO.gp_ucb_pe import GPUCB_PE

def run_hpc_ts(index, bounds, T, L, K):
    optimizer = HierarchicalPCBO(rosenbrock_3d, bounds, T, L, K, random_state=index)
    optimizer.optimize('ucb')
    return optimizer.y_train.flatten()

def run_gp_ucb_pe(index, bounds, T, K_GP):
    optimizer = GPUCB_PE(bounds, rosenbrock_3d, random_state=index)
    optimizer.optimize(T-1, K_GP, n_initial_points=8)
    return optimizer.Y_train.flatten()

def main():
    N = 2
    bounds = [[-2.048, 2.048], [-2.048, 2.048], [-2.048, 2.048]]
    T = 10
    L = 3
    K = [1, 2, 4]
    K_GP = 8

    y_results_hpc_ts = np.zeros((N, T * np.prod(K)))
    y_results_gp_ucb_pe = np.zeros((N, T * K_GP))

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures_hpc_ts = [executor.submit(run_hpc_ts, i, bounds, T, L, K) for i in range(N)]
        futures_gp_ucb_pe = [executor.submit(run_gp_ucb_pe, i, bounds, T, K_GP) for i in range(N)]

        for i, future in enumerate(futures_hpc_ts):
            y_results_hpc_ts[i, :] = future.result()
        for i, future in enumerate(futures_gp_ucb_pe):
            y_results_gp_ucb_pe[i, :] = future.result()

    np.savetxt("../results_hpcBO/gp_ucb_pe_results_rosenbrock_3d_test.csv", y_results_gp_ucb_pe, delimiter=",")
    np.savetxt("../results_hpcBO/hpc_ts_results_rosenbrock_3d_random_state_test.csv", y_results_hpc_ts, delimiter=",")

if __name__ == "__main__":
    main()
