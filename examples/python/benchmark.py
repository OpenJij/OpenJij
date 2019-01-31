import numpy as np
import matplotlib.pyplot as plt
import openjij as oj


if __name__ == "__main__":

    # make target instance
    N = 10
    h = {0: 1, 1: 1}
    J = {}
    for i in range(N-1):
        for j in range(i+1, N):
            J[(i, j)] = -1.0
    
    true_ground_state = [-1]*N
    sa_samp = oj.SASampler()
    
    ground_energy = oj.BinaryQuadraticModel(h, J).calc_energy(true_ground_state)

    # make benchmark target solver
    def solver(time_param, iteration):
        sa_samp.step_num = time_param 
        sa_samp.iteration = iteration
        return sa_samp.sample_ising(h, J)

    # benchmarking
    b_res = oj.benchmark([true_ground_state], ground_energy, solver, time_param_list=np.arange(1, 161, 50))

    plt.xlabel('annealing time')
    plt.ylabel('error probability')
    plt.plot(b_res['time'], b_res['error'])
    plt.show()
