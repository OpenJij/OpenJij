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
    bqm = oj.BinaryQuadraticModel(h, J)
    samp = oj.Sampler(bqm)
    
    ising_int = np.triu(bqm.ising_interactions())
    ground_energy = bqm.calc_energy(true_ground_state)

    # make benchmark target solver
    solver = lambda param, iterations: samp.simulated_annealing(step_num=param, iteration=iterations)

    # benchmarking
    b_res = oj.benchmark([true_ground_state], ground_energy, solver, param_list=np.arange(1, 161, 50))

    plt.xlabel('annealing time')
    plt.ylabel('error probability')
    plt.plot(b_res['time'], b_res['error'])
    plt.show()
