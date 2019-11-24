import numpy as np
import matplotlib.pyplot as plt
import openjij as oj


if __name__ == "__main__":

    # make target instance
    N = 50
    h = {0: 1, 1: 1}
    J = {}
    for i in range(N-1):
        for j in range(i+1, N):
            J[(i, j)] = -1.0

    ground_state = [-1]*N
    sa_samp = oj.SASampler()

    ground_energy = oj.BinaryQuadraticModel(
        h, J).calc_energy(ground_state)

    # make benchmark target solver
    def solver(time_param):
        return sa_samp.sample_ising(h, J, num_reads=100, num_sweeps=time_param)

    # benchmarking
    time_list = np.arange(1, 11, 1)
    print(time_list)
    b_res = oj.solver_benchmark(
        solver, time_list=time_list, solutions=[
            ground_state, list(-1*np.array(ground_state))]
    )
    # b_res = oj.benchmark([true_ground_state], ground_energy,
    #                      solver, time_param_list=np.arange(1, 161, 50))
    print(b_res.keys())
    plt.xlabel('annealing time')
    plt.ylabel('error probability')
    plt.plot(b_res['time'], 1-np.array(b_res['success_prob']), '.')
    plt.show()
