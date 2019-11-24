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

    N = 30
    h = {0: -10}
    J = {(i, i+1): 1 for i in range(N-1)}

    ground_state = correct_state = [(-1)**i for i in range(N)]  # [-1]*N
    sa_samp = oj.SASampler()

    ground_energy = oj.BinaryQuadraticModel(
        h, J).calc_energy(ground_state)

    # make benchmark target solver
    def solver(time_param):
        return sa_samp.sample_ising(h, J, num_reads=100, num_sweeps=time_param)

    # benchmarking
    time_list = list(range(10, 101, 10))
    b_res = oj.solver_benchmark(
        solver, time_list=time_list, solutions=[ground_state]
    )
    plt.xlabel('annealing time')
    plt.ylabel('TTS')
    plt.plot(b_res['time'], np.array(b_res['tts']), '.')
    plt.show()

    b_res = oj.solver_benchmark(
        solver, time_list=time_list, solutions=[ground_state]
    )
    plt.xlabel('annealing time')
    plt.ylabel('TTS')
    plt.plot(b_res['time'], np.array(b_res['tts']), '.')
    plt.show()

    plt.xlabel('annealing time')
    plt.ylabel('Success probability')
    plt.plot(b_res['time'], np.array(b_res['success_prob']), '.')
    plt.show()

    plt.xlabel('step')
    plt.ylabel('time')
    plt.plot(time_list, b_res['time'], '.')
    plt.show()
