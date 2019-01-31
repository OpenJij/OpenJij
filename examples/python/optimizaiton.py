import numpy as np
import openjij as oj


if __name__ == '__main__':
    h = {0: -1}
    J = {(0, 1): -1, (1, 2): -1}

    # Simulated annealing (classical) 10 times
    response = oj.SASampler(iteration=10).sample_ising(h, J)
    # show the lowest energy solution in ten times
    min_index = np.argmin(response.energies)
    print("SA results: ", response.states[min_index])
    # > SA results:  [1, 1, 1]


    # Simulated quantum annealing (quantum simulation) 10 times
    response = oj.SQASampler(iteration=10).sample_ising(h, J)
    # show the lowest energy solution in ten times
    min_index = np.argmin(response.energies)
    print("SQA results: ", response.states[min_index])
    # > SQA results:  [1, 1, 1]
