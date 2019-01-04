import numpy as np
import openjij as oj


if __name__ == '__main__':
    h = {0: -1}
    J = {(0, 1): -1, (1, 2): -1}

    bqm = oj.BinaryQuadraticModel(h, J) 
    samp = oj.Sampler(bqm)

    # Simulated annealing (classical) 10 times
    response = samp.simulated_annealing(iteration=10)
    # show the lowest energy solution in ten times
    min_index = np.argmin(response.energies)
    print("SA results: ", response.states[min_index])


    # Simulated quantum annealing (quantum simulation) 10 times
    response = samp.simulated_quantum_annealing(iteration=10)
    # show the lowest energy solution in ten times
    min_index = np.argmin(response.energies)
    print("SQA results: ", response.states[min_index])
