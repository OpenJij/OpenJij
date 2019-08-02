import numpy as np
import openjij as oj
import cxxjij as cj


def make_qubo():
    Q = {
        (0, 0): 1, (1, 1): -1, (2, 2): 2,
        (0, 1): 1, (1, 2): -1, (2, 0): -1
    }
    # solution is [0, 1, 0]
    return Q


def reverse_annealing():
    initial_state = [1, 0, 1]  # [0, 0, 0]
    qubo = make_qubo()
    sqa = oj.SASampler(beta_min=0.001, beta_max=100, iteration=20)
    res = sqa.sample_qubo(qubo, initial_state=initial_state)
    print(res.min_samples)
    model = oj.BinaryQuadraticModel(Q=qubo, var_type='BINARY')
    print(model.calc_energy(res.min_samples['min_states'][0]))


if __name__ == '__main__':
    h = {0: -1, 1: 1, 2: -1}
    J = {(0, 1): 1, (1, 2): 1,
         (2, 3): -1, (3, 1): -1}
    # This problem has degenerated solutions
    # [1,-1,1,-1] and [1,-1,1,1]

    # Simulated annealing (classical) 10 times
    response = oj.SASampler(iteration=10).sample_ising(h, J)
    # show the lowest energy solution in ten times
    print("SA results: \n", response.min_samples['min_states'])

    # Simulated quantum annealing (quantum simulation) 10 times
    response = oj.SQASampler(iteration=10).sample_ising(h, J)
    # show the lowest energy solution in ten times
    print("SQA results: \n", response.min_samples['min_states'])

    reverse_annealing()
