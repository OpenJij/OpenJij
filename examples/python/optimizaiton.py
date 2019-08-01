import numpy as np
import openjij as oj
import cxxjij as cj

if __name__ == '__main__':
    h = {0: -1, 1: 1, 2: -1}
    J = {(0, 1): 1, (1, 2): 1,
         (2, 3): -1, (3, 1): -1}
    # This problem has degenerated solutions
    # [1,-1,1,-1] and [1,-1,1,1]

    # Simulated annealing (classical) 10 times
    response = oj.SASampler(iteration=10).sample_ising(h, J)
    # show the lowest energy solution in ten times
    print("SA results: ", response.min_samples['min_states'])

    # Simulated quantum annealing (quantum simulation) 10 times
    response = oj.SQASampler(iteration=10).sample_ising(h, J)
    # show the lowest energy solution in ten times
    print("SQA results: ", response.min_samples['min_states'])
    # > SQA results:  [1, 1, 1]

    graph = cj.graph.Dense(10)
    system = cj.system.make_transverse_ising(graph.gen_spin(), graph, 1, 4)
    sche = cj.utility.make_transverse_field_schedule_list(10, 100, 100)
    cj.algorithm.Algorithm_SingleSpinFlip_run(system, sche)
    t_state = system.trotter_states
    print(t_state[0])
