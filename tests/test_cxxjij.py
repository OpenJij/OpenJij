import openjij.cxxjij as cj
import numpy as np


def test_sa():
    qubo = cj.SparseSymmetricGraph(
        row=np.array([0, 2]),
        col=np.array([1, 2]),
        values=np.array([1.0, 1.0])
    )

    state = np.array([1.0, 1, 1, 1])
    dE = np.zeros(len(state))
    qubo.calc_qubo_energy_gradient(state, dE)
    beta_schedule = np.array([0.01, 0.1, 1, 10])

    assert 2.0 == qubo.calc_energy(state)
    cj.sa_sparse_qubo_ssf(qubo, dE, state, beta_schedule)

    

def test_local_search():
    qubo = cj.SparseSymmetricGraph(
        row=np.array([0, 2]),
        col=np.array([1, 2]),
        values=np.array([1.0, 1.0])
    )

    state = np.array([1.0, 1, 1, 1])
    dE = np.zeros(len(state))
    qubo.calc_qubo_energy_gradient(state, dE)
    assert 2.0 == qubo.calc_energy(state)
    iter_count = cj.local_search_sparse_qubo_ssf(qubo, dE, state, 3)

    assert iter_count == 2

   
def test_bruteforce():
    qubo = cj.SparseSymmetricGraph(
        row=np.array([0, 2]),
        col=np.array([1, 2]),
        values=np.array([1.0, 1.0])
    )

    results, energies = cj.solve_qubo_bruteforce(qubo, 1e-8)
    for e in energies:
        assert abs(e - 0.0) < 1e-8
    
    assert len(results) == 3

