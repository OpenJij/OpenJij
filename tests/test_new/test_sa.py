import numpy as np
import openjij as oj


def test_simple_sa():
    sa_solver = oj.solver.SimulatedAnnealing()
    qubo = oj.solver.COOMatrix(
        row=np.array([0, 1, 2, 3], dtype=np.int32),
        col=np.array([1, 2, 3, 3], dtype=np.int32),
        val=np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64),
        size=4,
    )

    results = sa_solver.solve_coo_qubo(qubo, num_reads=10)
    assert len(results[0].value) == qubo.size
    assert sum([x.num_occurrences for x in results.data]) == 10


def test_sa_qubo_dict():
    sa_solver = oj.solver.SimulatedAnnealing()
    qubo = {(0, 1): 1.0, (1, 2): -1.0, (2, 3): -1.0, (3, 3): 1.0, (1, 4): 0.0}
    results = sa_solver.solve_qubo(qubo, num_reads=10)
    assert results.info["ignored_variables"] == set([4])


def test_sa_qubo_dict2():
    sa_solver = oj.solver.SimulatedAnnealing()
    qubo = {("a", "b"): 1.0, ("b", "c"): -1.0, ("c", "e"): -1.0, ("a", "f"): 0.0}
    results = sa_solver.solve_qubo(qubo, num_reads=1)
    assert results.info["ignored_variables"] == set(["f"])
    assert "a" in results[0].value

