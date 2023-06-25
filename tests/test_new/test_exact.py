import openjij as oj
import numpy as np


def test_simple_exact():
    solver = oj.solver.BruteForceSolver()
    qubo = oj.solver.COOMatrix(
        row=np.array([0, 1, 2, 3], dtype=np.int32),
        col=np.array([1, 2, 3, 3], dtype=np.int32),
        val=np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64),
        size=4,
    )

    results = solver.solve_coo_qubo(qubo)
    assert len(results[0].value) == qubo.size
    assert len(results.data) == 2

