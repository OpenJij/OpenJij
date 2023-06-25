import openjij as oj
import numpy as np


def test_compresse():
    sampleset = oj.solver.SampleSet.from_array(
        states=[np.array([1, 2, 3]), np.array([4, 5, 6])],
        energies=[1.0, 2.0],
        vartype=oj.solver.VarType.BINARY
    )

    sampleset = sampleset.compresse()

    assert len(sampleset) == 2

    sampleset = oj.solver.SampleSet.from_array(
        states=[np.array([1, 2, 3]), np.array([1, 2, 3])],
        energies=[1.0, 1.0],
        vartype=oj.solver.VarType.BINARY
    )

    sampleset = sampleset.compresse()

    assert len(sampleset) == 1
    assert sampleset[0].num_occurrences == 2


def test_compresse_degenerate():
    states = [
        np.array([0., 1., 1., 1., 1.]),
        np.array([0., 1., 1., 1., 1.]),
        np.array([0., 1., 1., 1., 1.]),
        np.array([0., 1., 1., 0., 1.]),
        np.array([0., 1., 1., 0., 1.]),
        np.array([0., 1., 1., 1., 1.]),
        np.array([0., 1., 1., 0., 1.]),
        np.array([0., 1., 1., 1., 1.]),
        np.array([0., 1., 1., 0., 1.]),
        np.array([0., 1., 1., 0., 1.])
    ]
    energies = [-1.0]*len(states)
    sampleset = oj.solver.SampleSet.from_array(
        states=states,
        energies=energies,
        vartype=oj.solver.VarType.BINARY
    )

    assert sum([x.num_occurrences for x in sampleset.data]) == 10

    sampleset = sampleset.compresse()

    assert sum([x.num_occurrences for x in sampleset.data]) == 10

