from __future__ import annotations

import typing as typ
import numpy as np

import openjij.cxxjij as cj

from .coomatrix import COOMatrix, KeyType
from .sampleset import SampleSet, VarType

class BruteForceSolver:
    def solve_coo_qubo(
        self,
        qubo: COOMatrix,
        constant: float = 0.0,
        *,
        epsilon = 1e-8,
        variable_labels: typ.Optional[list[KeyType]] = None,
    ) -> SampleSet:
        qubo_graph = qubo.get_cxxjij_graph()
        results, energies = cj.solve_qubo_bruteforce(qubo_graph, epsilon)

        sampleset = SampleSet.from_array(
            states=[x[:-1] for x in results],
            energies=(np.array(energies) + constant).tolist(),
            vartype=VarType.BINARY,
            variable_labels=variable_labels
        )

        return sampleset
