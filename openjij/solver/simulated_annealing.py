import dataclasses
import typing as typ
import enum
import uuid

import numpy as np
import numpy.typing as npt

import openjij.utils
import openjij.cxxjij as cj
from .sampleset import SampleSet, VarType



@dataclasses.dataclass
class COOMatrix:
    row: npt.NDArray[np.int64]
    col: npt.NDArray[np.int64]
    val: npt.NDArray[np.float64]
    size: int

    def get_cxxjij_graph(self) -> cj.SparseSymmetricGraph:
        return cj.SparseSymmetricGraph(
            row = self.row,
            col = self.col,
            val = self.val,
        )


@dataclasses.dataclass
class SAMetaInfo:
    beta_min: float
    beta_max: float
    num_steps: int
    schedule_type: typ.Literal['auto_geometric', 'geometric', 'custom']
    num_reads: int
    post_optimize_iter_max: int
    local_converged: bool


class SimulatedAnnealing:
    def solve_coo_qubo(
        self,
        qubo: COOMatrix,
        beta_min: typ.Optional[float] = None,
        beta_max: typ.Optional[float] = None,
        step_length: typ.Optional[int] = None,
        num_sweeps: typ.Optional[int] = None,
        beta_schedule: typ.Optional[list[float]] = None,
        num_reads: int = 1,
        post_optimize_iter_max: int = 10,
    ) -> SampleSet:
        
        qubo_graph = qubo.get_cxxjij_graph()

        states = [np.random.randint(0, 2, qubo.size).astype(np.float64) for _ in range(num_reads)]
        dE = np.zeros(qubo.size, dtype=np.float64)

        # local search for get good dE
        qubo_graph.calc_dE(states, dE)
        cj.local_search_sparse_qubo_ssf(qubo_graph, dE, states[0], 3)

        if num_sweeps is None:
            num_sweeps = min(1000, qubo.size * 10)

        energies = []
        meta_info_list = []
        for x in states:

            _beta_schedule = beta_scheduler_from_dE(dE, num_sweeps)

            qubo_graph.calc_dE(x, dE)
            cj.sa_sparse_qubo_ssf(qubo, dE, x, _beta_schedule)

            ls_iter_count = 0
            if post_optimize_iter_max > 0:
                ls_iter_count = cj.local_search_sparse_qubo_ssf(qubo_graph, dE, x, post_optimize_iter_max)
            
            meta_info_list.append(SAMetaInfo(
                beta_min = _beta_schedule[0],
                beta_max = _beta_schedule[-1],
                num_steps = len(_beta_schedule),
                schedule_type = 'custom',
                num_reads = num_reads,
                post_optimize_iter_max = post_optimize_iter_max,
                local_converged = ls_iter_count < post_optimize_iter_max,
            ))

            energies.append(qubo_graph.calc_energy(x))
        

        sampleset = SampleSet.from_states_and_energies(
            states=states,
            energies=energies,
            vartype=VarType.BINARY
        ).compresse()
        return sampleset



def beta_scheduler_from_dE(dE: npt.NDArray[np.float64], num_steps: int) -> npt.NDArray[np.float64]:
    posi_dE = dE[dE > 0]
    if len(posi_dE) == 0:
        posi_dE = np.abs(dE)

    dE_max = np.max(posi_dE)
    dE_min = np.min(posi_dE) + np.std(posi_dE) * 0.1
    if dE_max <= dE_min:
        dE_min = np.min(posi_dE) * 0.5

    # metropolice update: p = exp(-beta * dE) -> beta = -log(p) / dE
    # In initial step, p <= 1/2 -> beta >= log(2) / dE_max
    beta_min = np.log(2) / dE_max

    # In final step, p = 0.01
    beta_max = np.log(100) / dE_min

    beta_schedule = np.geomspace(beta_min, beta_max, num_steps, endpoint=True)

    return beta_schedule



