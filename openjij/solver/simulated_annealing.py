from __future__ import annotations
import dataclasses
import typing as typ
import enum
import uuid

import numpy as np
import numpy.typing as npt

import openjij.utils
import openjij.cxxjij as cj
from .coomatrix import COOMatrix
from .sampleset import SampleSet, VarType


KeyType = typ.TypeVar("KeyType", int, str)



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

    def solve_qubo(
        self,
        qubo: dict[tuple[KeyType, KeyType], float],
        constant: float = 0.0,
        *,
        beta_min: typ.Optional[float] = None,
        beta_max: typ.Optional[float] = None,
        step_length: typ.Optional[int] = None,
        num_sweeps: typ.Optional[int] = None,
        beta_schedule: typ.Optional[list[float]] = None,
        num_reads: int = 1,
        post_optimize_iter_max: int = 10,
    ) -> SampleSet:
        """Solve QUBO problem by simulated annealing.

        Args:
            qubo (dict[(int, int), float] or dict[(str, str), float]): QUBO matrix.
            constant (float): Constant energy offset. Defaults to 0.0.
            beta_min (float): Minimum beta value. Defaults to None.
            beta_max (float): Maximum beta value. Defaults to None.
            step_length (int): Step length. Defaults to None.
            num_sweeps (int): Number of sweeps. Defaults to None.
            beta_schedule (list[float]): Beta schedule. Defaults to None.
            num_reads (int): Number of reads. Defaults to 1.
            post_optimize_iter_max (int): Maximum number of iterations for post optimization. Defaults to 10.

        Returns:
            openjij.sampler.respo
        """
        coo_qubo, variable_map, ignored_vars = COOMatrix.from_dict(qubo)
        _inv_map = {v: k for k, v in variable_map.items()}
        variable_labels: list[KeyType] = [_inv_map[i] for i in range(len(variable_map))] 

        sampleset = self.solve_coo_qubo(
            coo_qubo,
            constant=constant,
            beta_min=beta_min,
            beta_max=beta_max,
            step_length=step_length,
            num_sweeps=num_sweeps,
            beta_schedule=beta_schedule,
            num_reads=num_reads,
            post_optimize_iter_max=post_optimize_iter_max,
            variable_labels=variable_labels,
        )
        sampleset.info["ignored_variables"] = ignored_vars
        return sampleset

    def solve_coo_qubo(
        self,
        qubo: COOMatrix,
        constant: float = 0.0,
        *,
        beta_min: typ.Optional[float] = None,
        beta_max: typ.Optional[float] = None,
        step_length: typ.Optional[int] = None,
        num_sweeps: typ.Optional[int] = None,
        beta_schedule: typ.Optional[list[float]] = None,
        num_reads: int = 1,
        post_optimize_iter_max: int = 10,
        variable_labels: typ.Optional[typ.Sequence[KeyType]] = None,
    ) -> SampleSet:
        
        qubo_graph = qubo.get_cxxjij_graph()

        states = [np.random.randint(0, 2, qubo.size+1).astype(np.float64) for _ in range(num_reads)]
        # set last element to 1.0
        # because the last element is used for representing the diagonal elements of QUBO matrix
        for x in states:
            x[-1] = 1.0

        dE = np.zeros(qubo.size+1, dtype=np.float64)

        # local search for get good dE
        qubo_graph.calc_qubo_energy_gradient(states[0], dE)
        cj.local_search_sparse_qubo_ssf(qubo_graph, dE, states[0], 3)

        if num_sweeps is None:
            num_sweeps = min(1000, qubo.size * 10)

        energies = []
        meta_info_list = []
        for x in states:

            _beta_schedule = beta_scheduler_from_dE(dE, num_sweeps)

            qubo_graph.calc_qubo_energy_gradient(x, dE)
            cj.sa_sparse_qubo_ssf(qubo_graph, dE, x, _beta_schedule)

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

        sampleset = SampleSet.from_array(
            states=[x[:-1] for x in states],
            energies=energies,
            vartype=VarType.BINARY,
            variable_labels=variable_labels
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
