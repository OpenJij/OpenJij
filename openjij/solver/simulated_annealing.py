import dataclasses
import typing as typ
import numpy as np
import numpy.typing as npt
import cxxjij


@dataclasses.dataclass
class COOMatrix:
    row: npt.NDArray[np.int64]
    col: npt.NDArray[np.int64]
    val: npt.NDArray[np.float64]
    size: int


class SimulatedAnnealing:
    def solve_coo_qubo(
        self,
        qubo: COOMatrix,
        beta_min: typ.Optional[float] = None,
        beta_max: typ.Optional[float] = None,
        step_length: typ.Optional[int] = None,
        num_sweeps: typ.Optional[int] = None,
        beta_schedule: typ.Optional[list[float]] = None,
        num_reads: int = 1
    ):
        
        state = np.random.randint(0, 2, qubo.size)
        beta_schedule_vec = np.array(beta_schedule, dtype=np.float64)

        cxxjij.sa_qubo_ssf_from_coo(
            row_indeices = qubo.row,
            col_indeices = qubo.col,
            values = qubo.val,
            state = state,
            beta_schedule = beta_schedule_vec,
        )
    

