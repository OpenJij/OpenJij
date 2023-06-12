from __future__ import annotations
try:
    from typing import Optional, Union
except ImportError:
    from typing_extensions import Optional, Union

import dimod
import numpy as np

from cimod.utils import get_state_and_energy
from dimod import BINARY

import openjij
import openjij as oj
import openjij.cxxjij as cxxjij

from openjij.sampler.sampler import BaseSampler
from openjij.utils.decorator import deprecated_alias


class SQASampler(BaseSampler):
    """Sampler with Simulated Quantum Annealing (SQA).

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.
    Hamiltonian

    .. math::

        H(s) = s H_p + \\Gamma (1-s)\\sum_i \\sigma_i^x

    where :math:`H_p` is the problem Hamiltonian we want to solve.

    Args:
        beta (float): Inverse temperature.
        gamma (float): Amplitude of quantum fluctuation.
        trotter (int): Trotter number.
        num_sweeps (int): number of sweeps
        schedule (list): schedule list
        num_reads (int): Number of iterations.
        schedule_info (dict): Information about a annealing schedule.

    Raises:
        ValueError: If the schedule violates as below.
        - not list or numpy.array.
        - schedule range is '0 <= s <= 1'.
    """

    @property
    def parameters(self):
        return {
            "beta": ["parameters"],
            "gamma": ["parameters"],
            "trotter": ["parameters"],
        }

    def __init__(self):

        # Set default parameters
        beta = 5.0
        gamma = 1.0
        num_sweeps = 1000
        num_reads = 1
        schedule = None
        trotter = 4

        self._default_params = {
            "beta": beta,
            "gamma": gamma,
            "num_sweeps": num_sweeps,
            "schedule": schedule,
            "trotter": trotter,
            "num_reads": num_reads,
        }

        self._params = {
            "beta": beta,
            "gamma": gamma,
            "num_sweeps": num_sweeps,
            "schedule": schedule,
            "trotter": trotter,
            "num_reads": num_reads,
        }

        self._make_system = {"singlespinflip": cxxjij.system.make_transverse_ising}
        self._algorithm = {
            "singlespinflip": cxxjij.algorithm.Algorithm_SingleSpinFlip_run
        }

    def _convert_validation_schedule(self, schedule, beta):
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        if isinstance(schedule[0], cxxjij.utility.TransverseFieldSchedule):
            return schedule

        # schedule validation  0 <= s <= 1
        sch = np.array(schedule).T[0]
        if not np.all((0 <= sch) & (sch <= 1)):
            raise ValueError("schedule range is '0 <= s <= 1'.")

        if len(schedule[0]) == 2:
            # schedule element: (s, one_mc_step) with beta fixed
            # convert to list of cxxjij.utility.TransverseFieldSchedule
            cxxjij_schedule = []
            for s, one_mc_step in schedule:
                _schedule = cxxjij.utility.TransverseFieldSchedule()
                _schedule.one_mc_step = one_mc_step
                _schedule.updater_parameter.beta = beta
                _schedule.updater_parameter.s = s
                cxxjij_schedule.append(_schedule)
            return cxxjij_schedule
        elif len(schedule[0]) == 3:
            # schedule element: (s, beta, one_mc_step)
            # convert to list of cxxjij.utility.TransverseFieldSchedule
            cxxjij_schedule = []
            for s, _beta, one_mc_step in schedule:
                _schedule = cxxjij.utility.TransverseFieldSchedule()
                _schedule.one_mc_step = one_mc_step
                _schedule.updater_parameter.beta = _beta
                _schedule.updater_parameter.s = s
                cxxjij_schedule.append(_schedule)
            return cxxjij_schedule
        else:
            raise ValueError(
                """schedule is list of tuple or list
                (annealing parameter s : float, step_length : int) or
                (annealing parameter s : float, beta: float, step_length : int)
                """
            )

    def _get_result(self, system, model):
        state, info = super()._get_result(system, model)

        q_state = system.trotter_spins[:-1].T.astype(int)
        c_energies = [get_state_and_energy(model, state)[1] for state in q_state]
        info["trotter_state"] = q_state
        info["trotter_energies"] = c_energies

        return state, info

    def sample(
        self,
        bqm: Union[
            "openjij.model.model.BinaryQuadraticModel", dimod.BinaryQuadraticModel
        ],
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        num_sweeps: Optional[int] = None,
        schedule: Optional[list] = None,
        trotter: Optional[int] = None,
        num_reads: Optional[int] = None,
        initial_state: Optional[Union[list, dict]] = None,
        updater: Optional[str] = None,
        sparse: Optional[bool] = None,
        reinitialize_state: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> "openjij.sampler.response.Response":
        """Sampling from the Ising model.

        Args:
            bqm (openjij.BinaryQuadraticModel) binary quadratic model
            beta (float, optional): inverse tempareture.
            gamma (float, optional): strangth of transverse field. Defaults to None.
            num_sweeps (int, optional): number of sweeps. Defaults to None.
            schedule (list[list[float, int]], optional): List of annealing parameter. Defaults to None.
            trotter (int): Trotter number.
            num_reads (int, optional): number of sampling. Defaults to 1.
            initial_state (list[int], optional): Initial state. Defaults to None.
            updater (str, optional): update method. Defaults to 'single spin flip'.
            sparse (bool): use sparse matrix or not.
            reinitialize_state (bool, optional): Re-initilization at each sampling. Defaults to True.
            seed (int, optional): Sampling seed. Defaults to None.

        Raises:
            ValueError:

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples:

            for Ising case::

                >>> h = {0: -1, 1: -1, 2: 1, 3: 1}
                >>> J = {(0, 1): -1, (3, 4): -1}
                >>> sampler = openjij.SQASampler()
                >>> res = sampler.sample_ising(h, J)

            for QUBO case::

                >>> Q = {(0, 0): -1, (1, 1): -1, (2, 2): 1, (3, 3): 1, (4, 4): 1, (0, 1): -1, (3, 4): 1}
                >>> sampler = openjij.SQASampler()
                >>> res = sampler.sample_qubo(Q)
        """

        # Set default parameters
        if sparse is None:
            sparse = True
        if reinitialize_state is None:
            reinitialize_state = True
        if updater is None:
            updater = "single spin flip"

        if isinstance(bqm, dimod.BinaryQuadraticModel):
            bqm = oj.model.model.BinaryQuadraticModel(
                dict(bqm.linear), dict(bqm.quadratic), bqm.offset, bqm.vartype
            )

        ising_graph, offset = bqm.get_cxxjij_ising_graph()

        self._set_params(
            beta=beta,
            gamma=gamma,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            trotter=trotter,
            schedule=schedule,
        )

        # set annealing schedule -------------------------------
        self._annealing_schedule_setting(
            bqm,
            self._params["beta"],
            self._params["gamma"],
            self._params["num_sweeps"],
            self._params["schedule"],
        )
        # ------------------------------- set annealing schedule

        # make init state generator --------------------------------
        if initial_state is None:

            def init_generator():
                return [
                    ising_graph.gen_spin(seed)
                    if seed is not None
                    else ising_graph.gen_spin()
                    for _ in range(self._params["trotter"])
                ]

        else:
            if isinstance(initial_state, dict):
                initial_state = [initial_state[k] for k in bqm.variables]

            # convert to spin variable
            if bqm.vartype == BINARY:
                temp_initial_state = []
                for v in initial_state:
                    if v != 0 and v != 1:
                        raise RuntimeError(
                            "The initial variables must be 0 or 1 if vartype is BINARY."
                        )
                    temp_initial_state.append(2 * v - 1)

                initial_state = temp_initial_state

            _init_state = np.array(initial_state)

            # validate initial_state size
            if len(initial_state) != ising_graph.size():
                raise ValueError(
                    "the size of the initial state should be {}".format(
                        ising_graph.size()
                    )
                )

            trotter_init_state = [_init_state for _ in range(self._params["trotter"])]

            def init_generator():
                return trotter_init_state

        # -------------------------------- make init state generator

        # choose updater -------------------------------------------
        _updater_name = updater.lower().replace("_", "").replace(" ", "")
        if _updater_name not in self._algorithm:
            raise ValueError('updater is one of "single spin flip"')
        algorithm = self._algorithm[_updater_name]
        sqa_system = self._make_system[_updater_name](
            init_generator(), ising_graph, self._params["gamma"]
        )
        # ------------------------------------------- choose updater

        response = self._cxxjij_sampling(
            bqm, init_generator, algorithm, sqa_system, reinitialize_state, seed
        )

        response.info["schedule"] = self.schedule_info

        return response

    def _annealing_schedule_setting(
        self, model, beta=None, gamma=None, num_sweeps=None, schedule=None
    ):
        if schedule:
            self._params["schedule"] = self._convert_validation_schedule(schedule, beta)
            self.schedule_info = {"schedule": "custom schedule"}
        else:

            self._params["schedule"], beta_gamma = quartic_ising_schedule(
                model=model, beta=beta, gamma=gamma, num_sweeps=num_sweeps
            )
            self.schedule_info = {
                "beta": beta_gamma[0],
                "gamma": beta_gamma[1],
                "num_sweeps": num_sweeps,
            }


def linear_ising_schedule(model, beta, gamma, num_sweeps):
    """Generate linear ising schedule.

    Args:
        model (:class:`openjij.model.model.BinaryQuadraticModel`): BinaryQuadraticModel
        beta (float): inverse temperature
        gamma (float): transverse field
        num_sweeps (int): number of steps
    Returns:
        generated schedule
    """
    schedule = cxxjij.utility.make_transverse_field_schedule_list(
        beta=beta, one_mc_step=1, num_call_updater=num_sweeps
    )
    return schedule, [beta, gamma]


# TODO: more optimal schedule?


def quartic_ising_schedule(model, beta, gamma, num_sweeps):
    """Generate quartic ising schedule based on S

    Morita and H. Nishimori,
    Journal of Mathematical Physics 49, 125210 (2008).

    Args:
        model (:class:`openjij.model.model.BinaryQuadraticModel`): BinaryQuadraticModel
        beta (float): inverse temperature
        gamma (float): transverse field
        num_sweeps (int): number of steps
    Returns:
        generated schedule
    """

    s = np.linspace(0, 1, num_sweeps)
    fs = s**4 * (35 - 84 * s + 70 * s**2 - 20 * s**3)
    schedule = [((beta, elem), 1) for elem in fs]
    return schedule, [beta, gamma]
