# Copyright 2023 Jij Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
from __future__ import annotations
try:
    from typing import Optional, Union
except ImportError:
    from typing_extensions import Optional, Union

import cimod
import dimod
import numpy as np

from dimod import BINARY, SPIN

import openjij
import openjij as oj
import openjij.cxxjij as cxxjij

from openjij.sampler.sampler import BaseSampler
from openjij.utils.graph_utils import qubo_to_ising
from openjij.sampler.base_sa_sample_hubo import base_sample_hubo

"""This module contains Simulated Annealing sampler."""


class SASampler(BaseSampler):
    """Sampler with Simulated Annealing (SA).

    Args:
        beta_min (float):
            Minmum beta (inverse temperature).
            You can overwrite in methods .sample_*.

        beta_max (float):
            Maximum beta (inverse temperature).
            You can overwrite in methods .sample_*.

        num_reads (int):
            number of sampling (algorithm) runs. defaults None.
            You can overwrite in methods .sample_*.

        num_sweeps (int):
            number of MonteCarlo steps during SA. defaults None.
            You can overwrite in methods .sample_*.

        schedule_info (dict):
            Information about an annealing schedule.

    Raises:
        ValueError: If schedules or variables violate as below.
        - not list or numpy.array.
        - not list of tuple (beta : float, step_length : int).
        - beta is less than zero.
    """

    @property
    def parameters(self):
        return {
            "beta_min": ["parameters"],
            "beta_max": ["parameters"],
        }

    def __init__(self):

        # Set default parameters
        num_sweeps = 1000
        num_reads = 1
        beta_min = None
        beta_max = None
        schedule = None

        self._default_params = {
            "beta_min": beta_min,
            "beta_max": beta_max,
            "num_sweeps": num_sweeps,
            "schedule": schedule,
            "num_reads": num_reads,
        }

        self._params = self._default_params.copy()

        self._make_system = {
            "singlespinflip": cxxjij.system.make_classical_ising,
            "singlespinflippolynomial": cxxjij.system.make_classical_ising_polynomial,
            "swendsenwang": cxxjij.system.make_classical_ising,
        }
        self._algorithm = {
            "singlespinflip": cxxjij.algorithm.Algorithm_SingleSpinFlip_run,
            "singlespinflippolynomial": cxxjij.algorithm.Algorithm_SingleSpinFlip_run,
            "swendsenwang": cxxjij.algorithm.Algorithm_SwendsenWang_run,
        }

    def _convert_validation_schedule(self, schedule):
        """Checks if the schedule is valid and returns cxxjij schedule."""
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        if isinstance(schedule[0], cxxjij.utility.ClassicalSchedule):
            return schedule

        if len(schedule[0]) != 2:
            raise ValueError(
                "schedule is list of tuple or list (beta : float, step_length : int)"
            )

        # schedule validation  0 <= beta
        beta = np.array(schedule).T[0]
        if not np.all(0 <= beta):
            raise ValueError("schedule beta range is '0 <= beta'.")

        # convert to cxxjij.utility.ClassicalSchedule
        cxxjij_schedule = []
        for beta, step_length in schedule:
            _schedule = cxxjij.utility.ClassicalSchedule()
            _schedule.one_mc_step = step_length
            _schedule.updater_parameter.beta = beta
            cxxjij_schedule.append(_schedule)

        return cxxjij_schedule

    def sample(
        self,
        bqm: Union[
            "openj.model.model.BinaryQuadraticModel", dimod.BinaryQuadraticModel
        ],
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        num_sweeps: Optional[int] = None,
        num_reads: Optional[int] = None,
        schedule: Optional[list] = None,
        initial_state: Optional[Union[list, dict]] = None,
        updater: Optional[str] = None,
        sparse: Optional[bool] = None,
        reinitialize_state: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> "oj.sampler.response.Response":
        """Sample Ising model.

        Args:
            bqm (openjij.model.model.BinaryQuadraticModel) binary quadratic model
            beta_min (float): minimal value of inverse temperature
            beta_max (float): maximum value of inverse temperature
            num_sweeps (int): number of sweeps
            num_reads (int): number of reads
            schedule (list): list of inverse temperature
            initial_state (dict): initial state
            updater(str): updater algorithm
            sparse (bool): use sparse matrix or not.
            reinitialize_state (bool): if true reinitialize state for each run
            seed (int): seed for Monte Carlo algorithm
        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples:

            for Ising case::

                >>> h = {0: -1, 1: -1, 2: 1, 3: 1}
                >>> J = {(0, 1): -1, (3, 4): -1}
                >>> sampler = openj.SASampler()
                >>> res = sampler.sample_ising(h, J)

            for QUBO case::

                >>> Q = {(0, 0): -1, (1, 1): -1, (2, 2): 1, (3, 3): 1, (4, 4): 1, (0, 1): -1, (3, 4): 1}
                >>> sampler = openj.SASampler()
                >>> res = sampler.sample_qubo(Q)
        """

        # Set default parameters
        if updater is None:
            updater = "single spin flip"
        if sparse is None:
            sparse = True
        if reinitialize_state is None:
            reinitialize_state = True

        _updater_name = updater.lower().replace("_", "").replace(" ", "")
        # swendsen wang algorithm runs only on sparse ising graphs.
        if _updater_name == "swendsenwang" or sparse:
            sparse = True
        else:
            sparse = False

        if isinstance(bqm, dimod.BinaryQuadraticModel):
            bqm = oj.model.model.BinaryQuadraticModel(
                dict(bqm.linear),
                dict(bqm.quadratic),
                bqm.offset,
                bqm.vartype,
                sparse=sparse,
            )

        if sparse == True and bqm.sparse == False:
            # convert to sparse bqm
            bqm = oj.model.model.BinaryQuadraticModel(
                bqm.linear, bqm.quadratic, bqm.offset, bqm.vartype, sparse=True
            )

        # alias
        model = bqm

        ising_graph, offset = model.get_cxxjij_ising_graph()

        self._set_params(
            beta_min=beta_min,
            beta_max=beta_max,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            schedule=schedule,
        )

        # set annealing schedule -------------------------------
        if self._params["schedule"] is None:
            self._params["schedule"], beta_range = geometric_ising_beta_schedule(
                cxxgraph=ising_graph,
                beta_max=self._params["beta_max"],
                beta_min=self._params["beta_min"],
                num_sweeps=self._params["num_sweeps"],
            )
            self.schedule_info = {
                "beta_max": beta_range[0],
                "beta_min": beta_range[1],
                "num_sweeps": self._params["num_sweeps"],
            }
        else:
            self._params["schedule"] = self._convert_validation_schedule(
                self._params["schedule"]
            )
            self.schedule_info = {"schedule": "custom schedule"}
        # ------------------------------- set annealing schedule

        # make init state generator --------------------------------
        if initial_state is None:

            def _generate_init_state():
                return (
                    ising_graph.gen_spin(seed)
                    if seed is not None
                    else ising_graph.gen_spin()
                )

        else:
            temp_initial_state = []
            if isinstance(initial_state, dict):
                if model.vartype == BINARY:
                    for k in model.variables:
                        v = initial_state[k]
                        if v != 0 and v != 1:
                            raise RuntimeError("The initial variables must be 0 or 1.")
                        temp_initial_state.append(2 * v - 1)
                elif model.vartype == SPIN:
                    for k in model.variables:
                        v = initial_state[k]
                        if v != -1 and v != 1:
                            raise RuntimeError(
                                "The initial variables must be -1 or +1."
                            )
                        temp_initial_state.append(v)
                else:
                    raise RuntimeError("Unknown vartype detected.")
            elif isinstance(initial_state, (list, tuple)):
                if model.vartype == BINARY:
                    for k in range(len(model.variables)):
                        v = initial_state[k]
                        if v != 0 and v != 1:
                            raise RuntimeError("The initial variables must be 0 or 1.")
                        temp_initial_state.append(2 * v - 1)
                elif model.vartype == SPIN:
                    for k in range(len(model.variables)):
                        v = initial_state[k]
                        if v != -1 and v != 1:
                            raise RuntimeError(
                                "The initial variables must be -1 or +1."
                            )
                        temp_initial_state.append(v)
                else:
                    raise RuntimeError("Unknown vartype detected.")
            else:
                raise RuntimeError("Unsupported type of initial_state.")

            _init_state = np.array(temp_initial_state)

            # validate initial_state size
            if len(initial_state) != ising_graph.size():
                raise ValueError(
                    "the size of the initial state should be {}".format(
                        ising_graph.size()
                    )
                )

            def _generate_init_state():
                return np.array(_init_state)

        # -------------------------------- make init state generator

        # choose updater -------------------------------------------
        _updater_name = updater.lower().replace("_", "").replace(" ", "")
        if _updater_name not in self._make_system:
            raise ValueError('updater is one of "single spin flip or swendsen wang"')
        algorithm = self._algorithm[_updater_name]
        sa_system = self._make_system[_updater_name](
            _generate_init_state(), ising_graph
        )
        # ------------------------------------------- choose updater
        response = self._cxxjij_sampling(
            model, _generate_init_state, algorithm, sa_system, reinitialize_state, seed
        )

        response.info["schedule"] = self.schedule_info

        return response
    
    def _sample_hubo_old(
        self,
        J: Union[
            dict, "openj.model.model.BinaryPolynomialModel", cimod.BinaryPolynomialModel
        ],
        vartype: Optional[str] = None,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        num_sweeps: Optional[int] = None,
        num_reads: Optional[int] = None,
        schedule: Optional[list] = None,
        initial_state: Optional[Union[list, dict]] = None,
        updater: Optional[str] = None,
        reinitialize_state: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> "openjij.sampler.response.Response":
        """Sampling from higher order unconstrainted binary optimization.

        Args:
            J (dict): Interactions.
            vartype (str, openjij.VarType): "SPIN" or "BINARY".
            beta_min (float, optional): Minimum beta (initial inverse temperature). Defaults to None.
            beta_max (float, optional): Maximum beta (final inverse temperature). Defaults to None.
            schedule (list, optional): schedule list. Defaults to None.
            num_sweeps (int, optional): number of sweeps. Defaults to None.
            num_reads (int, optional): number of reads. Defaults to 1.
            init_state (list, optional): initial state. Defaults to None.
            reinitialize_state (bool): if true reinitialize state for each run
            seed (int, optional): seed for Monte Carlo algorithm. Defaults to None.

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples::
            for Ising case::
                >>> sampler = openjij.SASampler()
                >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
                >>> response = sampler.sample_hubo(J, "SPIN")

            for Binary case::
                >>> sampler = ooenjij.SASampler()
                >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
                >>> response = sampler.sample_hubo(J, "BINARY")
        """

        # Set default parameters
        if reinitialize_state is None:
            reinitialize_state = True

        # Set model
        if str(type(J)) == str(type(oj.model.model.BinaryPolynomialModel({}, "SPIN"))):
            if vartype is not None:
                raise ValueError("vartype must not be specified")
            model = J
        elif str(type(J)) == str(type(cimod.BinaryPolynomialModel({}, "SPIN"))):
            if vartype is not None:
                raise ValueError("vartype must not be specified")
            model = J
        else:
            model = oj.model.model.BinaryPolynomialModel(J, vartype)

        # make init state generator --------------------------------
        if initial_state is None:
            if model.vartype == SPIN:

                def _generate_init_state():
                    return (
                        cxxjij.graph.Polynomial(model.num_variables).gen_spin(seed)
                        if seed is not None
                        else cxxjij.graph.Polynomial(model.num_variables).gen_spin()
                    )

            elif model.vartype == BINARY:

                def _generate_init_state():
                    return (
                        cxxjij.graph.Polynomial(model.num_variables).gen_binary(seed)
                        if seed is not None
                        else cxxjij.graph.Polynomial(model.num_variables).gen_binary()
                    )

            else:
                raise ValueError("Unknown vartype detected")
        else:
            if isinstance(initial_state, dict):
                initial_state = [initial_state[k] for k in model.indices]

            def _generate_init_state():
                return np.array(initial_state)

        # -------------------------------- make init state generator

        # determine system class and algorithm --------------------------------
        if model.vartype == SPIN:
            if updater is None or updater == "single spin flip":
                sa_system = cxxjij.system.make_classical_ising_polynomial(
                    _generate_init_state(), model.to_serializable()
                )
                algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
            elif updater == "k-local":
                raise ValueError(
                    "k-local update is only supported for binary variables"
                )
            else:
                raise ValueError("Unknown updater name")
        elif model.vartype == BINARY:
            if updater == "k-local":
                sa_system = cxxjij.system.make_k_local_polynomial(
                    _generate_init_state(), model.to_serializable()
                )
                algorithm = cxxjij.algorithm.Algorithm_KLocal_run
            elif updater is None or updater == "single spin flip":
                sa_system = cxxjij.system.make_classical_ising_polynomial(
                    _generate_init_state(), model.to_serializable()
                )
                algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
            else:
                raise ValueError("Unknown updater name")
        else:
            raise ValueError("Unknown vartype detected")
        # -------------------------------- determine system class and algorithm

        self._set_params(
            beta_min=beta_min,
            beta_max=beta_max,
            num_sweeps=num_sweeps,
            num_reads=num_reads,
            schedule=schedule,
        )

        # set annealing schedule -------------------------------
        if self._params["schedule"] is None:
            self._params["schedule"], beta_range = geometric_hubo_beta_schedule(
                sa_system,
                self._params["beta_max"],
                self._params["beta_min"],
                self._params["num_sweeps"],
            )
            self.schedule_info = {
                "beta_max": beta_range[0],
                "beta_min": beta_range[1],
                "num_sweeps": self._params["num_sweeps"],
            }
        else:
            self.schedule_info = {"schedule": "custom schedule"}
        # ------------------------------- set annealing schedule

        response = self._cxxjij_sampling(
            model, _generate_init_state, algorithm, sa_system, reinitialize_state, seed
        )

        response.info["schedule"] = self.schedule_info

        return response
    
    def sample_hubo(
        self,
        J: dict[tuple, float],
        vartype: Optional[str] = None,
        num_sweeps: int = 1000,
        num_reads: int = 1,
        num_threads: int = 1,
        beta_min: Optional[float] = None,
        beta_max: Optional[float] = None,
        updater: str = "METROPOLIS",
        random_number_engine: str = "XORSHIFT",
        seed: Optional[int] = None,
        temperature_schedule: str = "GEOMETRIC",
    ):  
        """Sampling from higher order unconstrainted binary optimization.

        Args:
            J (dict): Interactions.
            vartype (str): "SPIN" or "BINARY".
            num_sweeps (int, optional): The number of sweeps. Defaults to 1000.
            num_reads (int, optional): The number of reads. Defaults to 1.
            num_threads (int, optional): The number of threads. Parallelized for each sampling with num_reads > 1. Defaults to 1.
            beta_min (float, optional): Minimum beta (initial inverse temperature). Defaults to None.
            beta_max (float, optional): Maximum beta (final inverse temperature). Defaults to None.
            updater (str, optional): Updater. One can choose "METROPOLIS", "HEAT_BATH", or "k-local". Defaults to "METROPOLIS".
            random_number_engine (str, optional): Random number engine. One can choose "XORSHIFT", "MT", or "MT_64". Defaults to "XORSHIFT".            
            seed (int, optional): seed for Monte Carlo algorithm. Defaults to None.
            temperature_schedule (str, optional): Temperature schedule. One can choose "LINEAR", "GEOMETRIC". Defaults to "GEOMETRIC".

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples::
            for Ising case::
                >>> sampler = openjij.SASampler()
                >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
                >>> response = sampler.sample_hubo(J, "SPIN")

            for Binary case::
                >>> sampler = ooenjij.SASampler()
                >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
                >>> response = sampler.sample_hubo(J, "BINARY")
        """


        if updater=="k-local" or not isinstance(J, dict):
            # To preserve the correspondence with the old version.
            if updater=="METROPOLIS":
                updater="single spin flip"
            return self._sample_hubo_old(
                J=J,
                vartype=vartype,
                beta_min=beta_min,
                beta_max=beta_max,
                num_sweeps=num_sweeps,
                num_reads=num_reads,
                #schedule,
                #initial_state,
                updater=updater,
                #reinitialize_state,
                seed=seed
            )
        else:
            # To preserve the correspondence with the old version.
            if updater=="single spin flip":
                updater="METROPOLIS"
            return base_sample_hubo(
                hubo=J,
                vartype=vartype,
                num_sweeps=num_sweeps,
                num_reads=num_reads,
                num_threads=num_threads,
                beta_min=beta_min,
                beta_max=beta_max,
                update_method=updater,
                random_number_engine=random_number_engine,
                seed=seed,
                temperature_schedule=temperature_schedule
            )


def geometric_ising_beta_schedule(
    cxxgraph: Union[openjij.cxxjij.graph.Dense, openjij.cxxjij.graph.CSRSparse],
    beta_max=None,
    beta_min=None,
    num_sweeps=1000,
):
    """Make geometric cooling beta schedule.

    Args:
        cxxgraph (Union[openjij.cxxjij.graph.Dense, openjij.cxxjij.graph.CSRSparse]): Ising graph, must be either `Dense` or `CSRSparse`.
        beta_max (float, optional): [description]. Defaults to None.
        beta_min (float, optional): [description]. Defaults to None.
        num_sweeps (int, optional): [description]. Defaults to 1000.
    Returns:
        list of cxxjij.utility.ClassicalSchedule, list of beta range [max, min]
    """

 
    if beta_min is None or beta_max is None:
        # generate Ising matrix (with symmetric form)
        ising_interaction = cxxgraph.get_interactions()
        abs_ising_interaction = np.abs(ising_interaction)[:-1]
        # if `abs_ising_interaction` is empty, set min/max delta_energy to 1 (a trivial case).
        if abs_ising_interaction.shape[0] == 0:
            min_delta_energy = 1
            max_delta_energy = 1
        else:
            max_abs_ising_interaction = np.max(abs_ising_interaction)

            # automatical setting of min, max delta energy
            abs_bias = np.sum(abs_ising_interaction, axis=1)

            # apply threshold to avoid extremely large beta_max
            THRESHOLD = 1e-8


            min_delta_energy = np.min(
                abs_ising_interaction[
                    abs_ising_interaction > max_abs_ising_interaction * THRESHOLD
                ]
            )
            max_delta_energy = np.max(
                abs_bias[abs_bias > max_abs_ising_interaction * THRESHOLD]
            )

    # TODO: More optimal schedule ?

    if beta_min is None:
        beta_min = np.log(2) / max_delta_energy
    if beta_max is None:
        beta_max = np.log(100) / min_delta_energy

    num_sweeps_per_beta = max(1, num_sweeps // 1000)

    # set schedule to cxxjij
    schedule = cxxjij.utility.make_classical_schedule_list(
        beta_min=beta_min,
        beta_max=beta_max,
        one_mc_step=num_sweeps_per_beta,
        num_call_updater=num_sweeps // num_sweeps_per_beta,
    )

    return schedule, [beta_max, beta_min]


def geometric_hubo_beta_schedule(sa_system, beta_max, beta_min, num_sweeps):
    max_delta_energy = sa_system.get_max_effective_dE()
    min_delta_energy = sa_system.get_min_effective_dE()

    if beta_min is None:
        beta_min = np.log(2) / max_delta_energy

    if beta_max is None:
        beta_max = np.log(100) / min_delta_energy

    num_sweeps_per_beta = max(1, num_sweeps // 1000)

    schedule = cxxjij.utility.make_classical_schedule_list(
        beta_min=beta_min,
        beta_max=beta_max,
        one_mc_step=num_sweeps_per_beta,
        num_call_updater=num_sweeps // num_sweeps_per_beta,
    )

    return schedule, [beta_max, beta_min]
