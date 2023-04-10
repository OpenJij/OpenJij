# Copyright 2023 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines the abstract sampler (BaseSampler)."""
from __future__ import annotations
import time

import dimod
import numpy as np

from cimod.utils import get_state_and_energy
from dimod import BINARY, SPIN
from dimod.core.sampler import samplemixinmethod

import openjij
import openjij as oj
import openjij.cxxjij as cxxjij


def measure_time(func):
    """Decorator for measuring calculation time.

    Args:
        func: decorator function
    """

    def wrapper(*args, **kargs):
        start_time = time.perf_counter()

        func(*args, **kargs)

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return execution_time

    return wrapper


class BaseSampler(dimod.Sampler):
    """Base sampler class of python wrapper for cxxjij simulator."""

    parameters = dict()
    properties = dict()

    def _set_params(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self._default_params:
                raise ValueError("Unknown parameters detected")
            if value is None:
                self._params[key] = self._default_params[key]
            else:
                self._params[key] = value

    def _sampling(self, **kwargs):
        pass

    def _cxxjij_sampling(
        self,
        model,
        init_generator,
        algorithm,
        system,
        reinitialize_state=None,
        seed=None,
        offset=None,
    ):
        """Basic sampling function: for cxxjij sampling.

        Args:
            model (openjij.BinaryQuadraticModel): model has a information of instaunce (h, J, Q)
            init_generator (callable): return initial state, must have argument structure
            algorithm (callable): system algorithm of cxxjij
            system (:obj:): [description]
            reinitialize_state (bool, optional): [description]. Defaults to None.
            seed (int, optional): seed for algorithm. Defaults to None.
            offset (float): an offset which is added to the calculated energy

        Returns:
            :class:`openjij.sampler.response.Response`: results
        """

        if offset is None:
            offset = 0

        # set algorithm function and set random seed ----
        if seed is None:

            def sampling_algorithm(system):
                return algorithm(system, self._params["schedule"])

        else:

            def sampling_algorithm(system):
                return algorithm(system, seed, self._params["schedule"])

        # ---- set algorithm function and set random seed

        # setting of response class
        execution_time = []

        # define sampling execution function ---------------
        states, energies = [], []
        system_info = {"system": []}

        @measure_time
        def exec_sampling():
            for _ in range(self._params["num_reads"]):
                # Re-initialize at each sampling
                # In reverse annealing,
                # user can use previous result (if re-initilize is set to False)
                if reinitialize_state:
                    system.reset_spins(init_generator())
                # Run sampling algorithm
                # and measure execution time
                _exec_time = measure_time(sampling_algorithm)(system)
                execution_time.append(_exec_time)

                # get Ising result (state and system information)
                # ex. _sys_info save trotterized quantum state.
                result_state, _sys_info = self._get_result(system, model)

                # convert result_state to cimod style
                result_state, energy = get_state_and_energy(model, result_state, offset)

                # store result (state and energy)
                states.append(result_state)
                energies.append(energy)

                if _sys_info:
                    system_info["system"].append(_sys_info)

        # --------------- define sampling execution function

        # Execute sampling function
        sampling_time = exec_sampling()

        # construct response instance
        response = oj.sampler.response.Response.from_samples(
            states, model.vartype, energies, info=system_info
        )

        # save execution time
        response.info["sampling_time"] = sampling_time * 10**6  # micro sec
        response.info["execution_time"] = np.mean(execution_time) * 10**6  # micro sec
        response.info["list_exec_times"] = (
            np.array(execution_time) * 10**6
        )  # micro sec

        return response

    def _get_result(self, system, model):
        result = cxxjij.result.get_solution(system)
        sys_info = {}
        return result, sys_info

    @samplemixinmethod
    def sample(self, bqm, **parameters):
        """Sample from a binary quadratic model.

        Args:
            bqm (openjij.BinaryQuadraticModel):
                Binary Qudratic Model
            **parameters:
                See the implemented sampling for additional keyword definitions.

        Returns:
            :class:`openjij.sampler.response.Response`: results
        """
        if bqm.vartype == SPIN:
            if not getattr(self.sample_ising, "__issamplemixin__", False):
                # sample_ising is implemented
                h, J, offset = bqm.to_ising()
                sampleset = self.sample_ising(h, J, **parameters)
                sampleset.record.energy += offset
                return sampleset
            else:
                Q, offset = bqm.to_qubo()
                sampleset = self.sample_qubo(Q, **parameters)
                sampleset.change_vartype(dimod.SPIN, energy_offset=offset)
                return sampleset
        elif bqm.vartype == BINARY:
            if not getattr(self.sample_qubo, "__issamplemixin__", False):
                # sample_qubo is implemented
                Q, offset = bqm.to_qubo()
                sampleset = self.sample_qubo(Q, **parameters)
                sampleset.record.energy += offset
                return sampleset
            else:
                h, J, offset = bqm.to_ising()
                sampleset = self.sample_ising(h, J, **parameters)
                sampleset.change_vartype(dimod.BINARY, energy_offset=offset)
                return sampleset
        else:
            raise RuntimeError("binary quadratic model has an unknown vartype")

    @samplemixinmethod
    def sample_ising(self, h, J, **parameters):
        """Sample from an Ising model using the implemented sample method.

        Args:
            h (dict): Linear biases
            J (dict): Quadratic biases

        Returns:
            :class:`openjij.sampler.response.Response`: results
        """
        # default sparse mode: True
        sparse_option = parameters.pop("sparse", True)
        bqm = oj.model.model.BinaryQuadraticModel.from_ising(
            h, J, sparse=sparse_option
        )
        return self.sample(bqm, sparse=sparse_option, **parameters)

    @samplemixinmethod
    def sample_qubo(self, Q, **parameters):
        """Sample from a QUBO model using the implemented sample method.

        Args:
            Q (dict or numpy.ndarray): Coefficients of a quadratic unconstrained binary optimization

        Returns:
            :class:`openjij.sampler.response.Response`: results
        """
        if isinstance(Q, dict):
            sparse_option = parameters.pop("sparse", True)
            bqm = oj.model.model.BinaryQuadraticModel.from_qubo(
                Q, sparse=sparse_option
            )
            return self.sample(bqm, sparse=sparse_option, **parameters)
        elif isinstance(Q, np.ndarray):
            # apply np.ndarray disables sparse option
            sparse_option = parameters.pop("sparse", False)
            bqm = oj.model.model.BinaryQuadraticModel.from_numpy_matrix(
                Q, vartype="BINARY"
            )
            return self.sample(bqm, sparse=sparse_option, **parameters)
        else:
            raise TypeError("Q must be either dict or np.ndarray")
