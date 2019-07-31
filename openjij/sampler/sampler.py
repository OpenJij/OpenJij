# Copyright 2019 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cxxjij as cj
from openjij.model import BinaryQuadraticModel
import openjij
from .response import Response

import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        calc_time = time.time() - start
        return calc_time
    return wrapper


class BaseSampler:

    def _set_model(self, model):
        self.indices = model.indices
        self.size = model.size
        self.energy_bias = model.energy_bias
        self.var_type = model.var_type

    def _make_dense_graph(self, h=None, J=None, Q=None, var_type='SPIN'):
        if var_type == 'BINARY':
            if Q is None:
                raise ValueError('Input QUBO matrix: Q')
            model = BinaryQuadraticModel(Q=Q, var_type='BINARY')
        elif var_type == 'SPIN':
            if h is None or J is None:
                raise ValueError('Input h and J')
            model = BinaryQuadraticModel(h=h, J=J, var_type='SPIN')
        self.indices = model.indices
        self.N = len(model.indices)
        self.energy_bias = model.energy_bias

        self.var_type = model.var_type

        dense_graph = model.convert_to_dense_graph()
        self.model = model
        return dense_graph

    def _sampling_kwargs_setting(self, **kwargs):
        if 'iteration' in kwargs:
            self.iteration = kwargs['iteration']
        elif 'num_reads' in kwargs:
            self.iteration = kwargs['num_reads']


class SQASampler(BaseSampler):
    """Sampler with Simulated Quantum Annealing (SQA).

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.

    Args:
        beta (float):
            Inverse temperature.

        gamma (float):
            Amplitude of quantum fluctuation.

        trotter (int):
            Trotter number.

        step_length (int):
            Length of Monte Carlo step.

        step_num (int):
            Number of Monte Carlo step.

        schedule_info (dict):
            Information about a annealing schedule.

        iteration (int):
            Number of iterations.

    Attributes:
        energy_bias (float):
            Energy bias.

        var_type (str):    
            Type of variables: 'SPIN' or 'BINARY' which mean {-1, 1} or {0, 1}.

        indices (int):
            Indices of `openjij.model.model.BinaryQuadraticModel` object.

        N (int):
            Number of the indices.

        system_class (:class:):
            `cxxjij.system.QuantumIsing` class.

        sqa_kwargs (dict):
            Parameters of SQA: beta, gamma, and schedule_info.

    Raises:
        ValueError: If the schedule violates as below.
        - not list or numpy.array.
        - schedule range is '0 <= s < 1'.

    """

    def __init__(self, beta=5.0, gamma=1.0,
                 trotter=4, step_length=10, step_num=100, schedule=None, iteration=1):

        # make schedule
        if schedule is not None:
            self._validate_schedule(schedule)
            self.step_length = None
            self.step_num = None
            self.schedule_info = {'schedule': schedule}
        else:
            self.step_length = step_length
            self.step_num = step_num
            self.schedule_info = {
                'step_num': step_num, 'step_length': step_length}

        self.beta = beta
        self.gamma = gamma
        self.trotter = trotter

        self.iteration = iteration
        self.energy_bias = 0.0

        self.system_class = cj.system.QuantumIsing  # CPU Trotterize quantum system
        self.sqa_kwargs = dict(
            beta=self.beta, gamma=self.gamma, **self.schedule_info)

    def _validate_schedule(self, schedule):
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        # schedule validation  0 <= s < 1
        sch = np.array(schedule).T[0]
        if not np.all((0 <= sch) & (sch < 1)):
            raise ValueError("schedule range is '0 <= s < 1'.")

    def sample_ising(self, h, J, **kwargs):
        """Sample from the specified Ising model.

        Args:
            h (dict):
                Linear biases of the Ising model.

            J (dict):
                Quadratic biases of the Ising model.

            **kwargs:
                Optional keyword arguments for the sampling method.

        Returns:
            :obj:: `openjij.sampler.response.Response` object.

        Examples:
            This example submits a two-variable Ising problem.

            >>> import openjij as oj
            >>> sampler = oj.SQASampler()
            >>> response = sampler.sample_ising({0: -1, 1: 1}, {})
            >>> for sample in response.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 1, 1: -1}

        """

        var_type = 'SPIN'
        ising_dense_graph = self._make_dense_graph(h=h, J=J, var_type=var_type)
        return self._sampling(ising_dense_graph, var_type=var_type)

    def sample_qubo(self, Q, **kwargs):
        """Sample from the specified QUBO.

        Args:
            Q (dict):
                Coefficients of a quadratic unconstrained binary optimization (QUBO) model.

            **kwargs:
                Optional keyword arguments for the sampling method.

        Returns:
            :obj:: `openjij.sampler.response.Response` object.

        Examples:
            This example submits a two-variable QUBO model.

            >>> import openjij as oj
            >>> sampler = oj.SQASampler()
            >>> Q = {(0, 0): -1, (4, 4): -1, (0, 4): 2}
            >>> response = sampler.sample_qubo(Q)
            >>> for sample in response.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 0, 4: 1}

        """

        var_type = 'BINARY'
        ising_dense_graph = self._make_dense_graph(Q=Q, var_type=var_type)
        return self._sampling(ising_dense_graph, var_type=var_type)

    def _sampling(self, ising_graph, var_type, **kwargs):
        self._sampling_kwargs_setting(**kwargs)

        # system = self.system_class(ising_graph, num_trotter_slices=self.trotter)

        # to calculate energy
        int_mat = self.model.interactions()
        linear = np.diag(int_mat)
        quad = np.triu(int_mat) - np.diag(linear)

        q_states = []
        q_energies = []

        execution_time = []

        @measure_time
        def exec_sampling():
            for _ in range(self.iteration):
                # system.initialize_spins()  # not support yet on GPU
                system = self.system_class(
                    ising_graph, num_trotter_slices=self.trotter)
                _exec_time = measure_time(
                    system.simulated_quantum_annealing)(**self.sqa_kwargs)
                execution_time.append(_exec_time)
                q_state = self._post_process4state(system.get_spins())
                q_states.append(q_state)
                q_energies.append(
                    [state @ quad @ state + linear @ state + self.energy_bias for state in q_state])

        sampling_time = exec_sampling()

        response = Response(var_type=var_type, indices=self.indices)
        response.update_quantum_ising_states_energies(q_states, q_energies)

        response.info['sampling_time'] = sampling_time * \
            10**6              # micro sec
        response.info['execution_time'] = np.mean(
            execution_time) * 10**6   # micro sec
        response.info['list_exec_times'] = np.array(
            execution_time) * 10**6  # micro sec

        return response

    def _post_process4state(self, q_state):
        return q_state
