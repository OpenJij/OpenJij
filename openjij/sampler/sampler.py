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
import cxxjij
from openjij.model import BinaryQuadraticModel
import openjij
import dimod

import time


def measure_time(func):
    def wrapper(*args, **kargs):
        start_time = time.perf_counter()

        func(*args, **kargs)

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return execution_time
    return wrapper


class BaseSampler(dimod.Sampler):
    parameters = None
    properties = None

    def _set_model(self, model):
        self.indices = model.indices
        self.size = model.size
        self.energy_bias = model.energy_bias
        self.var_type = model.var_type

    def _sampling(self, model, init_generator,
                  algorithm, system,
                  initial_state=None, reinitialize_state=None,
                  seed=None, **kwargs):
        self._set_model(model)
        if seed is None:
            def sampling_algorithm(system): return algorithm(
                system, self._schedule)
        else:
            def sampling_algorithm(system): return algorithm(
                system, seed, self._schedule)

        # run algorithm
        execution_time = []
        response = openjij.Response(
            var_type=model.var_type, indices=self.indices)

        @measure_time
        def exec_sampling():
            previous_state = init_generator()
            for _ in range(self.num_reads):
                if reinitialize_state:
                    system.reset_spins(init_generator())
                else:
                    system.reset_spins(previous_state)
                _exec_time = measure_time(sampling_algorithm)(system)
                execution_time.append(_exec_time)
                previous_state = cxxjij.result.get_solution(system)
                self._post_save(previous_state, system, model, response)

        sampling_time = exec_sampling()

        response.info['sampling_time'] = sampling_time * 10**6  # micro sec
        response.info['execution_time'] = np.mean(
            execution_time) * 10**6  # micro sec
        response.info['list_exec_times'] = np.array(
            execution_time) * 10**6  # micro sec

        return response

    def _dict_to_model(self, var_type, h=None, J=None, Q=None, **kwargs):
        if var_type == openjij.SPIN:
            bqm = openjij.BinaryQuadraticModel(h, J, 0.0, var_type)
        elif var_type == openjij.BINARY:
            bqm = openjij.BinaryQuadraticModel.from_qubo(Q)
        else:
            raise ValueError(
                'var_type should be openjij.SPIN or openjij.BINARY')
        return bqm

    def _post_save(self, result_state, system, model, response):
        pass


# class BaseSampler:

#     def _set_model(self, model):
#         self.indices = model.indices
#         self.size = model.size
#         self.energy_bias = model.energy_bias
#         self.var_type = model.var_type

#     def _make_dense_graph(self, h=None, J=None, Q=None, var_type='SPIN'):
#         if var_type == 'BINARY':
#             if Q is None:
#                 raise ValueError('Input QUBO matrix: Q')
#             model = BinaryQuadraticModel(Q=Q, var_type='BINARY')
#         elif var_type == 'SPIN':
#             if h is None or J is None:
#                 raise ValueError('Input h and J')
#             model = BinaryQuadraticModel(h=h, J=J, var_type='SPIN')
#         self.indices = model.indices
#         self.N = len(model.indices)
#         self.energy_bias = model.energy_bias

#         self.var_type = model.var_type

#         dense_graph = model.convert_to_dense_graph()
#         self.model = model
#         return dense_graph

#     def _sampling_kwargs_setting(self, **kwargs):
#         if 'iteration' in kwargs:
#             self.iteration = kwargs['iteration']
#         elif 'num_reads' in kwargs:
#             self.iteration = kwargs['num_reads']

#     def _sampling(self, model, init_generator, algorithm, system,
#                   initial_state=None,
#                   reinitialize_state=True, seed=None, **kwargs):
#         self._sampling_kwargs_setting(**kwargs)
#         self._set_model(model)

#         # seed for MonteCarlo
#         if seed is None:
#             def sampling_algorithm(system): return algorithm(
#                 system, self.schedule)
#         else:
#             def sampling_algorithm(system): return algorithm(
#                 system, seed, self.schedule)

#         # run algorithm
#         execution_time = []
#         response = openjij.Response(
#             var_type=model.var_type, indices=self.indices)

#         @measure_time
#         def exec_sampling():
#             previous_state = init_generator()
#             for _ in range(self.iteration):
#                 if reinitialize_state:
#                     system.reset_spins(init_generator())
#                 else:
#                     system.reset_spins(previous_state)
#                 _exec_time = measure_time(sampling_algorithm)(system)
#                 execution_time.append(_exec_time)
#                 previous_state = cxxjij.result.get_solution(system)
#                 self._post_save(previous_state, system, model, response)

#         sampling_time = exec_sampling()
#         response.info['sampling_time'] = sampling_time * 10**6  # micro sec
#         response.info['execution_time'] = np.mean(
#             execution_time) * 10**6  # micro sec
#         response.info['list_exec_times'] = np.array(
#             execution_time) * 10**6  # micro sec

#         return response

#     def _post_save(self, result_state, system, model, response):
#         pass
