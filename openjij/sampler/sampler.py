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
    """Base sampler class of python wrapper for cxxjij simulator

    """
    parameters = None
    properties = None

    def _set_model(self, model):
        self.indices = model.indices
        self.size = model.size
        self.energy_bias = model.energy_bias
        self.var_type = model.var_type

    def _sampling(self, model, init_generator,
                  algorithm, system,
                  reinitialize_state=None,
                  seed=None, **kwargs):
        """Basic sampling function: for cxxjij sampling

        Args:
            model (openjij.BinaryQuadraticModel): model has a information of instaunce (h, J, Q) 
            init_generator (callable): return initial state 
            algorithm (callable): system algorithm of cxxjij
            system (:obj:): [description]
            reinitialize_state (bool, optional): [description]. Defaults to None.
            seed (int, optional): seed for algorithm. Defaults to None.

        Returns:
            [type]: [description]
        """

        self._set_model(model)

        # set algorithm function and set random seed
        if seed is None:
            def sampling_algorithm(system): return algorithm(
                system, self._schedule)
        else:
            def sampling_algorithm(system): return algorithm(
                system, seed, self._schedule)

        # setting of response class
        execution_time = []
        response = openjij.Response(var_type=model.var_type,
                                    indices=self.indices)

        # setting of sampling execution function
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

        # Execute sampling function
        sampling_time = exec_sampling()

        # save execution time
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
