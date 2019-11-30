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

import cxxjij
import openjij
from openjij.sampler import SASampler
from openjij.model import BinaryQuadraticModel, ChimeraModel
import numpy as np


class GPUSASampler(SASampler):
    """Sampler with Simulated Annealing (SA) on GPU.

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.

    Args:
        beta_min (float):
            Minimum inverse temperature.
        beta_max (float):
            Maximum inverse temperature.
        num_sweeps (int):
            Length of Monte Carlo step.

        schedule_info (dict):
            Information about a annealing schedule.

        num_reads (int):
            Number of iterations.

        unit_num_L (int):
            Length of one side of two-dimensional lattice
            in which chimera unit cells are arranged.

    Attributes:
        indices (int):
            Indices of `openjij.model.model.BinaryQuadraticModel` model.

        energy_bias (float):
            Energy bias.

        model (:obj:):
             `openjij.model.model.BinaryQuadraticModel` model.

        var_type (str):
            Type of variables: 'SPIN' or 'BINARY' which mean {-1, 1} or {0, 1}.

        system_class (:class:):
            `cxxjij.system.QuantumIsing` class.

        sqa_kwargs (dict):
            Parameters of SQA: beta, gamma, and schedule_info.

    Raises:
        ValueError: If variables violate as below.
        - trotter number is odd.
        - no input "unit_num_L" to an argument or this constructor.
        - given problem graph is incompatible with chimera graph.

        AttributeError: If GPU doesn't work.

    """

    def __init__(self,
                 beta_min=0.1, beta_max=5.0,
                 num_sweeps=10,
                 schedule=None, num_reads=1, unit_num_L=None):

        self.unit_num_L = unit_num_L

        super().__init__(beta_min, beta_max, num_reads=num_reads,
                         num_sweeps=num_sweeps, schedule=schedule)

    def _dict_to_model(self, var_type, h=None, J=None, Q=None, **kwargs):

        if 'unit_num_L' in kwargs:
            self.unit_num_L = kwargs['unit_num_L']
        elif not self.unit_num_L:
            raise ValueError(
                'Input "unit_num_L" to the argument or the constructor of GPUSASampler.')

        chimera = openjij.ChimeraModel(h=None, J=None, Q={(
            (0, 0, 1), (0, 0, 4)): -1}, var_type=openjij.BINARY, unit_num_L=2, gpu=True)

        return chimera

    def sampling(self, model,
                 initial_state=None,
                 reinitialize_state=True, seed=None,
                 **kwargs):
        # Check the system for GPU is compiled
        try:
            self.system_class = cxxjij.system.ChimeraClassicalGPU
        except AttributeError:
            raise AttributeError(
                'Does the computer you are running have a GPU? Compilation for the GPU has not been done. Please reinstall or compile.')

        # convert to ChimeraModel from normal BQM
        if isinstance(model, BinaryQuadraticModel):
            if 'unit_num_L' in kwargs:
                self.unit_num_L = kwargs['unit_num_L']
            elif not self.unit_num_L:
                raise ValueError(
                    'Input "unit_num_L" to the argument or the constructor of GPUSASampler.')
            chimera_model = ChimeraModel(
                model=model, unit_num_L=self.unit_num_L, gpu=True)
        else:
            chimera_model = model

        if chimera_model.unit_num_L % 2 != 0:
            raise ValueError('unit_num_L should be even number.')

        self.unit_num_L = chimera_model.unit_num_L
        self._set_model(chimera_model)

        chimera = self.model.get_chimera_graph()

        # use all spins ?
        self._use_all = len(model.indices) == (
            self.unit_num_L * self.unit_num_L * 8)

        if initial_state is None:
            def init_generator(): return chimera.gen_spin()
        else:
            if model.var_type == openjij.SPIN:
                _init_state = np.array(initial_state)
            else:  # BINARY
                _init_state = (2*np.array(initial_state)-1).astype(int)

            def init_generator(): return _init_state

        algorithm = cxxjij.algorithm.Algorithm_GPU_run

        sa_system = cxxjij.system.make_chimera_classical_gpu(
            init_generator(), chimera
        )

        response = self._sampling(
            chimera_model, init_generator,
            algorithm, sa_system, initial_state,
            reinitialize_state, seed, **kwargs
        )

        response.update_ising_states_energies(
            response.states, response.energies)

        return response

    def _post_save(self, result_state, system, model, response):
        if not self._use_all:
            if model.coordinate == 'chimera coordinate':
                indices = [model.to_index(
                    x, y, z, model.unit_num_L) for x, y, z in model.indices]
            else:
                indices = model.indices
            result_state = np.array(result_state)[indices]
        response.states.append(result_state)
        response.energies.append(model.calc_energy(
            result_state,
            need_to_convert_from_spin=True))

    def _set_model(self, model):
        self.model = model
        self.indices = model.indices
        self.energy_bias = model.energy_bias
        self.var_type = model.var_type
