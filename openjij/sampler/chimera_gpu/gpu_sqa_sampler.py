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
from openjij.sampler import SQASampler
from openjij.model import BinaryQuadraticModel, ChimeraModel
import numpy as np

from .GPUBaseSampler import GPUBaseSampler


class GPUSQASampler(SQASampler, GPUBaseSampler):
    """Sampler with Simulated Quantum Annealing (SQA) on GPU.

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.

    Args:
        beta (float):
            Inverse temperature.

        gamma (float):
            Amplitude of quantum fluctuation.

        trotter (int):
            Trotter number.

        num_sweeps (int):

        schedule_info (dict):
            Information about a annealing schedule.

        iteration (int):
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

    def __init__(self, beta=5.0, gamma=1.0,
                 trotter=4, num_sweeps=100,
                 schedule=None, num_reads=1, unit_num_L=None):
        # GPU Sampler allows only even trotter number
        if trotter % 2 != 0:
            raise ValueError('GPU Sampler allows only even trotter number')
        self.trotter = trotter

        self.unit_num_L = unit_num_L

        super().__init__(beta=beta, gamma=gamma, trotter=trotter,
                         num_reads=num_reads,
                         num_sweeps=num_sweeps, schedule=schedule)

    def _post_process4state(self, q_state):
        if self.model.coordinate == 'chimera coordinate':
            indices = [self.model.to_index(
                x, y, z, self.model.unit_num_L) for x, y, z in self.indices]
        else:
            indices = self.indices

        return [list(np.array(state)[indices]) for state in q_state]

    def sample(self, model,
               beta=None, gamma=None,
               num_sweeps=None, num_reads=1, schedule=None,
               initial_state=None, reinitialize_state=None,
               seed=None, **kwargs):

        self._setting_overwrite(
            beta=beta, gamma=gamma,
            num_sweeps=num_sweeps, num_reads=num_reads
        )

        self._annealing_schedule_setting(
            model, beta, gamma, num_sweeps, schedule)

        chimera = self._create_chimera_matrix(model, **kwargs)

        if initial_state is None:
            def init_generator(): return [chimera.gen_spin()
                                          for _ in range(self.trotter)]
        else:
            if model.var_type == openjij.SPIN:
                trotter_init_state = [np.array(initial_state)
                                      for _ in range(self.trotter)]
            else:  # BINARY
                trotter_init_state = [
                    (2*np.array(initial_state)-1).astype(int)
                    for _ in range(self.trotter)]

            def init_generator(): return trotter_init_state

        algorithm = cxxjij.algorithm.Algorithm_GPU_run

        def system_maker(init_state): return cxxjij.system.make_chimera_transverse_gpu(
            init_state, chimera, self.gamma
        )

        response = self._gpu_sampling(
            system_name="quantum",
            init_generator=init_generator,
            system_maker=system_maker
        )
        return response

    def _post_save(self, result_state, system, model, response):

        if not self._use_all:
            result_state = np.array(result_state)[model.indices]

        response.states.append(result_state)
        response.energies.append(model.calc_energy(
            result_state,
            need_to_convert_from_spin=True))
