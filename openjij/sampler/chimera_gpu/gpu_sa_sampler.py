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
from .base_gpu_chimera import BaseGPUChimeraSampler
from openjij.utils.graph_utils import chimera_to_ind
import numpy as np
import dimod


class GPUChimeraSASampler(SASampler, BaseGPUChimeraSampler):
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

    Raises:
        ValueError: [description]
        AttributeError: [description]
        ValueError: [description]
        ValueError: [description]

    """

    def __init__(self,
                 beta_min=None, beta_max=None,
                 num_sweeps=1000, schedule=None,
                 num_reads=1, unit_num_L=None,
                 **kwargs):

        super().__init__(beta_min=beta_min, beta_max=beta_max, 
                         num_reads=num_reads, num_sweeps=num_sweeps, 
                         schedule=schedule, **kwargs)

        self.unit_num_L = unit_num_L

        self._make_system = {
            'singlespinflip': cxxjij.system.make_chimera_classical_gpu
        }
        self._algorithm = {
            'singlespinflip': cxxjij.algorithm.Algorithm_GPU_run
        }

    def sample_ising(self, h, J, beta_min=None, beta_max=None,
                     num_sweeps=None, num_reads=1, schedule=None,
                     initial_state=None, updater='single spin flip',
                     reinitialize_state=True, seed=None, unit_num_L=None,
                     **kwargs):
        

        self.unit_num_L = unit_num_L if unit_num_L else self.unit_num_L

        model = openjij.ChimeraModel(linear=h, quadratic=J, var_type='SPIN', 
                                   unit_num_L=self.unit_num_L, gpu=True)
        return self._sampling(model, beta_min, beta_max,
                              num_sweeps, num_reads, schedule,
                              initial_state, updater,
                              reinitialize_state, seed, **kwargs)

        


    

