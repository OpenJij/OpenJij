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

import cxxjij as cj
from openjij.sampler import SQASampler
from openjij.model import BinaryQuadraticModel, ChimeraModel
from .response import Response
import numpy as np

class GPUSQASampler(SQASampler):
    def __init__(self, beta=5.0, gamma=1.0,
                 trotter=4, step_length=10, step_num=100, schedule=None, iteration=1, unit_num_L=None):
        # GPU Sampler allows only even trotter number
        if trotter % 2 != 0:
            raise ValueError('GPU Sampler allows only even trotter number')
        self.trotter = trotter

        self.unit_num_L = unit_num_L

        super().__init__(beta, gamma, trotter, step_length, step_num, schedule, iteration)

    def _post_process4state(self, q_state):
        if self.model.coordinate == 'chimera coordinate':
            indices = [self.model.to_index(x, y, z, self.model.unit_num_L) for x,y,z in self.indices]
        else:
            indices = self.indices
        
        return [list(np.array(state)[indices]) for state in q_state]

    def sampling(self, model, **kwargs):
        # Check the system for GPU is compiled
        try:
            self.system_class = cj.system.ChimeraGPUQuantum
        except AttributeError:
            raise AttributeError('Does the computer you are running have a GPU? Compilation for the GPU has not been done. Please reinstall or compile.')

        if isinstance(model, BinaryQuadraticModel):
            if 'unit_num_L' in kwargs:
                self.unit_num_L = kwargs['unit_num_L']
            elif not self.unit_num_L:
                raise ValueError('Input "unit_num_L" to the argument or the constructor of GPUSQASampler.')
            chimera_model = ChimeraModel(model=model, unit_num_L=self.unit_num_L)
        else:
            chiemra_model = model


        if chimera_model.unit_num_L % 2 != 0:
            raise ValueError('unit_num_L should be even number.')

        self.unit_num_L = chimera_model.unit_num_L
        self._set_model(chimera_model)

        chimera = self.model.get_chimera_graph() 

        response = self._sampling(ising_graph=chimera, var_type=self.var_type)

        return response

    def sample_ising(self, h, J, **kwargs):
        model = BinaryQuadraticModel(h=h, J=J, var_type='SPIN')
        self.var_type = 'SPIN'
        return self.sampling(model, **kwargs)

    def sample_qubo(self, Q, **kwargs):
        model = BinaryQuadraticModel(Q=Q, var_type='BINARY')
        self.var_type = 'BINARY'
        return self.sampling(model, **kwargs)
        
    def _set_model(self, model):
        self.model = model
        self.indices = model.indices
        self.energy_bias = model.energy_bias
        self.var_type = model.var_type

    
        

