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
from .sampler import Response
import numpy as np

class GPUSQASampler(SQASampler):
    def __init__(self, beta=5.0, gamma=1.0,
                 trotter=4, step_length=10, step_num=100, schedule=None, iteration=1):
        # GPU Sampler allows only even trotter number
        if trotter % 2 != 0:
            raise ValueError('GPU Sampler allows only even trotter number')
        self.trotter = trotter

        super().__init__(beta, gamma, trotter, step_length, step_num, schedule, iteration)

        try:
            self.system_class = cj.system.ChimeraGPUQuantum
        except AttributeError:
            raise AttributeError('Does the computer you are running have a GPU? Compilation for the GPU has not been done. Please reinstall or compile.')



    def sampling(self, model, chimera_L):

        if isinstance(model, BinaryQuadraticModel):
            chimera_model = ChimeraModel(model=model)

        chimera = self._chimera_graph(model, chimera_L)

        return self._sampling(ising_graph=chimera, var_type=self.var_type)


    def sample_ising(self, h, J, chimera_L):
        model = ChimeraModel(h=h, J=J, var_type='SPIN')
        self.var_type = 'SPIN'
        return self.sampling(model, chimera_L)

    def sample_qubo(self, Q, chimera_L):
        model = ChimeraModel(Q=Q, var_type='BINARY')
        self.var_type = 'BINARY'
        return self.sampling(model, chimera_L)
        
    def _set_model(self, model):
        self.model = model
        self.indices = model.indices
        self.energy_bias = model.energy_bias

    def _chimera_graph(self, model, chimera_L):
        self._set_model(model)

        if chimera_L % 2 != 0:
            raise ValueError('chimera_L should be even number.')

        if not model.validate_chimera(chimera_L):
            raise ValueError("Problem graph incompatible with chimera graph.")
        _h, _J = model.ising_dictionary()

        self.energy_bias = model.energy_bias

        chimera = cj.graph.Chimera(chimera_L, chimera_L)
        for i, hi in _h.items():
            xi, yi, zi = model.chimera_coordinate(i, unit_num_L=chimera_L)
        for (i, j), Jij in _J.items():
            xi, yi, zi = model.chimera_coordinate(i, unit_num_L=chimera_L)
            xj, yj, zj = model.chimera_coordinate(j, unit_num_L=chimera_L)
            if xi == xj and yi == yj:
                if zj in [0, 4]:
                    chimera[xi, yi, cj.graph.ChimeraDir.IN_0or4] = Jij
                elif zj in [1, 5]:
                    chimera[xi, yi, cj.graph.ChimeraDir.IN_1or5] = Jij
                elif zj in [2, 6]:
                    chimera[xi, yi, cj.graph.ChimeraDir.IN_2or6] = Jij
                else:
                    chimera[xi, yi, cj.graph.ChimeraDir.IN_1or7] = Jij
            elif xi == xj + 1:
                chimera[xi, yi, cj.graph.ChimeraDir.PLUS_R] = Jij
            elif xi == xj - 1:
                chimera[xi, yi, cj.graph.ChimeraDir.MINUS_R] = Jij 
            elif yi == yj + 1:
                chimera[xi, yi, cj.graph.ChimeraDir.PLUS_C] = Jij
            elif yi == yj - 1:
                chimera[xi, yi, cj.graph.ChimeraDir.MINUS_C] = Jij
        return chimera
        

