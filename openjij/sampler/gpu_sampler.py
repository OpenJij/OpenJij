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
from openjij.sampler import BaseSampler
from openjij.model import BinaryQuadraticModel
import numpy as np

class GPUSQASampler(BaseSampler):
    def __init__(self, graph=None, beta=5.0, gamma_min=0.1, gamma_max=10.0,
                 trotter=5, step_length=10, step_num=100, iteration=1):

        if graph is not None:
            if not isinstance(graph, cj.graph.Chimera):
                raise ValueError("'graph' should be cxxjij.graph.Chimera, not {}".format(type(graph)))
        
        self.beta = beta
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.trotter = trotter
        self.step_length = step_length
        self.step_num = step_num
        self.iteration = iteration

    def sample_ising(self, h, J, chimera_L):
        chimera = self._chimera_graph(h, J, chimera_L)
        gpu_sqa = cj.method.ChimeraGPUQuantum(chimera, num_trotter_slices=self.trotter)
        for _ in range(self.iteration):
            gpu_sqa.simulated_quantum_annealing(
                self.beta, 
                self.gamma_min, self.gamma_max,
                self.step_length, self.step_num)
            q_state = gpu_sqa.get_spins()
            energies = [chimera.calc_energy(state) + self.energy_bias for state in q_state]
            response.add_quantum_state_energy(q_state, energies)
        return response


    def _chimera_graph(self, h, J, chimera_L):
        model = BinaryQuadraticModel(h=h, J=J)
        if not model.validate_chimera():
            raise ValueError("Problem graph incompatible with chimera graph.")
        _h, _J = model.ising_dictionary()
        chimera = cj.graph.Chimera(chimera_L, chimera_L)
        for i, hi in _h.items():
            xi, yi, zi = model._index_chimera(i)
            chimera[xi, yi, zi] = hi
        for (i, j), Jij in _J.items():
            xi, yi, zi = model._index_chimera(i)
            xj, yj, zj = model._index_chimera(j)
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
        

    def sampling(self):
        pass
