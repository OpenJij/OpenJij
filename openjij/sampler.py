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
from .model import BinaryQuadraticModel


class Response:
    def __init__(self, states, energies, spin_type="ising"):
        self.states = states
        self.energies = energies

        if spin_type == "qubo":
            qubo_state = [list(np.array((np.array(state) + 1)/2).astype(np.int)) for state in states]
            self.states = qubo_state


class BaseSampler:
    def _make_int_mat(self, Q=None, h=None, J=None, spin_type='ising'):
        if spin_type=='qubo':
            if not isinstance(Q, dict):
                ValueError('Q should be dictionary.')
            h = {}
            J = {}
            for (i,j),qij in Q.items():
                if i==j:
                    h[i] = qij
                else:
                    J[(i, j)] = qij
            
        model = BinaryQuadraticModel(h, J, spin_type=spin_type)
        self.indices = model.indices
        self.N = len(model.indices)

        self.spin_type = model.spin_type

        # make interaction matrix
        ising_int = model.ising_interactions()

        # todo: Rewrite the following converting handle into more efficient code.
        # self.int_mat = cj.SquareMatrix(self.N, 0.0)
        self.graph = cj.graph.Dense(self.N)
        for i in range(self.N):
            for j in range(i, self.N):
                # self.int_mat.set(i, j, ising_int[i, j])
                # self.int_mat.set(j, i, ising_int[j, i])
                self.graph[i,j] = ising_int[i,j]

        self.samp = cj.Sampler(self.graph)
        self.resu = cj.Results()


class Sampler(BaseSampler):
    def __init__(self, beta, observe_num, burn_in=1000):
        self.beta = beta
        self.observe_num = observe_num
        self.burn_in = burn_in

    def sampling(self, h, J, spin_type='ising'):
        self._make_int_mat(h=h, J=J, spin_type=spin_type)
        self.samp.sampling(self.beta, self.burn_in, self.observe_num, self.resu)

        return Response(states=self.resu.states, energies=self.resu.energies, spin_type=spin_type)


class SASampler(BaseSampler):
    def __init__(self, beta_min=0.1, beta_max=5.0,
                 step_length=10, step_num=100, iteration=1):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.step_length = step_length
        self.step_num = step_num
        self.iteration = iteration

    def sample_ising(self, h, J):
        self._make_int_mat(h=h, J=J, spin_type='ising')
        self.samp.simulated_annealing(
            self.beta_min, self.beta_max, self.step_length,
            self.step_num, self.iteration, self.resu)
        return Response(states=self.resu.states, energies=self.resu.energies)

    def sample_qubo(self, Q):
        self._make_int_mat(Q, spin_type='qubo')
        self.samp.simulated_annealing(
            self.beta_min, self.beta_max, self.step_length,
            self.step_num, self.iteration, self.resu)
        return Response(states=self.resu.states, energies=self.resu.energies, spin_type='qubo')
        



class SQASampler(BaseSampler):
    def __init__(self, beta=5.0, gamma_min=0.1, gamma_max=10.0,
                 trotter=5, step_length=10, step_num=100, iteration=1):
        self.beta = beta
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.trotter = trotter
        self.step_length = step_length
        self.step_num = step_num
        self.iteration = iteration

    def sample_ising(self, h, J):
        self._make_int_mat(h=h, J=J, spin_type='ising')
        self.samp.simulated_quantum_annealing(
            self.beta, self.gamma_min, self.gamma_max, self.trotter,
            self.step_length, self.step_num, self.iteration, self.resu)
        return Response(states=self.resu.states, energies=self.resu.energies)

    def sample_qubo(self, Q):
        self._make_int_mat(Q, spin_type='qubo')
        self.samp.simulated_quantum_annealing(
            self.beta, self.gamma_min, self.gamma_max, self.trotter,
            self.step_length, self.step_num, self.iteration, self.resu)
        return Response(states=self.resu.states, energies=self.resu.energies, spin_type='qubo')
