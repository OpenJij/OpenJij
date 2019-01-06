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

class Response:
    def __init__(self, states, energies, var_type="spin"):
        self.states = states
        self.energies = energies

        if var_type == "bit":
            pass


class Sampler:
    def __init__(self, model):
        self.indices = model.indices
        self.N = len(model.indices)
        # make interaction matrix
        ising_int = model.ising_interactions()

        # todo: Rewrite the following converting handle into more efficient code.
        self.int_mat = cj.SquareMatrix(self.N, 0.0)
        for i in range(self.N):
            for j in range(i, self.N):
                self.int_mat.set(i, j, ising_int[i, j])
                self.int_mat.set(j, i, ising_int[j, i])

    def sampling(self, beta, observe_num, burn_in=1000):
        samp = cj.Sampler(self.int_mat)
        resu = cj.Results()
        samp.sampling(beta, burn_in, observe_num, resu)

        return Response(states=resu.states, energies=resu.energies)


    def simulated_annealing(self, beta_min=0.1, beta_max=5.0, step_length=10, step_num=100, iteration=1):
        samp = cj.Sampler(self.int_mat)
        resu = cj.Results()
        samp.simulated_annealing(beta_min, beta_max, step_length, step_num, iteration, resu)
        return Response(states=resu.states, energies=resu.energies) 


    def simulated_quantum_annealing(self, beta=5.0, gamma_min=0.1, gamma_max=10.0, trotter=5, step_length=10, step_num=100, iteration=1):
        samp = cj.Sampler(self.int_mat)
        resu = cj.Results()
        samp.simulated_quantum_annealing(beta, gamma_min, gamma_max, trotter, step_length, step_num, iteration, resu)
        return Response(states=resu.states, energies=resu.energies)
 
