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
import warnings

class BinaryQuadraticModel:
    def __init__(self, h, J, spin_type='ising'): 

        if not spin_type in ('ising', 'qubo'):
            raise ValueError('spin_type should be "ising" or "qubo"')

        index_set = set(h.keys())
        warning_called = False
        for v1, v2 in J.keys():
            indices_len = len(index_set)
            index_set.add(v1)
            index_set.add(v2)

            # When the same index add to index set, check the existence of inverse indices in the J
            if warning_called or (len(index_set) - indices_len < 2 and (v2, v1) in J):
                warn_message = 'Two connections J[(a, b)] and J[(b, a)] are defined. ' \
                               'Adopt the (lower index, higher index) connection. ' \
                               'Please pay attention to the symmetry of interaction J.'
                warnings.warn(warn_message, SyntaxWarning)
        self.indices = list(index_set)
        self.h = h
        self.J = J
        self.spin_type = spin_type
        self.energy_bias = 0.0

    def ising_interactions(self):
        interactions = self.interactions()
        if self.spin_type == 'qubo':
            self.energy_bias = (np.sum(np.triu(interactions)) + np.trace(interactions))/4.0
            for i in range(len(interactions)):
                interactions[i, i] = np.sum(interactions[i, :]) + interactions[i, i]
            interactions /= 4.0
        return interactions

    def interactions(self):
        system_size = len(self.indices)
        interactions = np.zeros((system_size, system_size))

        for i, i_index in enumerate(self.indices):
            interactions[i, i] = self.h[i_index] if i_index in self.h else 0.0
            for j, j_index in enumerate(self.indices[i+1:]):
                j += i+1
                if (i_index, j_index) in self.J:
                    jval = self.J[(i_index, j_index)]
                elif (j_index, i_index) in self.J:
                    jval = self.J[(j_index, i_index)]
                else:
                    jval = 0.0
                interactions[i, j] = jval
                interactions[j, i] = jval

        return interactions

    def calc_energy(self, state):
        if self.spin_type == 'ising':
            int_mat = self.ising_interactions()
        else: # spin_type == qubo
            int_mat = self.interactions()
        return np.dot(state, np.dot(int_mat, state)) + np.dot(np.diag(int_mat), -1+np.array(state))
