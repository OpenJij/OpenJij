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
import cxxjij.graph as cjg
import warnings

class BinaryQuadraticModel:
    def __init__(self, h=None, J=None, Q=None, spin_type='ising'): 

        if spin_type == 'ising':
            if (h is None) and (J is None):
                raise ValueError('Input h and J.')
        elif spin_type=='qubo':
            if not isinstance(Q, dict) or Q is None:
                raise ValueError('Q should be dictionary.')
            h = {}
            J = {}
            for (i,j),qij in Q.items():
                if i==j:
                    h[i] = qij
                else:
                    J[(i, j)] = qij
        else:
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
        if spin_type == 'ising':
            self.energy_bias = 0.0
        else:
            self.energy_bias = (sum(list(h.values()))*2 + sum(list(J.values())))/4


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


    def ising_dictionary(self):
        if self.spin_type == 'ising':
            return self.h, self.J
        elif self.spin_type == 'qubo':
            ising_int = self.ising_interactions()
            h = {}
            J = {(i,j): qij/4.0 for (i, j), qij in self.J.items()}
            for i in range(len(ising_int)):
                if ising_int[i][i] != 0:
                    h[self.indices[i]] = ising_int[i][i]
        return h, J

    def convert_to_dense_graph(self) -> cjg.Dense:
        """
        Convert to cxxjij.graph.Dense class from Python dictionary (h, J) or Q
        """
        N = len(self.indices)
        ising_int = self.ising_interactions()

        # cxxjij.graph.dense
        cxx_dense_ising = cjg.Dense(N)
        for i in range(N):
            if ising_int[i,i] != 0.0:
                cxx_dense_ising[i,i] = ising_int[i,i]
            for j in range(i+1, N):
                if ising_int[i,j] != 0.0:
                    cxx_dense_ising[i,j] = ising_int[i,j]
        
        return cxx_dense_ising

    def validate_chimera(self, unit_num_L):
        """
        Chimera coordinate: x, y, z
        One dimension coordinate: i
        Relation: i = 8Ly + 8x + z
        """
        # check chimera interaction
        for (i,j) in self.J.keys():
            z_i = i % 8
            x_i = (i - z_i) % unit_num_L / 8
            y_i = (i-(8*x_i + z_i))/(8 * unit_num_L)
            # list up indices which can connect i
            adj_list = []
            if z_i < 4:
                if y_i > 0:
                    adj_list.append(self._chimera_index(x_i, y_i-1, z_i, unit_num_L))
                if y_i < unit_num_L-1:
                    adj_list.append(self._chimera_index(x_i, y_i+1, z_i, unit_num_L))
                adj_list += [self._chimera_index(x_i, y_i, z, unit_num_L) for z in range(4, 8)]
            else:
                if x_i > 0:
                    adj_list.append(self._chimera_index(x_i, y_i, z_i-1, unit_num_L))
                if x_i < unit_num_L-1:
                    adj_list.append(self._chimera_index(x_i, y_i, z_i+1, unit_num_L)) 
                adj_list += [self._chimera_index(x_i, y_i, z, unit_num_L) for z in range(0, 4)]
            
            if not j in adj_list:
                return False
        return True

    def _chimera_index(self, x, y, z, L):
        """
        Chimera coordinate: x, y, z
        One dimension coordinate: i
        Relation: i = 8Ly + 8x + z
        """
        return 8*L*y + 8*x + z

    def _index_chimera(self, i, unit_num_L):
        z_i = i % 8
        x_i = (i - z_i) % unit_num_L / 8
        y_i = (i-(8*x_i + z_i))/(8 * unit_num_L)
        return int(x_i), int(y_i), int(z_i)