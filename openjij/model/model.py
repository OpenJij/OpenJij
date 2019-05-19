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
    def __init__(self, h=None, J=None, Q=None, var_type='SPIN'): 
        if var_type == 'SPIN':
            if (h is None) or (J is None):
                raise ValueError('Input h and J.')
            self.linear = h
            self.quad = J
        elif var_type=='BINARY':
            if not isinstance(Q, dict) or Q is None:
                raise ValueError('Q should be dictionary.')
            self.linear = {}
            self.quad = {}
            for (i,j),qij in Q.items():
                if i==j:
                    self.linear[i] = qij
                else:
                    self.quad[(i, j)] = qij
        else:
            raise ValueError('var_type should be "SPIN" or "BINARY"')

        index_set = set(self.linear.keys())
        for v1, v2 in self.quad.keys():
            indices_len = len(index_set)
            index_set.add(v1)
            index_set.add(v2)

            # When the same index add to index set, check the existence of inverse indices in the J
            if (len(index_set) - indices_len < 2 and (v2, v1) in self.quad):
                warn_message = 'Two connections J[(a, b)] and J[(b, a)] are defined. ' \
                               'Adopt the (lower index, higher index) connection. ' \
                               'Please pay attention to the symmetry of interaction J.'
                warnings.warn(warn_message, SyntaxWarning)
        self.indices = list(index_set)
        self.var_type = var_type
        if var_type == 'SPIN':
            self.energy_bias = 0.0
        else:
            self.energy_bias = (sum(list(self.linear.values()))*2 + sum(list(self.quad.values())))/4


        self._interaction_matrix = None  # calculated at interactions()


    def ising_interactions(self):
        interactions = self.interactions()
        if self.var_type == 'BINARY':
            self.energy_bias = (np.sum(np.triu(interactions)) + np.trace(interactions))/4.0
            for i in range(len(interactions)):
                interactions[i, i] = np.sum(interactions[i, :]) + interactions[i, i]
            interactions /= 4.0
        return interactions

    def interactions(self, re_calculate=False):

        if (self._interaction_matrix is not None) and (not re_calculate):
            return self._interaction_matrix

        system_size = len(self.indices)
        interactions = np.zeros((system_size, system_size))

        for i, i_index in enumerate(self.indices):
            interactions[i, i] = self.linear[i_index] if i_index in self.linear else 0.0
            for j, j_index in enumerate(self.indices[i+1:]):
                j += i+1
                if (i_index, j_index) in self.quad:
                    jval = self.quad[(i_index, j_index)]
                elif (j_index, i_index) in self.quad:
                    jval = self.quad[(j_index, i_index)]
                else:
                    jval = 0.0
                interactions[i, j] = jval
                interactions[j, i] = jval

        self._interaction_matrix = interactions

        return self._interaction_matrix

    def calc_energy(self, state):
        if self.var_type == 'SPIN':
            int_mat = self.ising_interactions()
        else: # var_type == qubo
            int_mat = self.interactions()
        return np.dot(state, np.dot(int_mat, state)) + np.dot(np.diag(int_mat), -1+np.array(state))


    def ising_dictionary(self):
        if self.var_type == 'SPIN':
            return self.linear, self.quad
        elif self.var_type == 'BINARY':
            ising_int = self.ising_interactions()
            h = {}
            J = {(i,j): qij/4.0 for (i, j), qij in self.quad.items()}
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



