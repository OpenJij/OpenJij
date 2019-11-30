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
import cxxjij
import openjij
import dimod
import warnings


class BinaryQuadraticModel(dimod.BinaryQuadraticModel):
    """Represents Binary quadratic model
    Attributes:
        var_type (openjij.VariableType): variable type SPIN or BINARY
        linear (dict): represents linear term
        quad (dict): represents quadratic term
        indices (list): labels of each variables sorted by results variables
        energy_bias (float): represents constant energy term when convert to SPIN from BINARY
        size (int): number of variables
    """

    def __init__(self, linear, quadratic, offset=0.0,
                 var_type=openjij.SPIN, **kwargs):
        super().__init__(linear, quadratic, offset, var_type, **kwargs)

        self.var_type = var_type

        index_set = set(self.linear.keys())
        for v1, v2 in self.quadratic.keys():
            index_set.add(v1)
            index_set.add(v2)
        self.indices = list(index_set)
        self._interaction_matrix = None  # calculated at interactions()
        self.size = len(self.indices)

        if var_type == openjij.SPIN:
            self.energy_bias = 0.0
        else:  # BINARY
            self.energy_bias = (sum(list(self.linear.values()))
                                * 2 + sum(list(self.quadratic.values())))/4

    def get_cxxjij_ising_graph(self, sparse=False):
        """
        Convert to cxxjij.graph.Dense or Sparse class from Python dictionary (h, J) or Q
        Args:
            sparse (bool): if true returns sparse graph
        Returns:
            openjij.graph.Dense openjij.graph.Sparse
        """

        if not sparse:
            GraphClass = cxxjij.graph.Dense
        else:
            GraphClass = cxxjij.graph.Sparse

        cxxjij_graph = GraphClass(self.size)

        ising_int = self.ising_interactions()
        # cxxjij.graph.dense
        for i in range(self.size):
            if ising_int[i, i] != 0.0:
                cxxjij_graph[i, i] = ising_int[i, i]
            for j in range(i+1, self.size):
                if ising_int[i, j] != 0.0:
                    cxxjij_graph[i, j] = ising_int[i, j]

        return cxxjij_graph

    def ising_interactions(self):
        """ Interactions in the Ising representation
        QUBO formulation to the Ising formulation
        We assumption Q is a triangular matrix.
        H = q ^ T Q q
          = 1/4 (1+s) ^ T Q(1+s)
          = s ^ T Q/4 s + 1 ^ T Q/4 s + s ^ T Q/4 1 + 1 ^ T Q/4 1
          = s ^ T nondiag(Q)/4 s + 1 ^ T Q/4 s + s ^ T Q/4 1 + 1 ^ T(Q + diag(Q))/4 1
          = \sum_{i < j} Q_{ij}/4 s_i s_j
            + \sum{i <= j}(Q_{ij} + Q_{ji}) s_i
            + sum_{i <= j}(Q_{ij} + Q_{ii})/4
        Therefore
          J_{ij} = Q_{ij}/4
          h_i = sum{i <= j}(Q_{ij} + Q_{ji})/4
          constant_term = sum_{i <= j} Q_{ij}/4 + Tr(Q)(energy bias)
        """
        if self.var_type == openjij.SPIN:
            return self.interactions()
        interactions = self.interactions()/4
        if self.var_type == openjij.BINARY:
            # convert to the Ising interaction
            self.energy_bias = (
                np.sum(np.triu(interactions)) + np.trace(interactions))
            for i in range(len(interactions)):
                interactions[i, i] = np.sum(
                    interactions[i, :]) + interactions[i, i]
        return interactions

    def interactions(self, re_calculate=False):
        """make interaction matrix
        The Ising model: E = ΣJ_ij σiσj + Σhiσi
            Interaction matrix -> H_ij = J_ij + J_ji, H_ii = hi
        QUBO: E = Σ1/2Q_ij q_iq_j + ΣQ_ii q_i

        Args:
            re_calculate(bool): Whether to force a recalculation
        Returns:
            numpy.ndarray: interactioin matrix H_{ij} or Q_{ij}
        """

        if (self._interaction_matrix is not None) and (not re_calculate):
            return self._interaction_matrix

        system_size = len(self.indices)
        interactions = np.zeros((system_size, system_size))

        for i, i_index in enumerate(self.indices):
            interactions[i, i] = self.linear[i_index] if i_index in self.linear else 0.0
            for j, j_index in enumerate(self.indices[i+1:]):
                j += i+1
                jval = 0.0
                if (i_index, j_index) in self.quadratic:
                    jval = self.quadratic[(i_index, j_index)]
                if (j_index, i_index) in self.quadratic:
                    jval = self.quadratic[(j_index, i_index)]
                interactions[i, j] = jval
                interactions[j, i] = jval

        self._interaction_matrix = interactions

        return self._interaction_matrix

    def calc_energy(self, state, need_to_convert_from_spin=False):
        """calculate energy from state
        Args:
            state(list, numpy.array): BINARY or SPIN state
            need_to_convet_to_spin(bool): if state is SPIN and need
                                           to convert to BINARY from SPIN
        """
        _state = np.array(state)
        if need_to_convert_from_spin and self.var_type == openjij.BINARY:
            _state = (1+_state)/2

        if self.var_type == openjij.BINARY:
            int_mat = np.triu(self.interactions())
            return np.dot(_state, np.dot(int_mat, _state))
        elif self.var_type == openjij.SPIN:
            int_mat = self.ising_interactions()
            linear_term = np.diag(int_mat)
            energy = (np.dot(_state, np.dot(int_mat, _state)) -
                      np.sum(linear_term))/2
            energy += np.dot(linear_term, _state)
            return energy

    def ising_dictionary(self):
        if self.var_type == openjij.SPIN:
            return self.linear, self.quadratic
        elif self.var_type == openjij.BINARY:
            ising_int = self.ising_interactions()
            h = {}
            J = {(i, j): qij/4.0 for (i, j), qij in self.quadratic.items()}
            for i in range(len(ising_int)):
                if ising_int[i][i] != 0:
                    h[self.indices[i]] = ising_int[i][i]
        return h, J

    @classmethod
    def from_qubo(cls, Q, offset=0.0, **kwargs):
        linear = {}
        quadratic = {}
        for (u, v), bias in Q.items():
            if u == v:
                linear[u] = bias
            else:
                quadratic[(u, v)] = bias
        return cls(linear, quadratic, offset,
                   var_type=openjij.BINARY, **kwargs)


# class BinaryQuadraticModel:
#     """Represents Binary quadratic model
#     Attributes:
#         var_type(openjij.VariableType): variable type SPIN or BINARY
#         linear(dict): represents linear term
#         quad(dict): represents quadratic term
#         indices(list): labels of each variables sorted by results variables
#         energy_bias(float): represents constant energy term when convert to SPIN from BINARY
#         size(int): number of variables
#     """

#     def __init__(self, h=None, J=None, Q=None, var_type=openjij.SPIN):

#         self.var_type = openjij.cast_var_type(var_type)

#         if self.var_type == openjij.SPIN:
#             if (h is None) or (J is None):
#                 raise ValueError('Input h and J.')
#             self.linear = h
#             self.quad = J
#         elif self.var_type == openjij.BINARY:
#             if not isinstance(Q, dict) or Q is None:
#                 raise ValueError('Q should be dictionary.')
#             self.linear = {}
#             self.quad = {}
#             for (i, j), qij in Q.items():
#                 if i == j:
#                     self.linear[i] = qij
#                 else:
#                     self.quad[(i, j)] = qij

#         index_set = set(self.linear.keys())
#         for v1, v2 in self.quad.keys():
#             index_set.add(v1)
#             index_set.add(v2)

#         self.indices = list(index_set)
#         if var_type == openjij.SPIN:
#             self.energy_bias = 0.0
#         else:  # BINARY
#             self.energy_bias = (sum(list(self.linear.values()))
#                                 * 2 + sum(list(self.quad.values())))/4

#         self._interaction_matrix = None  # calculated at interactions()
#         self.size = len(self.indices)

#     def get_cxxjij_ising_graph(self, sparse=False):
#         """
#         Convert to cxxjij.graph.Dense or Sparse class from Python dictionary(h, J) or Q
#         Args:
#             sparse(bool): if true returns sparse graph
#         Returns:
#             openjij.graph.Dense openjij.graph.Sparse
#         """
#         if not sparse:
#             GraphClass = cxxjij.graph.Dense
#         else:
#             GraphClass = cxxjij.graph.Sparse

#         cxxjij_graph = GraphClass(self.size)

#         ising_int = self.ising_interactions()
#         # cxxjij.graph.dense
#         for i in range(self.size):
#             if ising_int[i, i] != 0.0:
#                 cxxjij_graph[i, i] = ising_int[i, i]
#             for j in range(i+1, self.size):
#                 if ising_int[i, j] != 0.0:
#                     cxxjij_graph[i, j] = ising_int[i, j]

#         return cxxjij_graph

#     def ising_interactions(self):
#         """ Interactions in the Ising representation
#         QUBO formulation to the Ising formulation
#         We assumption Q is a triangular matrix.
#         H = q ^ T Q q
#           = 1/4 (1+s) ^ T Q(1+s)
#           = s ^ T Q/4 s + 1 ^ T Q/4 s + s ^ T Q/4 1 + 1 ^ T Q/4 1
#           = s ^ T nondiag(Q)/4 s + 1 ^ T Q/4 s + s ^ T Q/4 1 + 1 ^ T(Q + diag(Q))/4 1
#           = \sum_{i < j} Q_{ij}/4 s_i s_j
#             + \sum{i <= j}(Q_{ij} + Q_{ji}) s_i
#             + sum_{i <= j}(Q_{ij} + Q_{ii})/4
#         Therefore
#           J_{ij} = Q_{ij}/4
#           h_i = sum{i <= j}(Q_{ij} + Q_{ji})/4
#           constant_term = sum_{i <= j} Q_{ij}/4 + Tr(Q)(energy bias)
#         """
#         if self.var_type == openjij.SPIN:
#             return self.interactions()
#         interactions = self.interactions()/4
#         if self.var_type == openjij.BINARY:
#             # convert to the Ising interaction
#             self.energy_bias = (
#                 np.sum(np.triu(interactions)) + np.trace(interactions))
#             for i in range(len(interactions)):
#                 interactions[i, i] = np.sum(
#                     interactions[i, :]) + interactions[i, i]
#         return interactions

#     def interactions(self, re_calculate=False):
#         """make interaction matrix
#         The Ising model: E = ΣJ_ij σiσj + Σhiσi
#             Interaction matrix -> H_ij = J_ij + J_ji, H_ii = hi
#         QUBO: E = Σ1/2Q_ij q_iq_j + ΣQ_ii q_i

#         Args:
#             re_calculate(bool): Whether to force a recalculation
#         Returns:
#             numpy.ndarray: interactioin matrix H_{ij} or Q_{ij}
#         """

#         if (self._interaction_matrix is not None) and (not re_calculate):
#             return self._interaction_matrix

#         system_size = len(self.indices)
#         interactions = np.zeros((system_size, system_size))

#         for i, i_index in enumerate(self.indices):
#             interactions[i, i] = self.linear[i_index] if i_index in self.linear else 0.0
#             for j, j_index in enumerate(self.indices[i+1:]):
#                 j += i+1
#                 jval = 0.0
#                 if (i_index, j_index) in self.quad:
#                     jval += self.quad[(i_index, j_index)]
#                 if (j_index, i_index) in self.quad:
#                     jval += self.quad[(j_index, i_index)]
#                 interactions[i, j] = jval
#                 interactions[j, i] = jval

#         self._interaction_matrix = interactions

#         return self._interaction_matrix

#     def calc_energy(self, state, need_to_convert_from_spin=False):
#         """calculate energy from state
#         Args:
#             state(list, numpy.array): BINARY or SPIN state
#             need_to_convet_to_spin(bool): if state is SPIN and need
#                                            to convert to BINARY from SPIN
#         """
#         _state = np.array(state)
#         if need_to_convert_from_spin and self.var_type == openjij.BINARY:
#             _state = (1+_state)/2

#         if self.var_type == openjij.BINARY:
#             int_mat = np.triu(self.interactions())
#             return np.dot(_state, np.dot(int_mat, _state))
#         elif self.var_type == openjij.SPIN:
#             int_mat = self.ising_interactions()
#             linear_term = np.diag(int_mat)
#             energy = (np.dot(_state, np.dot(int_mat, _state)) -
#                       np.sum(linear_term))/2
#             energy += np.dot(linear_term, _state)
#             return energy

#     def ising_dictionary(self):
#         if self.var_type == openjij.SPIN:
#             return self.linear, self.quad
#         elif self.var_type == openjij.BINARY:
#             ising_int = self.ising_interactions()
#             h = {}
#             J = {(i, j): qij/4.0 for (i, j), qij in self.quad.items()}
#             for i in range(len(ising_int)):
#                 if ising_int[i][i] != 0:
#                     h[self.indices[i]] = ising_int[i][i]
#         return h, J
