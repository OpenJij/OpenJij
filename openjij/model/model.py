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
from openjij.utils.decorator import disabled
import cimod
import warnings
import sys


class BinaryQuadraticModel(cimod.BinaryQuadraticModel):
    """Represents Binary quadratic model. 
       Note that the indices are converted to the integers internally. 
       The dictionaries between indices and integers are self.ind_to_num (indices -> integers) and self.num_to_ind (integers -> indices).
       Indices are listed in self.indices.
    Attributes:
        var_type (cimod.VariableType): variable type SPIN or BINARY
        linear (dict): represents linear term
        quad (dict): represents quadratic term
        indices (list): labels of each variables sorted by results variables
        offset (float): represents constant energy term when convert to SPIN from BINARY
        size (int): number of variables
    """

    def __init__(self, linear, quadratic, offset=0.0,
                 var_type=openjij.SPIN, **kwargs):

        super().__init__(linear, quadratic, offset, vartype)


    def get_cxxjij_ising_graph(self, sparse=False):
        """
        Convert to cxxjij.graph.Dense or Sparse class from Python dictionary (h, J) or Q
        Args:
            sparse (bool): if true returns sparse graph
        Returns:
            openjij.graph.Dense openjij.graph.Sparse
        """

        if sparse:
            GraphClass = cxxjij.graph.Sparse
        else:
            GraphClass = cxxjij.graph.Dense

        return GraphClass(self.to_serializable())

    def get_linear(self, original_ind=True):
        """
        get linear
        Args:
            original_ind (bool): if true returns linear with index converted to original one
        Returns:
            linear (dict)
        """
        linear = super().get_linear()

        if original_ind:
            linear = self._conv_linear(linear, self.num_to_ind)

        return linear

    def get_quadratic(self, original_ind=True):
        """
        get quadratic
        Args:
            original_ind (bool): if true returns linear with index converted to original one
        Returns:
            quadratic (dict)
        """
        quadratic = super().get_quadratic()

        if original_ind:
            quadratic = self._conv_quadratic(quadratic, self.num_to_ind)

        return quadratic

    def get_adjacency(self, original_ind=True):
        """
        get adjacency
        Args:
            original_ind (bool): if true returns linear with index converted to original one
        Returns:
            adjacency (dict)
        """
        adjacency = super().get_adjacency()

        if original_ind:
            adjacency = self._conv_adjacency(adjacency, self.num_to_ind)

        return adjacency

    @property
    def linear(self):
        return self.get_linear()

    @property
    def quadratic(self):
        return self.get_quadratic()

    @property
    def vartype(self):
        return self.get_vartype()

    @property
    def offset(self):
        return self.get_offset()
    
    def to_qubo(self, original_ind=True):
        """
        Convert a binary quadratic model to QUBO format.
        Args:
            original_ind (bool): if true returns linear with index converted to original one
        Returns:
            Q (dict), offset
        """
        Q, offset = super().to_qubo()

        if original_ind:
            Q = self._conv_quadratic(Q, self.num_to_ind)

        return Q, offset

    def to_ising(self, original_ind=True):
        """
        Convert a binary quadratic model to Ising format.
        Args:
            original_ind (bool): if true returns linear with index converted to original one
        Returns:
            h (dict), J (dict), offset
        """

        h, J, offset = super().to_ising()

        if original_ind:
            h = self._conv_linear(h, self.num_to_ind)
            J = self._conv_quadratic(J, self.num_to_ind)

        return h, J, offset

    def interaction_matrix(self):
        """make Dense-type interaction matrix
        The Ising model: E = ΣJ_ij σiσj + Σhiσi
            Interaction matrix -> H_ij = J_ij + J_ji, H_ii = hi
        QUBO: E = Σ1/2Q_ij q_iq_j + ΣQ_ii q_i
        Returns:
            numpy.ndarray: interactioin matrix H_{ij} or Q_{ij}, energy_bias (float)
        """

        if self._re_calculate is True:

            # calculate interaction matrix
            system_size = len(self.indices)
            interactions = np.zeros((system_size, system_size))

            # variables with converted indices (integers)
            linear = self.get_linear(False)
            quadratic = self.get_quadratic(False)

            for i, i_index in enumerate(self.indices):
                interactions[i, i] = linear[i] if i in linear else 0.0
                for j, j_index in enumerate(self.indices[i+1:]):
                    j += i+1
                    jval = 0.0

                    # if the module is derived from dimod.BinaryQuadraticModel, the below should be 
                    # if (i, j) in quadratic:
                    #     jval = quadratic[(i, j)]
                    # if (j, i) in quadratic:
                    #     jval = quadratic[(j, i)]
                    # 
                    # since dimod.BinaryQuadraticModel internally sums up two tuples of indices with different order (such as (a,b) and (b,a)).

                    if (i, j) in quadratic:
                        jval += quadratic[(i, j)]
                    if (j, i) in quadratic:
                        jval += quadratic[(j, i)]

                    interactions[i, j] = jval
                    interactions[j, i] = jval

            self._interaction_matrix = interactions
            self._re_calculate = False

        return self._interaction_matrix

    def energy(self, sample, sparse=False, convert_sample=False):
        """Determine the energy of the specified sample of a binary quadratic model.
        Args:
            sample (dict): single sample. 
            sparse (bool): if true calculate energy by using adjacency matrix
            convert_sample (bool): if true the sample is automatically converted to self.vartype type.
        Returns:
            energy (float)
        """

        if isinstance(sample, dict):
            # convert int to num
            sample = self._conv_linear(sample, self.ind_to_num)

        # convert samples to SPIN or BINARY
        if convert_sample:
            for i in range(len(sample)):
                if sample[i] == -1 and self.vartype == openjij.BINARY:
                    sample[i] = 0
                if sample[i] == 0  and self.vartype == openjij.SPIN:
                    sample[i] = -1

        if sparse:
           # convert sample to dict
           if isinstance(sample, list) or isinstance(sample, np.ndarray):
               sample = {i:elem for i,elem in enumerate(sample)}
           return super().energy(sample)

        else:
            if isinstance(sample, dict):
                state = [0] * len(sample)
                for k,v in sample.items():
                    state[k] = v
                sample = state

            sample = np.array(sample)

            int_mat = self.interaction_matrix()

            # calculate 
            if self.get_vartype() == openjij.BINARY:
                return np.dot(sample, np.dot(np.triu(int_mat), sample)) + self.get_offset()
            elif self.get_vartype() == openjij.SPIN:
                linear_term = np.diag(int_mat)
                energy = (np.dot(sample, np.dot(int_mat, sample)) -
                      np.sum(linear_term))/2
                energy += np.dot(linear_term, sample)
                energy += self.get_offset()
            return energy 

    def energies(self, samples_like, **kwargs):
        en_vec = []

        for elem in samples_like:
            en_vec.append(self.energy(elem, **kwargs))

        return en_vec

    # compatible with the previous version
    def calc_energy(self, sample, **kwargs):
        return self.energy(sample, **kwargs)


    # disabled methods (TODO: implement these)
    @disabled
    def empty(*args, **kwargs):
        pass

    @disabled
    def add_variable(*args, **kwargs):
        pass

    @disabled
    def add_variables_from(*args, **kwargs):
        pass

    @disabled
    def add_interaction(*args, **kwargs):
        pass

    @disabled
    def add_interactions_from(*args, **kwargs):
        pass

    @disabled
    def remove_variable(*args, **kwargs):
        pass

    @disabled
    def remove_variables_from(*args, **kwargs):
        pass

    @disabled
    def remove_interaction(*args, **kwargs):
        pass

    @disabled
    def remove_interactions_from(*args, **kwargs):
        pass

    @disabled
    def add_offset(*args, **kwargs):
        pass

    @disabled
    def remove_offset(*args, **kwargs):
        pass

    @disabled
    def scale(*args, **kwargs):
        pass

    @disabled
    def normalize(*args, **kwargs):
        pass

    @disabled
    def fix_variable(*args, **kwargs):
        pass

    @disabled
    def fix_variables(*args, **kwargs):
        pass

    @disabled
    def flip_variable(*args, **kwargs):
        pass

    @disabled
    def update(*args, **kwargs):
        pass

    @disabled
    def contract_variables(*args, **kwargs):
        pass


    def change_vartype(self, vartype):
        """
        Create a binary quadratic model with the specified vartype
        Args:
            var_type (cimod.Vartype): SPIN or BINARY
        Returns:
            A new instance of the BinaryQuadraticModel class.
        """
        vartype = openjij.cast_var_type(vartype)
        bqm = super().change_vartype(vartype)
        linear = self._conv_linear(bqm.get_linear(), self.num_to_ind)
        quadratic = self._conv_quadratic(bqm.get_quadratic(), self.num_to_ind)
        return openjij.BinaryQuadraticModel(linear, quadratic, bqm.get_offset(), bqm.get_vartype())


    @classmethod
    def from_qubo(cls, Q, offset=0.0, **kwargs):
        """
        Create a binary quadratic model from a QUBO model.
        Args:
            Q (dict), offset
        Returns:
            A new instance of the BinaryQuadraticModel class.
        """
        linear = {}
        quadratic = {}
        for (u, v), bias in Q.items():
            if u == v:
                linear[u] = bias
            else:
                quadratic[(u, v)] = bias

        return cls(linear, quadratic, offset, var_type=openjij.BINARY, **kwargs)

    @classmethod
    def from_ising(cls, linear, quadratic, offset=0.0, **kwargs):
        """
        Create a binary quadratic model from a Ising model.
        Args:
            linear (dict), quadratic (dict), offset
        Returns:
            A new instance of the BinaryQuadraticModel class.
        """
        return cls(linear, quadratic, offset, var_type=openjij.SPIN, **kwargs)

    @staticmethod
    def _generate_indices_dict(linear=None, quadratic=None):
        """
        Generate indices dictionaries.
        Args:
            linear (dict), quadratic (dict)
        Returns:
            tuple of dictionaries (indices, num_to_ind, ind_to_num)
        """
        if linear is not None:
            index_set = set(linear.keys())
        else:
            index_set = set()


        if quadratic is not None:
            for v1, v2 in quadratic.keys():
                index_set.add(v1)
                index_set.add(v2)

        indices = list(index_set)

        # generate conversion map index <-> integer
        num_to_ind = {k:val for k,val in enumerate(indices)}
        ind_to_num = {val:k for k,val in enumerate(indices)}

        return indices, num_to_ind, ind_to_num


    def _conv_linear(self, dic, conv_dict):
        """
        Convert indices of dictionary (linear)
        Args:
            dic (dict): dictionary
            conv_dict (dict): convert dict (ind_to_num or num_to_ind)
        Returns:
            dictionaries with indices converted
        """
        return {conv_dict[k]:v for k,v in dic.items()}

    def _conv_quadratic(self, dic, conv_dict):
        """
        Convert indices of dictionary (quadratic)
        Args:
            dic (dict): dictionary
            conv_dict (dict): convert dict (ind_to_num or num_to_ind)
        Returns:
            dictionaries with indices converted
        """
        return {(conv_dict[k1], conv_dict[k2]):v for (k1,k2),v in dic.items()}

    def _conv_adjacency(self, dic, conv_dict):
        """
        Convert indices of dictionary (adjacency)
        Args:
            dic (dict): dictionary
            conv_dict (dict): convert dict (ind_to_num or num_to_ind)
        Returns:
            dictionaries with indices converted
        """
        return {conv_dict[index]:{conv_dict[k]:v for k,v in adj_dic.items()} for index,adj_dic in dic.items()}

