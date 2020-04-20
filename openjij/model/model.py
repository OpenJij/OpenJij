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

        vartype = openjij.cast_var_type(var_type)

        # set index array
        index_set = set(linear.keys())

        # set recalculate flag True
        # Be sure to enable this flag when variables are changed.
        self._re_calculate = True
        # interaction_matrix
        self._interaction_matrix = None

        for v1, v2 in quadratic.keys():
            index_set.add(v1)
            index_set.add(v2)

        self.indices = list(index_set)

        # generate conversion map index <-> integer
        self.num_to_ind = {k:val for k,val in enumerate(self.indices)}
        self.ind_to_num = {val:k for k,val in enumerate(self.indices)}

        # convert indices to integers and call super constructor
        linear      = self._conv_linear(linear, self.ind_to_num)
        quadratic   = self._conv_quadratic(quadratic, self.ind_to_num)
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

                    if (i, j) in quadratic:
                        jval += quadratic[(i, j)]
                    if (j, i) in quadratic:
                        jval += quadratic[(j, i)]

                    interactions[i, j] = jval
                    interactions[j, i] = jval

            self._interaction_matrix = interactions
            self._re_calculate = False

        return self._interaction_matrix

    def energy(self, sample, sparse=False):
        """Determine the energy of the specified sample of a binary quadratic model.
        Args:
            sample (dict): single sample
            sparse (bool): if true calculate energy by using adjacency matrix
        Returns:
            energy (float)
        """

        if isinstance(sample, dict):
            sample = self._conv_linear(sample, self.ind_to_num)

        if sparse:
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

    def energies(self, samples_like, sparse=False):
        en_vec = []

        for elem in samples_like:
            en_vec.append(self.energy(elem))

        return en_vec

    
    # deprecated methods (TODO: implement these)
    def disabled(func):
        def wrapper(*args, **kwargs):
            raise NotImplementedError("The function {} is disabled.".format(func.__name__))

        return wrapper


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


    @staticmethod
    def from_qubo(Q, offset=0.0):
        """
        Create a binary quadratic model from a QUBO model.
        Args:
            Q (dict), offset
        Returns:
            A new instance of the BinaryQuadraticModel class.
        """
        return cimod.BinaryQuadraticModel.from_qubo(Q, offset)

    @staticmethod
    def from_ising(linear, quadratic, offset=0.0):
        """
        Create a binary quadratic model from a Ising model.
        Args:
            linear (dict), quadratic (dict), offset
        Returns:
            A new instance of the BinaryQuadraticModel class.
        """
        return cimod.BinaryQuadraticModel.from_ising(linear, quadratic, offset)


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

