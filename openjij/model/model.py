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

        var_type = openjij.cast_var_type(var_type)

        # set index array
        index_set = set(linear.keys())

        # set recalculate flag True
        self._recalculate = True

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
        super().__init__(linear, quadratic, offset, var_type)


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
    
    # deprecated methods
    def empty(*args, **kwargs):
        pass
    def add_variable(*args, **kwargs):
        pass
    def add_variables_from(*args, **kwargs):
        pass
    def add_interaction(*args, **kwargs):
        pass
    def add_interactions_from(*args, **kwargs):
        pass
    def remove_variable(*args, **kwargs):
        pass
    def remove_variables_from(*args, **kwargs):
        pass
    def remove_interaction(*args, **kwargs):
        pass
    def remove_interactions_from(*args, **kwargs):
        pass
    def add_offset(*args, **kwargs):
        pass
    def remove_offset(*args, **kwargs):
        pass
    def scale(*args, **kwargs):
        pass
    def normalize(*args, **kwargs):
        pass
    def fix_variable(*args, **kwargs):
        pass
    def fix_variables(*args, **kwargs):
        pass
    def flip_variable(*args, **kwargs):
        pass
    def update(*args, **kwargs):
        pass
    def contract_variables(*args, **kwargs):
        pass


    @staticmethod
    def from_qubo(linear, quadratic, offset=0.0):
        """
        Create a binary quadratic model from a QUBO model.
        Args:
            original_ind (bool): if true returns linear with index converted to original one
        Returns:
            h (dict), J (dict), offset
        """
        linear = self._conv_linear(linear, self.ind_to_num)
        quadratic = self._conv_quadratic(quadratic, self.ind_to_num)
        return cimod.BinaryQuadraticModel.from_qubo(linear, quadratic, offset)

    @staticmethod
    def from_ising(linear, quadratic, offset=0.0):
        linear = self._conv_linear(linear, self.ind_to_num)
        quadratic = self._conv_quadratic(quadratic, self.ind_to_num)
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

