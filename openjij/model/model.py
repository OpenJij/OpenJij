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
    """Represents Binary quadratic model. Note that the indices are converted to integers internally. The dictionaries between indices and integers are in self.ind_to_num (indices -> integers) and self.num_to_ind (integers -> indices).
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

        var_type = openjij.cast_var_type(var_type)

        # set index array
        index_set = set(linear.keys())

        for v1, v2 in self.quadratic.keys():
            index_set.add(v1)
            index_set.add(v2)

        self.indices = list(index_set)

        # generate conversion map index <-> integer
        self.ind_to_num = {k:val for k,val in enumerate(self.indices)}
        self.num_to_ind = {val:k for k,val in enumerate(self.indices)}

        # convert indices to integers and call super constructor
        linear = self._conv_linear(linear, self.ind_to_num)
        quadratic = self._conv_quadratic(linear, self.ind_to_num)
        super().__init__(linear, quadratic, offset, var_type)


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

        return GraphClass(self.to_serializable())

    def contains(self, v):
        return super().contains(self._ind_to_num[v])

    def get_linear(convert=True):
        """
        get linear
        Args:
            convert (bool): if true returns linear with index converted to original one
        Returns:
            linear (dict)
        """
        linear = super().get_linear()

        if convert:
            return self._conv_linear(linear, self._num_to_ind)
        else:
            return linear

    def get_quadratic(self, convert=True):
        """
        get quadratic
        Args:
            convert (bool): if true returns linear with index converted to original one
        Returns:
            quadratic (dict)
        """
        quadratic = super().get_quadratic()

        if convert:
            return self._conv_quadratic(quadratic, self._num_to_ind)
        else:
            return quadratic

    def get_adjacency(self, convert=True):
        """
        get adjacency
        Args:
            convert (bool): if true returns linear with index converted to original one
        Returns:
            adjacency (dict)
        """
        adjacency = super().get_adjacency()

        if convert:
            return self._conv_adjacency(adjacency, self._num_to_ind)
        else:
            return adjacency

    def energy(self, sample):
        sample = {self._ind_to_num[k]:v for k,v in sample.items()}
        return super().energy(sample)

    def energies(self, samples_like):
        samples_like = [{self._ind_to_num[k]:v for k,v in sample.items()} for elem in samples_like]
        return super().energies(samples_like)

    def to_qubo(self):
        Q,offset = super().to_qubo()
        Q = self._conv_quadratic(Q, self._num_to_ind)
        return Q,offset

    def to_ising(self):
        h,J,offset = super().to_ising()
        h = self._conv_linear(h, self._num_to_ind)
        J = self._conv_quadratic(J, self._num_to_ind)
        return Q,offset






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

