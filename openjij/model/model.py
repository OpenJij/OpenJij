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
            return GraphClass(self.to_serializable())
        else:
            GraphClass = cxxjij.graph.Dense
            # graph type is dense.
            # initialize with interaction matrix.
            mat = self.interaction_matrix()
            # reshape matrix
            size = mat.shape[0]
            temp = np.zeros((size+1, size+1))
            temp[:size, :size] = mat
            temp[:size, size] = np.diag(mat)
            temp[size, :size] = np.diag(mat)
            np.fill_diagonal(temp, 0)
            temp[size, size] = 1
            temp[:size, :size] /= 2.0

            return GraphClass(temp)


    # compatible with the previous version
    def calc_energy(self, sample, **kwargs):
        return self.energy(sample, **kwargs)


