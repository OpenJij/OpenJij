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

        # set index array
        index_set = set(linear.keys())

        for v1, v2 in self.quadratic.keys():
            index_set.add(v1)
            index_set.add(v2)

        self.indices = list(index_set)



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
