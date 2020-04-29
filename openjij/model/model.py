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
import dimod
import warnings
import sys

def make_BinaryQuadraticModel(linear, quadratic):
    """ BinaryQuadraticModel factory
    Args:
        linear (dict): linear biases
        quadratic (dict): quadratic biases
    Returns:
        generated BinaryQuadraticModel class
    """
    class BinaryQuadraticModel(cimod.make_BinaryQuadraticModel(linear, quadratic)):
        """Represents Binary quadratic model. 
           Indices are listed in self.indices.
        Attributes:
            var_type (cimod.VariableType): variable type SPIN or BINARY
            linear (dict): represents linear term
            quadratic (dict): represents quadratic term
            adj (dict): represents adjacency
            indices (list): labels of each variables sorted by results variables
            offset (float): represents constant energy term when convert to SPIN from BINARY
        """
    
        def __init__(self, linear, quadratic, offset=0.0,
                     var_type=openjij.SPIN, gpu=False, **kwargs):
    
            super().__init__(linear, quadratic, offset, var_type, **kwargs)
            self.gpu = gpu
    
    
        def get_cxxjij_ising_graph(self, sparse=False):
            """
            Convert to cxxjij.graph.Dense or Sparse class from Python dictionary (h, J) or Q
            Args:
                sparse (bool): if true returns sparse graph
            Returns:
                openjij.graph.Dense openjij.graph.Sparse
            """
    
            if sparse:
                GraphClass = cxxjij.graph.Sparse if self.gpu == False else cxxjij.graph.SparseGPU
                return GraphClass(self.to_serializable())
            else:
                GraphClass = cxxjij.graph.Dense if self.gpu == False else cxxjij.graph.DenseGPU
                # initialize with interaction matrix.
                mat = self.interaction_matrix()
                size = mat.shape[0]
                dense = GraphClass(size)
                # graph type is dense.
                # reshape matrix
                temp = np.zeros((size+1, size+1))
                temp[:size, :size] = mat
                temp[:size, size] = np.diag(mat)
                temp[size, :size] = np.diag(mat)
                np.fill_diagonal(temp, 0)
                temp[size, size] = 1
                temp[:size, :size] /= 2.0
    
                dense.set_interaction_matrix(temp)
    
                return dense
    
    
        # compatible with the previous version
        def calc_energy(self, sample, **kwargs):
            return self.energy(sample, **kwargs)

    return BinaryQuadraticModel

def make_BinaryQuadraticModel_from_JSON(obj):
    """ BinaryQuadraticModel factory for JSON
    Args:
        obj (dict): JSON object
    Returns:
        generated BinaryQuadraticModel class
    """
    label = obj['variable_labels'][0]
    if isinstance(label, list):
        #convert to tuple
        label = tuple(label)

    mock_linear = {label:1.0}

    return make_BinaryQuadraticModel(mock_linear, {})

def BinaryQuadraticModel(linear, quadratic, offset=0.0,
        var_type=dimod.SPIN, **kwargs):

    Model = make_BinaryQuadraticModel(linear, quadratic)

    return Model(linear, quadratic, offset, var_type, **kwargs)

#classmethods
BinaryQuadraticModel.from_qubo = \
        lambda Q, offset=0.0, **kwargs: make_BinaryQuadraticModel({}, Q).from_qubo(Q, offset, **kwargs)

BinaryQuadraticModel.from_ising = \
        lambda linear, quadratic, offset=0.0, **kwargs: make_BinaryQuadraticModel(linear, quadratic).from_ising(linear, quadratic, offset, **kwargs)

BinaryQuadraticModel.from_serializable = \
        lambda obj: make_BinaryQuadraticModel_from_JSON(obj).from_serializable(obj)
















