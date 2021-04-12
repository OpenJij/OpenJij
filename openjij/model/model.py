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
"""
| This module defines the BinaryQuadraticModel with the Hamiltonian,

.. math:: 

    H = \\sum_{i\\neq j} J_{ij}\sigma_i \sigma_j + \\sum_{i} h_{i}\sigma_i,

| in an Ising form and

.. math:: 

    H = \\sum_{ij} Q_{ij}x_i x_j + \\sum_{i} H_{i}x_i,

| in a QUBO form.
| The methods and usage are basically the same as `dimod <https://github.com/dwavesystems/dimod>`_.

"""

import numpy as np
import cxxjij
import openjij
from openjij.utils.graph_utils import qubo_to_ising
import cimod
import dimod
import warnings
import sys

def make_BinaryQuadraticModel(linear: dict, quadratic: dict):
    """BinaryQuadraticModel factory.

    Returns:
        generated BinaryQuadraticModel class
    """

    class BinaryQuadraticModel(cimod.make_BinaryQuadraticModel(linear, quadratic)):
        """Represents Binary quadratic model. 
           Indices are listed in self.indices.
        Attributes:
            var_type (dimod.Vartype): variable type SPIN or BINARY
            linear (dict): represents linear term
            quadratic (dict): represents quadratic term
            adj (dict): represents adjacency
            indices (list): labels of each variables sorted by results variables
            offset (float): represents constant energy term when convert to SPIN from BINARY
        """
    
        def __init__(self, linear: dict, quadratic: dict, offset: float=0.0,
                var_type=openjij.SPIN, gpu: bool=False, **kwargs):
            """BinaryQuadraticModel constructor.

            Args:
                linear (dict): linear biases.
                quadratic (dict): quadratic biases
                offset (float): offset
                var_type (openjij.variable_type.Vartype): var_type
                gpu (bool): if true, this can be used for gpu samplers.
                kwargs:
            """
    
            super().__init__(linear, quadratic, offset, var_type, **kwargs)
            self.gpu = gpu
    
    
        def get_cxxjij_ising_graph(self, sparse: bool=False):
            """generate cxxjij Ising graph from the interactions.

            Args:
                sparse (bool): if true, this function returns cxxjij.graph.Sparse. Otherwise it returns cxxjij.graph.Dense.
            Returns:
                cxxjij.graph.Dense or cxxjij.graph.Sparse: 
            """
    
            if sparse:
                GraphClass = cxxjij.graph.Sparse if self.gpu == False else cxxjij.graph.SparseGPU
                return GraphClass(self.to_serializable())
            else:
                GraphClass = cxxjij.graph.Dense if self.gpu == False else cxxjij.graph.DenseGPU
                # initialize with interaction matrix.
                mat = self.interaction_matrix()

                if (self.vartype == openjij.BINARY):
                    # convert to ising matrix
                    qubo_to_ising(mat)

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
    
                dense.set_interaction_matrix(temp)
    
                return dense
    
    
        # compatible with the previous version
        def calc_energy(self, sample, **kwargs):
            return self.energy(sample, **kwargs)

    return BinaryQuadraticModel

def make_BinaryQuadraticModel_from_JSON(obj: dict):
    """make BinaryQuadraticModel from JSON.

    Returns:
        corresponding BinaryQuadraticModel type
    """

    label = obj['variable_labels'][0]
    if isinstance(label, list):
        #convert to tuple
        label = tuple(label)

    mock_linear = {label:1.0}

    return make_BinaryQuadraticModel(mock_linear, {})

def BinaryQuadraticModel(linear: dict, quadratic: dict, offset: float=0.0,
        var_type=dimod.SPIN, gpu: bool=False, **kwargs):
    """generate BinaryQuadraticModel object.

    Attributes:
        var_type (dimod.Vartype): variable type SPIN or BINARY
        linear (dict): represents linear term
        quadratic (dict): represents quadratic term
        adj (dict): represents adjacency
        indices (list): labels of each variables sorted by results variables.
        offset (float): represents constant energy term when convert to SPIN from BINARY
    Args:
        linear (dict): linear biases
        quadratic (dict): quadratic biases
        offset (float): offset
        var_type (openjij.variable_type.Vartype): vartype ('SPIN' or 'BINARY')
        gpu (bool): if true, this can be used for gpu samplers.
        kwargs:
    Returns:
        generated BinaryQuadraticModel
    Examples:
        BinaryQuadraticModel can be initialized by specifing h and J::

            >>> h = {0: 1, 1: -2}
            >>> J = {(0, 1): -1, (1, 2): -3, (2, 3): 0.5}
            >>> bqm = oj.BinaryQuadraticModel(self.h, self.J)

        You can also use strings and tuples of integers (up to 4 elements) as indices::

            >>> h = {'a': 1, 'b': -2}
            >>> J = {('a', 'b'): -1, ('b', 'c'): -3, ('c', 'd'): 0.5}
            >>> bqm = oj.BinaryQuadraticModel(self.h, self.J)
    """

    Model = make_BinaryQuadraticModel(linear, quadratic)

    return Model(linear, quadratic, offset, var_type, gpu, **kwargs)

#classmethods
BinaryQuadraticModel.from_qubo = \
        lambda Q, offset=0.0, **kwargs: make_BinaryQuadraticModel({}, Q).from_qubo(Q, offset, **kwargs)

BinaryQuadraticModel.from_ising = \
        lambda linear, quadratic, offset=0.0, **kwargs: make_BinaryQuadraticModel(linear, quadratic).from_ising(linear, quadratic, offset, **kwargs)

BinaryQuadraticModel.from_serializable = \
        lambda obj: make_BinaryQuadraticModel_from_JSON(obj).from_serializable(obj)


def make_BinaryPolynomialModel(interactions: dict):

    class BinaryPolynomialModel(cimod.make_BinaryPolynomialModel(interactions)):

        def __init__(self, interactions: dict, var_type = openjij.SPIN, **kwargs):
            super().__init__(interactions, var_type, **kwargs)

        def get_cxxjij_ising_graph(self):
            return cxxjij.graph.Polynomial(self.to_serializable())

        def calc_energy(self, sample, **kwargs):
            return self.energy(sample, **kwargs)

    return BinaryPolynomialModel

def make_BinaryPolynomialModel_from_JSON(obj: dict):
    label = obj['variable_labels'][0]
    if isinstance(label, list):
        #convert to tuple
        label = tuple(label)
    mock_linear = {(label,):1.0}
    return make_BinaryPolynomialModel(mock_linear)

def BinaryPolynomialModel(interactions: dict, var_type = openjij.SPIN, **kwargs):
    Model = make_BinaryPolynomialModel(interactions)
    return Model(interactions, var_type, **kwargs)

#classmethods
BinaryPolynomialModel.from_pubo = \
        lambda P, **kwargs: make_BinaryPolynomialModel(P).from_pubo(P, **kwargs)

BinaryPolynomialModel.from_ising = \
        lambda polynomial, **kwargs: make_BinaryPolynomialModel(polynomial).from_ising(polynomial, **kwargs)

BinaryPolynomialModel.from_serializable = \
        lambda obj: make_BinaryPolynomialModel_from_JSON(obj).from_serializable(obj)
