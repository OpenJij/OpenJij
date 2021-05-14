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
import cxxcimod
import dimod
from functools import singledispatch
import warnings
import sys

def make_BinaryQuadraticModel(linear: dict, quadratic: dict, sparse):
    """BinaryQuadraticModel factory.

    Returns:
        generated BinaryQuadraticModel class
    """

    class BinaryQuadraticModel(cimod.make_BinaryQuadraticModel(linear, quadratic, sparse)):
        """Represents Binary quadratic model. 
           Indices are listed in self.indices.
        Attributes:
            vartype (cimod.VariableType): variable type SPIN or BINARY
            linear (dict): represents linear term
            quadratic (dict): represents quadratic term
            offset (float): represents constant energy term when convert to SPIN from BINARY
            num_variables (int): represents constant energy term when convert to SPIN from BINARY
            variables (list): represents constant energy term when convert to SPIN from BINARY
        """
    
        def __init__(self, *args, **kwargs):
            """BinaryQuadraticModel constructor.

            Args:
                linear (dict): linear biases.
                quadratic (dict): quadratic biases
                offset (float): offset
                vartype (openjij.variable_type.Vartype): vartype
                gpu (bool): if true, this can be used for gpu samplers.
                kwargs:
            """
            # extract gpu keyword
            gpu = kwargs.pop('gpu', False)
            super().__init__(*args, **kwargs)
            self.gpu = gpu
    
    
        def get_cxxjij_ising_graph(self):
            """generate cxxjij Ising graph from the interactions.

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
                dense.set_interaction_matrix(mat)
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

    if obj['version']['bqm_schema'] == '3.0.0-dense':
        sparse = False
    elif obj['version']['bqm_schema'] == '3.0.0':
        sparse = True 
    else:
        raise TypeError("Invalid bqm_schema")

    return make_BinaryQuadraticModel(mock_linear, {}, sparse)

def BinaryQuadraticModel(linear, quadratic, *args, **kwargs):
    """generate BinaryQuadraticModel object.

    Attributes:
        vartype (cimod.VariableType): variable type SPIN or BINARY
        linear (dict): represents linear term
        quadratic (dict): represents quadratic term
        offset (float): represents constant energy term when convert to SPIN from BINARY
        num_variables (int): represents constant energy term when convert to SPIN from BINARY
        variables (list): represents constant energy term when convert to SPIN from BINARY
    Args:
        linear (dict): linear biases
        quadratic (dict): quadratic biases
        offset (float): offset
        vartype (openjij.variable_type.Vartype): vartype ('SPIN' or 'BINARY')
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

    Model = make_BinaryQuadraticModel(linear, quadratic, kwargs.pop('sparse', False))

    # offset and vartype
    if len(args) == 2:
        [offset, vartype] = args
        return Model(linear, quadratic, offset, vartype, **kwargs)
    elif len(args) == 1 and 'vartype' in kwargs:
        [offset] = args
        vartype = kwargs.pop('vartype')
        return Model(linear, quadratic, offset, vartype, **kwargs)
    elif len(args) == 1:
        [vartype] = args
        return Model(linear, quadratic, 0.0, vartype, **kwargs)
    elif len(args) == 0 and 'vartype' in kwargs:
        vartype = kwargs.pop('vartype')
        return Model(linear, quadratic, 0.0, vartype, **kwargs)
    else:
        raise TypeError("invalid args for BinaryQuadraticModel. please check arguments")


#classmethods
def bqm_from_numpy_matrix(mat, variables: list=None, offset=0.0, vartype='BINARY', **kwargs):
    if variables is None:
        # generate array
        num_variables = mat.shape[0]
        variables = list(range(num_variables))

    return make_BinaryQuadraticModel({variables[0]: 1.0}, {}, kwargs.pop('sparse', False)).from_numpy_matrix(mat, variables, offset, to_cxxcimod(vartype), True, **kwargs)

BinaryQuadraticModel.from_numpy_matrix = bqm_from_numpy_matrix

BinaryQuadraticModel.from_qubo = \
lambda Q, offset=0.0, **kwargs: make_BinaryQuadraticModel({}, Q, kwargs.pop('sparse', False)).from_qubo(Q, offset, **kwargs)

BinaryQuadraticModel.from_qubo = \
lambda Q, offset=0.0, **kwargs: make_BinaryQuadraticModel({}, Q, kwargs.pop('sparse', False)).from_qubo(Q, offset, **kwargs)

BinaryQuadraticModel.from_ising = \
lambda linear, quadratic, offset=0.0, **kwargs: make_BinaryQuadraticModel(linear, quadratic, kwargs.pop('sparse', False)).from_ising(linear, quadratic, offset, **kwargs)

BinaryQuadraticModel.from_serializable = \
lambda obj, **kwargs: make_BinaryQuadraticModel_from_JSON(obj).from_serializable(obj, **kwargs)


def make_BinaryPolynomialModel(polynomial, index_type = None, tuple_size = 0):

    class BinaryPolynomialModel(cimod.make_BinaryPolynomialModel(polynomial, index_type, tuple_size)):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_cxxjij_ising_graph(self):
            return cxxjij.graph.Polynomial(self.to_serializable())

        def calc_energy(self, sample, omp_flag = True):
            return self.energy(sample, omp_flag)

    return BinaryPolynomialModel

def make_BinaryPolynomialModel_from_JSON(obj):
    if(obj["type"] != "BinaryPolynomialModel"):
        raise Exception("Type must be \"BinaryPolynomialModel\"")
    mock_polynomial = {}
    if obj["index_type"] == 'IndexType.INT':
        mock_polynomial = {(0,1):1}
    elif obj["index_type"] == 'IndexType.STRING':
        mock_polynomial = {("a","b"):1}
    elif obj["index_type"] == 'IndexType.INT_TUPLE_2':
        mock_polynomial = {((0,1),(1,2)):1}
    elif obj["index_type"] == 'IndexType.INT_TUPLE_3':
        mock_polynomial = {((0,1,2),(1,2,3)):1}
    elif obj["index_type"] == 'IndexType.INT_TUPLE_4':
        mock_polynomial = {((0,1,2,3),(1,2,3,4)):1}
    else:
        raise TypeError("Invalid types of polynomial")
    return make_BinaryPolynomialModel(mock_polynomial)

@singledispatch
def BinaryPolynomialModel(polynomial: dict, vartype):
    Model = make_BinaryPolynomialModel(polynomial)
    return Model(polynomial, _to_cxxcimod(vartype))

@BinaryPolynomialModel.register
def _BinaryPolynomialModel_from_list(keys: list, values: list, vartype):
    if len(keys) == 0:
        Model = make_BinaryPolynomialModel({})
        return Model(keys, values, _to_cxxcimod(vartype))

    i = 0
    label = None
    while i < len(keys):
        if len(keys[i]) > 0:
            label = keys[i][0]
            break
        i += 1

    if label == None:
        Model = make_BinaryPolynomialModel({():1.0})
        return Model(keys, values, _to_cxxcimod(vartype))
    else:
        if isinstance(label, list):
            label = tuple(label)
        mock_polynomial = {(label,):1.0}
        Model = make_BinaryPolynomialModel(mock_polynomial)
        return Model(keys, values, _to_cxxcimod(vartype))

@singledispatch
def make_BinaryPolynomialModel_from_hising(polynomial: dict):
    return make_BinaryPolynomialModel(polynomial).from_hising(polynomial)

@make_BinaryPolynomialModel_from_hising.register
def _make_BinaryPolynomialModel_from_hising_from_list(keys: list, values: list):
    if len(keys) == 0:
        return make_BinaryPolynomialModel({}).from_hising(keys, values)

    i = 0
    label = None
    while i < len(keys):
        if len(keys[i]) > 0:
            label = keys[i][0]
            break
        i += 1

    if label == None:
        return make_BinaryPolynomialModel({():1.0}).from_hising(keys, values)
    else:
        if isinstance(label, list):
            label = tuple(label)
        mock_polynomial = {(label,):1.0}
        return make_BinaryPolynomialModel(mock_polynomial).from_hising(keys, values)

@singledispatch
def make_BinaryPolynomialModel_from_hubo(polynomial: dict):
    return make_BinaryPolynomialModel(polynomial).from_hubo(polynomial)

@make_BinaryPolynomialModel_from_hubo.register
def _make_BinaryPolynomialModel_from_hubo_from_list(keys: list, values: list):
    if len(keys) == 0:
        return make_BinaryPolynomialModel({}).from_hubo(keys, values)

    i = 0
    label = None
    while i < len(keys):
        if len(keys[i]) > 0:
            label = keys[i][0]
            break
        i += 1

    if label == None:
        return make_BinaryPolynomialModel({():1.0}).from_hubo(keys, values)
    else:
        if isinstance(label, list):
            label = tuple(label)
        mock_polynomial = {(label,):1.0}
        return make_BinaryPolynomialModel(mock_polynomial).from_hubo(keys, values)

def _to_cxxcimod(vartype):
    # convert to cxxcimod type
    if isinstance(vartype, cxxcimod.Vartype):
        return vartype
    
    vartype = dimod.as_vartype(vartype)
    if vartype == dimod.SPIN:
        return cxxcimod.Vartype.SPIN
    if vartype == dimod.BINARY:
        return cxxcimod.Vartype.BINARY

#classmethods
BinaryPolynomialModel.from_serializable = lambda obj: make_BinaryPolynomialModel_from_JSON(obj).from_serializable(obj)
BinaryPolynomialModel.from_hising       = lambda *args, **kwargs: make_BinaryPolynomialModel_from_hising(*args, **kwargs)
BinaryPolynomialModel.from_hubo         = lambda *args, **kwargs: make_BinaryPolynomialModel_from_hubo(*args, **kwargs)
