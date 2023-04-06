# Copyright 2023 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""| This module defines the BinaryQuadraticModel with the Hamiltonian,

.. math::

    H = \\sum_{i\\neq j} J_{ij}\\sigma_i \\sigma_j + \\sum_{i} h_{i}\\sigma_i,

| in an Ising form and

.. math::

    H = \\sum_{ij} Q_{ij}x_i x_j + \\sum_{i} H_{i}x_i,

| in a QUBO form.
| The methods and usage are basically the same as `dimod <https://github.com/dwavesystems/dimod>`_.
"""
from __future__ import annotations
import cimod
import cimod.cxxcimod as cxxcimod
import dimod

import openjij.cxxjij as cxxjij


def make_BinaryQuadraticModel(linear: dict, quadratic: dict, sparse):
    """BinaryQuadraticModel factory.

    Returns:
        generated BinaryQuadraticModel class
    """

    Base = cimod.make_BinaryQuadraticModel(linear, quadratic, sparse)

    class BinaryQuadraticModel(Base):
        """Represents Binary quadratic model.

        Indices are listed in self.indices.
        Attributes:
            vartype (dimod.Vartype): variable type SPIN or BINARY
            linear (dict): represents linear term
            quadratic (dict): represents quadratic term
            offset (float): represents constant energy term when convert to SPIN from BINARY
            num_variables (int): represents number of variables in the model
            variables (list): represents variables of the binary quadratic model
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
            gpu = kwargs.pop("gpu", False)
            super().__init__(*args, **kwargs)
            self.gpu = gpu
            self.model_type = "openjij.BinaryQuadraticModel"

        def get_cxxjij_ising_graph(self):
            """Generate cxxjij Ising graph from the interactions.

            Returns:
                cxxjij.graph.Dense or cxxjij.graph.Sparse:
                offset (float): offset of the energy due to qubo->ising transformation
            """

            # if sparse is true, select `cxxjij.graph.CSRSparse` graph type,
            # else, select `cxxjij.graph.Dense` graph type.
            if sparse:
                old_vartype = self.vartype
                self.change_vartype("SPIN")

                GraphClass = (
                    cxxjij.graph.CSRSparse if self.gpu == False else cxxjij.graph.CSRSparseGPU
                )
                offset = self.offset
                sparse_mat = self.interaction_matrix()

                self.change_vartype(old_vartype)
                return GraphClass(sparse_mat), offset
            else:
                old_vartype = self.vartype
                self.change_vartype("SPIN")

                GraphClass = (
                    cxxjij.graph.Dense if self.gpu == False else cxxjij.graph.DenseGPU
                )
                # initialize with interaction matrix.
                mat = self.interaction_matrix()
                num_variables = mat.shape[0] - 1
                dense = GraphClass(num_variables)
                dense.set_interaction_matrix(mat)
                offset = self.offset

                self.change_vartype(old_vartype)
                return dense, offset

        # compatible with the previous version

        def calc_energy(self, sample, **kwargs):
            return self.energy(sample, **kwargs)

        # compatible with the previous version
        @property
        def indices(self):
            return self.variables

    return BinaryQuadraticModel


def make_BinaryQuadraticModel_from_JSON(obj: dict):
    """Make BinaryQuadraticModel from JSON.

    Returns:
        corresponding BinaryQuadraticModel type
    """
    label = obj["variable_labels"][0]
    if isinstance(label, list):
        # convert to tuple
        label = tuple(label)

    mock_linear = {label: 1.0}

    if obj["version"]["bqm_schema"] == "3.0.0-dense":
        sparse = False
    elif obj["version"]["bqm_schema"] == "3.0.0":
        sparse = True
    else:
        raise TypeError("Invalid bqm_schema")

    return make_BinaryQuadraticModel(mock_linear, {}, sparse)


def BinaryQuadraticModel(linear, quadratic, *args, **kwargs):
    """Generate BinaryQuadraticModel object.

    Attributes:
        vartype (dimod.Vartype): variable type SPIN or BINARY
        linear (dict): represents linear term
        quadratic (dict): represents quadratic term
        offset (float): represents constant energy term when convert to SPIN from BINARY
        num_variables (int): represents number of variables in the model
        variables (list): represents variables of the binary quadratic model
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

    sparse_option = kwargs.pop("sparse", False)

    Model = make_BinaryQuadraticModel(linear, quadratic, sparse_option)

    def __extract_offset_and_vartype(*args, **kwargs):
        if kwargs == {}:
            if len(args) == 0:
                raise TypeError(
                    f"Offset or vartype is configured incorrectly. Vartype must be set."
                )
            elif len(args) == 1:
                offset = 0.0
                [vartype] = args
            elif len(args) == 2:
                [offset, vartype] = args
            else:
                raise TypeError(
                    f"Offset or vartype is configured incorrectly. Vartype must be set."
                )
        else:
            if "offset" in kwargs and "vartype" in kwargs:
                offset = kwargs["offset"]
                vartype = kwargs["vartype"]
            elif "offset" in kwargs:
                if len(args) != 1:
                    raise TypeError(
                        f"Offset or vartype is configured incorrectly. Vartype must be set."
                    )
                offset = kwargs["offset"]
                [vartype] = args
            elif "vartype" in kwargs:
                if len(args) >= 2:
                    raise TypeError(
                        f"Offset or vartype is configured incorrectly. Vartype must be set."
                    )
                elif len(args) == 0:
                    offset = 0.0
                elif len(args) == 1:
                    [offset] = args
                vartype = kwargs["vartype"]
            else:
                raise TypeError(
                    f"Offset or vartype is configured incorrectly. Vartype must be set."
                )

        return offset, vartype

    offset, vartype = __extract_offset_and_vartype(*args, **kwargs)

    return Model(linear, quadratic, offset, vartype)


# classmethods
def bqm_from_numpy_matrix(
    mat, variables: list = None, offset=0.0, vartype="BINARY", **kwargs
):
    if variables is None:
        # generate array
        num_variables = mat.shape[0]
        variables = list(range(num_variables))

    sparse_option = kwargs.pop("sparse", False)

    return make_BinaryQuadraticModel(
        {variables[0]: 1.0}, {}, sparse_option
    ).from_numpy_matrix(mat, variables, offset, vartype, True, **kwargs)


BinaryQuadraticModel.from_numpy_matrix = bqm_from_numpy_matrix

def bqm_from_qubo(Q, offset=0.0, **kwargs):
    sparse_option = kwargs.pop("sparse", False)
    return make_BinaryQuadraticModel(
        {}, Q, sparse_option
    ).from_qubo(Q, offset, **kwargs)

BinaryQuadraticModel.from_qubo = bqm_from_qubo


def bqm_from_ising(linear, quadratic, offset=0.0, **kwargs):
    sparse_option = kwargs.pop("sparse", False)
    return make_BinaryQuadraticModel(
        linear, quadratic, sparse_option
    ).from_ising(linear, quadratic, offset, **kwargs)

BinaryQuadraticModel.from_ising = bqm_from_ising


BinaryQuadraticModel.from_serializable = (
    lambda obj, **kwargs: make_BinaryQuadraticModel_from_JSON(obj).from_serializable(
        obj, **kwargs
    )
)


def make_BinaryPolynomialModel(polynomial, index_type=None, tuple_size=0):
    class BinaryPolynomialModel(
        cimod.make_BinaryPolynomialModel(polynomial, index_type, tuple_size)
    ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_type = "openjij.BinaryPolynomialModel"

        def get_cxxjij_ising_graph(self):
            return cxxjij.graph.Polynomial(self.to_serializable())

        def calc_energy(self, sample, omp_flag=True):
            return self.energy(sample, omp_flag)

    return BinaryPolynomialModel


def make_BinaryPolynomialModel_from_JSON(obj):
    if obj["type"] != "BinaryPolynomialModel":
        raise Exception('Type must be "BinaryPolynomialModel"')
    mock_polynomial = {}
    if obj["index_type"] == "IndexType.INT":
        mock_polynomial = {(0, 1): 1}
    elif obj["index_type"] == "IndexType.STRING":
        mock_polynomial = {("a", "b"): 1}
    elif obj["index_type"] == "IndexType.INT_TUPLE_2":
        mock_polynomial = {((0, 1), (1, 2)): 1}
    elif obj["index_type"] == "IndexType.INT_TUPLE_3":
        mock_polynomial = {((0, 1, 2), (1, 2, 3)): 1}
    elif obj["index_type"] == "IndexType.INT_TUPLE_4":
        mock_polynomial = {((0, 1, 2, 3), (1, 2, 3, 4)): 1}
    else:
        raise TypeError("Invalid types of polynomial")
    return make_BinaryPolynomialModel(mock_polynomial)


def BinaryPolynomialModel(*args, **kwargs):
    if kwargs == {}:
        if len(args) <= 1:
            raise TypeError("Invalid argument for this function")
        elif len(args) == 2:
            if isinstance(args[0], dict):
                return _BinaryPolynomialModel_from_dict(args[0], _to_cxxcimod(args[1]))
            else:
                raise TypeError("Invalid argument for this function")
        elif len(args) == 3:
            key_condition = isinstance(args[0], list) or isinstance(args[0], tuple)
            val_condition = isinstance(args[1], list) or isinstance(args[1], tuple)
            if key_condition and val_condition:
                return _BinaryPolynomialModel_from_list(
                    args[0], args[1], _to_cxxcimod(args[2])
                )
            else:
                raise TypeError("Invalid argument for this function")
        else:
            raise TypeError("Invalid argument for this function")
    else:
        if "keys" in kwargs and "values" in kwargs and "vartype" in kwargs:
            key_condition = isinstance(kwargs["keys"], list) or isinstance(
                kwargs["keys"], tuple
            )
            val_condition = isinstance(kwargs["values"], list) or isinstance(
                kwargs["values"], tuple
            )
            if key_condition and val_condition:
                return _BinaryPolynomialModel_from_list(
                    kwargs["keys"], kwargs["values"], _to_cxxcimod(kwargs["vartype"])
                )
            else:
                raise TypeError("Invalid argument for this function")
        elif "polynomial" in kwargs and "vartype" in kwargs:
            if isinstance(kwargs["polynomial"], dict):
                return _BinaryPolynomialModel_from_dict(
                    kwargs["polynomial"], _to_cxxcimod(kwargs["vartype"])
                )
            else:
                raise TypeError("Invalid argument for this function")
        elif "values" in kwargs and "vartype" in kwargs:
            if len(args) != 1:
                raise TypeError("Invalid argument for this function")
            key_condition = isinstance(args[0], list) or isinstance(args[0], tuple)
            val_condition = isinstance(kwargs["values"], list) or isinstance(
                kwargs["values"], tuple
            )
            if key_condition and val_condition:
                return _BinaryPolynomialModel_from_list(
                    args[0], kwargs["values"], _to_cxxcimod(kwargs["vartype"])
                )
            else:
                raise TypeError("Invalid argument for this function")
        elif "vartype" in kwargs:
            if len(args) == 1:
                if isinstance(args[0], dict):
                    return _BinaryPolynomialModel_from_dict(
                        args[0], _to_cxxcimod(kwargs["vartype"])
                    )
                else:
                    raise TypeError("Invalid argument for this function")
            elif len(args) == 2:
                key_condition = isinstance(args[0], list) or isinstance(args[0], tuple)
                val_condition = isinstance(args[1], list) or isinstance(args[1], tuple)
                if key_condition and val_condition:
                    return _BinaryPolynomialModel_from_list(
                        args[0], args[1], _to_cxxcimod(kwargs["vartype"])
                    )
                else:
                    raise TypeError("Invalid argument for this function")
            else:
                raise TypeError("Invalid argument for this function")
        else:
            raise TypeError("Invalid argument for this function")


def _BinaryPolynomialModel_from_dict(polynomial: dict, vartype):
    Model = make_BinaryPolynomialModel(polynomial)
    return Model(polynomial, _to_cxxcimod(vartype))


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
    if label is None:
        Model = make_BinaryPolynomialModel({(): 1.0})
        return Model(keys, values, _to_cxxcimod(vartype))
    else:
        if isinstance(label, list):
            label = tuple(label)
        mock_polynomial = {(label,): 1.0}
        Model = make_BinaryPolynomialModel(mock_polynomial)
        return Model(keys, values, _to_cxxcimod(vartype))


def make_BinaryPolynomialModel_from_hising(*args, **kwargs):
    if kwargs == {}:
        if len(args) == 0:
            raise TypeError("Invalid argument for this function")
        elif len(args) == 1:
            if isinstance(args[0], dict):
                return _make_BinaryPolynomialModel_from_hising_from_dict(args[0])
            else:
                raise TypeError("Invalid argument for this function")
        elif len(args) == 2:
            key_condition = isinstance(args[0], list) or isinstance(args[0], tuple)
            val_condition = isinstance(args[1], list) or isinstance(args[1], tuple)
            if key_condition and val_condition:
                return _make_BinaryPolynomialModel_from_hising_from_list(
                    args[0], args[1]
                )
            else:
                raise TypeError("Invalid argument for this function")
        else:
            raise TypeError("Invalid argument for this function")
    else:
        if "keys" in kwargs and "values" in kwargs:
            key_condition = isinstance(kwargs["keys"], list) or isinstance(
                kwargs["keys"], tuple
            )
            val_condition = isinstance(kwargs["values"], list) or isinstance(
                kwargs["values"], tuple
            )
            if key_condition and val_condition:
                return _make_BinaryPolynomialModel_from_hising_from_list(
                    kwargs["keys"], kwargs["values"]
                )
            else:
                raise TypeError("Invalid argument for this function")
        elif "values" in kwargs:
            if len(args) != 1:
                raise TypeError("Invalid argument for this function")
            key_condition = isinstance(args[0], list) or isinstance(args[0], tuple)
            val_condition = isinstance(kwargs["values"], list) or isinstance(
                kwargs["values"], tuple
            )
            if key_condition and val_condition:
                return _make_BinaryPolynomialModel_from_hising_from_list(
                    args[0], kwargs["values"]
                )
            else:
                raise TypeError("Invalid argument for this function")
        elif "polynomial" in kwargs:
            if len(args) != 0:
                raise TypeError("Invalid argument for this function")
            if isinstance(kwargs["polynomial"], dict):
                _make_BinaryPolynomialModel_from_hising_from_dict(kwargs["polynomial"])
            else:
                raise TypeError("Invalid argument for this function")
        else:
            raise TypeError("Invalid argument for this function")


def _make_BinaryPolynomialModel_from_hising_from_dict(polynomial: dict):
    return make_BinaryPolynomialModel(polynomial).from_hising(polynomial)


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

    if label is None:
        return make_BinaryPolynomialModel({(): 1.0}).from_hising(keys, values)
    else:
        if isinstance(label, list):
            label = tuple(label)
        mock_polynomial = {(label,): 1.0}
        return make_BinaryPolynomialModel(mock_polynomial).from_hising(keys, values)


def make_BinaryPolynomialModel_from_hubo(*args, **kwargs):
    if kwargs == {}:
        if len(args) == 0:
            raise TypeError("Invalid argument for this function")
        elif len(args) == 1:
            if isinstance(args[0], dict):
                return _make_BinaryPolynomialModel_from_hubo_from_dict(args[0])
            else:
                raise TypeError("Invalid argument for this function")
        elif len(args) == 2:
            key_condition = isinstance(args[0], list) or isinstance(args[0], tuple)
            val_condition = isinstance(args[1], list) or isinstance(args[1], tuple)
            if key_condition and val_condition:
                return _make_BinaryPolynomialModel_from_hubo_from_list(args[0], args[1])
            else:
                raise TypeError("Invalid argument for this function")
        else:
            raise TypeError("Invalid argument for this function")
    else:
        if "keys" in kwargs and "values" in kwargs:
            key_condition = isinstance(kwargs["keys"], list) or isinstance(
                kwargs["keys"], tuple
            )
            val_condition = isinstance(kwargs["values"], list) or isinstance(
                kwargs["values"], tuple
            )
            if key_condition and val_condition:
                return _make_BinaryPolynomialModel_from_hubo_from_list(
                    kwargs["keys"], kwargs["values"]
                )
            else:
                raise TypeError("Invalid argument for this function")
        elif "values" in kwargs:
            if len(args) != 1:
                raise TypeError("Invalid argument for this function")
            key_condition = isinstance(args[0], list) or isinstance(args[0], tuple)
            val_condition = isinstance(kwargs["values"], list) or isinstance(
                kwargs["values"], tuple
            )
            if key_condition and val_condition:
                return _make_BinaryPolynomialModel_from_hubo_from_list(
                    args[0], kwargs["values"]
                )
            else:
                raise TypeError("Invalid argument for this function")
        elif "polynomial" in kwargs:
            if len(args) != 0:
                raise TypeError("Invalid argument for this function")
            if isinstance(kwargs["polynomial"], dict):
                _make_BinaryPolynomialModel_from_hubo_from_dict(kwargs["polynomial"])
            else:
                raise TypeError("Invalid argument for this function")
        else:
            raise TypeError("Invalid argument for this function")


def _make_BinaryPolynomialModel_from_hubo_from_dict(polynomial: dict):
    return make_BinaryPolynomialModel(polynomial).from_hubo(polynomial)


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

    if label is None:
        return make_BinaryPolynomialModel({(): 1.0}).from_hubo(keys, values)
    else:
        if isinstance(label, list):
            label = tuple(label)
        mock_polynomial = {(label,): 1.0}
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


# classmethods
BinaryPolynomialModel.from_serializable = (
    lambda obj: make_BinaryPolynomialModel_from_JSON(obj).from_serializable(obj)
)
BinaryPolynomialModel.from_hising = (
    lambda *args, **kwargs: make_BinaryPolynomialModel_from_hising(*args, **kwargs)
)
BinaryPolynomialModel.from_hubo = (
    lambda *args, **kwargs: make_BinaryPolynomialModel_from_hubo(*args, **kwargs)
)
