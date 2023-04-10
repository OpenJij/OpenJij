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

from __future__ import annotations

from dimod import SPIN

import openjij.cxxjij as cj

from openjij.model.model import make_BinaryQuadraticModel


def make_ChimeraModel(linear, quadratic):
    """ChimeraModel factory.

    Returns:
        generated ChimeraModel class
    """
    
    class ChimeraModel(make_BinaryQuadraticModel(linear, quadratic)):
        """Binary quadnratic model dealing with chimera graph This model deal

        with chimera graph. ChimeraModel provide methods to verify whether a
        given interaction graph matches a Chimera graph and to convert it to
        cxxjij.graph.Chimera.

        Examples::
            # This interactions satisfy chimera topology.
            >>> Q={(0, 4): -1, (4, 12): -1}
            >>> chimera_model = ChimeraModel(Q, unit_num_L=2)  # make
            >>> chimera_self.validate_chimera()
        """
        
        def __init__(
                self,
            linear=None,
            quadratic=None,
            offset=0.0,
            vartype=SPIN,
            unit_num_L=None,
            model=None,
            gpu=False,
        ):
            self.gpu = gpu
            if model:
                super().__init__(
                    model.linear, model.quadratic, model.offset, model.vartype, gpu=gpu
                )
            else:
                super().__init__(linear, quadratic, offset, vartype, gpu=gpu)
            if not unit_num_L:
                raise ValueError(
                    "Input unit_num_L which is the length of the side of the two-dimensional grid where chimera unit cells are arranged."
                )
            self.unit_num_L = unit_num_L

            # check the type of indices is valid.
            self.coordinate = self._validate_indices(self.indices)

            # _chimera_index: 1-D index i,L -> chimera coordinate x,y,z
            # _to_index: chimera coordinate x,y,z,L -> 1-D index i
            if self.coordinate == "index":
                self._chimera_index = lambda i, L: self.chimera_coordinate(i, L)
                self._to_index = lambda x, y, z, L: self.to_index(x, y, z, L)
            elif self.coordinate == "chimera coordinate":
                self._chimera_index = lambda i, L: i
                self._to_index = lambda x, y, z, L: self.to_index(x, y, z, L)
        
        def _validate_indices(self, indices):
            """Check if the type of indices is valid.

            Args:
                indices (list(int) or list(tuple))
            Return:
                the type of indices ('chimera coordinate or index')
            """
            if isinstance(indices[0], int):
                return "index"
            elif isinstance(indices[0], (tuple, list)):
                if len(indices[0]) == len(indices[-1]) == 3:
                    return "chimera coordinate"

            raise ValueError(
                "In the chimera graph, index should be int or tuple or list."
            )

        def validate_chimera(self):
            """Check if the Chimera connectivity is valid.

            Chimera coordinate: r, c, z
            One dimension coordinate: i
            Relation: i = 8Lr + 8c + z

            Chimera unit cell (column reprezentation)
            0 - 4
            1 - 5
            2 - 6
            3 - 7
            """
            # check chimera interaction
            for (i, j) in self.quadratic.keys():
                r_i, c_i, z_i = self._chimera_index(i, self.unit_num_L)
                # list up indices which can connect i
                adj_list = []
                if z_i >= 4:
                    # part of right side of a Chimera unit cell (in the column representation).
                    if c_i > 0:
                        adj_list.append(
                            self._to_index(r_i, c_i - 1, z_i, self.unit_num_L)
                        )
                    if c_i < self.unit_num_L - 1:
                        adj_list.append(
                            self._to_index(r_i, c_i + 1, z_i, self.unit_num_L)
                        )
                    adj_list += [
                        self._to_index(r_i, c_i, z, self.unit_num_L)
                        for z in range(0, 4)
                    ]
                else:
                    # part of left side of a Chimera unit cell (in the column representation).
                    if r_i > 0:
                        adj_list.append(
                            self._to_index(r_i - 1, c_i, z_i, self.unit_num_L)
                        )
                    if r_i < self.unit_num_L - 1:
                        adj_list.append(
                            self._to_index(r_i + 1, c_i, z_i, self.unit_num_L)
                        )
                    adj_list += [
                        self._to_index(r_i, c_i, z, self.unit_num_L)
                        for z in range(4, 8)
                    ]

                connect_i = (
                    j if isinstance(j, int) else self._to_index(*j, self.unit_num_L)
                )
                if connect_i not in adj_list:
                    incomp_part = "The connectable nodes of {} are {}, not {}.".format(
                        i, adj_list, j
                    )
                    raise ValueError(
                        "Problem graph incompatible with chimera graph.\n" + incomp_part
                    )
                    return False
            return True

        def to_index(self, r, c, i, unit_num_L):
            """
            Chimera coordinate: r, c, i

            One dimension coordinate: i
            Relation: i = 8*L*r + 8*c + i
            Args:
                r (int): Row index of 2-D Chimera grid.
                c (int): Column index of 2-D Chimera grid.
                i (int): index in Chimera unit cell.
                unit_num_L (int): Row and Column length of 2-D Chimera grid.
            """
            return 8 * unit_num_L * r + 8 * c + i

        def chimera_coordinate(self, i, unit_num_L):
            """Convert 1-d index to chimera corrdinate

            Args:
                i (int): 1-D index 0~L*L*8-1
                unit_num_L (int): number of chimera grid size
            Returns:
                (int, int, int): chimera corrdinate
            """
            z_i = i % 8
            c_i = (i % (8 * unit_num_L) - z_i) / 8
            r_i = (i - 8 * c_i - z_i) / (8 * unit_num_L)
            return int(r_i), int(c_i), int(z_i)

        def get_cxxjij_ising_graph(self):
            """Get cxxjij.graph.Chimera type instance

            Args:
                i (int): 1-D index 0~L*L*8-1
                unit_num_L (int): number of chimera grid size
            Returns:
                object (cxxjij.graph.Chimera)
            """
            chimera_L = self.unit_num_L

            if not self.validate_chimera():
                raise ValueError("Problem graph incompatible with chimera graph.")
            _h, _J, _offset = self.to_ising()

            if self.gpu:
                chimera = cj.graph.ChimeraGPU(chimera_L, chimera_L)
            else:
                chimera = cj.graph.Chimera(chimera_L, chimera_L)

            for i, hi in _h.items():
                r_i, c_i, zi = self._chimera_index(i, L = chimera_L)
                if not self._index_validate(i, chimera_L):
                    raise ValueError(
                            "Problem graph incompatible with chimera graph. Node {}.".format(
                                    i
                                    )
                            )
                chimera[r_i, c_i, zi] = hi
            for (i, j), Jij in _J.items():
                r_i, c_i, zi = self._chimera_index(i, L = chimera_L)
                r_j, c_j, zj = self._chimera_index(j, L = chimera_L)
    
                # validate connection
                error_msg = f"In the {chimera_L}*{chimera_L} Chimera grid, "
                error_msg += f"there is no connection between node {i} and node {j}."
                linear_vldt = self._index_validate(
                        i, chimera_L
                        ) and self._index_validate(j, chimera_L)
                if not (
                        linear_vldt
                        and self._validate((r_i, c_i, zi), (r_j, c_j, zj), chimera_L)
                ):
                    raise ValueError(
                            "Problem graph incompatible with chimera graph.\n" + error_msg
                            )

                if r_i == r_j and c_i == c_j:
                    # connection in Chimera unit cell
                    if zj in [0, 4]:
                        chimera[r_i, c_i, zi, cj.graph.ChimeraDir.IN_0or4] = Jij
                    elif zj in [1, 5]:
                        chimera[r_i, c_i, zi, cj.graph.ChimeraDir.IN_1or5] = Jij
                    elif zj in [2, 6]:
                        chimera[r_i, c_i, zi, cj.graph.ChimeraDir.IN_2or6] = Jij
                    else:
                        chimera[r_i, c_i, zi, cj.graph.ChimeraDir.IN_3or7] = Jij
                # connection between Chimera unit cells
                elif r_i - r_j == -1:
                    chimera[r_i, c_i, zi, cj.graph.ChimeraDir.PLUS_R] = Jij
                elif r_i - r_j == 1:
                    chimera[r_i, c_i, zi, cj.graph.ChimeraDir.MINUS_R] = Jij
                elif c_i - c_j == -1:
                    chimera[r_i, c_i, zi, cj.graph.ChimeraDir.PLUS_C] = Jij
                elif c_i - c_j == 1:
                    chimera[r_i, c_i, zi, cj.graph.ChimeraDir.MINUS_C] = Jij

            return chimera

        def energy(self, sample, convert_sample=False):
            """Calc energy of the BinaryQuadraticModel.

            Args:
                sample: samples
                convert_sample: if true, the type of sample is automatically converted to self.vartype.
            """
            return super().energy(sample, sparse=True, convert_sample=convert_sample)

        def energies(self, samples_like, convert_sample=False):
            return super().energies(
                    samples_like, sparse = True, convert_sample = convert_sample
                    )

        def _validate(self, rcz1, rcz2, L):
            """Check if the connectivity is valid.

            Args:
                rcz1 (int), rcz2(int), L(int)
            Returns:
                result (bool)
            """
            r1, c1, z1 = rcz1
            r2, c2, z2 = rcz2
            left_side = [0, 1, 2, 3]
            right_side = [4, 5, 6, 7]
            if r1 == r2 and c1 == c2:
                if (z1 in left_side) and (z2 in right_side):
                    return True
                elif (z2 in left_side) and (z1 in right_side):
                    return True
            elif (c1 == c2 and abs(r1 - r2) == 1) or (r1 == r2 and abs(c1 - c2) == 1):
                return True
            return False

        def _index_validate(self, i, L):
            """Check if the index is valid.

            Args:
                i(int), L(int)
            Returns:
                result (bool)
            """
            if isinstance(i, tuple):
                two_d_bool = (i[0] < self.unit_num_L) and (i[1] < self.unit_num_L)
                return two_d_bool and (i[2] < 8)
            max_index = 8 * L * L
            return 0 <= i < max_index

    return ChimeraModel


def make_ChimeraModel_from_JSON(obj):
    """Make ChimeraModel from JSON.

    Returns:
        corresponding ChimeraModel type
    """
    label = obj["variable_labels"][0]
    if isinstance(label, list):
        # convert to tuple
        label = tuple(label)

    mock_linear = {label: 1.0}

    return make_ChimeraModel(mock_linear, {})


def ChimeraModel(
        linear: dict = None,
        quadratic: dict = None,
        offset: float = 0.0,
        vartype = SPIN,
        unit_num_L: int = None,
        model = None,
        gpu: bool = False,
        ):
    """Generate ChimeraModel object

    This model deal with chimera graph.
    ChimeraModel provide methods to verify whether a given interaction graph
    matches a Chimera graph and to convert it to cxxjij.graph.Chimera.

    Args:
        linear (dict): linear biases
        quadratic (dict): quadratic biases
        offset (float): offset
        vartype: vartype ('SPIN' or 'BINARY')
        unit_num_L (int): unit_num_L
        model (BinaryQuadraticModel): if model is not None, the object is initialized by model.
        gpu (bool): if true, this can be used for gpu samplers.
    Returns:
        generated ChimeraModel

    Examples:
        Example shows how to initialize ChimeraModel.::

            # This interactions satisfy chimera topology.
            >>> Q={(0, 4): -1, (4, 12): -1}
            >>> chimera_model = ChimeraModel(Q, unit_num_L=2)  # make
            >>> chimera_self.validate_chimera()
    """

    Model = make_ChimeraModel(linear, quadratic)

    return Model(linear, quadratic, offset, vartype, unit_num_L, model, gpu)


# classmethods
ChimeraModel.from_qubo = lambda Q, offset=0.0, **kwargs: make_ChimeraModel(
    {}, Q
).from_qubo(Q, offset, **kwargs)

ChimeraModel.from_ising = (
    lambda linear, quadratic, offset=0.0, **kwargs: make_ChimeraModel(
        linear, quadratic
    ).from_ising(linear, quadratic, offset, **kwargs)
)

ChimeraModel.from_serializable = lambda obj: make_ChimeraModel_from_JSON(
    obj
).from_serializable(obj)
