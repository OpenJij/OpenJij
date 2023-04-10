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

import openjij
import openjij as oj

from openjij.model.model import make_BinaryQuadraticModel


def make_KingGraph(linear=None, quadratic=None, king_graph=None):
    """KingGraph factory

    Returns:
        generated KingGraph class
    """
    mock_linear = {}
    mock_quadratic = {}

    if linear is not None:
        mock_linear = linear

    if quadratic is not None:
        mock_quadratic = quadratic

    if mock_linear == {} and mock_quadratic == {}:
        # no elements in linear and quadratic
        # fetch first element of the king_graph
        label = (king_graph[0][0], king_graph[0][0])
        # add to linear
        mock_linear[label] = 1.0

    class KingGraph(make_BinaryQuadraticModel(mock_linear, mock_quadratic)):
        """
        BQM for king graph of HITACHI CMOS Annealer

        Attributes:
            xrange (list(int)): represents hardware (CMOS) restricts for coordinate. [xmin, xmax]
            yrange (list(int)): represents hardware (CMOS) restricts for coordinate. [ymin, ymax]
            prange (list(int)): represents hardware (CMOS) restricts for the strength of interactions 'p'. [pmin, pmax]
            king_graph (list(int)):
                Annealing cloud Web API format representation of interaction coefficients
                Quadratic term [x1, y1, x2, y2, value]
                Linear term    [x1, y1, x1, y1, value]
        """

        def __init__(
            self,
            linear: dict = None,
            quadratic: dict = None,
            offset: float = 0.0,
            king_graph=None,
            vartype=SPIN,
            machine_type: str = "",
        ):
            """__init__.

            Args:
                linear (dict): linear biases
                quadratic (dict): quadratic biases
                offset (float): offset
                king_graph: represents ising or QUBO interaction.
                    Each spins are decided by 2-d corrdinate x, y.
                    Quadratic term [x1, y1, x2, y2, value]
                    Linear term    [x1, y1, x1, y1, value]
                vartype: 'SPIN' or 'BINARY'
                machine_type (str): choose 'ASIC' or 'FPGA'
            """
            vartype = oj.variable_type.cast_vartype(vartype)

            # set parameter ranges
            self.machine_type = machine_type
            if self.machine_type == "ASIC":
                self.xrange = [0, 351 + 1]
                self.yrange = [0, 175 + 1]
                self.prange = [-3, 3]
            elif self.machine_type == "FPGA":
                self.xrange = [0, 79 + 1]
                self.yrange = [0, 79 + 1]
                self.prange = [-127, 127]
            else:
                raise ValueError("machine type should be ASIC or FPGA")

            # convert format h, J, Q and initilize BQM
            if king_graph is not None:
                linear, quadratic = self._convert_to_BQM_format(king_graph, vartype)
            super().__init__(linear, quadratic, offset=offset, vartype=vartype)

            # reformat to ising king graph (which is Web API format)
            if king_graph is not None and vartype == SPIN:
                self._ising_king_graph = king_graph

            else:
                # generate Ising h and J and create ising_king_graph format
                lin, quad, _ = self.to_ising()
                self._ising_king_graph = []
                for index, h in lin.items():
                    if h != 0:
                        x, y = self._convert_to_xy(index)
                        self._ising_king_graph.append([x, y, x, y, h])
                for (i, j), J in quad.items():
                    if J != 0:
                        x1, y1 = self._convert_to_xy(i)
                        x2, y2 = self._convert_to_xy(j)
                        self._ising_king_graph.append([x1, y1, x2, y2, J])

            self._validation_ising_king_graph()

        def _convert_to_BQM_format(self, king_graph, vartype):
            linear, quad = {}, {}
            for x1, y1, x2, y2, value in king_graph:
                if (x1, y1) == (x2, y2):
                    linear[(x1, y1)] = value
                else:
                    quad[(x1, y1), (x2, y2)] = value
            return linear, quad

        def get_ising_king_graph(self):
            return self._ising_king_graph

        def king_indices(self):
            if isinstance(self.indices[0], tuple):
                return self.indices
            else:
                return [self._convert_to_xy(i) for i in self.indices]

        def _convert_to_xy(self, index):
            if isinstance(index, tuple):
                return index[0], index[1]
            else:
                y = int(index / self.xrange[1])
                return int(index - y * self.xrange[1]), y

        def convert_to_index(self, x, y):
            return y * self.xrange[1] + x

        def _validation_ising_king_graph(self):
            for xi, yi, xj, yj, p in self._ising_king_graph:
                if yi >= self.yrange[1] or yj >= self.yrange[1]:
                    raise ValueError(
                        "Graph is incomplete xi: {}, yi: {}, xj: {}, yj: {}, p:{}".format(
                            xi, yi, xj, yj, p
                        )
                    )
                if not (xi in [xj, xj - 1, xj + 1]) or not (yi in [yj, yj - 1, yj + 1]):
                    raise ValueError(
                        "Graph is incomplete xi: {}, yi: {}, xj: {}, yj: {}, p:{}".format(
                            xi, yi, xj, yj, p
                        )
                    )
                if not (self.prange[0] <= p <= self.prange[1]):
                    raise ValueError(
                        "Graph is incomplete xi: {}, yi: {}, xj: {}, yj: {}, p: {}".format(
                            xi, yi, xj, yj, p
                        )
                    )

        def energy(self, sample):
            return super().energy(sample, sparse=True)

        def energies(self, samples_like):
            return super().energies(sample_like, sparse=True)

    return KingGraph


def make_KingGraph_from_JSON(obj):
    """KingGraph factory for JSON

    Args:
        obj (dict): JSON object
    Returns:
        generated KingGraph class
    """
    label = obj["variable_labels"][0]
    if isinstance(label, list):
        # convert to tuple
        label = tuple(label)

    mock_linear = {label: 1.0}

    return make_KingGraph(mock_linear, {})


def KingGraph(
    linear=None,
    quadratic=None,
    offset=0.0,
    king_graph=None,
    vartype=SPIN,
    machine_type="",
):
    """Generate KingGraph model.

    Args:
        linear (dict): linear biases
        quadratic (dict): quadratic biases
        offset (float): offset
        king_graph: represents ising or QUBO interaction.
            Each spins are decided by 2-d corrdinate x, y.

            * Quadratic term: [x1, y1, x2, y2, value]
            * Linear term:    [x1, y1, x1, y1, value]
        vartype: 'SPIN' or 'BINARY'
        machine_type (str): choose 'ASIC' or 'FPGA'
    Returns:
        generated KingGraphModel
    Examples:
        The following code shows intialization of KingGraph::

            >>> h = {}
            >>> J = {(0, 1): -1.0, (1, 2): -3.0}
            >>> king_graph = oj.KingGraph(machine_type="ASIC", linear=h, quadratic=J)

        You can initialize it from `king_interaction`::

            >>> king_interaction = [[0, 0, 1, 0, -1.0], [1, 0, 2, 0, -3.0]]
            >>> king_graph = oj.KingGraph(machine_type="ASIC", king_graph=king_interaction)
    """

    Model = make_KingGraph(linear, quadratic, king_graph)

    return Model(linear, quadratic, offset, king_graph, vartype, machine_type)


# classmethods
KingGraph.from_qubo = lambda Q, offset=0.0, **kwargs: make_KingGraph({}, Q).from_qubo(
    Q, offset, **kwargs
)

KingGraph.from_ising = lambda linear, quadratic, offset=0.0, **kwargs: make_KingGraph(
    linear, quadratic
).from_ising(linear, quadratic, offset, **kwargs)

KingGraph.from_serializable = lambda obj: make_KingGraph_from_JSON(
    obj
).from_serializable(obj)
