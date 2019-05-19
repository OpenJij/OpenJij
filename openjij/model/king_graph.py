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

from .model import BinaryQuadraticModel

class KingGraph(BinaryQuadraticModel):
    """
    BQM for king graph of HITACHI CMOS Annealer
    Attributes
    ---------
    xrange : list of int
        represents hardware (CMOS) restricts for coordinate. [xmin, xmax]
    yrange : list of int
        represents hardware (CMOS) restricts for coordinate. [ymin, ymax]
    prange : list of int
        represents hardware (CMOS) restricts for the strength of interactions 'p'. [pmin, pmax]
    king_graph : list of list of int
        Annealing cloud Web API format representation of interaction coefficients
        Quadratic term [x1, y1, x2, y2, value]
        Linear term    [x1, y1, x1, y1, value]
    """
    def __init__(self, machine_type, h=None, J=None, Q=None, king_graph=None, var_type='SPIN'):
        """
        The constructor reformat interactions to Web API format (ising king graph),
        and validates that the interaction is in King Graph. 
        ----------
        machine_type : int
            choose 'ASIC' or 'FPGA'
        king_graph : list of list
            represents ising or QUBO interaction.
            Each spins are decided by 2-d corrdinate x, y.
            Quadratic term [x1, y1, x2, y2, value]
            Linear term    [x1, y1, x1, y1, value]
        """
        # set parameter ranges
        self.machine_type = machine_type
        if self.machine_type == "ASIC":
            self.xrange = [0, 351+1]
            self.yrange = [0, 175+1]
            self.prange = [-3, 3]
        elif self.machine_type == "FPGA":
            self.xrange = [0, 79+1]
            self.yrange = [0, 79+1]
            self.prange = [-127, 127]
        else:
            raise ValueError('machine type should be ASIC or FPGA')

        # convert format h, J, Q and initilize BQM
        if king_graph is not None:
            h, J, Q = self._convert_to_BQM_format(king_graph, var_type)
        super().__init__(h=h, J=J, Q=Q, var_type=var_type)

        # reformat to ising king graph (which is Web API format)
        if king_graph is not None and var_type == "SPIN":
            self._ising_king_graph = king_graph
        elif var_type == "SPIN":
            self._ising_king_graph = []
            for index, h in self.linear.items():
                x, y = self._convert_to_xy(index)
                self._ising_king_graph.append([x,y ,x,y, h])
            for (i, j), J in self.quad.items():
                x1, y1 = self._convert_to_xy(i)
                x2, y2 = self._convert_to_xy(j)
                self._ising_king_graph.append([x1, y1, x2, y2, J])
        else:
            ising_int = self.ising_interactions()
            sys_size = len(ising_int)
            self._ising_king_graph = []
            for i in range(sys_size):
                for j in range(i, sys_size):
                    if ising_int[i][j] == 0:
                        continue
                    x1, y1 = self._convert_to_xy(self.indices[i])
                    x2, y2 = self._convert_to_xy(self.indices[j])
                    self._ising_king_graph.append([x1, y1, x2, y2, ising_int[i][j]])

        self._validation_ising_king_graph()
        
    def _convert_to_BQM_format(self, king_graph, var_type):
        h, J, Q = None, None, None
        if var_type == "SPIN":
            h, J = {}, {}
            for x1, y1, x2, y2, value in king_graph:
                if (x1, y1) == (x2, y2):
                    h[(x1, y1)] = value
                else:
                    J[(x1, y1), (x2, y2)] = value
        else:  # qubo
            Q = {((x1, y1), (x2, y2)): value for x1, y1, x2, y2, value in king_graph}
        return h, J, Q

    def get_ising_king_graph(self):
        return self._ising_king_graph
        
    def king_indices(self):
        if isinstance(self.indices[0], tuple):
            return self.indices
        else:
            return [self.convert_to_xy(i) for i in self.indices]
            
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
                raise ValueError('Graph is incomplete xi: {}, yi: {}, xj: {}, yj: {}, p:{}'
                .format(xi, yi, xj, yj, p))
            if not (xi in [xj, xj-1, xj+1]) or not (yi in [yj, yj-1, yj+1]):
                raise ValueError('Graph is incomplete xi: {}, yi: {}, xj: {}, yj: {}, p:{}'
                .format(xi, yi, xj, yj, p))
            if not (self.prange[0] <= p <= self.prange[1]):
                raise ValueError('Graph is incomplete xi: {}, yi: {}, xj: {}, yj: {}, p: {}'
                .format(xi, yi, xj, yj, p))
            
            
    def convert_ising(self):
        interactions = {(x + y*self.xrange[1], xn + yn*self.xrange[1]): p for x,y, xn, yn, p in self.king_graph}
        return interactions