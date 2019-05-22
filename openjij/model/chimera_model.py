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

from openjij.model import BinaryQuadraticModel
import cxxjij as cj

class ChimeraModel(BinaryQuadraticModel):
    """Binary quadnratic model dealing with chimera graph
    This model deal with chimera graph.
    ChimeraModel provide methods to verify whether a given interaction graph matches a Chimera graph and to convert it to cxxjij.graph.Chimera.

    Examples:
        >>> Q={(0, 4): -1, (4, 12): -1}  # This interactions satisfy chimera topology.
        >>> chimera_model = ChimeraModel(Q, unit_num_L=2)  # make 
        >>> chimera_self.validate_chimera()
    """
    def __init__(self, h=None, J=None, Q=None, unit_num_L=None, model=None, var_type='SPIN'):
        if model:
            self.linear = model.linear
            self.quad = model.quad
            self.indices = model.indices
            self.energy_bias = model.energy_bias
            self.var_type = model.var_type
            self._interaction_matrix = None
        else:
            super().__init__(h=h, J=J, Q=Q, var_type=var_type)

        if not unit_num_L:
            raise ValueError('Input unit_num_L which is the length of the side of the two-dimensional grid where chimera unit cells are arranged.')
        self.unit_num_L = unit_num_L
        
        self.coordinate = self._validate_indices(self.indices)
        if self.coordinate == 'index':
            self._chimera_index = lambda x, y, z, L: self.to_index(x, y, z, L)
            self._to_index = lambda i, L: self.chimera_coordinate(i, L)
        elif self.coordinate == 'chimera coordinate':
            self._chimera_index = lambda x, y, z, L: (x, y, z)
            self._to_index = lambda i, L: i

    def _validate_indices(self, indices):
        if isinstance(indices[0], int):
            return 'index'
        elif isinstance(indices[0], (tuple, list)): 
            if len(indices[0]) == len(indices[-1]) == 3:
                return 'chimera coordinate'
        
        raise ValueError('In the chimera graph, index should be int or tuple or list.')

    def full_indices(self):
        if self.coordinate == 'index':
            return list(range(0, 8 * self.unit_num_L * self.unit_num_L))
        else:
            L = self.unit_num_L
            return [(x, y, i) for y in range(0, L*L) for x in range(0, L*L) for i in range(0,8)]
        
    def validate_chimera(self):
        """
        Chimera coordinate: r, c, z 
        One dimension coordinate: i
        Relation: i = 8Lr + 8c + z

        Chimera unit cell (column reprezentation)
        0 - 4
          ×
        1 - 5
          ×
        2 - 6
          ×
        3 - 7
        """
        # check chimera interaction
        for (i,j) in self.quad.keys():
            r_i, c_i, z_i = self._to_index(i, self.unit_num_L)
            # list up indices which can connect i
            adj_list = []
            if z_i >= 4:
                # part of right side of a Chimera unit cell (in the column representation).
                if c_i > 0:
                    adj_list.append(self._chimera_index(r_i, c_i-1, z_i, self.unit_num_L))
                if c_i < self.unit_num_L-1:
                    adj_list.append(self._chimera_index(r_i, c_i+1, z_i, self.unit_num_L))
                adj_list += [self._chimera_index(r_i, c_i, z, self.unit_num_L) for z in range(0, 4)]
            else:
                # part of left side of a Chimera unit cell (in the column representation).
                if r_i > 0:
                    adj_list.append(self._chimera_index(r_i-1, c_i, z_i, self.unit_num_L))
                if r_i < self.unit_num_L-1:
                    adj_list.append(self._chimera_index(r_i+1, c_i, z_i, self.unit_num_L)) 
                adj_list += [self._chimera_index(r_i, c_i, z, self.unit_num_L) for z in range(4, 8)]
            
            if not j in adj_list:
                incomp_part = 'The connectable nodes of {} are {}, not {}.'.format(i, adj_list, j)
                raise ValueError('Problem graph incompatible with chimera graph.\n' + incomp_part)
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
        return 8*unit_num_L*r + 8*c + i 

    def chimera_coordinate(self, i, unit_num_L):
        z_i = i % 8
        c_i = (i % (8 * unit_num_L) - z_i)/8
        r_i = (i - 8*c_i - z_i) / (8 * unit_num_L)
        return int(r_i), int(c_i), int(z_i)


    def get_chimera_graph(self):
        chimera_L = self.unit_num_L

        if not self.validate_chimera():
            raise ValueError("Problem graph incompatible with chimera graph.")
        _h, _J = self.ising_dictionary()

        self.energy_bias = self.energy_bias

        chimera = cj.graph.Chimera(chimera_L, chimera_L)
        for i, hi in _h.items():
            r_i, c_i, zi = self.chimera_coordinate(i, unit_num_L=chimera_L)
            chimera[r_i, c_i, zi] = hi
        for (i, j), Jij in _J.items():
            r_i, c_i, zi = self.chimera_coordinate(i, unit_num_L=chimera_L)
            r_j, c_j, zj = self.chimera_coordinate(j, unit_num_L=chimera_L)
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
            elif r_i == r_j + 1:
                chimera[r_i, c_i, zi, cj.graph.ChimeraDir.PLUS_R] = Jij
            elif r_i == r_j - 1:
                chimera[r_i, c_i, zi, cj.graph.ChimeraDir.MINUS_R] = Jij 
            elif c_i == c_j + 1:
                chimera[r_i, c_i, zi, cj.graph.ChimeraDir.PLUS_C] = Jij
            elif c_i == c_j - 1:
                chimera[r_i, c_i, zi, cj.graph.ChimeraDir.MINUS_C] = Jij
        return chimera