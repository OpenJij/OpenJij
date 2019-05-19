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

class ChimeraModel(BinaryQuadraticModel):
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
        Chimera coordinate: x, y, z
        One dimension coordinate: i
        Relation: i = 8Ly + 8x + z

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
            x_i, y_i, z_i = self._to_index(i, self.unit_num_L)
            # list up indices which can connect i
            adj_list = []
            if z_i < 4:
                # part of left side (in the column representation) of a Chimera unit cell.
                if y_i > 0:
                    adj_list.append(self._chimera_index(x_i, y_i-1, z_i, self.unit_num_L))
                if y_i < self.unit_num_L-1:
                    adj_list.append(self._chimera_index(x_i, y_i+1, z_i, self.unit_num_L))
                adj_list += [self._chimera_index(x_i, y_i, z, self.unit_num_L) for z in range(4, 8)]
            else:
                # part of right side (in the column representation) of a Chimera unit cell.
                if x_i > 0:
                    adj_list.append(self._chimera_index(x_i-1, y_i, z_i, self.unit_num_L))
                if x_i < self.unit_num_L-1:
                    adj_list.append(self._chimera_index(x_i+1, y_i, z_i, self.unit_num_L)) 
                adj_list += [self._chimera_index(x_i, y_i, z, self.unit_num_L) for z in range(0, 4)]
            
            if not j in adj_list:
                incomp_part = 'The connectable nodes of {} are {}.'.format(j, adj_list)
                raise ValueError('Problem graph incompatible with chimera graph.\n' + incomp_part)
                return False
        return True

    def to_index(self, x, y, z, unit_num_L):
        """
        Chimera coordinate: x, y, z
        One dimension coordinate: i
        Relation: i = 8Ly + 8x + z
        """
        return 8*unit_num_L*y + 8*x + z

    def chimera_coordinate(self, i, unit_num_L):
        z_i = i % 8
        x_i = (i % (8 * unit_num_L) - z_i)/8
        y_i = (i - 8*x_i - z_i) / (8 * unit_num_L)
        return int(x_i), int(y_i), int(z_i)