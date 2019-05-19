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
    def __init__(self, h=None, J=None, Q=None, model=None, var_type='SPIN'):
        if model:
            print(model)
            self.linear = model.linear
            self.quad = model.quad
            self.indices = model.indices
            self.energy_bias = model.energy_bias
        else:
            super().__init__(h=h, J=J, Q=Q, var_type=var_type)
        
        coordinate = self._validate_indices(self.indices)
        if coordinate == 'index':
            self._chimera_index = lambda x, y, z, L: self.to_index(x, y, z, L)
            self._to_index = lambda i, L: self.chimera_coordinate(i, L)
        elif coordinate == 'chimera coordinate':
            self._chimera_index = lambda x, y, z, L: (x, y, z)
            self._to_index = lambda i, L: i


    def _validate_indices(self, indices):
        if isinstance(indices[0], int):
            return 'index'
        elif isinstance(indices[0], (tuple, list)): 
            if len(indices[0]) == len(indices[-1]) == 3:
                return 'chimera coordinate'
        
        raise ValueError('In the chimera graph, index should be int or tuple or list.')
        
        
    def validate_chimera(self, unit_num_L):
        """
        Chimera coordinate: x, y, z
        One dimension coordinate: i
        Relation: i = 8Ly + 8x + z
        """
        # check chimera interaction
        for (i,j) in self.quad.keys():
            x_i, y_i, z_i = self._to_index(i, unit_num_L)
            # list up indices which can connect i
            adj_list = []
            if z_i < 4:
                if y_i > 0:
                    adj_list.append(self._chimera_index(x_i, y_i-1, z_i, unit_num_L))
                if y_i < unit_num_L-1:
                    adj_list.append(self._chimera_index(x_i, y_i+1, z_i, unit_num_L))
                adj_list += [self._chimera_index(x_i, y_i, z, unit_num_L) for z in range(4, 8)]
            else:
                if x_i > 0:
                    adj_list.append(self._chimera_index(x_i, y_i, z_i-1, unit_num_L))
                if x_i < unit_num_L-1:
                    adj_list.append(self._chimera_index(x_i, y_i, z_i+1, unit_num_L)) 
                adj_list += [self._chimera_index(x_i, y_i, z, unit_num_L) for z in range(0, 4)]
            
            if not j in adj_list:
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
        x_i = (i - z_i) % unit_num_L / 8
        y_i = (i-(8*x_i + z_i))/(8 * unit_num_L)
        return int(x_i), int(y_i), int(z_i)