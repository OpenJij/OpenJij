# Copyright 2023 Jij Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from __future__ import annotations
import dimod

from openjij.utils.graph_utils import chimera_to_ind


class BaseGPUChimeraSampler(dimod.Structured):
    """Abstract GPUChimera Sampler."""

    @property
    def parameters(self):
        param = super().parameters
        param["unit_num_L"] = ["system parameter"]
        return param

    @property
    def nodelist(self):
        if "unit_num_L" not in dir(self) or self.unit_num_L is None:
            raise ValueError("this object is not set unit_num_L yet.")

        return list(range(0, 8 * 8 * self.unit_num_L * self.unit_num_L))

    @property
    def edgelist(self):
        if "unit_num_L" not in dir(self) or self.unit_num_L is None:
            raise ValueError("this object is not set unit_num_L yet.")

        edges = []
        for r in range(self.unit_num_L):
            for c in range(self.unit_num_L):
                for z in range(8):
                    i = chimera_to_ind(r, c, z, self.unit_num_L)

                    if z in [0, 1, 2, 3]:
                        edges += [
                            (i, chimera_to_ind(r, c, _z, self.unit_num_L))
                            for _z in [4, 5, 6, 7]
                        ]
                        if r < self.unit_num_L - 1:
                            edges.append(
                                (i, chimera_to_ind(r + 1, c, z, self.unit_num_L))
                            )

                    elif z in [4, 5, 6, 7]:
                        if c < self.unit_num_L - 1:
                            edges.append(
                                (i, chimera_to_ind(r, c + 1, z, self.unit_num_L))
                            )
        return edges
