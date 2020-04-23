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

import numpy as np
import dimod
import openjij
from openjij.variable_type import cast_to_dimod


class Response(dimod.SampleSet):
    @property
    def states(self):
        return self.record.sample

    @property
    def indices(self):
        return list(self.variables)

    @property
    def energies(self):
        return self.record.energy

    @property
    def min_sampels(self):
        return self.record.first

    def update_trotter_ising_states_energies(self, trotter_states, q_energies):
        pass

    def update_ising_states_energies(self, states, energies):
        pass

    @classmethod
    def from_samples(cls, samples_like, vartype, energy, **kwargs):
        return super().from_samples(samples_like, cast_to_dimod(vartype), energy, **kwargs)

    @classmethod
    def from_samples_bqm(cls, samples_like, bqm, **kwargs):
        linear = bqm.linear
        quadratic = bqm.quadratic
        vartype = cast_to_dimod(bqm.vartype)
        offset = bqm.offset
        return super().from_samples_bqm(samples_like, 
                openjij.BinaryQuadraticModel(linear, quadratic, offset, vartype), **kwargs)
