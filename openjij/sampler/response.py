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
"""| OpenJij response class.

| The usage is almost the same as :class:`dimod.SampleSet`.
"""
from __future__ import annotations
import dimod


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
    def min_samples(self):
        return self.first.sample
