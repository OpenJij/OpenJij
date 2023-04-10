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
import dimod

SPIN = dimod.SPIN
BINARY = dimod.BINARY

Vartype = dimod.Vartype


def cast_vartype(vartype):
    if isinstance(vartype, dimod.Vartype):
        return vartype
    elif isinstance(vartype, str):
        if vartype.upper() == "SPIN":
            return SPIN
        elif vartype.upper() == "BINARY":
            return BINARY
