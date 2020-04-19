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

import cimod
import dimod

SPIN = cimod.SPIN
BINARY = cimod.BINARY
Vartype = cimod.Vartype

def cast_var_type(var_type):

    if isinstance(var_type, cimod.Vartype):
        return var_type
    else:
        var_type = dimod.as_vartype(var_type)

        if var_type == dimod.SPIN:
            return cimod.SPIN

        elif var_type == dimod.BINARY:
            return cimod.BINARY

