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
import openjij as oj


def convert_response(response):
    if isinstance(response, oj.sampler.response.Response):
        return response

    try:
        from dimod.response import Response
        from dimod.sampleset import SampleSet
    except ImportError:
        raise ImportError('Import dwave dimod : "pip install dimod"')

    if isinstance(response, Response) or isinstance(response, SampleSet):
        from dimod.vartypes import BINARY

        vartype = "BINARY" if response.vartype == BINARY else "SPIN"
        o_res = oj.sampler.response.Response(
            vartype=vartype, indices=list(response.variables)
        )
        states = []
        energies = []
        for rec in response.record:
            for _ in range(rec[2]):
                states.append(rec[0])
                energies.append(rec[1])
        o_res.update_ising_states_energies(states=states, energies=energies)

        o_res.info = response.info
        if "qpu_sampling_time" in response.info:
            o_res.info["sampling_time"] = response.info["qpu_sampling_time"]
        if "anneal_time_per_run" in response.info:
            o_res.info["execution_time"] = response.info["anneal_time_per_run"]
        o_res.info["dimod"] = response
        return o_res
    else:
        raise TypeError(
            "response is dimod.response.Response, SampleSet or openjij.sampler.response.Response"
        )
