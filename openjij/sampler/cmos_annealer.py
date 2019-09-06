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

import requests
import json

from openjij.sampler import BaseSampler, Response
from openjij.model import KingGraph

import numpy as np

class CMOSAnnealer(BaseSampler):
    """Sampler with CMOS Annealer.

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.

    To get More details about CMOS Annealer API,
    please access the reference (https://annealing-cloud.com/web-api/reference.html).

    Args:
        token (str):
            API token of COMS Annealer.

        machine_type (str):
            Type of CMOS Annealer: 'ASIC' or 'FPGA'.

        beta_min (float):
            Minimum beta (inverse temperature).

        beta_max (float):
            Maximum beta (inverse temperature).

        step_length (int):
            Length of Monte Carlo step.

        step_num (int):
            Number of Monte Carlo step.

        schedule_info (dict):
            Information about an annealing schedule.

        iteration (int):
            Number of iterations.

        **kwargs:
            Optional keyword arguments for CMOS Annealer.

    Attributes:
        cmos_parameters (dict):
            Parameters of CMOS Annealer.

    Raises:
        ValueError:
            - only valid input: h, J, Q, or king graph.
            - CMOS Annealer API raises error.

    """

    def __init__(self, token, machine_type="ASIC", beta_min=0.1, beta_max=5.0, step_length=10, step_num=100, iteration=1, **kwargs):
        
        self.token = token
        self.machine_type = machine_type
        
        self.cmos_parameters = {"temperature_num_steps": step_num,
                                "temperature_step_length": step_length,
                                "temperature_initial": 1/beta_min,
                                "temperature_target": 1/beta_max,
                                "num_executions": iteration}
        
        self.cmos_parameters.update(kwargs)
        
    def sample_ising(self, h=None, J=None, king_graph=None):
        """Sample from the specified Ising model.

        Args:
            h (dict):
                Linear biases of the Ising model.

            J (dict):
                Quadratic biases of the Ising model.

            **kwargs:
                Optional keyword arguments for the sampling method.

        Returns:
            :obj:: `openjij.sampler.response.Response` object.

        Examples:
            This example submits a two-variable Ising problem.

            >>> import openjij as oj
            >>> sampler = oj.CMOSAnnealer(token="YOUR_TOKEN", iteration=10)
            >>> response = sampler.sample_ising({0: 1}, {(0, 1): -1, (1, 2): -1, (0, 80): 3})
            >>> print(response)
            number of state: 10, minimun energy: -4.0, spin_type: ising
            info:
                averaged_spins: [[2, 0, 1.0], [0, 1, -1.0], [0, 0, 1.0], [1, 0, 1.0]]
                averaged_energy: -4.0
                execution_time: 58769171
                job_id: XXXXXXXXXXXXXXXXX

        """

        var_type = 'SPIN'
        if king_graph is not None:
            _king_graph = KingGraph(machine_type=self.machine_type, king_graph=king_graph, var_type=var_type)
            return self._sampling(_king_graph, var_type=var_type, token=self.token)
        elif (h is not None) and (J is not None):
            _king_graph = KingGraph(machine_type=self.machine_type, h=h, J=J, var_type=var_type)
            return self._sampling(_king_graph, var_type=var_type, token=self.token)
        else:
            raise ValueError('intput "h and J" or king_graph model')

    def sample_qubo(self, Q=None, king_graph=None):
        """Sample from the specified QUBO.

        Args:
            Q (dict):
                Coefficients of a quadratic unconstrained binary optimization (QUBO) model.

            **kwargs:
                Optional keyword arguments for the sampling method.

        Returns:
            :obj:: `openjij.sampler.response.Response` object.

        Examples:
            This example submits a two-variable QUBO model.

            >>> import openjij as oj
            >>> sampler = oj.CMOSAnnealer(token="YOUR_TOKEN", iteration=10)
            >>> response = sampler.sample_qubo({(0, 0): 1, (0, 1): -1, (1, 2): -1, (0, 80): 3})
            >>> print(response)
            number of state: 10, minimun energy: -4.0, spin_type: binary
            info:
                averaged_spins: [[2, 0, 1.0], [0, 1, -1.0], [0, 0, 1.0], [1, 0, 1.0]]
                averaged_energy: -4.0
                execution_time: 58769171
                job_id: XXXXXXXXXXXXXXXXX

        """

        var_type = 'BINARY'
        if king_graph is not None:
            _king_graph = KingGraph(machine_type=self.machine_type, king_graph=king_graph, var_type=var_type)
            return self._sampling(_king_graph, var_type=var_type, token=self.token)
        elif Q is not None:
            _king_graph = KingGraph(machine_type=self.machine_type, Q=Q, var_type=var_type)
            return self._sampling(_king_graph, var_type=var_type, token=self.token)
        else:
            raise ValueError('intput Q or king_graph model')

    
    def _sampling(self, king_graph, var_type, token):
        indices = king_graph.indices
        response = Response(var_type=var_type, indices=indices)
        headers, request = self.make_json_request(king_graph.get_ising_king_graph(), token)
        
        url = 'https://annealing-cloud.com/api/v2/solve'
        res = requests.post(url, data=json.dumps(request), headers=headers)
        res_dict = res.json()
        
        if res_dict['status'] != 0:
            raise ValueError('Error status: {}, message: {}'.format(res_dict['status'], res_dict['message']))
            
        if var_type == "SPIN":
            response.states = [[s for x, y, s in spins] for spins in res_dict['result']['spins']]
        else: #qubo
            response.states = [[int((s+1)/2) for x, y, s in spins] for spins in res_dict['result']['spins']]
        response.indices = [king_graph.convert_to_index(x, y) for x, y, s in res_dict['result']['spins'][0]]
        response.energies = np.array(res_dict['result']['energies']) + king_graph.energy_bias


        # more infomation see : https://annealing-cloud.com/web-api/reference/v2.html
        response.info = {
            "averaged_spins": res_dict['result']["averaged_spins"],
            "averaged_energy": res_dict['result']["averaged_energy"],
            'execution_time' : res_dict['result']['execution_time'] * 10**(-3),
            'job_id': res_dict['job_id'],
        }
        
        return response
    
    def make_json_request(self, model, token):
        """Make request for CMOS Annealer API.
 
        Args:
            model (list):
                A list of 5 integer values representing vertices
                or interactions of Ising model

            token (str):
                API token of COMS Annealer.

        """

        headers = {"Authorization": "Bearer " + token}
        headers.update({ "Accept": "application/json" })
        headers.update({ 'content-type': 'application/json' })

        request = {}
        request["model"] = model # modelのみ必須項目
        request["type"] = 1 if self.machine_type == "ASIC" else 2 # FPGA
        request["num_executions"] = self.cmos_parameters["num_executions"]
        request["parameter"] = {"temperature_num_steps": self.cmos_parameters["temperature_num_steps"],
                                                    "temperature_step_length": self.cmos_parameters['temperature_step_length'],
                                                    "temperature_initial": self.cmos_parameters['temperature_initial'],
                                                    "temperature_target": self.cmos_parameters["temperature_target"]}
        request["outputs"] = {"energies": True,
                                                "spins": True,
                                                "execution_time": True,
                                                "num_outputs": 0,
                                                "averaged_spins": True,
                                                "averaged_energy": True}

        return headers, request
