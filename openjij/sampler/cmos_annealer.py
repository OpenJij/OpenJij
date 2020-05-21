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

from openjij.sampler import BaseSampler
import openjij

import numpy as np
import dimod


class CMOSAnnealer(BaseSampler, dimod.Structured):
    """Sampler with CMOS Annealer.

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.

    To get More details about CMOS Annealer API,
    please access the reference (https://annealing-cloud.com/web-api/reference.html).

    Args:
        token (str): API token of COMS Annealer.
        machine_type (str): Type of CMOS Annealer: 'ASIC' or 'FPGA'.
        beta_min (float): Minimum beta (inverse temperature).
        beta_max (float): Maximum beta (inverse temperature).
        step_length (int): Length of Monte Carlo step.
        num_sweeps (int): Number of Monte Carlo step.
        num_reads (int): Number of iterations.
        **kwargs: Optional keyword arguments for CMOS Annealer.

    Attributes:
        cmos_parameters (dict): Parameters of CMOS Annealer.

    Raises:
        ValueError:
            - only valid input: h, J, Q, or king graph.
            - CMOS Annealer API raises error.

    """

    def __init__(self, token, machine_type="ASIC",
                 beta_min=None, beta_max=None,
                 num_sweeps=10, step_length=100, num_reads=1, **kwargs):

        self.token = token
        self.machine_type = machine_type
        # setting machine type and 2D Grid size
        # more information see: https://annealing-cloud.com/web-api/reference/v2.html
        if machine_type == 'ASIC':
            self._X, self._Y = 352, 176
            self.max_abs = 3
        elif machine_type == 'FPGA':
            self._X, self._Y = 80, 80
            self.max_abs = 127
        else:
            raise ValueError("machine_type is 'ASIC' or 'FPGA'.")

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_sweeps = num_sweeps
        self.num_reads = num_reads
        self._schedule_setting = {
            'beta_min': beta_min,
            'beta_max': beta_max,
            'num_sweeps': num_sweeps,
            'num_reads': num_reads,
            'step_length': step_length
        }

    @property
    def parameters(self):
        return {'num_reads': ['1<=n<=10', lambda n: 1 <= n <= 10],
                'num_sweeps': ['1<=n<=100', lambda n: 1 <= n <= 100],
                'step_length': ['1<=n<=1000', lambda n: 1 <= n <= 1000],
                'beta_min': ['1/2000<=beta', lambda beta: 1/2000 <= beta],
                'beta_max': ['1/2000<=beta', lambda beta: 1/2000 <= beta]}

    @property
    def properties(self):
        return {'System size': "self._X, self._Y are size of 2D grid King's graph."}

    @property
    def nodelist(self):
        """2-D lattice nodes

        Returns:
            [list of int]: 2-D lattice nodes
        """
        return list(range(self._X*self._Y))

    @property
    def edgelist(self):
        """King's graph

        Returns:
            [list of tuple]: King's graph
        """
        edges = []
        for _x in range(self._X-1):
            for _y in range(self._Y-1):
                i = _x + self._X * _y
                edges += [
                    (i, _x + 1 + self._X * _y),
                    (i, _x + self._X * (_y + 1)),
                    (i, _x + 1 + self._X * (_y + 1)),
                ]
        return edges

    def _validate_schedule(self):
        for k, v in self._schedule_setting.items():
            if not self.parameters[k][1](v):
                raise ValueError(
                    'The value of {} is invalid.'
                    ' "{}: {}".'.format(k, k, self.parameters[k][0]))

    def sample_ising(self, h, J, num_reads=1,
                     num_sweeps=None, step_length=None,
                     beta_min=None, beta_max=None,):
        """sample Ising model.

        Args:
            h (dict): linear biases
            J (dict): quadratic biases
            num_reads (int): number of reads
            num_sweeps (int): number of sweeps
            step_length (int): number of Monte Carlo steop for each step
            beta_min (float): minimal value of inverse temperature
            beta_max (float): maximum value of inverse temperature
        """

        # settting annealing parameters and validation -------------
        self._setting_overwrite(
            beta_min=beta_min, beta_max=beta_max,
            num_sweeps=num_sweeps, num_reads=num_reads,
            step_length=step_length
        )
        # ------------- settting annealing parameters and validation

        # Convert to CMOS Annealing Cloud Web API format [xi,yi,xj,yj,pij] ---
        # from the Ising interaction dictionary.
        cmos_model = _cmos_annealer_format(
            h, J, self._X, self._Y, self.max_abs)
        beta_max, beta_min = defalut_beta_range(
            cmos_model,
            beta_max=self._schedule_setting['beta_max'],
            beta_min=self._schedule_setting['beta_min']
        )
        self._schedule_setting['beta_max'] = beta_max
        self._schedule_setting['beta_min'] = beta_min
        self._validate_schedule()
        headers, request = self.make_json_request(
            cmos_model, self.token
        )
        # --------------------------------------------------------------------

        # POST to API -------------------------------------
        url = 'https://annealing-cloud.com/api/v2/solve'
        res = requests.post(url, data=json.dumps(
            request, cls=NumpyEncoder), headers=headers)
        res_dict = res.json()
        # ------------------------------------- POST to API

        # API Error handling
        if res_dict['status'] != 0:
            raise ValueError('Error status: {}, message: {}'.format(
                res_dict['status'], res_dict['message']))

        # more infomation see : https://annealing-cloud.com/web-api/reference/v2.html
        info = {
            "schedule": {
                'beta_max': self._schedule_setting['beta_max'],
                'beta_min': self._schedule_setting['beta_min']
            },
            "averaged_spins": res_dict['result']["averaged_spins"],
            "averaged_energy": res_dict['result']["averaged_energy"],
            'execution_time': res_dict['result']['execution_time'] * 10**(-3),
            'job_id': res_dict['job_id'],
            'posted_model': cmos_model
        }

        bqm = openjij.BinaryQuadraticModel(h, J)
        samples = [{x+self._X*y: s for x, y, s in spins}
                   for spins in res_dict['result']['spins']]
        response = openjij.Response.from_samples_bqm(
            samples, bqm, info=info
        )
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
        headers.update({"Accept": "application/json"})
        headers.update({'content-type': 'application/json'})

        request = {}
        request["model"] = model  # modelのみ必須項目
        request["type"] = 1 if self.machine_type == "ASIC" else 2  # FPGA
        request["num_executions"] = self.num_reads
        request["parameter"] = {
            "temperature_num_steps": self._schedule_setting['num_sweeps'],
            "temperature_step_length": self._schedule_setting['step_length'],
            "temperature_initial": 1.0/self._schedule_setting['beta_min'],
            "temperature_target": 1.0/self._schedule_setting['beta_max']}
        request["outputs"] = {"energies": True,
                              "spins": True,
                              "execution_time": True,
                              "num_outputs": 0,
                              "averaged_spins": True,
                              "averaged_energy": True}

        return headers, request


def _cmos_annealer_format(h, J, X, Y, max_abs):

    interactions = []

    def _2d_pos(i, j, pij):
        xi, xj = i % X, j % X
        yi, yj = int((i-xi)/X), int((j-xj)/X)
        interactions.append(pij)
        return [xi, yi, xj, yj, pij]

    annealer_format = [_2d_pos(i, j, Jij) for (i, j), Jij in J.items()]
    annealer_format += [_2d_pos(i, i, hi) for i, hi in h.items()]
    # degitalize to integer
    max_int = np.max(np.abs(interactions))
    _interactions = np.array(annealer_format).T
    _interactions[-1] = _interactions[-1] * max_abs/max_int
    _interactions = [list(q) for q in _interactions.T.astype(int)]
    return _interactions


def defalut_beta_range(cmos_model, beta_max, beta_min):

    if beta_min is None or beta_max is None:
        interactions = {}

        def add_interactions(i, v):
            if i in interactions:
                interactions[i].append(np.abs(v))
            else:
                interactions[i] = [np.abs(v)]

        for xi, yi, xj, yj, Jij in cmos_model:
            add_interactions((xi, yi), Jij)
            add_interactions((xj, yj), Jij)

        abs_bias = np.array([np.sum(v) for v in interactions.values()])
        ising_interaction = np.abs(np.array(cmos_model).T[-1])
        min_delta_energy = np.min(ising_interaction[ising_interaction > 0])
        max_delta_energy = np.max(abs_bias[abs_bias > 0])

    beta_min = np.log(
        2) / max_delta_energy if beta_min is None else beta_min
    beta_max = np.log(
        100) / min_delta_energy if beta_max is None else beta_max

    beta_min = max(beta_min, 1/2000)
    beta_max = max(beta_max, 1/2000)

    return beta_max, beta_min


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
