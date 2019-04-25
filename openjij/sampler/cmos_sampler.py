import requests
import json

from openjij.sampler import BaseSampler, Response
from openjij.model import KingGraph

import numpy as np

class CMOSAnnealer(BaseSampler):
    def __init__(self, token, machine_type="ASIC", beta_min=0.1, beta_max=5.0, step_length=10, step_num=100, iteration=1, **args):
        
        self.token = token
        self.machine_type = machine_type
        
        self.cmos_parameters = {"temperature_num_steps": step_num,
                                "temperature_step_length": step_length,
                                "temperature_initial": 1/beta_min,
                                "temperature_target": 1/beta_max,
                                "num_executions": iteration}
        
        self.cmos_parameters.update(args)
        
    def sample_ising(self, h=None, J=None, king_graph=None):
        spin_type = 'ising'
        if king_graph is not None:
            _king_graph = KingGraph(machine_type=self.machine_type, king_graph=king_graph, spin_type=spin_type)
            return self._sampling(_king_graph, spin_type=spin_type, token=self.token)
        elif (h is not None) and (J is not None):
            _king_graph = KingGraph(machine_type=self.machine_type, h=h, J=J, spin_type=spin_type)
            return self._sampling(_king_graph, spin_type=spin_type, token=self.token)
        else:
            raise ValueError('should set "h and J" or king_graph model')
    
    def _sampling(self, king_graph, spin_type, token):
        indices = king_graph.indices
        response = Response(spin_type=spin_type, indices=indices)
        headers, request = self.make_json_request(king_graph.get_ising_king_graph(), token)
        
        url = 'https://annealing-cloud.com/api/v2/solve'
        res = requests.post(url, data=json.dumps(request), headers=headers)
        res_dict = res.json()
        
        if res_dict['status'] != 0:
            raise ValueError('Error status: {}, message: {}'.format(res_dict['status'], res_dict['message']))
        
        response.states = [[s for x, y, s in spins] for spins in res_dict['result']['spins']]
        response.indices = [king_graph.convert_to_index(x, y) for x, y, s in res_dict['result']['spins'][0]]
        response.energies = np.array(res_dict['result']['energies']) + king_graph.energy_bias

        if spin_type == "qubo":
            response.state_reformat_to_qubo()
        
        response.info = {
            "averaged_spins": res_dict['result']["averaged_spins"],
            "averaged_energy": res_dict['result']["averaged_energy"],
            'execution_time' : res_dict['result']['execution_time'],
            'job_id': res_dict['job_id'],
        }
        
        return response
    
    def make_json_request(self, model, token):
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