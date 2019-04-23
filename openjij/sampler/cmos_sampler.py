import requests
import json

from openjij.sampler import BaseSampler, Response
from openjij.model import BinaryQuadraticModel

class KingGraph(BinaryQuadraticModel):
    def __init__(self, machine_type, h=None, J=None, Q=None, king_graph=None, spin_type='ising'):
        
        self.machine_type = machine_type
        if self.machine_type == "ASIC":
            self.xrange = [0, 351+1]
            self.yrange = [0, 175+1]
            self.prange = [-3, 3]
        elif self.machine_type == "FPGA":
            self.xrange = [0, 79+1]
            self.yrange = [0, 79+1]
            self.prange = [-127, 127]
        else:
            raise ValueError('machine type should be ASIC or FPGA')
        
        self.king_graph = king_graph
        if king_graph is not None:
            _indices = set([])
            for xi, yi, xj, yj, p in king_graph:
                _indices.add((xi, yi))
                _indices.add((xj, yj))
            self.indices = list(_indices)
        if Q is not None:
            super().__init__(Q=Q, spin_type=spin_type)
        elif (h is not None) and (J is not None):
            super().__init__(h=h, J=J, spin_type=spin_type)
            
    def king_indices(self):
        if isinstance(self.indices[0], tuple):
            return self.indices
        else:
            return [self.convert_to_xy(i) for i in self.indices]
            
    def convert_to_xy(self, index):
            y = int(index / self.xrange[1])
            return index - y * self.xrange[1], y
        
    def convert_to_index(self, x, y):
        return y * self.xrange[1] + x
        
        
    def validation_king_graph(self):
        
        def validate_xy(xi, yi, xj, yj, p, error_index):
            if yi >= self.yrange[1] or yj >= self.yrange[1]:
                raise ValueError('Graph is incomplete {}'.format(error_index))
            if not (xi in [xj, xj-1, xj+1]) or not (yi in [yj, yj-1, yj+1]):
                raise ValueError('Graph is incomplete {}'.format(error_index) )
            if not (self.prange[0] <= p <= self.prange[1]):
                raise ValueError('Graph is incomplete {} p {}'.format(error_index, p) )
        
        if self.king_graph is None:
            self.king_graph = []
            for (i, j), p in self.J.items():
                xi, yi = self.convert_to_xy(i)
                xj, yj = self.convert_to_xy(j)
                
                validate_xy(xi, yi, xj, yj, p, "i {} j {}".format(i, j))
                self.king_graph.append([xi, yi, xj, yj, p])

            for i, h in self.h.items():
                xi, yi = self.convert_to_xy(i)
                validate_xy(xi, yi, xi, yi, p, "i {} j {}".format(i, j))
                self.king_graph.append([xi, yi, xi, yi, h])
        else:
            for xi, yi, xj, yj, p in self.king_graph:
                validate_xy(xi, yi, xi, yi, p, "xi {} yi {}".format(xi, yi))
            
            
    def convert_ising(self):
        interactions = {(x + y*self.xrange[1], xn + yn*self.xrange[1]): p for x,y, xn, yn, p in self.king_graph}
        return interactions

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
            _king_graph.validation_king_graph()
            return self._sampling(_king_graph, spin_type=spin_type, token=self.token)
        elif (h is not None) and (J is not None):
            _king_graph = KingGraph(machine_type=self.machine_type, h=h, J=J, spin_type=spin_type)
            _king_graph.validation_king_graph()
            return self._sampling(_king_graph, spin_type=spin_type, token=self.token)
        else:
            raise ValueError('should set "h and J" or king_graph model')
    
    def _sampling(self, king_graph, spin_type, token):
        indices = king_graph.indices
        response = Response(spin_type=spin_type, indices=indices)
        headers, request = self.make_json_request(king_graph.king_graph, token)
        
        url = 'https://annealing-cloud.com/api/v2/solve'
        res = requests.post(url, data=json.dumps(request), headers=headers)
        res_dict = res.json()
        
        if res_dict['status'] != 0:
            raise ValueError('Error status: {}, message: {}'.format(res_dict['status'], res_dict['message']))
        
#         _states = [{(x, y):s for x, y, s in spin} for spin in res_dict['result']['spins']]
#         _states = [{index: state[x, y] for index, (x, y) in zip(king_graph.indices, king_graph.king_indices())} for state in _states]
        response.states = [[s for x, y, s in spins] for spins in res_dict['result']['spins']]
        response.indices = [king_graph.convert_to_index(x, y) for x, y, s in res_dict['result']['spins'][0]]
        response.energies = res_dict['result']['energies']
        
        response.info = {
            "averaged_spins": res_dict['result']["averaged_spins"],
            "averaged_energy": res_dict['result']["averaged_energy"],
            'execution_time' : res_dict['result']['execution_time'],
            'job_id': res_dict['job_id'],
        }
        
        return response
    
    def make_json_request(self, model, token):
        """ ヘッダとJSONリクエストデータを作成 """
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