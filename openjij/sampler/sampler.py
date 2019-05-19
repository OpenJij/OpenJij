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
import cxxjij as cj
from openjij.model import BinaryQuadraticModel


class Response:
    def __init__(self, var_type, indices):
        self.states = []
        self.energies = []
        self.var_type = var_type

        self.q_states = []
        self.q_energies = []
        self.indices = indices
        self.min_samples = {}
        self.info = {}

    def __repr__(self):
        min_energy_index = np.argmin(self.energies) if len(self.energies) != 0 else None
        ground_energy = self.energies[min_energy_index]
        ground_state = self.states[min_energy_index]
        ret_str = "number of state : {}, minimum energy : {}, var_type : {}\n".format(
            len(self.states), ground_energy, self.var_type)
        ret_str += "indices: {} \nminmum energy state sample : {}".format(self.indices, ground_state)
        return ret_str

    def update_ising_states_energies(self, states, energies):
        if self.var_type == 'SPIN':
            self.states = states
        else:
            self.states = [list(np.array((np.array(state) + 1)/2).astype(np.int)) for state in states]
        self.energies = energies
        self.min_samples = self._minmum_sample()

    def update_quantum_ising_states_energies(self, trotter_states, q_energies):
        if self.var_type == 'SPIN':
            self.q_states = trotter_states
        else:
            self.q_states = [[list(np.array((np.array(state) + 1)/2).astype(np.int)) for state in t_state] for t_state in trotter_states]
        self.q_energies = q_energies
        # save minimum energy of each trotter_state
        min_e_indices = np.argmin(q_energies, axis=1)
        self.energies = [q_e[min_ind] for min_ind, q_e in zip(min_e_indices, q_energies)]
        self.states = [list(t_state[min_ind]) for min_ind, t_state in zip(min_e_indices, self.q_states)]
        self.min_samples = self._minmum_sample()

    def _minmum_sample(self):
        min_energy_ind = np.argmin(self.energies) if len(self.energies) != 0 else None
        min_energy = self.energies[min_energy_ind]
        min_e_indices = np.where(np.array(self.energies) == min_energy)[0]
        min_states = np.array(self.states)[min_e_indices]
        min_states, counts = np.unique(min_states, axis=0, return_counts=True)
        return {'min_states': min_states, 'num_occurrences': counts, 'min_energy': min_energy}

    @property
    def samples(self):
        return [dict(zip(self.indices, state)) for state in self.states]


class BaseSampler:
    def _make_dense_graph(self, h=None, J=None, Q=None, var_type='SPIN'):
        if var_type=='BINARY':
            if Q is None:
                raise ValueError('Input QUBO matrix: Q')
            model = BinaryQuadraticModel(Q=Q, var_type='BINARY')
        elif var_type=='SPIN':
            if h is None or J is None:
                raise ValueError('Input h and J')
            model = BinaryQuadraticModel(h=h, J=J, var_type='SPIN')
        self.indices = model.indices
        self.N = len(model.indices)
        self.energy_bias = model.energy_bias

        self.var_type = model.var_type

        dense_graph = model.convert_to_dense_graph()
        return dense_graph

class SASampler(BaseSampler):
    def __init__(self, beta_min=0.1, beta_max=5.0, step_length=10, step_num=100, schedule=None, iteration=1):

        if schedule:
            self._validation_schedule(schedule)
            self.beta_min = None
            self.beta_max = None
            self.step_length = None
            self.step_num = None
            self.schedule_info = {'schedule': schedule}
        else:
            self.beta_min = beta_min
            self.beta_max = beta_max
            self.step_length = step_length
            self.step_num = step_num
            self.schedule_info = {
                'beta_min': beta_min, 'beta_max': beta_max,
                'step_length': step_length, 'step_num': step_num
                }
        self.iteration = iteration

    def _validation_schedule(self, schedule):
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        if not isinstance(schedule[0], tuple):
            raise ValueError("schedule is list of tuple (beta : float, step_length : int)")

         # schedule validation  0 <= beta
        beta = np.array(schedule).T[0]
        if not np.all(0 <= beta):
            raise ValueError("schedule beta range is '0 <= beta'.")


    def sample_ising(self, h, J):
        var_type = 'SPIN'
        ising_dense_graph = self._make_dense_graph(h=h, J=J, var_type=var_type)
        return self._sampling(ising_dense_graph, var_type=var_type)

    def sample_qubo(self, Q):
        var_type = 'BINARY'
        ising_dense_graph = self._make_dense_graph(Q=Q, var_type=var_type)
        return self._sampling(ising_dense_graph, var_type=var_type)

    def _sampling(self, ising_dense_graph, var_type):
        sa_system = cj.system.ClassicalIsing(ising_dense_graph)
        states = []
        energies = []
        for _ in range(self.iteration):
            sa_system.initialize_spins()
            sa_system.simulated_annealing(**self.schedule_info)
            state = sa_system.get_spins()
            states.append(state)
            energies.append(ising_dense_graph.calc_energy(state) + self.energy_bias)
        response = Response(var_type=var_type, indices=self.indices)
        response.update_ising_states_energies(states, energies)
        return response

class SQASampler(BaseSampler):
    def __init__(self, beta=5.0, gamma=1.0,
                 trotter=4, step_length=10, step_num=100, schedule=None, iteration=1):

        # make schedule
        if schedule is not None:
            self._validate_schedule(schedule)
            self.step_length = None
            self.step_num = None
            self.schedule_info = {'schedule': schedule}
        else:
            self.step_length = step_length
            self.step_num = step_num
            self.schedule_info = {'step_num': step_num, 'step_length': step_length}

        self.beta = beta
        self.gamma = gamma
        self.trotter = trotter

        self.iteration = iteration
        self.energy_bias = 0.0
        self.var_type = 'SPIN'

    def _validate_schedule(self, schedule):
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        # schedule validation  0 <= s < 1
        sch = np.array(schedule).T[0]
        if not np.all((0 <= sch) & (sch < 1)):
            raise ValueError("schedule range is '0 <= s < 1'.")

    def sample_ising(self, h, J):
        var_type = 'SPIN'
        ising_dense_graph = self._make_dense_graph(h=h, J=J, var_type=var_type)
        return self._sampling(ising_dense_graph, var_type=var_type)

    def sample_qubo(self, Q):
        var_type = 'BINARY'
        ising_dense_graph = self._make_dense_graph(Q=Q, var_type=var_type)
        return self._sampling(ising_dense_graph, var_type=var_type)

    def _sampling(self, ising_dense_graph, var_type):
        system = cj.system.QuantumIsing(ising_dense_graph, num_trotter_slices=self.trotter)
        q_states = []
        q_energies = []
        for _ in range(self.iteration):
            system.initialize_spins()
            system.simulated_quantum_annealing(
                beta = self.beta, gamma=self.gamma,
                **self.schedule_info
            )
            q_state = system.get_spins()
            q_states.append(q_state)
            q_energies.append([ising_dense_graph.calc_energy(state) + self.energy_bias for state in q_state])
        
        response = Response(var_type=var_type, indices=self.indices)
        response.update_quantum_ising_states_energies(q_states, q_energies)
        return response



