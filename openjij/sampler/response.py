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
import openjij


class Response:
    """A class of response from samplers.

    Args:
        var_type (str):
            Type of variables: 'SPIN' or 'BINARY' which mean {-1, 1} or {0, 1}.

        indices (int):
            Indices of `openjij.sampler.response.Response` object.

    Attributes:
        states (list):
            States of the system.

        energies (list):
            Energies for the states.

        q_states (list):
            Quantum states of the system.

        q_energies (list):
            Quantum energies for the quantum states.

        min_samples (list):
            Samples with minimum energy.

        info (dict):
            Other information.

    """

    def __init__(self, var_type, indices):
        self.states = []
        self.energies = []
        self.var_type = openjij.cast_var_type(var_type)

        self.q_states = []
        self.q_energies = []
        self.indices = indices
        self.min_samples = {}
        self.info = {}

    def __repr__(self):
        if len(self.states) == 0:
            return "Response\n\tvar_type: {}\n\tstates: empty".format(self.var_type)

        if len(self.min_samples) == 0:
            self.min_samples = self._minimum_sample()
        min_energy_index = np.argmin(self.energies) if len(
            self.energies) != 0 else None
        ground_energy = self.energies[min_energy_index]
        ground_state = self.states[min_energy_index]
        res_str = "Response\n\titeration : {},\n\t".format(len(self.states))
        res_str += "minimum energy: {}\n\t".format(ground_energy)
        res_str += "var_type: {}\n\t".format(self.var_type)
        res_str += "indices: {} \n\tminimum energy state : {}".format(
            self.indices, ground_state)
        return res_str

    def update_ising_states_energies(self, states, energies):
        """Update states and energies.

        Args:
            states (list):
                Updated states.

            energies (list):
                Updated energies.

        Attributes:
            min_samples (dict):
                Minimun energies, states, and number of occurrences.

        """

        if self.var_type == openjij.SPIN:
            self.states = states
        else:
            self.states = [
                list(np.array((np.array(state) + 1)/2).astype(np.int)) for state in states]
        self.energies = energies
        self.min_samples = self._minimum_sample()

    def update_trotter_ising_states_energies(self, trotter_states, q_energies):
        """Update quantum states and energies.

        Args:
            trotter_states (list):
                Updated trotter states.

            q_energies (list):
                Updated quantum energies.

        Attributes:
            min_samples (dict):
                Minimun energies, states, and number of occurrences.

        """

        if self.var_type == openjij.SPIN:
            self.q_states = list(np.array(trotter_states).astype(np.int))
        else:
            self.q_states = [[list(np.array((np.array(state) + 1)/2).astype(np.int))
                              for state in t_state] for t_state in trotter_states]
        self.q_energies = q_energies
        # save minimum energy of each trotter_state
        min_e_indices = np.argmin(q_energies, axis=1)
        self.energies = [q_e[min_ind]
                         for min_ind, q_e in zip(min_e_indices, q_energies)]
        self.states = [list(t_state[min_ind]) for min_ind,
                       t_state in zip(min_e_indices, self.q_states)]
        self.min_samples = self._minimum_sample()
    def _minimum_sample(self):
        min_energy_ind = np.argmin(self.energies) if len(
            self.energies) != 0 else None
        min_energy = self.energies[min_energy_ind]
        min_e_indices = np.where(np.array(self.energies) == min_energy)[0]
        min_states = np.array(self.states)[min_e_indices]
        min_states, counts = np.unique(min_states, axis=0, return_counts=True)
        return {'states': min_states, 'num_occurrences': counts, 'min_energy': min_energy}

    @property
    def samples(self):
        """Returns samples as list.

        Returns:
            list: all the samples.

        """

        return [dict(zip(self.indices, state)) for state in self.states]
