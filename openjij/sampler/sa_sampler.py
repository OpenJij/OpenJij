# Copyright 2019 Jij Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import random
import sys
import numpy as np
import openjij
import openjij.model
from openjij.sampler import BaseSampler
from openjij.utils.decorator import deprecated_alias
from openjij.utils.graph_utils import qubo_to_ising
import cxxjij

"""
This module contains Simulated Annealing sampler.
"""

class SASampler(BaseSampler):
    """Sampler with Simulated Annealing (SA).

    Args:
        beta_min (float):
            Minmum beta (inverse temperature).
            You can overwrite in methods .sample_*.

        beta_max (float):
            Maximum beta (inverse temperature).
            You can overwrite in methods .sample_*.

        num_reads (int):
            number of sampling (algorithm) runs. defaults None.
            You can overwrite in methods .sample_*.

        num_sweeps (int):
            number of MonteCarlo steps during SA. defaults None.
            You can overwrite in methods .sample_*.

        schedule_info (dict):
            Information about an annealing schedule.

    Raises:
        ValueError: If schedules or variables violate as below.
        - not list or numpy.array.
        - not list of tuple (beta : float, step_length : int).
        - beta is less than zero.

    """

    @property
    def parameters(self):
        return {
            'beta_min': ['parameters'],
            'beta_max': ['parameters'],
        }

    def __init__(self,
                 beta_min=None, beta_max=None,
                 num_sweeps=1000, schedule=None,
                 num_reads=1):

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_sweeps = num_sweeps
        self.schedule = schedule
        self.num_reads = num_reads
        self._schedule_setting = {
            'beta_min': beta_min,
            'beta_max': beta_max,
            'num_sweeps': num_sweeps,
            'num_reads': num_reads,
        }

        self._make_system = {
            'singlespinflip': cxxjij.system.make_classical_ising,
            'singlespinflippolynomial': cxxjij.system.make_classical_ising_polynomial,
            'swendsenwang': cxxjij.system.make_classical_ising
        }
        self._algorithm = {
            'singlespinflip': cxxjij.algorithm.Algorithm_SingleSpinFlip_run,
            'singlespinflippolynomial': cxxjij.algorithm.Algorithm_SingleSpinFlip_run,
            'swendsenwang': cxxjij.algorithm.Algorithm_SwendsenWang_run
        }


    def _convert_validation_schedule(self, schedule):
        """Checks if the schedule is valid and returns cxxjij schedule
        """
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        if isinstance(schedule[0], cxxjij.utility.ClassicalSchedule):
            return schedule

        if len(schedule[0]) != 2:
            raise ValueError(
                "schedule is list of tuple or list (beta : float, step_length : int)")

        # schedule validation  0 <= beta
        beta = np.array(schedule).T[0]
        if not np.all(0 <= beta):
            raise ValueError("schedule beta range is '0 <= beta'.")

        # convert to cxxjij.utility.ClassicalSchedule
        cxxjij_schedule = []
        for beta, step_length in schedule:
            _schedule = cxxjij.utility.ClassicalSchedule()
            _schedule.one_mc_step = step_length
            _schedule.updater_parameter.beta = beta
            cxxjij_schedule.append(_schedule)

        return cxxjij_schedule

    def sample_ising(self, h, J, beta_min=None, beta_max=None,
                     num_sweeps=None, num_reads=1, schedule=None,
                     initial_state=None, updater='single spin flip',
                     sparse=False,
                     reinitialize_state=True, seed=None,
                     ):
        """sample Ising model.

        Args:
            h (dict): linear biases
            J (dict): quadratic biases
            beta_min (float): minimal value of inverse temperature
            beta_max (float): maximum value of inverse temperature
            num_sweeps (int): number of sweeps
            num_reads (int): number of reads
            schedule (list): list of inverse temperature
            initial_state (dict): initial state
            updater(str): updater algorithm
            reinitialize_state (bool): if true reinitialize state for each run
            seed (int): seed for Monte Carlo algorithm
        Returns:
            :class:`openjij.sampler.response.Response`: results
            
        Examples:
            
            for Ising case::

                >>> h = {0: -1, 1: -1, 2: 1, 3: 1}
                >>> J = {(0, 1): -1, (3, 4): -1}
                >>> sampler = oj.SASampler()
                >>> res = sampler.sample_ising(h, J)

            for QUBO case::

                >>> Q = {(0, 0): -1, (1, 1): -1, (2, 2): 1, (3, 3): 1, (4, 4): 1, (0, 1): -1, (3, 4): 1}
                >>> sampler = oj.SASampler()
                >>> res = sampler.sample_qubo(Q)
            
        """

        model = openjij.BinaryQuadraticModel(
            linear=h, quadratic=J, var_type='SPIN'
        )
        return self._sampling(model, beta_min, beta_max,
                              num_sweeps, num_reads, schedule,
                              initial_state, updater,
                              sparse=sparse,
                              reinitialize_state=reinitialize_state, 
                              seed=seed)

    def _sampling(self, model, beta_min=None, beta_max=None,
                     num_sweeps=None, num_reads=1, schedule=None,
                     initial_state=None, updater='single spin flip',
                     sparse=False,
                     reinitialize_state=True, seed=None, structure=None, 
                     ):
        """sampling by using specified model
        Args:
            model (openjij.BinaryQuadraticModel): BinaryQuadraticModel
            beta_min (float): minimal value of inverse temperature
            beta_max (float): maximum value of inverse temperature
            num_sweeps (int): number of sweeps
            num_reads (int): number of reads
            schedule (list): list of inverse temperature
            initial_state (dict): initial state
            updater(str): updater algorithm
            reinitialize_state (bool): if true reinitialize state for each run
            seed (int): seed for Monte Carlo algorithm
            structure (dict): specify the structure. 
            This argument is necessary if the model has a specific structure (e.g. Chimera graph) and the updater algorithm is structure-dependent.
            structure must have two types of keys, namely "size" which shows the total size of spins and "dict" which is the map from model index (elements in model.indices) to the number.
        Returns:
            :class:`openjij.sampler.response.Response`: results
        """
        _updater_name = updater.lower().replace('_', '').replace(' ', '')
        # swendsen wang algorithm runs only on sparse ising graphs.
        if _updater_name == 'swendsenwang' or sparse:
            ising_graph = model.get_cxxjij_ising_graph(sparse=True)
        else:
            ising_graph = model.get_cxxjij_ising_graph()


        self._setting_overwrite(
            beta_min=beta_min, beta_max=beta_max,
            num_sweeps=num_sweeps, num_reads=num_reads
        )


        # set annealing schedule -------------------------------
        if schedule or self.schedule:
            self._schedule = self._convert_validation_schedule(
                schedule if schedule else self.schedule
            )
            self.schedule_info = {'schedule': 'custom schedule'}
        else:
            self._schedule, beta_range = geometric_ising_beta_schedule(
                model=model,
                beta_max=self._schedule_setting['beta_max'],
                beta_min=self._schedule_setting['beta_min'],
                num_sweeps=self._schedule_setting['num_sweeps']
            )
            self.schedule_info = {
                'beta_max': beta_range[0],
                'beta_min': beta_range[1],
                'num_sweeps': self._schedule_setting['num_sweeps']
            }
        # ------------------------------- set annealing schedule

        # make init state generator --------------------------------
        if initial_state is None:
            def _generate_init_state(): return ising_graph.gen_spin(seed) if seed != None else ising_graph.gen_spin()
        else:
            if isinstance(initial_state, dict):
                initial_state = [initial_state[k] for k in model.indices]
            _init_state = np.array(initial_state)

            if structure == None:
                # validate initial_state size
                if len(initial_state) != ising_graph.size():
                    raise ValueError(
                        "the size of the initial state should be {}"
                        .format(ising_graph.size()))
            else:
                # resize _initial_state
                temp_state = [1]*int(structure['size'])
                for k,ind in enumerate(model.indices):
                    temp_state[structure['dict'][ind]] = _init_state[k]
                _init_state = temp_state


            def _generate_init_state(): return np.array(_init_state)
        # -------------------------------- make init state generator

        # choose updater -------------------------------------------
        _updater_name = updater.lower().replace('_', '').replace(' ', '')
        if _updater_name not in self._make_system:
            raise ValueError('updater is one of "single spin flip or swendsen wang"')
        algorithm = self._algorithm[_updater_name]
        sa_system = self._make_system[_updater_name](_generate_init_state(), ising_graph)
        # ------------------------------------------- choose updater
        response = self._cxxjij_sampling(
            model, _generate_init_state,
            algorithm, sa_system,
            reinitialize_state, seed, structure
        )

        response.info['schedule'] = self.schedule_info

        return response


    def sample_hubo(self, J, var_type = openjij.SPIN, 
                    beta_min = None, beta_max = None, schedule = None,
                    num_sweeps = None, num_reads = 1,
                    initial_state = None, reinitialize_state=True, seed = None):

        """sampling from higher order unconstrainted binary optimization.

        Args:
            J (dict): Interactions.
            var_type (str, openjij.VarType): "SPIN" or "BINARY". Defaults to "SPIN".
            beta_min (float, optional): Minimum beta (initial inverse temperature). Defaults to None.
            beta_max (float, optional): Maximum beta (final inverse temperature). Defaults to None.
            schedule (list, optional): schedule list. Defaults to None.
            num_sweeps (int, optional): number of sweeps. Defaults to None.
            num_reads (int, optional): number of reads. Defaults to 1.
            init_state (list, optional): initial state. Defaults to None.
            reinitialize_state (bool): if true reinitialize state for each run
            seed (int, optional): seed for Monte Carlo algorithm. Defaults to None.

        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples::
            for Ising case::
                >>> sampler = oj.SASampler()
                >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
                >>> response = sampler.sample_hubo(J)

            for Binary case::
                >>> sampler = oj.SASampler()
                >>> J = {(0,): -1, (0, 1): -1, (0, 1, 2): 1}
                >>> response = sampler.sample_hubo(J, var_type = "BINARY")
        """
        #if seed is None:
            #random.seed
            #seed = random.randint(0, sys.maxsize)

        if var_type == "SPIN":
            var_type = openjij.SPIN
        elif var_type == "BINARY":
            var_type = openjij.BINARY

        bhom = openjij.BinaryPolynomialModel(interactions = J, var_type = var_type)

        return self._sampling_hubo(bhom, beta_min, beta_max,num_sweeps, num_reads, schedule, initial_state, reinitialize_state, seed)

    def _sampling_hubo(self, model, beta_min=None, beta_max=None,
                     num_sweeps=None, num_reads=1, schedule=None,
                     initial_state=None,
                     reinitialize_state=True, seed=None):

        ising_graph = model.get_cxxjij_ising_graph()

        # make init state generator --------------------------------
        if initial_state is None:
            if model.vartype == openjij.SPIN:
                def _generate_init_state(): return ising_graph.gen_spin(seed)   if seed != None else ising_graph.gen_spin()
            elif model.vartype == openjij.BINARY:
                def _generate_init_state(): return ising_graph.gen_binary(seed) if seed != None else ising_graph.gen_binary()
            else:
                raise ValueError("Unknown var_type detected")
        else:
            if isinstance(initial_state, dict):
                initial_state = [initial_state[k] for k in model.indices]
            _init_state = np.array(initial_state)
            def _generate_init_state(): return _init_state
        # -------------------------------- make init state generator
        
        sa_system = self._make_system['singlespinflippolynomial'](_generate_init_state(), ising_graph)
        
        self._setting_overwrite(
            beta_min=beta_min, beta_max=beta_max,
            num_sweeps=num_sweeps, num_reads=num_reads
        )

        # set annealing schedule -------------------------------
        if schedule or self.schedule:
            self._schedule = self._convert_validation_schedule(
                schedule if schedule else self.schedule
            )
            self.schedule_info = {'schedule': 'custom schedule'}
        else:
            self._schedule, beta_range = geometric_hubo_beta_schedule(
                sa_system=sa_system,
                beta_max=self._schedule_setting['beta_max'],
                beta_min=self._schedule_setting['beta_min'],
                num_sweeps=self._schedule_setting['num_sweeps']
            )
            self.schedule_info = {
                'beta_max': beta_range[0],
                'beta_min': beta_range[1],
                'num_sweeps': self._schedule_setting['num_sweeps']
            }
        # ------------------------------- set annealing schedule

        algorithm = self._algorithm['singlespinflippolynomial']
        response = self._cxxjij_sampling(
            model, _generate_init_state,
            algorithm, sa_system,
            reinitialize_state, seed
        )

        response.info['schedule'] = self.schedule_info

        return response


def geometric_ising_beta_schedule(model: openjij.model.BinaryQuadraticModel,
                                  beta_max=None, beta_min=None,
                                  num_sweeps=1000):
    """make geometric cooling beta schedule

    Args:
        model (openjij.BinaryQuadraticModel)
        beta_max (float, optional): [description]. Defaults to None.
        beta_min (float, optional): [description]. Defaults to None.
        num_sweeps (int, optional): [description]. Defaults to 1000.
    Returns:
        list of cxxjij.utility.ClassicalSchedule, list of beta range [max, min]
    """
    if beta_min is None or beta_max is None:
        # generate Ising matrix
        ising_interaction = model.interaction_matrix()
        if (model.vartype == openjij.BINARY):
            # convert to ising matrix
            qubo_to_ising(ising_interaction)

        ising_interaction = np.abs(ising_interaction)

        #automatical setting of min, max delta energy
        abs_bias = np.sum(ising_interaction, axis=1)

        min_delta_energy = np.min(ising_interaction[ising_interaction > 0])
        max_delta_energy = np.max(abs_bias[abs_bias > 0])

    # TODO: More optimal schedule ?

    beta_min = np.log(2) / max_delta_energy if beta_min is None else beta_min
    beta_max = np.log(100) / min_delta_energy if beta_max is None else beta_max

    num_sweeps_per_beta = max(1, num_sweeps // 1000)

    # set schedule to cxxjij
    schedule = cxxjij.utility.make_classical_schedule_list(
        beta_min=beta_min, beta_max=beta_max,
        one_mc_step=num_sweeps_per_beta,
        num_call_updater=num_sweeps//num_sweeps_per_beta
    )

    return schedule, [beta_max, beta_min]

def geometric_hubo_beta_schedule(sa_system: cxxjij.system.make_classical_ising_polynomial,
                                beta_max=None, beta_min=None,
                                num_sweeps=1000,
                                ):

    max_delta_energy = sa_system.get_max_dE()
    min_delta_energy = sa_system.get_min_dE()

    beta_min = np.log(2)   / max_delta_energy if beta_min is None else beta_min   
    beta_max = np.log(100) / min_delta_energy if beta_max is None else beta_max

    num_sweeps_per_beta = max(1, num_sweeps // 1000)

    schedule = cxxjij.utility.make_classical_schedule_list(
        beta_min=beta_min, beta_max=beta_max,
        one_mc_step=num_sweeps_per_beta,
        num_call_updater=num_sweeps//num_sweeps_per_beta
    )

    return schedule, [beta_max, beta_min]



