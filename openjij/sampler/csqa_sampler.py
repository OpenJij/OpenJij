import numpy as np
import openjij
from openjij.sampler import measure_time
from openjij.sampler import SQASampler
from openjij.utils.decorator import deprecated_alias
import cxxjij


class CSQASampler(SQASampler):
    """Sampler with continuous-time simulated quantum annealing (CSQA) using Hamiltonian

    .. math:: 

        H(s) = s H_p + \\Gamma (1-s)\\sum_i \\sigma_i^x

    where :math:`H_p` is the problem Hamiltonian we want to solve.

    Args:
        beta (float): Inverse temperature.
        gamma (float): Amplitude of quantum fluctuation.
        schedule (list): schedule list
        step_num (int): Number of Monte Carlo step.
        schedule_info (dict): Information about a annealing schedule.
        num_reads (int): Number of iterations.
        num_sweeps (int): number of sweeps
        num_reads (int): Number of iterations.
        schedule_info (dict): Information about a annealing schedule.

    """
    def __init__(self,
                 beta=5.0, gamma=1.0,
                 num_sweeps=1000, schedule=None,
                 num_reads=1):

        self.beta = beta
        self.gamma = gamma
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.schedule = schedule
        self.energy_bias = 0.0
        self._schedule_setting = {
            'beta': beta,
            'gamma': gamma,
            'num_sweeps': num_sweeps,
            'num_reads': num_reads,
        }

    def _get_result(self, system, model):
        info = {}
        info['spin_config'] = system.spin_config

        state = cxxjij.result.get_solution(system)

        return state, info

    def sample_ising(self, h, J,
                     beta=None, gamma=None,
                     num_sweeps=None, schedule=None,
                     num_reads=1,
                     initial_state=None, updater='swendsenwang',
                     reinitialize_state=True, seed=None):
        """Sampling from the Ising model.

        Args:
            h (dict): linear biases
            J (dict): quadratic biases
            beta (float, optional): inverse temperature
            gamma (float, optional): strength of transverse field
            num_sweeps (int, optional): number of sampling.
            schedule (list, optional): schedule list
            num_reads (int, optional): number of iterations
            initial_state (optional): initial state of spins
            updater (str, optional): updater algorithm
            reinitialize_state (bool, optional): Re-initilization at each sampling. Defaults to True.
            seed (int, optional): Sampling seed.
            structure (int, optional): specify the structure. 
            This argument is necessary if the model has a specific structure (e.g. Chimera graph) and the updater algorithm is structure-dependent.
            structure must have two types of keys, namely "size" which shows the total size of spins and "dict" which is the map from model index (elements in model.indices) to the number.
        
        Returns:
            :class:`openjij.sampler.response.Response`: results

        Examples:
            
            for Ising case::

                >>> h = {0: -1, 1: -1, 2: 1, 3: 1}
                >>> J = {(0, 1): -1, (3, 4): -1}
                >>> sampler = oj.CSQASampler()
                >>> res = sampler.sample_ising(h, J)

            for QUBO case::

                >>> Q = {(0, 0): -1, (1, 1): -1, (2, 2): 1, (3, 3): 1, (4, 4): 1, (0, 1): -1, (3, 4): 1}
                >>> sampler = oj.CSQASampler()
                >>> res = sampler.sample_qubo(Q)

        """

        bqm = openjij.BinaryQuadraticModel(
            linear=h, quadratic=J, var_type='SPIN'
        )

        #Continuous time ising system only supports sparse ising graph

        ising_graph = bqm.get_cxxjij_ising_graph(sparse=True)

        self._setting_overwrite(
            beta=beta, gamma=gamma,
            num_sweeps=num_sweeps, num_reads=num_reads
        )
        self._annealing_schedule_setting(
            bqm, beta, gamma, num_sweeps, schedule)

        # make init state generator --------------------------------
        if initial_state is None:
            def init_generator():
                spin_config = np.random.choice([1,-1], len(bqm.indices))
                return list(spin_config)
        else:
            def init_generator(): return initial_state
        # -------------------------------- make init state generator

        # choose updater -------------------------------------------
        sqa_system = cxxjij.system.make_continuous_time_ising(
            init_generator(), ising_graph, self.gamma
        )
        _updater_name = updater.lower().replace('_', '').replace(' ', '')
        if _updater_name == 'swendsenwang':
            algorithm = cxxjij.algorithm.Algorithm_ContinuousTimeSwendsenWang_run
        else:
            raise ValueError('updater is one of "swendsen wang"')
        # ------------------------------------------- choose updater

        response = self._cxxjij_sampling(
            bqm, init_generator,
            algorithm, sqa_system,
            reinitialize_state, seed
        )

        response.info['schedule'] = self.schedule_info

        return response
