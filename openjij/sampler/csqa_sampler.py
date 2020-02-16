import numpy as np
import openjij
from openjij.sampler import measure_time
from openjij.sampler import SQASampler
from openjij.utils.decorator import deprecated_alias
import cxxjij


class CSQASampler(SQASampler):
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
                     reinitialize_state=True, seed=None, **kwargs):

        bqm = openjij.BinaryQuadraticModel(
            linear=h, quadratic=J, var_type='SPIN'
        )

        ising_graph = bqm.get_cxxjij_ising_graph()

        self._setting_overwrite(
            beta=beta, gamma=gamma,
            num_sweeps=num_sweeps, num_reads=num_reads
        )
        self._annealing_schedule_setting(
            bqm, beta, gamma, num_sweeps, schedule)

        # make init state generator --------------------------------
        if initial_state is None:
            def init_generator():
                n = len(bqm.indices)
                init_num_cut = 10
                c_spins = ising_graph.gen_spin()
                _cut = np.random.uniform(0, 1, (n, init_num_cut))
                spin_config = [[
                    (t, s**(_ti+1))
                    for _ti, t in enumerate(np.sort(_cut[i]))]
                    for i, s in enumerate(c_spins)]
                return spin_config
        else:
            def init_generator(): return initial_state
        # -------------------------------- make init state generator

        # choose updater -------------------------------------------
        sqa_system = cxxjij.system.ContinuousTimeIsing_Dense(
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
            reinitialize_state, seed, **kwargs
        )

        response.info['schedule'] = self.schedule_info

        return response
