import numpy as np
import openjij
from openjij.sampler import measure_time
from openjij.sampler import BaseSampler
from openjij.utils.decorator import deprecated_alias
import cxxjij


class SQASampler(BaseSampler):
    """Sampler with Simulated Quantum Annealing (SQA).

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.

    Args:
        beta (float):
            Inverse temperature.

        gamma (float):
            Amplitude of quantum fluctuation.

        trotter (int):
            Trotter number.

        step_length (int):
            Length of Monte Carlo step.

        step_num (int):
            Number of Monte Carlo step.

        schedule_info (dict):
            Information about a annealing schedule.

        iteration (int):
            Number of iterations.

    Attributes:
        energy_bias (float):
            Energy bias.

        var_type (str):
            Type of variables: 'SPIN' or 'BINARY' which mean {-1, 1} or {0, 1}.

        indices (int):
            Indices of `openjij.model.model.BinaryQuadraticModel` object.

        N (int):
            Number of the indices.

        system_class (:class:):
            `cxxjij.system.QuantumIsing` class.

        sqa_kwargs (dict):
            Parameters of SQA: beta, gamma, and schedule_info.

    Raises:
        ValueError: If the schedule violates as below.
        - not list or numpy.array.
        - schedule range is '0 <= s <= 1'.

    """
    @deprecated_alias(iteration='num_reads')
    def __init__(self,
                 beta=5.0, gamma=1.0,
                 num_sweeps=1000, schedule=None,
                 trotter=4,
                 num_reads=1):

        self.beta = beta
        self.gamma = gamma
        self.trotter = trotter
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

    def _setting_overwrite(self, **kwargs):
        for key, value in kwargs.items():
            if value:
                self._schedule_setting[key] = value

    def _convert_validation_schedule(self, schedule, beta):
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        if isinstance(schedule[0], cxxjij.utility.TransverseFieldSchedule):
            return schedule
        if len(schedule[0]) != 2:
            raise ValueError(
                """schedule is list of tuple or list
                (annealing parameter s : float, step_length : int)""")

        # schedule validation  0 <= s <= 1
        sch = np.array(schedule).T[0]
        if not np.all((0 <= sch) & (sch <= 1)):
            raise ValueError("schedule range is '0 <= s <= 1'.")

        # convert to list of cxxjij.utility.TransverseFieldSchedule
        cxxjij_schedule = []
        for s, one_mc_step in schedule:
            _schedule = cxxjij.utility.TransverseFieldSchedule()
            _schedule.one_mc_step = one_mc_step
            _schedule.updater_parameter.beta = beta
            _schedule.updater_parameter.s = s
            cxxjij_schedule.append(_schedule)
        return cxxjij_schedule

    def _dict_to_model(self, var_type, h=None, J=None, Q=None, **kwargs):
        if var_type == openjij.SPIN:
            bqm = openjij.BinaryQuadraticModel(h, J, 0.0, var_type)
        elif var_type == openjij.BINARY:
            bqm = openjij.BinaryQuadraticModel.from_qubo(Q)
        else:
            raise ValueError(
                'var_type should be openjij.SPIN or openjij.BINARY')
        return bqm

    def _post_save(self, result_state, system, model, response):
        # trotter_spins is transposed because it is stored as [Spin ​​space][Trotter].
        # [-1] is excluded because it is a tentative spin of s = 1 for convenience in SQA.
        q_state = system.trotter_spins[:-1].T.astype(np.int)
        # calculate classical energy at each trotter slices
        c_energies = [model.calc_energy(
            state, need_to_convert_from_spin=True) for state in q_state]

        response.q_states.append(q_state)
        response.q_energies.append(c_energies)

    def sample(self, bqm,
               beta=None, gamma=None,
               num_sweeps=None, schedule=None,
               num_reads=1,
               initial_state=None, updater='single spin flip',
               reinitialize_state=True, seed=None, **kwargs):

        bqm = openjij.BinaryQuadraticModel(
            linear=bqm.linear, quadratic=bqm.quadratic,
            offset=bqm.offset, var_type=bqm.vartype
        )

        self._setting_overwrite(
            beta=beta, gamma=gamma,
            num_sweeps=num_sweeps, num_reads=num_reads
        )

        ising_graph = bqm.get_cxxjij_ising_graph()

        self._annealing_schedule_setting(
            bqm, beta, gamma, num_sweeps, schedule)

        if initial_state is None:
            def init_generator(): return [ising_graph.gen_spin()
                                          for _ in range(self.trotter)]
        else:
            if bqm.var_type == openjij.SPIN:
                trotter_init_state = [np.array(initial_state)
                                      for _ in range(self.trotter)]
            else:  # BINARY
                trotter_init_state = [
                    (2*np.array(initial_state)-1).astype(int)
                    for _ in range(self.trotter)]

                def init_generator(): return trotter_init_state

        sqa_system = cxxjij.system.make_transverse_ising_Eigen(
            init_generator(), ising_graph, self.gamma
        )

        # choose updater
        _updater_name = updater.lower().replace('_', '').replace(' ', '')
        if _updater_name == 'singlespinflip':
            algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
        else:
            raise ValueError('updater is one of "single spin flip"')

        response = self._sampling(
            bqm, init_generator,
            algorithm, sqa_system, initial_state,
            reinitialize_state, seed, **kwargs
        )

        response.update_trotter_ising_states_energies(
            response.q_states, response.q_energies)

        response.info['schedule'] = self.schedule_info

        return response

    def _annealing_schedule_setting(self, model,
                                    beta=None, gamma=None,
                                    num_sweeps=None,
                                    schedule=None):
        self.beta = beta if beta else self.beta
        self.gamma = gamma if gamma else self.gamma
        if schedule or self.schedule:
            self._schedule = self._convert_validation_schedule(
                schedule if schedule else self.schedule, self.beta
            )
            self.schedule_info = {'schedule': 'custom schedule'}
        else:

            self.num_sweeps = num_sweeps if num_sweeps else self.num_sweeps
            self._schedule, beta_gamma = linear_ising_schedule(
                model=model,
                beta=self._schedule_setting['beta'],
                gamma=self._schedule_setting['gamma'],
                num_sweeps=self._schedule_setting['num_sweeps']
            )
            self.schedule_info = {
                'beta': beta_gamma[0],
                'gamma': beta_gamma[1],
                'num_sweeps': self._schedule_setting['num_sweeps']
            }


def linear_ising_schedule(model, beta, gamma, num_sweeps):
    schedule = cxxjij.utility.make_transverse_field_schedule_list(
        beta=beta, one_mc_step=1, num_call_updater=num_sweeps
    )
    return schedule, [beta, gamma]

    # def sampling(self, model,
    #              initial_state=None, updater='single spin flip',
    #              reinitialize_state=True, seed=None,
    #              **kwargs):

    #     ising_graph = model.get_cxxjij_ising_graph()

    #     if initial_state is None:
    #         def init_generator(): return [ising_graph.gen_spin()
    #                                       for _ in range(self.trotter)]
    #     else:
    #         if model.var_type == openjij.SPIN:
    #             trotter_init_state = [np.array(initial_state)
    #                                   for _ in range(self.trotter)]
    #         else:  # BINARY
    #             trotter_init_state = [
    #                 (2*np.array(initial_state)-1).astype(int)
    #                 for _ in range(self.trotter)]

    #         def init_generator(): return trotter_init_state

    #     sqa_system = cxxjij.system.make_transverse_ising_Eigen(
    #         init_generator(), ising_graph, self.gamma
    #     )

    #     # choose updater
    #     _updater_name = updater.lower().replace('_', '').replace(' ', '')
    #     if _updater_name == 'singlespinflip':
    #         algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
    #     else:
    #         raise ValueError('updater is one of "single spin flip"')

    #     response = self._sampling(
    #         model, init_generator,
    #         algorithm, sqa_system, initial_state,
    #         reinitialize_state, seed, **kwargs
    #     )

    #     response.update_trotter_ising_states_energies(
    #         response.q_states, response.q_energies)

    #     return response
