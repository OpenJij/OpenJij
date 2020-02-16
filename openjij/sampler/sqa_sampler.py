import numpy as np
import openjij
from openjij.sampler import measure_time
from openjij.sampler import BaseSampler
from openjij.utils.decorator import deprecated_alias
import cxxjij


class SQASampler(BaseSampler):
    """Sampler with Simulated Quantum Annealing (SQA).

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.
    Hamiltonian
    $$H(s) = s H_p + (1-s)\sum_i \sigma_i^x$$
    which $H_p$ is problem Hamiltonian that we want solve.

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

    def _convert_validation_schedule(self, schedule, beta):
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        if isinstance(schedule[0], cxxjij.utility.TransverseFieldSchedule):
            return schedule

        # schedule validation  0 <= s <= 1
        sch = np.array(schedule).T[0]
        if not np.all((0 <= sch) & (sch <= 1)):
            raise ValueError("schedule range is '0 <= s <= 1'.")

        if len(schedule[0]) == 2:
            # convert to list of cxxjij.utility.TransverseFieldSchedule
            cxxjij_schedule = []
            for s, one_mc_step in schedule:
                _schedule = cxxjij.utility.TransverseFieldSchedule()
                _schedule.one_mc_step = one_mc_step
                _schedule.updater_parameter.beta = beta
                _schedule.updater_parameter.s = s
                cxxjij_schedule.append(_schedule)
            return cxxjij_schedule
        elif len(schedule[0]) == 3:
            # convert to list of cxxjij.utility.TransverseFieldSchedule
            cxxjij_schedule = []
            for s, _beta, one_mc_step in schedule:
                _schedule = cxxjij.utility.TransverseFieldSchedule()
                _schedule.one_mc_step = one_mc_step
                _schedule.updater_parameter.beta = _beta
                _schedule.updater_parameter.s = s
                cxxjij_schedule.append(_schedule)
            return cxxjij_schedule
        else:
            raise ValueError(
                """schedule is list of tuple or list
                (annealing parameter s : float, step_length : int) or
                (annealing parameter s : float, beta: float, step_length : int)
                """)

    def _dict_to_model(self, var_type, h=None, J=None, Q=None, **kwargs):
        if var_type == openjij.SPIN:
            bqm = openjij.BinaryQuadraticModel(h, J, 0.0, var_type)
        elif var_type == openjij.BINARY:
            bqm = openjij.BinaryQuadraticModel.from_qubo(Q)
        else:
            raise ValueError(
                'var_type should be openjij.SPIN or openjij.BINARY')
        return bqm

    def _get_result(self, system, model):
        state, info = super()._get_result(system, model)

        q_state = system.trotter_spins[:-1].T.astype(np.int)
        c_energies = [model.calc_energy(
            state, need_to_convert_from_spin=True) for state in q_state]
        info['trotter_state'] = q_state
        info['trotter_energies'] = c_energies

        return state, info

    def sample_ising(self, h, J,
                     beta=None, gamma=None,
                     num_sweeps=None, schedule=None,
                     num_reads=1,
                     initial_state=None, updater='single spin flip',
                     reinitialize_state=True, seed=None, **kwargs):
        """Sampling from the Ising model

        Args:
            h (dict): Linear term of the target Ising model. 
            J (dict): Quadratic term of the target Ising model. 
            beta (float, optional): inverse tempareture.
            gamma (float, optional): strangth of transverse field. Defaults to None.
            num_sweeps (int, optional): number of sweeps. Defaults to None.
            schedule (list[list[float, int]], optional): List of annealing parameter. Defaults to None.
            num_reads (int, optional): number of sampling. Defaults to 1.
            initial_state (list[int], optional): Initial state. Defaults to None.
            updater (str, optional): update method. Defaults to 'single spin flip'.
            reinitialize_state (bool, optional): Re-initilization at each sampling. Defaults to True.
            seed (int, optional): Sampling seed. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """

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
            def init_generator(): return [ising_graph.gen_spin()
                                          for _ in range(self.trotter)]
        else:
            trotter_init_state = [np.array(initial_state)
                                  for _ in range(self.trotter)]

            def init_generator(): return trotter_init_state
        # -------------------------------- make init state generator

        # choose updater -------------------------------------------
        sqa_system = cxxjij.system.make_transverse_ising_Eigen(
            init_generator(), ising_graph, self.gamma
        )
        _updater_name = updater.lower().replace('_', '').replace(' ', '')
        if _updater_name == 'singlespinflip':
            algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
        else:
            raise ValueError('updater is one of "single spin flip"')
        # ------------------------------------------- choose updater

        response = self._cxxjij_sampling(
            bqm, init_generator,
            algorithm, sqa_system,
            reinitialize_state, seed, **kwargs
        )

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
