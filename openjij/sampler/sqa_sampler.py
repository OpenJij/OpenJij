import numpy as np
import openjij
from openjij.sampler import measure_time
from openjij.sampler import BaseSampler
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

    def __init__(self, beta=5.0, gamma=1.0,
                 trotter=4, step_length=10,
                 step_num=100, schedule=None, iteration=1):

        # make schedule
        if schedule is not None:
            self.schedule = self._convert_validate_schedule(schedule, beta)
            self.step_length = None
            self.step_num = None
            self.schedule_info = {'schedule': schedule}
        else:
            self.step_length = step_length
            self.step_num = step_num
            self.schedule_info = {
                'step_num': step_num, 'step_length': step_length}
            self.schedule = cxxjij.utility.make_transverse_field_schedule_list(
                beta=beta, one_mc_step=step_length,
                num_call_updater=step_num
            )

        self.beta = beta
        self.gamma = gamma
        self.trotter = trotter

        self.iteration = iteration
        self.energy_bias = 0.0

        self.sqa_kwargs = dict(
            beta=self.beta, gamma=self.gamma, **self.schedule_info)

    def _convert_validate_schedule(self, schedule, beta):
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

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

    def sample_ising(self, h, J,
                     initial_state=None, updater='single spin flip',
                     reinitilize_state=True, seed=None, **kwargs):
        """Sample from the specified Ising model.

        Args:
            h (dict):
                Linear biases of the Ising model.

            J (dict):
                Quadratic biases of the Ising model.

            **kwargs:
                Optional keyword arguments for the sampling method.

        Returns:
            :obj:: `openjij.sampler.response.Response` object.

        Examples:
            This example submits a two-variable Ising problem.

            >>> import openjij as oj
            >>> sampler = oj.SQASampler()
            >>> response = sampler.sample_ising({0: -1, 1: 1}, {})
            >>> for sample in response.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 1, 1: -1}

        """

        var_type = openjij.SPIN
        bqm = openjij.BinaryQuadraticModel(h=h, J=J, var_type=var_type)
        return self.sampling(bqm,
                             initial_state=initial_state, updater=updater,
                             reinitilize_state=reinitilize_state,
                             seed=seed,
                             **kwargs
                             )

    def sample_qubo(self, Q,
                    initial_state=None, updater='single spin flip',
                    reinitilize_state=True, seed=None, **kwargs):
        """Sample from the specified QUBO.

        Args:
            Q (dict):
                Coefficients of a quadratic unconstrained binary optimization (QUBO) model.

            **kwargs:
                Optional keyword arguments for the sampling method.

        Returns:
            :obj:: `openjij.sampler.response.Response` object.

        Examples:
            This example submits a two-variable QUBO model.

            >>> import openjij as oj
            >>> sampler = oj.SQASampler()
            >>> Q = {(0, 0): -1, (4, 4): -1, (0, 4): 2}
            >>> response = sampler.sample_qubo(Q)
            >>> for sample in response.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 0, 4: 1}

        """

        var_type = openjij.BINARY
        bqm = openjij.BinaryQuadraticModel(Q=Q, var_type=var_type)
        return self.sampling(bqm,
                             initial_state=initial_state, updater=updater,
                             reinitilize_state=reinitilize_state,
                             seed=seed,
                             **kwargs
                             )

    def _post_save(self, result_state, system, model, response):
        # trotter_spins is transposed because it is stored as [Spin ​​space][Trotter].
        # [-1] is excluded because it is a tentative spin of s = 1 for convenience in SQA.
        q_state = system.trotter_spins[:-1].T.astype(np.int)
        # calculate classical energy at each trotter slices
        c_energies = [model.calc_energy(
            state, need_to_convert_from_spin=True) for state in q_state]

        response.q_states.append(q_state)
        response.q_energies.append(c_energies)

    def sampling(self, model,
                 initial_state=None, updater='single spin flip',
                 reinitialize_state=True, seed=None,
                 **kwargs):

        ising_graph = model.get_cxxjij_ising_graph()

        if initial_state is None:
            def init_generator(): return [ising_graph.gen_spin()
                                          for _ in range(self.trotter)]
        else:
            if model.var_type == openjij.SPIN:
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
            model, init_generator,
            algorithm, sqa_system, initial_state,
            reinitialize_state, seed, **kwargs
        )

        response.update_trotter_ising_states_energies(
            response.q_states, response.q_energies)

        return response
