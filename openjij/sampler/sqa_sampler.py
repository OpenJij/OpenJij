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

    def sampling(self, model,
                 initial_state=None, updater='single spin flip',
                 reinitilize_state=True, seed=None,
                 **kwargs):
        self._sampling_kwargs_setting(**kwargs)
        self._set_model(model)
        ising_graph = model.get_cxxjij_ising_graph()

        if initial_state is None:
            if not reinitilize_state:
                raise ValueError(
                    'You need initial_state if reinitilize_state is False.')

            def _generate_init_state(): return [ising_graph.gen_spin()
                                                for _ in range(self.trotter)]
        else:
            if model.var_type == openjij.SPIN:
                trotter_init_state = [np.array(initial_state)
                                      for _ in range(self.trotter)]
            else:  # BINARY
                trotter_init_state = [
                    (2*np.array(initial_state)-1).astype(int)
                    for _ in range(self.trotter)]

            def _generate_init_state(): return trotter_init_state

        sqa_system = cxxjij.system.make_transverse_ising_Eigen(
            _generate_init_state(), ising_graph, self.gamma
        )

        # choose updater
        _updater_name = updater.lower().replace('_', '').replace(' ', '')
        if _updater_name == 'singlespinflip':
            algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
        else:
            raise ValueError('updater is one of "single spin flip"')

        # seed for MonteCarlo
        if seed is None:
            def sqa_algorithm(system): return algorithm(
                system, self.schedule)
        else:
            def sqa_algorithm(system): return algorithm(
                system, seed, self.schedule)

        q_states = []    # store trotterized_states
        c_energies = []  # store classical energy at each trotter slices

        execution_time = []

        @measure_time
        def exec_sampling():
            previous_state = _generate_init_state()
            for _ in range(self.iteration):
                if reinitilize_state:
                    sqa_system.reset_spins(_generate_init_state())
                else:
                    sqa_system.reset_spins(previous_state)
                _exec_time = measure_time(sqa_algorithm)(sqa_system)
                execution_time.append(_exec_time)
                self._post_process4state(sqa_system, model, q_states=q_states, c_energies=c_energies)
#                 q_state = self._post_process4state(
#                     sqa_system.trotter_spins[:-1].T)
#                 q_energies.append(
#                     [model.calc_energy(state,
#                                        need_to_convert_from_spin=True)
#                      for state in q_state])
#                 q_states.append(q_state.astype(np.int))

        sampling_time = exec_sampling()
        response = openjij.Response(
            var_type=model.var_type, indices=self.indices)
        response.update_trotter_ising_states_energies(q_states, c_energies)

        response.info['sampling_time'] = sampling_time * \
            10**6              # micro sec
        response.info['execution_time'] = np.mean(
            execution_time) * 10**6   # micro sec
        response.info['list_exec_times'] = np.array(
            execution_time) * 10**6  # micro sec

        return response

    def _post_process4state(self, q_system, model, **args):
        # trotter_spins is transposed because it is stored as [Spin ​​space][Trotter].
        # [-1] is excluded because it is a tentative spin of s = 1 for convenience in SQA.
        q_state = q_system.trotter_spins[:-1].T.astype(np.int)
        # calculate classical energy at each trotter slices
        c_energies = [model.calc_energy(state, need_to_convert_from_spin=True) for state in q_state]

        args["q_states"].append(q_state.astype(np.int))
        args["c_energies"].append(c_energies)

