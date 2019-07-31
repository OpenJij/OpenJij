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
        - schedule range is '0 <= s < 1'.

    """

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
            self.schedule_info = {
                'step_num': step_num, 'step_length': step_length}

        self.beta = beta
        self.gamma = gamma
        self.trotter = trotter

        self.iteration = iteration
        self.energy_bias = 0.0

        self.system_class = cj.system.QuantumIsing  # CPU Trotterize quantum system
        self.sqa_kwargs = dict(
            beta=self.beta, gamma=self.gamma, **self.schedule_info)

    def _validate_schedule(self, schedule):
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        # schedule validation  0 <= s < 1
        sch = np.array(schedule).T[0]
        if not np.all((0 <= sch) & (sch < 1)):
            raise ValueError("schedule range is '0 <= s < 1'.")

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
        return self.sampling(bqm, var_type=var_type,
                             initial_state=initial_state, updater=updater,
                             reinitilize_state=reinitilize_state,
                             seed=seed,
                             **kwargs
                             )

    def sample_qubo(self, Q,
                    initial_state=None, updater='single spin flip',
                    reinitilize_state=False, seed=None, **kwargs):
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
        return self.sampling(bqm, var_type=var_type,
                             initial_state=initial_state, updater=updater,
                             reinitilize_state=reinitilize_state,
                             seed=seed,
                             **kwargs
                             )

    def sampling(self, model, var_type,
                 initial_state=None, updater='single spin flip',
                 reinitilize_state=True, seed=None,
                 **kwargs):
        self._sampling_kwargs_setting(**kwargs)
        self._set_model(model)
        ising_graph = model.get_cxxjjij_ising_graph()

        if initial_state is None:
            if not reinitilize_state:
                raise ValueError(
                    'You need initial_state if reinitilize_state is False.')

            def _init_state(): return [ising_graph.gen_spin()
                                       for _ in range(self.trotter)]
        else:
            trotter_init_state = [np.array(initial_state)
                                  for _ in range(self.trotter)]

            def _init_state(): return trotter_init_state

        sqa_system = cxxjij.system.make_transverse_ising(
            _init_state(), ising_graph
        )

        # choose updater
        _updater_name = updater.lower().replace('_', '').replace(' ', '')
        if _updater_name == 'singlespinflip':
            algorithm = cxxjij.algorith.Algorithm_SingleSpinFlip_run
        else:
            raise ValueError('updater is one of "single spin flip"')

        # seed for MonteCarlo
        if seed is None:
            def simulated_annealing(system): return algorithm(
                system, self.schedule)
        else:
            def simulated_annealing(system): return algorithm(
                system, seed, self.schedule)

        q_states = []
        q_energies = []

        execution_time = []

        @measure_time
        def exec_sampling():
            previous_state = _init_state()
            for _ in range(self.iteration):
                if reinitilize_state:
                    sqa_system.reset_spins(_init_state())
                else:
                    sqa_system.reset_spins(previous_state)
                _exec_time = measure_time(algorithm)(sqa_system)
                execution_time.append(_exec_time)
                # trotter_spins is transposed because it is stored as [Spin ​​space][Trotter].
                # [-1] is excluded because it is a tentative spin of s = 1 for convenience in SQA.
                q_state = self._post_process4state(
                    sqa_system.trotter_spins[:-1].T)
                q_states.append(q_state)
                q_energies.append([ising_graph.calc_energy(
                    state) + self.energy_bias for state in q_state])

        sampling_time = exec_sampling()

        response = openjij.Response(var_type=var_type, indices=self.indices)
        response.update_quantum_ising_states_energies(q_states, q_energies)

        response.info['sampling_time'] = sampling_time * \
            10**6              # micro sec
        response.info['execution_time'] = np.mean(
            execution_time) * 10**6   # micro sec
        response.info['list_exec_times'] = np.array(
            execution_time) * 10**6  # micro sec

        return response

    def _post_process4state(self, q_state):
        return q_state
