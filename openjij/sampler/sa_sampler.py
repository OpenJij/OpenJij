import numpy as np
import openjij
from openjij.sampler import measure_time
from openjij.sampler import BaseSampler
import cxxjij


class SASampler(BaseSampler):
    """Sampler with Simulated Annealing (SA).

    Inherits from :class:`openjij.sampler.sampler.BaseSampler`.

    Args:
        beta_min (float):
            Minmum beta (inverse temperature).

        beta_max (float):
            Maximum beta (inverse temperature).

        step_length (int):
            Length of Monte Carlo step.

        step_num (int):
            Number of Monte Carlo step.

        schedule_info (dict):
            Information about an annealing schedule.

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

    Raises:
        ValueError: If schedules or variables violate as below.
        - not list or numpy.array.
        - not list of tuple (beta : float, step_length : int).
        - beta is less than zero.

    """

    def __init__(self,
                 beta_min=0.1, beta_max=5.0,
                 step_length=10, step_num=100, schedule=None, iteration=1):
        if schedule:
            self.schedule = self._convert_validation_schedule(schedule)
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
            self.schedule = cxxjij.utility.make_classical_schedule_list(
                beta_min=beta_min, beta_max=beta_max,
                one_mc_step=step_length,
                num_call_updater=step_num
            )
            self.schedule_info = {
                'beta_min': beta_min, 'beta_max': beta_max,
                'step_length': step_length, 'step_num': step_num
            }
        self.iteration = iteration

    def _convert_validation_schedule(self, schedule):
        if not isinstance(schedule, (list, np.array)):
            raise ValueError("schedule should be list or numpy.array")

        if isinstance(schedule[0], cxxjij.utility.ClassicalSchedule):
            return schedule

        if not isinstance(schedule[0], tuple):
            raise ValueError(
                "schedule is list of tuple (beta : float, step_length : int)")

        # schedule validation  0 <= beta
        beta = np.array(schedule).T[0]
        if not np.all(0 <= beta):
            raise ValueError("schedule beta range is '0 <= beta'.")

        # convert to cxxjij.utility.ClassicalSchedule
        cxxjij_schedule = []
        for beta, step_length in schedule:
            _schedule = cxxjij.utility.ClassicalSchedule()
            _schedule.one_mc_step = step_length
            _schedule.updater_parameter = beta
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

            initial_state (list):
                The initial state of simulated annealing

            updater (str):
                Monte Carlo update algorithm : 'single spin flip' or 'swendsenwang'

            reinitilize_state (bool):
                Reinitialize the initial state for every anneal-readout cycle.

            seed (:obj:`int`, optional):
                seed for Monte Carlo step

            **kwargs:
                Optional keyword arguments for the sampling method.

        Returns:
            :obj:: `openjij.sampler.response.Response` object.

        Examples:
            This example submits a two-variable Ising problem.

            >>> import openjij as oj
            >>> sampler = oj.SASampler()
            >>> response = sampler.sample_ising({0: -1, 1: 1}, {})
            >>> for sample in response.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 1, 1: -1}

        """

        var_type = openjij.SPIN
        model = openjij.BinaryQuadraticModel(h=h, J=J, var_type=var_type)
        return self.sampling(model,
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

            initial_state (list):
                The initial state of simulated annealing

            updater (str):
                Monte Carlo update algorithm : 'single spin flip' or 'swendsenwang'

            reinitilize_state (bool):
                Reinitialize the initial state for every anneal-readout cycle.

            seed (:obj:`int`, optional):
                seed for Monte Carlo step

            **kwargs:
                Optional keyword arguments for the sampling method.

        Returns:
            :obj:: `openjij.sampler.response.Response` object.

        Examples:
            This example submits a two-variable QUBO model.

            >>> import openjij as oj
            >>> sampler = oj.SASampler()
            >>> Q = {(0, 0): -1, (4, 4): -1, (0, 4): 2}
            >>> response = sampler.sample_qubo(Q)
            >>> for sample in response.samples():    # doctest: +SKIP
            ...    print(sample)
            ...
            {0: 0, 4: 1}

        """

        var_type = openjij.BINARY
        model = openjij.BinaryQuadraticModel(Q=Q, var_type=var_type)
        return self.sampling(model,
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

            def _init_state(): return ising_graph.gen_spin()
        else:
            def _init_state(): return np.array(initial_state)

        sa_system = cxxjij.system.make_classical_ising_Eigen(
            _init_state(), ising_graph)

        # choose updater
        _updater_name = updater.lower().replace('_', '').replace(' ', '')
        if _updater_name == 'singlespinflip':
            algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
        elif _updater_name == 'swendsenwang':
            algorithm = cxxjij.algorithm.Algorithm_SwendsenWang_run
        else:
            raise ValueError(
                'updater is one of "single spin flip or swendsen wang"')

        # seed for MonteCarlo
        if seed is None:
            def simulated_annealing(system): return algorithm(
                system, self.schedule)
        else:
            def simulated_annealing(system): return algorithm(
                system, seed, self.schedule)

        states = []
        energies = []

        execution_time = []

        @measure_time
        def exec_sampling():
            previous_state = _init_state()
            for _ in range(self.iteration):
                if reinitilize_state:
                    sa_system.reset_spins(_init_state())
                else:
                    sa_system.reset_spins(previous_state)

                _exec_time = measure_time(simulated_annealing)(sa_system)
                execution_time.append(_exec_time)
                previous_state = cxxjij.result.get_solution(sa_system)
                states.append(previous_state)
                energies.append(ising_graph.calc_energy(
                    previous_state) + self.energy_bias)

        sampling_time = exec_sampling()

        response = openjij.Response(
            var_type=model.var_type, indices=self.indices)
        response.update_ising_states_energies(states, energies)

        response.info['sampling_time'] = sampling_time * \
            10**6              # micro sec
        response.info['execution_time'] = np.mean(
            execution_time) * 10**6   # micro sec
        response.info['list_exec_times'] = np.array(
            execution_time) * 10**6  # micro sec

        return response
