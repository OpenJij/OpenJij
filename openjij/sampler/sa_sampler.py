import numpy as np
import openjij
from openjij.sampler import measure_time
from openjij.sampler import BaseSampler
from openjij.utils.decorator import deprecated_alias
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
            Number of annealing temperature divisions

        step_num (int):
            Number of steps per temperature.

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

    # @renames.renamed_kwarg(old_name='step_length', new_name='num_call_updater', removal_version='0.1.0')
    # @renames.renamed_kwarg(old_name='step_num', new_name='one_mc_step', removal_version='0.1.0')
    @deprecated_alias(iteration='num_reads')
    def __init__(self,
                 beta_min=None, beta_max=None,
                 num_sweeps=1000, schedule=None,
                 num_reads=1,
                 **kwargs):

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_sweeps = num_sweeps
        self.schedule = schedule
        self.num_reads = num_reads

    def _convert_validation_schedule(self, schedule):
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

    @deprecated_alias(iteration='num_reads')
    def sample(self, model, beta_min=None, beta_max=None,
               num_sweeps=None, num_reads=1, schedule=None,
               initial_state=None, updater='single spin flip',
               reinitialize_state=True, seed=None,
               **kwargs):

        model = openjij.BinaryQuadraticModel(
            linear=model.linear, quadratic=model.quadratic,
            offset=model.offset, var_type=model.vartype
        )

        ising_graph = model.get_cxxjij_ising_graph()

        self.num_reads = num_reads if num_reads > 1 else self.num_reads

        # set annealing schedule -------------------------------
        if schedule or self.schedule:
            self.schedule = self._convert_validation_schedule(
                schedule if schedule else self.schedule
            )
            self.schedule_info = {'schedule': 'custom schedule'}
        else:
            self.beta_min = beta_min if beta_min else self.beta_min
            self.beta_max = beta_max if beta_max else self.beta_max
            self.num_sweeps = num_sweeps if num_sweeps else self.num_sweeps
            self.schedule, beta_range = geometric_ising_beta_schedule(
                model=model,
                beta_max=self.beta_max, beta_min=self.beta_min,
                num_sweeps=self.num_sweeps
            )
            self.schedule_info = {
                'beta_max': beta_range[0],
                'beta_min': beta_range[1],
                'num_sweeps': self.num_sweeps
            }
        # ------------------------------- set annealing schedule

        # make init state generator
        if initial_state is None:
            if not reinitialize_state:
                raise ValueError(
                    'You need initial_state if reinitilize_state is False.')

            def _generate_init_state(): return ising_graph.gen_spin()
        else:
            # validate initial_state size
            if len(initial_state) != ising_graph.size():
                raise ValueError(
                    "the size of the initial state should be {}"
                    .format(ising_graph.size()))
            if model.var_type == openjij.SPIN:
                _init_state = np.array(initial_state)
            else:  # BINARY
                _init_state = (2*np.array(initial_state)-1).astype(np.int)

            def _generate_init_state(): return np.array(_init_state)

        # choose updater
        _updater_name = updater.lower().replace('_', '').replace(' ', '')
        if _updater_name == 'singlespinflip':
            algorithm = cxxjij.algorithm.Algorithm_SingleSpinFlip_run
            sa_system = cxxjij.system.make_classical_ising_Eigen(
                _generate_init_state(), ising_graph)
        elif _updater_name == 'swendsenwang':
            # swendsen-wang is not support Eigen system
            algorithm = cxxjij.algorithm.Algorithm_SwendsenWang_run
            sa_system = cxxjij.system.make_classical_ising(
                _generate_init_state(), ising_graph)
        else:
            raise ValueError(
                'updater is one of "single spin flip or swendsen wang"')

        response = self._sampling(
            model, _generate_init_state,
            algorithm, sa_system, initial_state,
            reinitialize_state, seed, **kwargs
        )

        response.update_ising_states_energies(
            response.states, response.energies)

        response.info['schedule'] = self.schedule_info

        return response

    def _post_save(self, result_state, system, model, response):
        response.states.append(result_state)
        response.energies.append(model.calc_energy(
            result_state,
            need_to_convert_from_spin=True))


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
        ising_interaction = np.abs(model.ising_interactions())
        abs_bias = np.sum(ising_interaction, axis=1)

        min_delta_energy = np.min(ising_interaction[ising_interaction > 0])
        max_delta_energy = np.max(abs_bias[abs_bias > 0])

    beta_min = np.log(2) / max_delta_energy if beta_min is None else beta_min
    beta_max = np.log(100) / min_delta_energy if beta_max is None else beta_max

    schedule = cxxjij.utility.make_classical_schedule_list(
        beta_min=beta_min, beta_max=beta_max,
        one_mc_step=1, num_call_updater=num_sweeps
    )

    return schedule, [beta_max, beta_min]
