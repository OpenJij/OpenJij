# Copyright 2023 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import inspect

from logging import getLogger

import numpy as np

logger = getLogger(__name__)


def solver_benchmark(
    solver,
    time_list,
    solutions=[],
    args={},
    p_r=0.99,
    ref_energy=0,
    measure_with_energy=False,
    time_name="execution_time",
):
    """Calculate 'success probability', 'TTS', 'Residual energy','Standard Error' with computation time

    Args:
        solver (callable): returns openjij.Response, and solver has arguments 'time' and '**args'
        time_list (list):
        solutions (list(list(int)), list(int)): true solution or list of solution (if solutions are degenerated).
        args (dict): Arguments for solver.
        p_r (float): Thereshold probability for time to solutions.
        ref_energy (float): The ground (reference to calculate success probability and the residual energy) energy.
        measure_with_energy (bool): use a energy as measure for success
    Returns:
        dict: dictionary which has the following keys:

        * **time**: list of compuation time
        * **success_prob** list of success probability at each computation time
        * **tts**: list of time to solusion at each computation time
        * **residual_energy**: list of residual energy at each computation time
        * **se_lower_tts**: list of tts's lower standard error at each computation time
        * **se_upper_tts**: list of tts's upper standard error at each computation time
        * **se_success_prob**: list of success probability's standard error at each computation time
        * **se_residual_energy**: list of residual_energy's standard error at each computation time
        * **info** (dict): Parameter information for the benchmark
    """

    if not measure_with_energy:
        if solutions == []:
            raise ValueError("need input 'solutions': (list(list))")

    logger.info("function " + inspect.currentframe().f_code.co_name + " start")

    computation_times = []
    success_probabilities = []
    tts_list = []
    residual_energies = []

    se_lower_tts_list = []
    se_upper_tts_list = []
    se_success_prob_list = []
    se_residual_energy_list = []

    for time in time_list:
        response = solver(time, **args)

        comp_time = response.info[time_name]
        computation_times.append(comp_time)

        ps = success_probability(response, solutions, ref_energy, measure_with_energy)
        tts = time_to_solution(ps, comp_time, p_r)

        success_probabilities.append(ps)
        tts_list.append(tts)
        residual_energies.append(residual_energy(response, ref_energy))

        se_ps = se_success_probability(
            response, solutions, ref_energy, measure_with_energy
        )

        se_success_prob_list.append(se_ps)
        se_lower_tts_list.append(se_lower_tts(tts, ps, comp_time, p_r, se_ps))
        se_upper_tts_list.append(se_upper_tts(tts, ps, comp_time, p_r, se_ps))
        se_residual_energy_list.append(se_residual_energy(response, ref_energy))

    return {
        "time": computation_times,
        "success_prob": success_probabilities,
        "tts": tts_list,
        "residual_energy": residual_energies,
        "se_lower_tts": se_lower_tts_list,
        "se_upper_tts": se_upper_tts_list,
        "se_success_prob": se_success_prob_list,
        "se_residual_energy": se_residual_energy_list,
        "info": {
            "tts_threshold_prob": p_r,
            "ref_energy": ref_energy,
            "measure_with_energy": measure_with_energy,
        },
    }


def residual_energy(response, ref_energy):
    """Calculate redisual energy from measure energy

    Args:
        response (openjij.Response): response from solver (or sampler).
        ref_energy (float): the reference energy (usually use the ground energy)
    Returns:
        float: Residual energy which is defined as :math:`\\langle E \\rangle - E_0` (:math:`\\langle...\\rangle` represents average, :math:`E_0` is the reference energy (usually use the ground energy)).
    """
    return np.mean(response.energies) - ref_energy


def se_residual_energy(response, ref_energy):
    """Calculate redisual energy's standard error from measure energy

    Args:
        response (openjij.Response): response from solver (or sampler).
        ref_energy (float): the reference energy (usually use the ground energy)
    Returns:
        float: redisual energy's standard error from measure energy
    """
    return np.std(response.energies, ddof=1)


def success_probability(response, solutions, ref_energy=0, measure_with_energy=False):
    """Calculate success probability from openjij.response

    Args:
        response (openjij.Response): response from solver (or sampler).
        solutions (list[int]): true solutions.
    Returns:
        float: Success probability.

        * When measure_with_energy is False, success is defined as getting the same state as solutions.
        * When measure_with_energy is True, success is defined as getting a state which energy is below reference energy
    """

    if measure_with_energy:
        suc_prob = np.count_nonzero(np.array(response.energies) <= ref_energy) / len(
            response.energies
        )
    else:
        if isinstance(solutions[0], dict):
            sampled_states = response.samples()
            suc_prob = np.mean(
                [1 if dict(state) in solutions else 0 for state in sampled_states]
            )
        else:
            sampled_states = response.states
            suc_prob = np.mean(
                [1 if list(state) in solutions else 0 for state in sampled_states]
            )

    return suc_prob


def se_success_probability(
    response, solutions, ref_energy=0, measure_with_energy=False
):
    """Calculate success probability's standard error from openjij.response

    Args:
        response (openjij.Response): response from solver (or sampler).
        solutions (list[int]): true solutions.
    Returns:
        float: Success probability's standard error.

        * When measure_with_energy is False, success is defined as getting the same state as solutions.
        * When measure_with_energy is True, success is defined as getting a state which energy is below reference energy
    """

    if measure_with_energy:
        se_suc_prob = np.sqrt(
            np.count_nonzero(np.array(response.energies) <= ref_energy)
            / (len(response.energies) - 1)
        )
    else:
        if isinstance(solutions[0], dict):
            sampled_states = response.samples()
            se_suc_prob = np.std(
                [1 if dict(state) in solutions else 0 for state in sampled_states]
            )
        else:
            sampled_states = response.states
            se_suc_prob = np.std(
                [1 if list(state) in solutions else 0 for state in sampled_states]
            )

    return se_suc_prob


def time_to_solution(success_prob, computation_time, p_r):
    """
    Args:

        success_prob (float): success probability.
        computation_time (float):
        p_r (float): thereshold probability to calculate time to solution.
    Returns:
        float: time to solution :math:`\\tau * \\log(1-pr)/\\log(1-ps)` which pr is thereshold probability, ps is success probability and :math:`tau` is computation time.
    """

    if success_prob == 1.0:
        tts = 0.0
    elif success_prob == 0.0:
        tts = np.inf
    else:
        tts = computation_time * np.log(1 - p_r) / np.log(1 - success_prob)

    return tts


def se_lower_tts(tts, success_prob, computation_time, p_r, se_success_prob):
    """
    Args:

        success_prob (float): success probability.
        computation_time (float):
        p_r (float): thereshold probability to calculate time to solution.
    Returns:
        float: time to solution :math:`\\tau * \\log(1-pr)/\\log(1-ps)` 's standard error which pr is thereshold probability, ps is success probability and :math:`tau` is computation time.
    """

    if 1 - (success_prob + se_success_prob) <= 0.0:
        tts_low_error = 0.0
    elif success_prob == 0.0:
        tts_low_error = 0.0
    else:
        tts_low_error = (
            computation_time
            * np.log(1 - p_r)
            / np.log(1 - (success_prob + se_success_prob))
        )

    se_lower_tts = abs(tts_low_error - tts)

    return se_lower_tts


def se_upper_tts(tts, success_prob, computation_time, p_r, se_success_prob):
    """
    Args:

        success_prob (float): success probability.
        computation_time (float):
        p_r (float): thereshold probability to calculate time to solution.
    Returens:
        float: time to solution :math:`\\tau * \\log(1-pr)/\\log(1-ps)` 's standard error which pr is thereshold probability, ps is success probability and :math:`tau` is computation time.
    """

    if success_prob == 1.0:
        tts_up_error = 0.0
    elif success_prob == 0.0:
        tts_up_error = 0.0
    else:
        tts_up_error = (
            computation_time
            * np.log(1 - p_r)
            / np.log(1 - (success_prob - se_success_prob))
        )

    se_upper_tts = abs(tts_up_error - tts)

    return se_upper_tts
