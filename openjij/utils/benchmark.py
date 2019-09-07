# Copyright 2019 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import time
import inspect
from logging import getLogger


logger = getLogger(__name__)

def solver_benchmark(solver, time_list, solutions=[], args={}, p_r=0.99, ref_energy=0, measure_with_energy=False, time_name='execution_time'):
    """Calculate 'success probability', 'TTS', 'Residual energy' with computation time
    Args:
        solver (callable): returns openjij.Response, and solver has arguments 'time' and '**args'
        time_list (list):
        solutions (list(list(int)), list(int)): true solution or list of solution (if solutions are degenerated).
        args (dict): Arguments for solver.
        p_r (float): Thereshold probability for time to solutions.
        ref_energy (float): The ground (reference to calculate success probability and the residual energy) energy.
        measure_with_energy (bool): use a energy as measure for success

    Returns:
        dict: {
                "time": list of compuation time,
                "success_prob" list of success probability at each computation time
                "tts": list of time to solusion at each computation time
                "residual_energy": list of residual energy at each computation time
                "info" (dict): Parameter information for the benchmark
            }
    """

    if not measure_with_energy:
        if solutions == []:
            raise ValueError("need input 'solutions': (list(list))")

    logger.info( 'function ' + inspect.currentframe().f_code.co_name + ' start')

    computation_times = []
    success_probabilities = []
    tts_list = []
    residual_energies = []

    for time in time_list:
        response = solver(time, **args)


        comp_time = response.info[time_name]
        computation_times.append(comp_time)

        ps = success_probability(response, solutions, ref_energy, measure_with_energy)

        success_probabilities.append(ps)
        tts_list.append(time_to_solution(ps, comp_time, p_r))
        residual_energies.append(residual_energy(response, ref_energy))

    return {
        "time": computation_times,
        "success_prob": success_probabilities, 
        "tts": tts_list, 
        "residual_energy": residual_energies,
        "info":{
            "tts_threshold_prob": p_r,
            "ref_energy": ref_energy,
            "measure_with_energy": measure_with_energy
            }
        }


def residual_energy(response, ref_energy):
    """Calculate redisual energy from measure energy
    Args:
        response (openjij.Response): response from solver (or sampler).
        ref_energy (float): the reference energy (usually use the ground energy)

    Returns:
        float: Residual energy which is defined as follow
               <E> - E_0
               (<...> represents average, E_0 is the reference energy (usually use the ground energy)).
    """
    return np.mean(response.energies) - ref_energy



def success_probability(response, solutions,ref_energy=0, measure_with_energy=False):
    """Calculate success probability from openjij.response
    Args:
        response (openjij.Response): response from solver (or sampler).
        solutions (list[int]): true solutions.
    Returns:
        float: Success probability.
              When measure_with_energy is False, success is defined as getting the same state as solutions.
              When measure_with_energy is True, success is defined as getting a state which energy is below reference energy
    """

    if measure_with_energy:
        suc_prob = np.count_nonzero(np.array(response.energies) <= ref_energy)
    else:
        suc_prob = np.mean([1 if state in solutions else 0 for state in response.states])

    return suc_prob


def time_to_solution(success_prob, computation_time, p_r):
    """
    Args:
        success_prob (float): success probability.
        computation_time (float):
        p_r (float): thereshold probability to calculate time to solution.
    Returens:
        float: time to solution `tau * log(1-pr)/log(1-ps)` which pr is thereshold probability, ps is success probability and tau is computation time.
    """

    if success_prob == 1.0:
        tts = 0.0
    elif success_prob == 0.0:
        tts = np.inf
    else:
        tts = computation_time * np.log(1 - p_r) / np.log(1-success_prob)
    
    return tts

