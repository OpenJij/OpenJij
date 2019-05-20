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

def benchmark(true_ground_states, ground_energy, solver, time_param_list, p_d=0.9, iteration=100, time_name='execution_time'):

    logger = getLogger('openjij')

    logger.info( 'function ' + inspect.currentframe().f_code.co_name + ' start')
    
    if not (isinstance(true_ground_states, list) and isinstance(true_ground_states[0], list)):
        raise ValueError("true_ground_states should be list of list which stored each spin state.")
    
    error_prob_list = []
    e_res_list = []
    tts_list = []
    
    tlist = []

    for time_param in time_param_list:
        logger.info('Time parameter: {}'.format(time_param))
        logger.info('\tsampling...')
        response = convert_response(solver(time_param, iteration=iteration))
        if not time_name in response.info:
            raise ValueError('the solver should return openjij.Response object which has info[{}] for measurement computation time.'.format(time_name))

        tlist.append(response.info[time_name])

        logger.info('sampling time : '.format(tlist[-1]))

        # error probability
        error_prob = 1.0 - sum([1 if state in true_ground_states else 0 for state in response.states]) /iteration

        # residual energy
        e_res = (np.mean(response.energies) - ground_energy)

        # Time to solution (TTS)
        t = tlist[-1]
        if error_prob == 0.0:
            tts = 0.0
        elif error_prob == 1.0:
            tts = np.inf
        else:
            tts = t/iteration * np.log(1-p_d)/np.log(error_prob) 
        
        error_prob_list.append(error_prob)
        e_res_list.append(e_res)
        tts_list.append(tts)

    logger.info('benchmark end')
        
    return {'time':tlist, 'error':error_prob_list, 'e_res':e_res_list, 'tts':tts_list, 
             'tts_threshold_prob': p_d}

    

import openjij as oj

def convert_response(response):

    if isinstance(response, oj.Response):
        return response

    try:
        from dimod.response import Response
        from dimod.sampleset import SampleSet
    except ImportError:
        raise ImportError('Import dwave dimod : "pip install dimod"')
    
    if isinstance(response, Response) or isinstance(response, SampleSet):
        from dimod.vartypes import BINARY, SPIN
        var_type = 'BINARY' if response.vartype == BINARY else 'SPIN'
        o_res = oj.Response(var_type=var_type, indices=list(response.variables))
        states = []
        energies = []
        for rec in response.record:
            for _ in range(rec[2]):
                states.append(rec[0])
                energies.append(rec[1])
        o_res.update_ising_states_energies(states=states, energies=energies)
                
        o_res.info = response.info
        if 'qpu_sampling_time' in response.info:
            o_res.info['sampling_time'] = response.info['qpu_sampling_time']
        if 'anneal_time_per_run' in response.info:
            o_res.info['execution_time'] = response.info['anneal_time_per_run']
        o_res.info['dimod'] = response
        return o_res
    else:
        raise TypeError('response is dimod.response.Response, SampleSet or openjij.Response')