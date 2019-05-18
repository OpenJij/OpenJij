import numpy as np
import time
import inspect
from logging import getLogger

def benchmark(true_ground_states, ground_energy, solver, time_param_list, p_d=0.9, iteration=1000):

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
        start = time.time()
        logger.info('\tsampling...')
        response = convert_response(solver(time_param, iteration=iteration))
        tlist.append( time.time() - start )

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
        o_res.info['dimod'] = response
        return o_res
    else:
        raise TypeError('response is dimod.response.Response, SampleSet or openjij.Response')