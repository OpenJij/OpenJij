import numpy as np
import time 

def benchmark(true_ground_states, ground_energy, solver, time_param_list, p_d=0.9, iteration=1000):
    
    if not (isinstance(true_ground_states, list) and isinstance(true_ground_states[0], list)):
        raise ValueError("true_ground_states should be list of list which stored each spin state.")
    
    error_prob_list = []
    e_res_list = []
    tts_list = []
    
    tlist = []
    
    for time_param in time_param_list:
        print('time parameter: ', time_param)
        start = time.time()
        response = solver(time_param, iteration=iteration)
        tlist.append( time.time() - start )

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
            tts = t/iteration * np.log(1-pd)/np.log(error_prob) 
        
        error_prob_list.append(error_prob)
        e_res_list.append(e_res)
        tts_list.append(tts)
        
    return {'time':tlist, 'error':error_prob_list, 'e_res':e_res_list, 'tts':tts_list, 
             'tts_threshold_prob': pd}

    

import openjij as oj

def convert_oj_response(response):
    """
    Converter to openjij response class from response class of dwave tools (dimod) 
    """
    try:
        from dimod.response import Response
    except ImportError:
        raise ImportError('Import dwave dimod : "pip install dimod"')
    if isinstance(response, Response):
        from dimod.vartypes import BINARY, SPIN
        spin_type = 'qubo' if response.vartype == BINARY else 'spin'
        o_res = oj.Response(spin_type=spin_type, indices=list(response.variable_labels))
        for rec in response.record:
            for _ in range(rec[2]):
                o_res.add_state_energy(rec[0], rec[1])
        return o_res
    elif isinstance(response, oj.Response):
        return response
    else:
        raise TypeError('type of response is dimod.response.Response or openjij.Response')