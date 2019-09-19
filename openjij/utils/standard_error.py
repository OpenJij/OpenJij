import numpy as np
import scipy as sp
from scipy import stats


def standard_error(solver, time_list, iteration_list, args={}):
    """Calculate 'Standard Error' with iteration
    Args:
        solver (callable): returns openjij.Response, and solver has arguments 'time' and '**args'
        time_list (list):
        iteration_list (list):
        args (dict): Arguments for solver.

    Returns:
        dict: {
                "se": list of standard error at each iteration
                "info" (dict): Parameter information for the benchmark
            }
    """

    se_list = []
    response_list = []

    i = 0
    for iteration in iteration_list:
        for time in time_list:
            response = solver(time, **args)
            response_list.append(response)
        se_list[i] = sp.std(response_list[i].energies, ddof=1)
        i = i + 1


    return {
        "se": standard_error,
        "info":{
            "iteration_list": iteration_list,
            }
        }

