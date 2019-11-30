import numpy as np
from openjij.model import BinaryHigherOrderModel
from openjij import SPIN, BINARY
from dimod import as_vartype


def default_schedule(bhom, beta_min, beta_max, num_sweeps):

    _beta_max, _beta_min = _default_beta_schedule(bhom)

    beta_max = beta_max if beta_max else _beta_max
    beta_min = beta_min if beta_min else _beta_min

    num_sweeps_per_beta = max(1, num_sweeps // 1000.0)
    r = (beta_max/beta_min)**(1/num_sweeps)

    return [[beta_min * r ** n, num_sweeps_per_beta] for n in range(num_sweeps)]


def _default_beta_schedule(hubo):
    adj_dict = hubo.adj_dict()
    linear = hubo.interactions[0]

    de_list = []
    for i, adj in adj_dict.items():
        de = linear[i]
        for _, v in adj:
            de += np.abs(v)
        de_list.append(de)

    n = len(linear)
    beta_max = np.log(n)/np.max(de_list)

    abs_linear = np.abs(list(linear.values())) + 0.01
    beta_min = np.log(5)/np.min(abs_linear)

    return beta_min, beta_max


def hubo_simulated_annealing(bhom: BinaryHigherOrderModel, state: list,
                             schedule: list, var_type):

    adj_dict = bhom.adj_dict()
    state = np.array(state)

    if SPIN == as_vartype(var_type):
        for beta, mc_steps in schedule:
            for _ in range(mc_steps):
                for i in bhom.indices:
                    de = -2*state[i]*np.sum([d*np.prod(state[_inds])
                                             for _inds, d in adj_dict[i]])
                    de += -2*state[i]*bhom.interactions[0][i]
                    if de < 0 or np.random.uniform(0, 1) < np.exp(-beta*de):
                        state[i] *= -1
        return state
    elif BINARY == as_vartype(var_type):
        for beta, mc_steps in schedule:
            for _ in range(mc_steps):
                for i in bhom.indices:
                    de = (1-2*state[i])*np.sum([d*np.prod(state[_inds])
                                                for _inds, d in adj_dict[i]])
                    de += (1-2*state[i])*bhom.interactions[0][i]
                    if de <= 0 or np.random.uniform(0, 1) < np.exp(-beta*de):
                        state[i] = 0 if state[i] == 1 else 1
        return state
    else:
        raise ValueError("var_type should be SPIN or BINARY")
