import numpy as np


class BinaryHigherOrderModel:
    def __init__(self, interactions: list):
        self.interactions = interactions

        indices = set(self.interactions[0].keys())
        for coeff in self.interactions[1:]:
            for _inds in coeff.keys():
                indices = indices | set(_inds)

        self.indices = list(indices)

        for i in self.indices:
            if i not in self.interactions[0]:
                self.interactions[0][i] = 0.0

    def adj_dict(self):
        """adjgency list of each variables

        Returns:
            dict: key (variables key), value (list of tuple represents connected indices)
        """
        adj_dict = {i: [] for i in self.indices}
        for coeff in self.interactions[1:]:
            for _inds, value in coeff.items():
                for i in _inds:
                    _inds_list = list(_inds)
                    _inds_list.remove(i)
                    adj_dict[i].append([_inds_list, value])
        return adj_dict

    def calc_energy(self, state):
        """calculate energy of state

        Args:
            state (list of int): list of SPIN or BINARY 

        Returns:
            float: energy of state
        """
        energy = 0.0
        state = np.array(state)
        for coeff in self.interactions[1:]:
            for _inds, value in coeff.items():
                energy += value * np.prod(state[list(_inds)])
        for i, hi in self.interactions[0].items():
            energy += hi * state[i]

        return energy
