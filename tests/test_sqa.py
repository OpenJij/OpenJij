import unittest
import openjij as oj


class TestSQA(unittest.TestCase):
    def setUp(self):
        self.num_ind = {
            'h': {0: -1, 1: -1, 2: 1, 3: 1},
            'J': {(0, 1): -1, (3, 4): -1}
        }
        str_ind = ['a', 'b', 'c', 'd', 'e']
        self.str_ising = {
            'h': {str_ind[i] for i in self.num_ind['h'].keys()},
            'J': {(str_ind[i], str_ind[j]) for i, j in self.num_ind['J'].keys()}
        }
        self.ground_state = [1, 1, -1, -1, -1]
        self.e_g = -1-1-1-1 + (-1-1)
        self.g_sample = {i: self.ground_state[i]
                         for i in range(len(self.ground_state))}
        self.g_samp_str = {k: self.ground_state[i]
                           for i, k in enumerate(str_ind)}

        self.qubo = {
            (0, 0): -1, (1, 1): -1, (2, 2): 1, (3, 3): 1, (4, 4): 1,
            (0, 1): -1, (3, 4): 1
        }
        self.str_qubo = {(str_ind[i], str_ind[j]): qij
                         for (i, j), qij in self.qubo.items()}
        self.ground_q = [1, 1, 0, 0, 0]
        self.e_q = -1-1-1

    #def test_sqa(self):
    #    sampler = oj.SQASampler()
