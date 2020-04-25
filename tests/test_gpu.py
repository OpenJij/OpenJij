import openjij as oj
import cxxjij as cj
import numpy as np

import unittest


class TestGPUSampler(unittest.TestCase):
    def setUp(self):
        self.num_ind = {
            'h': {0: -1, 1: -1, 2: 1, 3: 1},
            'J': {(0, 4): -1, (2, 5): -1}
        }
        str_ind = ['a', 'b', 'c', 'd', 'e', 'd']
        self.str_ising = {
            'h': {str_ind[i]: v for i, v in self.num_ind['h'].items()},
            'J': {(str_ind[i], str_ind[j]): v for (i, j), v in self.num_ind['J'].items()}
        }
        self.ground_state = [1, 1, -1, -1, 1, -1]
        self.e_g = -1-1-1-1 + (-1-1)
        # self.g_sample = {i: self.ground_state[i]
        #                  for i in range(len(self.ground_state))}
        # self.g_samp_str = {k: self.ground_state[i]
        #                    for i, k in enumerate(str_ind)}

        self.qubo = {
            (0, 0): -1, (1, 1): -1, (2, 2): -1, (3, 3): 1,
            (0, 4): -1, (2, 5): -1
        }
        # self.str_qubo = {(str_ind[i], str_ind[j]): qij
        #                  for (i, j), qij in self.qubo.items()}
        self.ground_q = [1, 1, 1, 0, 1, 1]
        self.e_q = -1-1-1 + (-1-1)

    def samplers(self, sampler, init_state=None):
        res = sampler.sample_ising(
            self.num_ind['h'], self.num_ind['J'],
            initial_state=init_state, seed=1)
        self._test_response(res, self.e_g, self.ground_state)
        res = sampler.sample_qubo(self.qubo,
                                  initial_state=init_state, seed=1)
        
        self._test_response(res, self.e_q, self.ground_q)

        # embedding composit
        # from dwave.system.composites import EmbeddingComposite
        # samp = EmbeddingComposite(sampler)
        # res = samp.sample_ising(
        #     self.num_ind['h'], self.num_ind['J']
        # )
        # 
        # res = samp.sample_ising(
        #     self.str_ising['h'], self.str_ising['J']
        # )

    def _test_response(self, res, e_g, s_g):
        # test openjij response interface
        self.assertEqual(len(res.states), 1)
        self.assertListEqual(s_g, list(res.states[0]))
        self.assertEqual(res.energies[0], e_g)
        # test dimod interface
        self.assertEqual(len(res.record.sample), 1)
        self.assertListEqual(s_g, list(res.record.sample[0]))
        self.assertEqual(res.record.energy[0], e_g)

    def _test_response_num(self, res, num_reads):
        # test openjij response interface
        self.assertEqual(len(res.states), num_reads)
        self.assertEqual(len(res.energies), num_reads)
        # test dimod interface
        self.assertEqual(len(res.record.sample), num_reads)
        self.assertEqual(len(res.record.energy), num_reads)

    def _test_num_reads(self, sampler_cls):
        num_reads = 10
        sampler = sampler_cls()
        res = sampler.sample_ising(
            self.num_ind['h'], self.num_ind['J'],
            num_reads=num_reads,
            seed=1
        )
        self._test_response_num(res, num_reads)

        sampler = sampler_cls(num_reads=num_reads)
        res = sampler.sample_ising(
            self.num_ind['h'], self.num_ind['J'],
        )
        self._test_response_num(res, num_reads)

    def test_sa(self):
        sampler = oj.GPUChimeraSASampler(unit_num_L=2)
        self.samplers(sampler)
        self._test_num_reads(oj.SASampler)

    def test_sqa(self):
        sampler = oj.GPUChimeraSQASampler(unit_num_L=2)
        self.samplers(sampler)
        self._test_num_reads(oj.SQASampler)



if __name__ == '__main__':
    unittest.main()
