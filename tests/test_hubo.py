import unittest
import openjij as oj

class HUBOTest(unittest.TestCase):
    def test_hubo_sampling(self):
        sampler = oj.SASampler()

        # make HUBO
        h = {0: -1}
        J = {(0, 1): -1}
        K = {(0, 1, 2): 1}

        response = sampler.sample_hubo([h, J, K], var_type="SPIN")

        print(response)