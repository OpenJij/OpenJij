import unittest
import openjij as oj

class HUBOTest(unittest.TestCase):
    def test_hubo_sampling(self):
        sampler = oj.SASampler()

        # make HUBO
        K = {(0, 1, 2): 1}

        response = sampler.sample_hubo(K, var_type="SPIN")
