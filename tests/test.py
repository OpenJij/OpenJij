import unittest
import numpy as np

import openjij as oj

class ModelTest(unittest.TestCase):
    def test_bqm(self):
        h = {}
        J = {(0,1): -1.0, (1,2): -3.0}
        bqm = oj.BinaryQuadraticModel(h=h, J=J)

        self.assertEqual(type(bqm.ising_interactions()), np.ndarray)
        correct_mat = np.array([[0, -1, 0,],[-1, 0, -3],[0, -3, 0]])
        np.testing.assert_array_equal(bqm.ising_interactions(), correct_mat.astype(np.float))

class SamplerOptimizeTest(unittest.TestCase):

    def setUp(self):
        h = {0: -1, 1: -1}
        J = {(0,1): -1.0, (1,2): -1.0}
        self.bqm = oj.BinaryQuadraticModel(h=h, J=J)  
        self.samp = oj.Sampler(model=self.bqm)

    def test_sa(self):
        response = self.samp.simulated_annealing()
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [1,1,1])

    def test_sqa(self):
        response = self.samp.simulated_quantum_annealing()
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [1,1,1])


if __name__ == '__main__':
    unittest.main()