import unittest
import numpy as np

import openjij as oj
import cxxjij as cj

# class UtilsTest(unittest.TestCase):


#     def test_benchmark(self):
#         h = {0: 1}
#         J = {(0, 1):-1.0, (1,2): -1.0}

#         sa_samp = oj.SASampler()

#         def solver(time_param, iteration):
#             sa_samp.step_num = time_param 
#             sa_samp.iteration = iteration
#             return sa_samp.sample_ising(h, J)

#         ground_state = [-1, -1, -1]
#         ground_energy = oj.BinaryQuadraticModel(h, J).calc_energy(ground_state)
#         step_num_list = np.linspace(10, 50, 5, dtype=np.int)
#         bm_res = oj.benchmark([ground_state], ground_energy, solver, time_param_list=step_num_list)
#         self.assertTrue(set(bm_res) >= {'time', 'error', 'e_res', 'tts', 'tts_threshold_prob'})

#         self.assertEqual(len(bm_res) ,len(step_num_list))


class ModelTest(unittest.TestCase):
    def test_bqm(self):
        h = {}
        J = {(0,1): -1.0, (1,2): -3.0}
        bqm = oj.BinaryQuadraticModel(h=h, J=J)

        self.assertEqual(type(bqm.ising_interactions()), np.ndarray)
        correct_mat = np.array([[0, -1, 0,],[-1, 0, -3],[0, -3, 0]])
        np.testing.assert_array_equal(bqm.ising_interactions(), correct_mat.astype(np.float))
    
    def test_chimera(self):
        h = {}
        J = {(0,4): -1.0, (6,2): -3.0}
        bqm = oj.BinaryQuadraticModel(h=h, J=J)
        self.assertTrue(bqm.validate_chimera(unit_num_L=3))

        J = {(0, 1): -1}
        bqm = oj.BinaryQuadraticModel(h=h, J=J)
        self.assertFalse(bqm.validate_chimera(unit_num_L=3))

    def test_ising_dict(self):
        Q = {(0,4): -1.0, (6,2): -3.0}
        bqm = oj.BinaryQuadraticModel(Q=Q, spin_type='qubo')


class SamplerOptimizeTest(unittest.TestCase):

    def setUp(self):
        self.h = {0: 1, 1: 1, 2: 1}
        self.J = {(0,1): -1.0, (1,2): -1.0}
        self.Q = {(i,i): hi for i, hi in self.h.items()}
        self.Q.update(self.J)

    def test_sa(self):
        response = oj.SASampler().sample_ising(self.h, self.J)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [-1,-1,-1])

        response = oj.SASampler().sample_qubo(self.Q)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [0,0,0])

    def test_sqa(self):
        response = oj.SQASampler().sample_ising(self.h, self.J)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [-1,-1,-1])

        response = oj.SQASampler().sample_qubo(self.Q)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [0,0,0])

    def test_gpu_sqa(self):
        gpu_sampler = oj.GPUSQASampler()
        h = {0: -1}
        J = {(0, 4): -1, (0, 5): -1, (2, 5): -1}
        model = oj.BinaryQuadraticModel(h, J, spin_type='ising')
        chimera = gpu_sampler._chimera_graph(model, chimera_L=10)


class CXXTest(unittest.TestCase):
    def test_system(self):
        N = 10
        graph = cj.graph.Dense(N)
        q_ising = cj.method.QuantumIsing(graph, 3)
        spins = q_ising.get_spins()
        q_ising.initialize_spins()
        new_spins = q_ising.get_spins()
        for spin, n_spin in zip(spins, new_spins):
            not_eq = spins == new_spins
            self.assertFalse(not_eq)

        initial_state = [1] * N
        q_ising.set_spins(initial_state)
        for spin in q_ising.get_spins():
            self.assertEqual(initial_state, spin)


if __name__ == '__main__':
    # test is currently disabled. TODO: write test!
    unittest.main()
