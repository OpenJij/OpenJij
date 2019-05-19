from logging import getLogger, StreamHandler, INFO

import unittest
import numpy as np

import openjij as oj
import cxxjij as cj

class UtilsTest(unittest.TestCase):

    def test_benchmark(self):
        h = {0: 1}
        J = {(0, 1):-1.0, (1,2): -1.0}

        def solver(time_param, iteration):
            sa_samp = oj.SASampler()
            sa_samp.step_num = time_param 
            sa_samp.iteration = iteration
            return sa_samp.sample_ising(h, J)

        # logger setting
        logger = getLogger('openjij')
        stream_handler = StreamHandler()
        stream_handler.setLevel(INFO)
        logger.addHandler(stream_handler)

        ground_state = [-1, -1, -1]
        ground_energy = oj.BinaryQuadraticModel(h, J).calc_energy(ground_state)
        step_num_list = np.linspace(1, 5, 5, dtype=np.int)
        bm_res = oj.benchmark([ground_state], ground_energy, solver, time_param_list=step_num_list)
        self.assertTrue(set(bm_res) >= {'time', 'error', 'e_res', 'tts', 'tts_threshold_prob'})

        self.assertEqual(len(bm_res) ,len(step_num_list))

    def test_response_converter(self):
        try:
            from dimod.sampleset import SampleSet
            import neal
        except ImportError:
            print(' skip')
            return
        
        neal_sampler = neal.SimulatedAnnealingSampler()
        Q = {(1,2):-1, (2,3):-1}
        response = neal_sampler.sample_qubo(Q)
        oj_res = oj.convert_response(response)

class CXXTest(unittest.TestCase):
    def setUp(self):
        self.N = 10
        self.dense = cj.graph.Dense(self.N)
        for i in range(self.N):
            for j in range(i+1, self.N):
                self.dense[i, j] = -1
    def test_cxx_sa(self):
        sa = cj.system.ClassicalIsing(self.dense)
        sa.simulated_annealing(beta_min=0.1, beta_max=10.0, step_length=10, step_num=10)
        ground_spins = sa.get_spins()

        sa.simulated_annealing(schedule=[[0.1, 20]])
        spins = sa.get_spins()

        self.assertNotEqual(ground_spins, spins)

    def test_cxx_sqa(self):
        # 1-d model
        one_d = cj.graph.Dense(self.N)
        for i in range(self.N):
            one_d[i, (i+1)%self.N] = -1
            one_d[i, i] = -1
        sqa = cj.system.QuantumIsing(one_d, num_trotter_slices=5)
        sqa.simulated_quantum_annealing(beta=1.0, gamma=2.0, step_length=10, step_num=10)
        ground_spins = sqa.get_spins()

        sqa.simulated_quantum_annealing(beta=1.0, gamma=2.0, schedule=[[0.5, 200]])
        spins = sqa.get_spins()

        self.assertNotEqual(ground_spins, spins)



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
        bqm = oj.ChimeraModel(h=h, J=J)
        self.assertTrue(bqm.validate_chimera(unit_num_L=3))

        J = {(0, 1): -1}
        bqm = oj.ChimeraModel(h=h, J=J)
        self.assertFalse(bqm.validate_chimera(unit_num_L=3))

    def test_ising_dict(self):
        Q = {(0,4): -1.0, (6,2): -3.0}
        bqm = oj.ChimeraModel(Q=Q, var_type='BINARY')

    def test_king_graph(self):
        h = {}
        J = {(0,1): -1.0, (1,2): -3.0}
        king_interaction = [[0,0, 1,0, -1.0], [1,0, 2,0, -3.0]]

        king_graph = oj.KingGraph(machine_type="ASIC", h=h, J=J)
        correct_mat = np.array([[0, -1, 0,],[-1, 0, -3],[0, -3, 0]])
        np.testing.assert_array_equal(king_graph.ising_interactions(), correct_mat.astype(np.float))
        np.testing.assert_array_equal(king_interaction, king_graph._ising_king_graph)

        king_graph = oj.KingGraph(machine_type="ASIC", king_graph=king_interaction)
        np.testing.assert_array_equal(king_interaction, king_graph._ising_king_graph)


        king_graph = oj.KingGraph(machine_type="ASIC", Q={(0,1): -1}, var_type="BINARY")
        king_interaction = [[0, 0, 0, 0, -0.25], [0,0,1,0,-0.25], [1,0,1,0,-0.25]]
        np.testing.assert_array_equal(king_interaction, king_graph._ising_king_graph)



class SamplerOptimizeTest(unittest.TestCase):

    def setUp(self):
        self.h = {0: 1, 1: 1, 2: 1}
        self.J = {(0,1): -1.0, (1,2): -1.0}
        self.Q = {(i,i): hi for i, hi in self.h.items()}
        self.Q.update(self.J)

    def test_sa(self):
        response = oj.SASampler(beta_max=100).sample_ising(self.h, self.J)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [-1,-1,-1])

        response = oj.SASampler(beta_max=100).sample_qubo(self.Q)
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
        model = oj.ChimeraModel(h, J, var_type='SPIN')
        chimera = gpu_sampler._chimera_graph(model, chimera_L=10)

    def test_cmos(self):
        cmos = oj.CMOSAnnealer(token="")

    # def test_gpu(self):
    #     h = {0: -1}
    #     J = {(0,4):-1,(0,5):-1,(2,5):-1}
    #     sampler=oj.GPUSQASampler(iteration=10,step_num=100)
    #     response=sampler.sample_ising(h,J,chimera_L=10)
        
if __name__ == '__main__':
    # test is currently disabled. TODO: write test!
    unittest.main()
