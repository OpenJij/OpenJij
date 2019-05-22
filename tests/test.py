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

        sa.simulated_annealing(schedule=[[0.01, 20]])
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

    def test_chimera_converter(self):
        h = {}
        J = {(0,4): -1.0, (6,2): -3.0, (16, 0): 4}
        chimera = oj.ChimeraModel(h=h, J=J, unit_num_L=2)
        self.assertEqual(chimera.chimera_coordinate(4, unit_num_L=2), (0,0,4))
        self.assertEqual(chimera.chimera_coordinate(12, unit_num_L=2), (0,1,4))
        self.assertEqual(chimera.chimera_coordinate(16, unit_num_L=2), (1,0,0))
        

    def test_chimera(self):
        h = {}
        J = {(0,4): -1.0, (6,2): -3.0}
        bqm = oj.ChimeraModel(h=h, J=J, unit_num_L=3)
        self.assertTrue(bqm.validate_chimera())

        J = {(0, 1): -1}
        bqm = oj.ChimeraModel(h=h, J=J, unit_num_L=3)
        with self.assertRaises(ValueError):
            bqm.validate_chimera()
        
        J = {(4, 12): -1}
        bqm = oj.ChimeraModel(h=h, J=J, unit_num_L=2)
        self.assertTrue(bqm.validate_chimera())

        J = {(0,4): -1, (5, 13):1, (24, 8):2, (18,20): 1, (16,0):0.5, (19, 23): -2}
        h = {13: 2}
        chimera = oj.ChimeraModel(h, J, unit_num_L=2)
        self.assertEqual(chimera.to_index(1,1,1, unit_num_L=2), 25)

        self.assertTrue(chimera.validate_chimera())

    def test_chimera_graph(self):
        L = 2
        to_ind = lambda r,c,i: 8*L*r + 8*c + i

        left_side = [0,1,2,3]
        right_side = [4,5,6,7]

        Q = {}
        # Set to -1 for all bonds in each chimera unit
        for c in range(L):
            for r in range(L):
                for z_l in left_side:
                    for z_r in right_side:
                        Q[to_ind(r,c,z_l), to_ind(r,c,z_r)] = -1

                        # linear term
                        Q[to_ind(r,c,z_l), to_ind(r,c,z_l)] = -1
                    #linear term
                    Q[to_ind(r,c,z_r), to_ind(r,c,z_r)] = -1

        # connect all chimera unit
        # column direction
        for c in range(L-1):
            for r in range(L):
                for z_r in right_side:
                    Q[to_ind(r,c,z_r), to_ind(r,c+1,z_r)] = +0.49
        # row direction
        for r in range(L-1):
            for c in range(L):
                for z_l in left_side:
                    Q[to_ind(r,c,z_l), to_ind(r+1,c,z_l)] = 0.49

        chimera = oj.ChimeraModel(Q=Q, unit_num_L=2, var_type='BINARY')
        self.assertTrue(chimera.validate_chimera())




    def test_ising_dict(self):
        Q = {(0,4): -1.0, (6,2): -3.0}
        bqm = oj.ChimeraModel(Q=Q, var_type='BINARY', unit_num_L=3)

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
        self.h = {0: 5, 1: 5, 2: 5}
        self.J = {(0,1): -1.0, (1,2): -1.0, (2, 0): -1.0}
        self.Q = {(i,i): hi for i, hi in self.h.items()}
        self.Q.update({(0,1): 1.0, (1,2): 1.0, (2, 0): 1.0})

    def test_sa(self):
        response = oj.SASampler(beta_max=100).sample_ising(self.h, self.J)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [-1,-1,-1])

        response = oj.SASampler(beta_max=100).sample_qubo(self.Q)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [0,0,0])

        valid_sche = [(beta, 1) for beta in np.linspace(-1, 1, 5)]
        with self.assertRaises(ValueError):
            sampler = oj.SASampler(schedule=valid_sche)

    def test_time_sa(self):
        fast_res = oj.SASampler(beta_max=100, step_num=10, iteration=10).sample_ising(self.h, self.J)
        slow_res = oj.SASampler(beta_max=100, step_num=50, iteration=10).sample_ising(self.h, self.J)

        self.assertEqual(len(fast_res.info['list_exec_times']), 10)
        self.assertTrue(fast_res.info['execution_time'] < slow_res.info['execution_time'])



    def test_sqa(self):
        response = oj.SQASampler().sample_ising(self.h, self.J)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [-1,-1,-1])
        self.assertEqual(response.energies[0], -18)

        response = oj.SQASampler().sample_qubo(self.Q)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [0,0,0])

        schedule = [(s, 10) for s in np.arange(0, 1, 5)] + [(0.99, 100)]
        response = oj.SQASampler(schedule=schedule).sample_qubo(self.Q)
        self.assertListEqual(response.states[0], [0,0,0])

        vaild_sche = [(s, 10) for s in np.linspace(0, 1, 5)]
        with self.assertRaises(ValueError):
            sampler = oj.SQASampler(schedule=vaild_sche)

    def test_time_sqa(self):
        fast_res = oj.SQASampler(step_num=10, iteration=10).sample_ising(self.h, self.J)
        slow_res = oj.SQASampler(step_num=50, iteration=10).sample_ising(self.h, self.J)

        self.assertEqual(len(fast_res.info['list_exec_times']), 10)
        self.assertTrue(fast_res.info['execution_time'] < slow_res.info['execution_time'])



    def test_gpu_sqa(self):
        gpu_sampler = oj.GPUSQASampler()
        h = {0: -1}
        J = {(0, 4): -1, (0, 5): -1, (2, 5): -2, (4, 12): 0.5, (16, 0): 2}
        model = oj.ChimeraModel(h, J, var_type='SPIN', unit_num_L=3)


        model = oj.ChimeraModel(h, J, var_type='SPIN', unit_num_L=2)
        chimera = model.get_chimera_graph() 

        self.assertEqual(chimera[0,0,0], h[0])
        self.assertEqual(chimera[0,0,0,cj.graph.ChimeraDir.IN_0or4], J[0, 4])
        self.assertEqual(chimera[0,0,0,cj.graph.ChimeraDir.IN_1or5], J[0, 5])
        self.assertEqual(chimera[0,0,2,cj.graph.ChimeraDir.IN_1or5], J[2, 5])
        self.assertEqual(chimera[0,0,4,cj.graph.ChimeraDir.PLUS_C], J[4, 12])
        self.assertEqual(chimera[1,0,0,cj.graph.ChimeraDir.MINUS_R], J[16, 0])

        # should satisfy symmetry
        self.assertEqual(chimera[1,0,0,cj.graph.ChimeraDir.MINUS_R], chimera[0,0,0,cj.graph.ChimeraDir.PLUS_R])


    def test_cmos(self):
        cmos = oj.CMOSAnnealer(token="")

    # def test_gpu(self):
    #     h = {0: -1}
    #     J = {(0,4):-1,(0,5):-1,(2,5):-1}
    #     sampler=oj.GPUSQASampler(iteration=10,step_num=100)
    #     response=sampler.sample_ising(h,J,unit_num_L=10)
        
if __name__ == '__main__':
    # test is currently disabled. TODO: write test!
    unittest.main()
