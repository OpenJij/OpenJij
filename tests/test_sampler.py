import unittest
import numpy as np

import cxxjij as cj
import openjij as oj


class VarTypeTest(unittest.TestCase):
    def test_vartype(self):
        self.assertTrue(isinstance(oj.SPIN, oj.VariableType))
        self.assertTrue(isinstance(oj.BINARY, oj.VariableType))

    def test_cast(self):
        spin_var = oj.cast_var_type('SPIN')
        self.assertEqual(spin_var, oj.SPIN)
        binary_var = oj.cast_var_type('BINARY')
        self.assertEqual(binary_var, oj.BINARY)


class SamplerOptimizeTest(unittest.TestCase):

    def setUp(self):
        self.h = {0: 5, 1: 5, 2: 5}
        self.J = {(0, 1): -1.0, (1, 2): -1.0, (2, 0): -1.0}
        self.Q = {(i, i): hi for i, hi in self.h.items()}
        self.Q.update({(0, 1): 1.0, (1, 2): 1.0, (2, 0): 1.0})

        indices = set(self.h.keys())
        for i, j in self.J.keys():
            indices.add(i)
            indices.add(j)

        self.size = len(indices)

    def test_seed(self):
        initial_state = [1 for _ in range(self.size)]
        sampler = oj.SASampler(iteration=10)
        response = sampler.sample_ising(
            h=self.h, J=self.J,
            initial_state=initial_state
        )
        print('unique ', np.unique(response.energies))
        print('energy ', response.energies[0])
        print('state ', response.states[0])

        model = oj.BinaryQuadraticModel(h=self.h, J=self.J)
        schedule = cj.utility.make_classical_schedule_list(
            0.1, 5.0, 10, 100)
        graph = model.get_cxxjij_ising_graph()
        system = cj.system.make_classical_ising_Eigen(
            graph.gen_spin(), graph)

        seed = None
        algorithm = cj.algorithm.Algorithm_SingleSpinFlip_run
        if seed is None:
            def simulated_annealing(system): return algorithm(
                system, schedule)
        else:
            def simulated_annealing(system): return algorithm(
                system, seed, schedule)

        for _ in range(10):
            system.spin = initial_state + [1]
            simulated_annealing(system)
            print('spin ', system.spin)
            print(cj.result.get_solution(system))

    # def test_sa(self):
    #     initial_state = [1 for _ in range(self.size)]

    #     response = oj.SASampler().sample_ising(
    #         self.h, self.J, initial_state=initial_state, seed=1)
    #     print('energy ', response.energies[0])
    #     self.assertEqual(len(response.states), 1)
    #     self.assertListEqual(response.states[0], [-1, -1, -1])

    #     response = oj.SASampler(beta_max=100).sample_qubo(self.Q)
    #     self.assertEqual(len(response.states), 1)
    #     self.assertListEqual(response.states[0], [0, 0, 0])

    #     valid_sche = [(beta, 1) for beta in np.linspace(-1, 1, 5)]
    #     with self.assertRaises(ValueError
        # print('unique', np.unique(response.energies))):
    #         sampler = oj.SASampler(schedule=valid_sche)

    # def test_time_sa(self):
    #     fast_res = oj.SASampler(beta_max=100, step_num=10,
    #                             iteration=10).sample_ising(self.h, self.J)
    #     slow_res = oj.SASampler(beta_max=100, step_num=50,
    #                             iteration=10).sample_ising(self.h, self.J)

    #     self.assertEqual(len(fast_res.info['list_exec_times']), 10)
    #     self.assertTrue(fast_res.info['execution_time']
    #                     < slow_res.info['execution_time'])

    # def test_sqa(self):
    #     response = oj.SQASampler().sample_ising(self.h, self.J)
    #     self.assertEqual(len(response.states), 1)
    #     self.assertListEqual(response.states[0], [-1, -1, -1])
    #     self.assertEqual(response.energies[0], -18)

    #     response = oj.SQASampler().sample_qubo(self.Q)
    #     self.assertEqual(len(response.states), 1)
    #     self.assertListEqual(response.states[0], [0, 0, 0])

    #     schedule = [(s, 10) for s in np.arange(0, 1, 5)] + [(0.99, 100)]
    #     response = oj.SQASampler(schedule=schedule).sample_qubo(self.Q)
    #     self.assertListEqual(response.states[0], [0, 0, 0])

    #     vaild_sche = [(s, 10) for s in np.linspace(0, 1, 5)]
    #     with self.assertRaises(ValueError):
    #         sampler = oj.SQASampler(schedule=vaild_sche)

    # def test_time_sqa(self):
    #     fast_res = oj.SQASampler(
    #         step_num=10, iteration=10).sample_ising(self.h, self.J)
    #     slow_res = oj.SQASampler(
    #         step_num=50, iteration=10).sample_ising(self.h, self.J)

    #     self.assertEqual(len(fast_res.info['list_exec_times']), 10)
    #     self.assertTrue(fast_res.info['execution_time']
    #                     < slow_res.info['execution_time'])

    # def test_gpu_sqa(self):
    #     gpu_sampler = oj.GPUSQASampler()
    #     h = {0: -1}
    #     J = {(0, 4): -1, (0, 5): -1, (2, 5): -2, (4, 12): 0.5, (16, 0): 2}
    #     model = oj.ChimeraModel(h, J, var_type='SPIN', unit_num_L=3)

    #     model = oj.ChimeraModel(h, J, var_type='SPIN', unit_num_L=2)
    #     chimera = model.get_chimera_graph()

    #     self.assertEqual(chimera[0, 0, 0], h[0])
    #     self.assertEqual(
    #         chimera[0, 0, 0, cj.graph.ChimeraDir.IN_0or4], J[0, 4])
    #     self.assertEqual(
    #         chimera[0, 0, 0, cj.graph.ChimeraDir.IN_1or5], J[0, 5])
    #     self.assertEqual(
    #         chimera[0, 0, 2, cj.graph.ChimeraDir.IN_1or5], J[2, 5])
    #     self.assertEqual(
    #         chimera[0, 0, 4, cj.graph.ChimeraDir.PLUS_C], J[4, 12])
    #     self.assertEqual(
    #         chimera[1, 0, 0, cj.graph.ChimeraDir.MINUS_R], J[16, 0])

    #     # should satisfy symmetry
    #     self.assertEqual(chimera[1, 0, 0, cj.graph.ChimeraDir.MINUS_R],
    #                      chimera[0, 0, 0, cj.graph.ChimeraDir.PLUS_R])

    # def test_cmos(self):
    #     cmos = oj.CMOSAnnealer(token="")

    # def test_gpu(self):
    #     h = {0: -1}
    #     J = {(0,4):-1,(0,5):-1,(2,5):-1}
    #     sampler=oj.GPUSQASampler(iteration=10,step_num=100)
    #     response=sampler.sample_ising(h,J,unit_num_L=10)
        #


if __name__ == '__main__':
    unittest.main()
