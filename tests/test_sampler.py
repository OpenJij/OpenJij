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
        # the following interactions has
        # the ground state [-1, -1, -1]
        # and ground energy -18
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
        _J = self.J.copy()
        _J[(1, 2)] = 1
        sampler = oj.SASampler(iteration=10)
        response = sampler.sample_ising(
            h=self.h, J=_J,
            initial_state=initial_state,
            seed=1
        )
        unique_energy = np.unique(response.energies)
        self.assertEqual(len(unique_energy), 1)

    def test_openjij_cxxjij_compare(self):
        seed_for_mc = 1
        Q = {
            (0, 0): 1, (1, 1): -1, (2, 2): 2,
            (0, 1): 1, (1, 2): -1, (2, 0): -1
        }
        # solution is [0, 1, 0]

        init_binary = [1, 0, 1]
        init_spin = [1, -1, 1]

        # openjij
        sampler = oj.SASampler(
            beta_min=0.01, beta_max=10,
            step_length=10, step_num=100
        )
        res = sampler.sample_qubo(
            Q=Q, initial_state=init_binary,
            seed=seed_for_mc
        )

        # cxxjij
        model = oj.BinaryQuadraticModel.from_qubo(Q=Q)
        graph = model.get_cxxjij_ising_graph()
        system = cj.system.make_classical_ising_Eigen(init_spin, graph)
        sch = cj.utility.make_classical_schedule_list(
            beta_min=0.01, beta_max=10,
            one_mc_step=10, num_call_updater=100
        )
        cj.algorithm.Algorithm_SingleSpinFlip_run(
            system, seed_for_mc, sch
        )

        self.assertListEqual(
            res.states[0], list((system.spin[:-1]+1)/2)
        )

    def test_sa(self):
        initial_state = [1 for _ in range(self.size)]

        response = oj.SASampler().sample_ising(
            self.h, self.J, initial_state=initial_state, seed=1)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [-1, -1, -1])

        response = oj.SASampler(beta_max=100).sample_qubo(self.Q, seed=1)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [0, 0, 0])

        vaild_sche = [(beta, 1) for beta in np.linspace(-1, 1, 5)]
        with self.assertRaises(ValueError):
            sampler = oj.SASampler(schedule=vaild_sche)
            sampler.sample_ising({}, {})

    def test_sa_sweeps(self):
        iteration = 10
        sampler = oj.SASampler()
        res = sampler.sample_ising(self.h, self.J, num_reads=iteration)
        self.assertEqual(iteration, len(res.energies))

        sampler = oj.SASampler(num_reads=iteration)
        res = sampler.sample_ising(self.h, self.J)
        self.assertEqual(iteration, len(res.energies))

    def test_swendsenwang(self):
        sampler = oj.SASampler()
        initial_state = [1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1]
        h = {0: -1, 10: -1}
        J = {(i, i+1): -1 for i in range(10)}
        res = sampler.sample_ising(h, J,
                                   updater="swendsenwang",
                                   seed=1, initial_state=initial_state)
        self.assertListEqual(res.states[0], [1]*11)

    def test_time_sa(self):
        fast_res = oj.SASampler(beta_max=100, num_sweeps=5,
                                iteration=10).sample_ising(self.h, self.J)
        slow_res = oj.SASampler(beta_max=100, num_sweeps=100,
                                iteration=10).sample_ising(self.h, self.J)

        self.assertEqual(len(fast_res.info['list_exec_times']), 10)
        self.assertTrue(fast_res.info['execution_time']
                        < slow_res.info['execution_time'])

    def test_sqa_response(self):
        iteration = 10
        trotter = 4
        sampler = oj.SQASampler(iteration=iteration, trotter=trotter)
        response = sampler.sample_ising(h=self.h, J=self.J)

        self.assertEqual(len(response.states), iteration)
        self.assertEqual(len(response.q_states), iteration)
        self.assertEqual(len(response.q_states[0]), trotter)
        self.assertTrue(isinstance(
            response.q_states[0][0][0], (int, np.int, np.int64)))

    def test_sqa(self):
        response = oj.SQASampler().sample_ising(self.h, self.J)
        self.assertEqual(len(response.states), 1)
        self.assertEqual(response.var_type, oj.SPIN)
        self.assertListEqual(response.states[0], [-1, -1, -1])
        self.assertEqual(response.energies[0], -18)

        response = oj.SQASampler().sample_qubo(self.Q)
        self.assertEqual(len(response.states), 1)
        self.assertListEqual(response.states[0], [0, 0, 0])

        schedule = [(s, 10) for s in np.arange(0, 1, 5)] + [(0.99, 100)]
        response = oj.SQASampler(schedule=schedule).sample_qubo(self.Q, seed=1)
        self.assertListEqual(response.states[0], [0, 0, 0])

        vaild_sche = [(s, 10) for s in np.linspace(0, 1.1, 5)]
        with self.assertRaises(ValueError):
            sampler = oj.SQASampler()
            _ = sampler.sample_ising({}, {}, schedule=vaild_sche)

    def test_time_sqa(self):
        fast_res = oj.SQASampler(
            num_sweeps=10, iteration=10).sample_ising(self.h, self.J, seed=1)
        slow_res = oj.SQASampler(
            num_sweeps=100, iteration=10).sample_ising(self.h, self.J, seed=1)

        self.assertEqual(len(fast_res.info['list_exec_times']), 10)
        self.assertTrue(fast_res.info['execution_time']
                        < slow_res.info['execution_time'])

    def test_reverse_annealing(self):
        seed_for_mc = 1
        initial_state = [0, 0, 0]
        qubo = {
            (0, 0): 1, (1, 1): -1, (2, 2): 2,
            (0, 1): 1, (1, 2): -1, (2, 0): -1
        }
        # solution is [0, 1, 0]
        solution = [0, 1, 0]

        # Reverse simulated annealing
        # beta, step_length
        reverse_schedule = [
            [10, 3], [1, 3], [0.5, 3], [1, 3], [10, 5]
        ]
        rsa_sampler = oj.SASampler(schedule=reverse_schedule, iteration=10)
        res = rsa_sampler.sample_qubo(
            qubo, initial_state=initial_state, seed=seed_for_mc)
        self.assertListEqual(
            solution,
            list(res.min_samples['states'][0])
        )

        # Reverse simulated quantum annealing
        # annealing parameter s, step_length
        reverse_schedule = [
            [1, 1], [0.3, 3], [0.1, 5], [0.3, 3], [1, 3]
        ]
        rqa_sampler = oj.SQASampler(schedule=reverse_schedule, iteration=10)
        res = rqa_sampler.sample_qubo(
            qubo, initial_state=initial_state, seed=seed_for_mc)
        self.assertListEqual(
            solution,
            list(res.min_samples['states'][0])
        )

    def test_hubo_sampler(self):
        sampler = oj.SASampler()
        h = {0: -1}
        J = {(0, 1): -1}
        K = {(0, 1, 2): 1}
        response = sampler.sample_hubo([h, J, K], var_type='SPIN')
        print(response.info)
        self.assertListEqual([1, 1, -1], list(response.states[0]))

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
