import openjij as oj
import numpy as np
import dimod
import cimod
import unittest


class TestUtils(unittest.TestCase):

    def test_vartype(self):
        #self.assertTrue(dimod.BINARY == oj.BINARY)
        #self.assertTrue(dimod.SPIN == oj.SPIN)

        # latest version of OpenJij replaces dimod.xxx with cimod.xxx

        self.assertTrue(cimod.BINARY == oj.BINARY)
        self.assertTrue(cimod.SPIN == oj.SPIN)

    def test_benchmark(self):
        h = {0: 1}
        J = {(0, 1): -1.0, (1, 2): -1.0}

        def solver(time_param, *args):
            sa_samp = oj.SASampler()
            sa_samp.num_sweeps = time_param
            return sa_samp.sample_ising(h, J, num_reads=10)

        # logger setting
        ground_state = [-1, -1, -1]
        ground_energy = oj.BinaryQuadraticModel(h, J, 'SPIN').calc_energy(ground_state)
        step_num_list = np.linspace(1, 9, 9, dtype=int)
        bm_res = oj.solver_benchmark(
            solver=solver,
            time_list=step_num_list,
            solutions=[ground_state])
        self.assertTrue(
            set(bm_res) >= {'time', 'success_prob', 'residual_energy', 'tts', 'info'})
        self.assertEqual(len(bm_res), len(step_num_list))

        bench = oj.solver_benchmark(
            solver=solver,
            time_list=step_num_list,
            ref_energy=ground_energy, measure_with_energy=True)
        self.assertTrue(
            set(bench) >= {'time', 'success_prob', 'residual_energy', 'tts', 'info'})

    def test_str_key_success_prob(self):
        solutions = [{'a': 1, 'b': -1, 'c': -1}]

        response = oj.Response.from_samples(
            ([[-1, -1, 1], [1, -1, -1], [1, -1, -1]], ['c', 'b', 'a']),
            oj.SPIN, [0, 0, 0]
        )

        ps = oj.utils.success_probability(response, solutions)

        self.assertEqual(ps, 1/3)


if __name__ == '__main__':
    unittest.main()
