import openjij as oj
import numpy as np
import unittest



class TestUtils(unittest.TestCase):
    def test_benchmark(self):
        h = {0: 1}
        J = {(0, 1):-1.0, (1,2): -1.0}

        def solver(time_param, *args):
            sa_samp = oj.SASampler()
            sa_samp.step_num = time_param 
            sa_samp.iteration = 10
            return sa_samp.sample_ising(h, J)

        # logger setting
        ground_state = [-1, -1, -1]
        ground_energy = oj.BinaryQuadraticModel(h, J).calc_energy(ground_state)
        step_num_list = np.linspace(1, 5, 5, dtype=np.int)
        bm_res = oj.solver_benchmark(
            solver=solver,
            time_list=step_num_list,
            solutions=[ground_state])
        self.assertTrue(set(bm_res) >= {'time', 'success_prob', 'residual_energy', 'tts', 'info'})
        self.assertEqual(len(bm_res) ,len(step_num_list))

        bench = oj.solver_benchmark(
            solver=solver,
            time_list=step_num_list,
            ref_energy=ground_energy, measure_with_energy=True)
        self.assertTrue(set(bench) >= {'time', 'success_prob', 'residual_energy', 'tts', 'info'})

        print(bench)



if __name__ == '__main__':
    unittest.main()