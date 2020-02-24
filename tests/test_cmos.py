import unittest

from openjij.sampler.cmos_annealer import _cmos_annealer_format
from openjij import CMOSAnnealer

TOKEN = ''

if len(TOKEN) > 0:

    class TestCMOS(unittest.TestCase):
        def test_cmos_format(self):
            J = {(0, 1): -1, (1, 2): -0.5}
            h = {0: -1.5, 2: -1.2}
            X, Y = 5, 5
            max_abs = 3
            result = _cmos_annealer_format(h, J, X, Y, max_abs)
            self.assertListEqual(
                result, [[0, 0, 1, 0, -2], [1, 0, 2, 0, -1],
                         [0, 0, 0, 0, -3], [2, 0, 2, 0, -2]]
            )

            # FPGA case
            X, Y = 80, 80
            h, J = {X*Y-1: -1}, {(0, 1): -1, (X*Y-2, X*Y-1): -1}
            _ = _cmos_annealer_format(h, J, X, Y, 3)

        def test_cmos_sampler(self):
            sampler = CMOSAnnealer(token=TOKEN)
            h, J = {0: 1}, {(0, 1): -2, (1, 2): -1}
            num_reads = 10
            res = sampler.sample_ising(h, J, num_reads=num_reads)
            self.assertEqual(len(res.record), num_reads)

            sampler = CMOSAnnealer(token=TOKEN, num_reads=num_reads)
            res = sampler.sample_ising(h, J, num_reads=num_reads)
            self.assertEqual(len(res.record), num_reads)

            # QUBO Test
            q = {(0, 0): -1, (0, 1): -1, (1, 2): 1}
            res = sampler.sample_qubo(q)
            self.assertEqual(res.first.sample, {0: 1, 1: 1, 2: 0})

            res = sampler.sample_ising(h, J, beta_min=0.1, beta_max=10.0)
            self.assertEqual(res.info['schedule']['beta_min'], 0.1)
            self.assertEqual(res.info['schedule']['beta_max'], 10.0)

        def _grid_test(self, X, Y, sampler):
            h, J = {X*Y-1: -1}, {(0, 1): -1, (X*Y-2, X*Y-1): -1}
            _ = sampler.sample_ising(h, J)  # validate Ising
            h, J = disconnect_in_Kings()
            with self.assertRaises(ValueError):
                _ = sampler.sample_ising(h, J)
            h, J = over_grid(X, Y)
            with self.assertRaises(ValueError):
                _ = sampler.sample_ising(h, J)

        def test_gird_validate(self):
            sampler = CMOSAnnealer(token=TOKEN)
            # check ASIC Grid 352 * 176
            self._grid_test(X=352, Y=176, sampler=sampler)

            # check FPGA Grid 80*80
            sampler = CMOSAnnealer(token=TOKEN, machine_type='FPGA')
            self._grid_test(X=80, Y=80, sampler=sampler)

        def test_parameter_validate(self):
            h, J = {0: -1}, {(0, 1): -1, (1, 2): -1}
            sampler = CMOSAnnealer(token=TOKEN)
            with self.assertRaises(ValueError):
                _ = sampler.sample_ising(h, J, num_reads=15)
            with self.assertRaises(ValueError):
                _ = sampler.sample_ising(h, J, num_sweeps=101)
            with self.assertRaises(ValueError):
                _ = sampler.sample_ising(h, J, beta_max=0)
            with self.assertRaises(ValueError):
                _ = sampler.sample_ising(h, J, beta_max=0)
            with self.assertRaises(ValueError):
                _ = sampler.sample_ising(h, J, beta_min=0)
            with self.assertRaises(ValueError):
                _ = sampler.sample_ising(h, J, step_length=1001)

        # def test_embbed(self):
        #     from dwave.system.composites import EmbeddingComposite
        #     sampler = EmbeddingComposite(CMOSAnnealer(token=TOKEN))
        #     q = {('a', 'b'): -1, ('b', 'c'): 1, ('c', 'a'): 1}
        #     res = sampler.sample_qubo(q)
        #     self.assertEqual(res.first.sample, {'a': 1, 'b': 1, 'c': 0})

    def disconnect_in_Kings():
        J = {(0, 2): -1, (0, 1): 1}
        h = {0: 1}
        return h, J

    def over_grid(X, Y):
        J = {(0, 1): -1}
        h = {X*Y: -1}
        return h, J

    if __name__ == '__main__':
        unittest.main()
