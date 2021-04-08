import unittest
import openjij as oj

class HUBOTest(unittest.TestCase):

    def setUp(self):
        self.J_quad = {(0,): 1, (1,): -2, (0, 1): -1, (1, 2): -3, (2, 3): 0.5}
        self.spins = {0: 1, 1: -1, 2: 1, 3: 1}

        self.Q_quad = {(0,): 1, (1, 2): -1, (2, 0): -0.2, (1, 3): 3}
        self.binaries = {0: 0, 1: 1, 2: 1, 3: 0}

    def gen_testcase_polynomial(self):
        J = {}
        J[(0, 1, 2, 3, 4)] = 0.0686616367121328
        J[(0, 2, 3, 4   )] = 0.0682112165613232
        J[(2, 3, 4      )] = -0.1763027211493039
        J[(0, 1, 3, 4   )] = -0.0907800090462850
        J[(1, 3, 4      )] = 0.1318413458843757
        J[(0, 3, 4      )] = 0.1316587643599703
        J[(3, 4         )] = 0.1460080982070779
        J[(4,           )] = -0.0171180762893237
        J[(1, 2, 3      )] = 0.0137655628870602
        J[(0, 2, 4      )] = 0.1211030013829714
        J[(1,           )] = -0.1487502208910776
        J[(0, 1, 2      )] = 0.0678984161788189
        J[(0, 1, 2, 3   )] = 0.1655848090229992
        J[(1, 2, 4      )] = -0.1628796758769616
        J[(3,           )] = 0.1742156290818721
        J[(0, 2, 3      )] = -0.1081691119002069
        J[(1, 4         )] = 0.1756511179861042
        J[(0, 1, 3      )] = 0.0098192651462946
        J[(1, 3         )] = -0.0746905947645014
        J[(0, 3         )] = 0.1385243673379363
        J[(0, 4         )] = -0.0277205719092218
        J[(0, 1, 4      )] = 0.1113556942155680
        J[(0, 2         )] = -0.0413677095349563
        J[(0, 1, 2, 4   )] = 0.0072610193576964
        J[(2,           )] = -0.1055644094807323
        J[(0, 1         )] = 0.1996162061861095
        J[(2, 3         )] = -0.0226188424784269
        J[(1, 2, 3, 4   )] = 0.0372262067253093
        J[(0,           )] = 0.1730229445472662
        J[(2, 4         )] = 0.0863882044144668
        J[(1, 2         )] = -0.0448357038957756
        J[(             )] =0.198873923292106
        true_energy = -1.3422641349549371
        return J, true_energy

    def test_SASampler_hubo(self):
        sampler = oj.SASampler()
        K, true_energy = self.gen_testcase_polynomial()
        response = sampler.sample_hubo(K, var_type="SPIN", seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])

    def test_hubo_constructor(self):
        hubo_spin = oj.BinaryPolynomialModel(self.J_quad)
        self.assertEqual(hubo_spin.vartype, oj.SPIN)

        hubo_binary = oj.BinaryPolynomialModel(self.Q_quad, "BINARY")
        self.assertEqual(hubo_binary.vartype, oj.BINARY)

    def test_zero_interaction(self):
        sampler = oj.SASampler()
        response = sampler.sample_hubo({(1,2,3):0.0})



if __name__ == '__main__':
    unittest.main()