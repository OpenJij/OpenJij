import unittest

import numpy as np
import openjij as oj
import openjij.cxxjij as cj


def calculate_ising_energy(h, J, spins):
    energy = 0.0
    for (i, j), Jij in J.items():
        energy += Jij*spins[i]*spins[j]
    for i, hi in h.items():
        energy += hi * spins[i]
    return energy


def calculate_qubo_energy(Q, binary):
    energy = 0.0
    for (i, j), Qij in Q.items():
        energy += Qij*binary[i]*binary[j]
    return energy


class VariableTypeTest(unittest.TestCase):
    def test_variable_type(self):
        spin = oj.cast_vartype('SPIN')
        self.assertEqual(spin, oj.SPIN)

        binary = oj.cast_vartype('BINARY')
        self.assertEqual(binary, oj.BINARY)


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.h = {0: 1, 1: -2}
        self.J = {(0, 1): -1, (1, 2): -3, (2, 3): 0.5}
        self.spins = {0: 1, 1: -1, 2: 1, 3: 1}

        self.Q = {(0, 0): 1, (1, 2): -1, (2, 0): -0.2, (1, 3): 3}
        self.binaries = {0: 0, 1: 1, 2: 1, 3: 0}

    def test_bqm_constructor(self):
        # Test BinaryQuadraticModel constructor
        bqm = oj.BinaryQuadraticModel(self.h, self.J, 'SPIN', sparse=False)
        self.assertEqual(type(bqm.interaction_matrix()), np.ndarray)

        self.assertEqual(bqm.vartype, oj.SPIN)

        dense_graph,offset = bqm.get_cxxjij_ising_graph()
        self.assertTrue(isinstance(dense_graph, cj.graph.Dense))
        self.assertEqual(offset, 0)

        bqm_qubo = oj.BinaryQuadraticModel.from_qubo(Q=self.Q)
        self.assertEqual(bqm_qubo.vartype, oj.BINARY)

    def test_interaction_matrix(self):
        bqm = oj.BinaryQuadraticModel(self.h, self.J, 'SPIN')
        ising_matrix = np.array([
            [0, -1,  0,  0, 1],
            [0, 0, -3, 0, -2],
            [0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ])
        np.testing.assert_array_equal(
            bqm.interaction_matrix(), ising_matrix
        )

    def test_transfer_to_cxxjij(self):
        bqm = oj.BinaryQuadraticModel(self.h, self.J, 'SPIN', sparse=False)
        # to Dense
        ising_graph,offset = bqm.get_cxxjij_ising_graph()
        self.assertEqual(ising_graph.size(), len(bqm.variables))
        for i in range(len(bqm.variables)+1):
            for j in range(i+1, len(bqm.variables)+1):
                    self.assertAlmostEqual(bqm.interaction_matrix()[i,j], ising_graph.get_interactions()[i, j])
                    self.assertAlmostEqual(ising_graph.get_interactions()[i, j], ising_graph.get_interactions()[j, i])

        #with offset
        bqm_qubo = oj.BinaryQuadraticModel.from_qubo(Q=self.Q)
        ising_graph,offset = bqm_qubo.get_cxxjij_ising_graph()
        bqm = bqm_qubo.change_vartype('SPIN', inplace=False)
        self.assertEqual(ising_graph.size(), len(bqm.variables))
        for i in range(len(bqm.variables)+1):
            for j in range(i+1, len(bqm.variables)+1):
                    self.assertAlmostEqual(bqm.interaction_matrix()[i,j], ising_graph.get_interactions()[i, j])
                    self.assertAlmostEqual(ising_graph.get_interactions()[i, j], ising_graph.get_interactions()[j, i])

        self.assertAlmostEqual(offset, bqm.offset)

        # to Sparse
        bqm = oj.BinaryQuadraticModel(self.h, self.J, 'SPIN', sparse=True)
        ising_graph,offset = bqm.get_cxxjij_ising_graph()
        self.assertEqual(ising_graph.size(), len(bqm.variables))
        for i in range(len(bqm.variables)+1):
            for j in range(i+1, len(bqm.variables)+1):
                    self.assertAlmostEqual(bqm.interaction_matrix()[i,j], ising_graph.get_interactions()[i, j])
                    self.assertAlmostEqual(ising_graph.get_interactions()[i, j], ising_graph.get_interactions()[j, i])


    def test_bqm_calc_energy(self):
        # Test to calculate energy

        # Test Ising energy
        bqm = oj.BinaryQuadraticModel(self.h, self.J, 'SPIN')
        ising_energy_bqm = bqm.energy(self.spins)
        true_ising_e = calculate_ising_energy(self.h, self.J, self.spins)
        self.assertEqual(ising_energy_bqm, true_ising_e)

        # Test QUBO energy
        bqm = oj.BinaryQuadraticModel.from_qubo(Q=self.Q)
        qubo_energy_bqm = bqm.energy(self.binaries)
        true_qubo_e = calculate_qubo_energy(self.Q, self.binaries)
        self.assertEqual(qubo_energy_bqm, true_qubo_e)

        # QUBO == Ising
        spins = {0: 1, 1: 1, 2: -1, 3: 1}
        binary = {0: 1, 1: 1, 2: 0, 3: 1}
        qubo_bqm = oj.BinaryQuadraticModel.from_qubo(Q=self.Q)
        # ising_mat = qubo_bqm.ising_interactions()
        # h, J = {}, {}
        # for i in range(len(ising_mat)-1):
        #     for j in range(i, len(ising_mat)):
        #         if i == j:
        #             h[i] = ising_mat[i][i]
        #         else:
        #             J[(i, j)] = ising_mat[i][j]

        qubo_energy = qubo_bqm.energy(binary)
        qubo_bqm.change_vartype('SPIN')

        self.assertEqual(qubo_energy, qubo_bqm.energy(spins))

    def test_energy_consistency(self):
        bqm = oj.BinaryQuadraticModel(self.h, self.J, vartype='SPIN', sparse=False)
        dense_ising_graph,offset = bqm.get_cxxjij_ising_graph()
        bqm = oj.BinaryQuadraticModel(self.h, self.J, vartype='SPIN', sparse=True)
        sparse_ising_graph,offset = bqm.get_cxxjij_ising_graph()
        spins = {0: -1, 1: -1, 2: -1, 3: -1}
        self.assertAlmostEqual(dense_ising_graph.calc_energy([spins[i] for i in range(len(spins))]), bqm.energy(spins))
        self.assertAlmostEqual(sparse_ising_graph.calc_energy([spins[i] for i in range(len(spins))]), bqm.energy(spins))

    def test_bqm(self):
        h = {}
        J = {(0, 1): -1.0, (1, 2): -3.0}
        bqm = oj.BinaryQuadraticModel(h, J, 'SPIN')
        
        self.assertEqual(J, bqm.get_quadratic())

        self.assertEqual(type(bqm.interaction_matrix()), np.ndarray)
        correct_mat = np.array([[0, -1, 0, 0], [0, 0, -3, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_array_equal(
            bqm.interaction_matrix(), correct_mat.astype(float))

if __name__ == '__main__':
    unittest.main()
