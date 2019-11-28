import unittest

import numpy as np
import openjij as oj
import cxxjij as cj


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
        spin = oj.cast_var_type('SPIN')
        self.assertEqual(spin, oj.SPIN)

        binary = oj.cast_var_type('BINARY')
        self.assertEqual(binary, oj.BINARY)


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.h = {0: 1, 1: -2}
        self.J = {(0, 1): -1, (1, 2): -3, (2, 3): 0.5}
        self.spins = [1, -1, 1, 1]

        self.Q = {(0, 0): 1, (1, 2): -1, (2, 0): -0.2, (1, 3): 3}
        self.binaries = [0, 1, 1, 0]

    def test_bqm_constructor(self):
        # Test BinaryQuadraticModel constructor
        bqm = oj.BinaryQuadraticModel(self.h, self.J)
        self.assertEqual(type(bqm.ising_interactions()), np.ndarray)

        self.assertEqual(bqm.var_type, oj.SPIN)

        dense_graph = bqm.get_cxxjij_ising_graph(sparse=False)
        self.assertTrue(isinstance(dense_graph, cj.graph.Dense))

        bqm_qubo = oj.BinaryQuadraticModel.from_qubo(self.Q)
        self.assertEqual(bqm_qubo.var_type, oj.BINARY)

    def test_interaction_matrix(self):
        bqm = oj.BinaryQuadraticModel(self.h, self.J)
        ising_matrix = np.array([
            [1, -1,  0,  0],
            [-1, -2, -3, 0],
            [0, -3, 0, 0.5],
            [0, 0, 0.5, 0]
        ])
        np.testing.assert_array_equal(
            bqm.ising_interactions(), ising_matrix
        )

        # check Hij = Jij + Jji
        J = self.J.copy()
        J[0, 1] /= 3
        J[1, 0] = J[0, 1] * 2
        bqm = oj.BinaryQuadraticModel(self.h, J)
        np.testing.assert_array_equal(
            bqm.ising_interactions(), ising_matrix
        )

    def test_bqm_calc_energy(self):
        # Test to calculate energy

        # Test Ising energy
        bqm = oj.BinaryQuadraticModel(self.h, self.J)
        ising_energy_bqm = bqm.calc_energy(self.spins)
        true_ising_e = calculate_ising_energy(self.h, self.J, self.spins)
        self.assertEqual(ising_energy_bqm, true_ising_e)

        # Test QUBO energy
        bqm = oj.BinaryQuadraticModel.from_qubo(Q=self.Q)
        qubo_energy_bqm = bqm.calc_energy(self.binaries)
        true_qubo_e = calculate_qubo_energy(self.Q, self.binaries)
        self.assertEqual(qubo_energy_bqm, true_qubo_e)

        # QUBO == Ising
        spins = [1, 1, -1, 1]
        binary = [1, 1, 0, 1]
        qubo_bqm = oj.BinaryQuadraticModel.from_qubo(Q=self.Q)
        # ising_mat = qubo_bqm.ising_interactions()
        # h, J = {}, {}
        # for i in range(len(ising_mat)-1):
        #     for j in range(i, len(ising_mat)):
        #         if i == j:
        #             h[i] = ising_mat[i][i]
        #         else:
        #             J[(i, j)] = ising_mat[i][j]

        qubo_energy = qubo_bqm.calc_energy(binary)

        self.assertEqual(qubo_energy, qubo_bqm.calc_energy(
            spins, need_to_convert_from_spin=True))

    def test_bqm(self):
        h = {}
        J = {(0, 1): -1.0, (1, 2): -3.0}
        bqm = oj.BinaryQuadraticModel(h, J)

        self.assertEqual(J, bqm.quadratic)

        self.assertEqual(type(bqm.ising_interactions()), np.ndarray)
        correct_mat = np.array([[0, -1, 0, ], [-1, 0, -3], [0, -3, 0]])
        np.testing.assert_array_equal(
            bqm.ising_interactions(), correct_mat.astype(np.float))

    def test_chimera_converter(self):
        h = {}
        J = {(0, 4): -1.0, (6, 2): -3.0, (16, 0): 4}
        chimera = oj.ChimeraModel(h, J, offset=0, unit_num_L=2)
        self.assertEqual(chimera.chimera_coordinate(
            4, unit_num_L=2), (0, 0, 4))
        self.assertEqual(chimera.chimera_coordinate(
            12, unit_num_L=2), (0, 1, 4))
        self.assertEqual(chimera.chimera_coordinate(
            16, unit_num_L=2), (1, 0, 0))

    def test_chimera(self):
        h = {}
        J = {(0, 4): -1.0, (6, 2): -3.0}
        bqm = oj.ChimeraModel(h, J, offset=0, unit_num_L=3)
        self.assertTrue(bqm.validate_chimera())

        J = {(0, 1): -1}
        bqm = oj.ChimeraModel(h, J, unit_num_L=3)
        with self.assertRaises(ValueError):
            bqm.validate_chimera()

        J = {(4, 12): -1}
        bqm = oj.ChimeraModel(h, J, unit_num_L=2)
        self.assertTrue(bqm.validate_chimera())

        J = {(0, 4): -1, (5, 13): 1, (24, 8): 2,
             (18, 20): 1, (16, 0): 0.5, (19, 23): -2}
        h = {13: 2}
        chimera = oj.ChimeraModel(h, J, unit_num_L=2)
        self.assertEqual(chimera.to_index(1, 1, 1, unit_num_L=2), 25)

        self.assertTrue(chimera.validate_chimera())

    def test_ising_dict(self):
        Q = {(0, 4): -1.0, (6, 2): -3.0}
        bqm = oj.ChimeraModel.from_qubo(Q=Q, unit_num_L=3)

    def test_king_graph(self):
        h = {}
        J = {(0, 1): -1.0, (1, 2): -3.0}
        king_interaction = [[0, 0, 1, 0, -1.0], [1, 0, 2, 0, -3.0]]

        king_graph = oj.KingGraph(machine_type="ASIC", linear=h, quadratic=J)
        correct_mat = np.array([[0, -1, 0, ], [-1, 0, -3], [0, -3, 0]])
        np.testing.assert_array_equal(
            king_graph.ising_interactions(), correct_mat.astype(np.float))
        np.testing.assert_array_equal(
            king_interaction, king_graph._ising_king_graph)

        king_graph = oj.KingGraph(
            machine_type="ASIC", king_graph=king_interaction)
        np.testing.assert_array_equal(
            king_interaction, king_graph._ising_king_graph)

        king_graph = oj.KingGraph.from_qubo(machine_type="ASIC", Q={
            (0, 1): -1})
        king_interaction = [[0, 0, 0, 0, -0.25],
                            [0, 0, 1, 0, -0.25], [1, 0, 1, 0, -0.25]]
        np.testing.assert_array_equal(
            king_interaction, king_graph._ising_king_graph)

    def test_get_chimera_graph(self):
        c_model = oj.ChimeraModel.from_qubo(
            Q={(0, 4): -1, (1, 1): -1, (1, 5): 1}, unit_num_L=2)
        chimera = c_model.get_chimera_graph()
        self.assertIsInstance(chimera, cj.graph.Chimera)

        c_model = oj.ChimeraModel.from_qubo(
            Q={((0, 0, 1), (0, 0, 4)): -1, ((0, 0, 4), (0, 0, 2)): -1},
            unit_num_L=2)
        chimera = c_model.get_chimera_graph()
        self.assertIsInstance(chimera, cj.graph.Chimera)


if __name__ == '__main__':
    unittest.main()
