import unittest

import numpy as np
import openjij as oj


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.h = {0: 1}
        self.J = {(0, 1): -1, (1, 2): -3}

    def test_bqm_construct(self):
        # Test BinaryQuadraticModel constructor
        bqm = oj.BinaryQuadraticModel(h=self.h, J=self.J)
        self.assertEqual(type(bqm.ising_interactions()), np.ndarray)

    def test_bqm(self):
        h = {}
        J = {(0, 1): -1.0, (1, 2): -3.0}
        bqm = oj.BinaryQuadraticModel(h=h, J=J)

        self.assertEqual(type(bqm.ising_interactions()), np.ndarray)
        correct_mat = np.array([[0, -1, 0, ], [-1, 0, -3], [0, -3, 0]])
        np.testing.assert_array_equal(
            bqm.ising_interactions(), correct_mat.astype(np.float))

    def test_chimera_converter(self):
        h = {}
        J = {(0, 4): -1.0, (6, 2): -3.0, (16, 0): 4}
        chimera = oj.ChimeraModel(h=h, J=J, unit_num_L=2)
        self.assertEqual(chimera.chimera_coordinate(
            4, unit_num_L=2), (0, 0, 4))
        self.assertEqual(chimera.chimera_coordinate(
            12, unit_num_L=2), (0, 1, 4))
        self.assertEqual(chimera.chimera_coordinate(
            16, unit_num_L=2), (1, 0, 0))

    def test_chimera(self):
        h = {}
        J = {(0, 4): -1.0, (6, 2): -3.0}
        bqm = oj.ChimeraModel(h=h, J=J, unit_num_L=3)
        self.assertTrue(bqm.validate_chimera())

        J = {(0, 1): -1}
        bqm = oj.ChimeraModel(h=h, J=J, unit_num_L=3)
        with self.assertRaises(ValueError):
            bqm.validate_chimera()

        J = {(4, 12): -1}
        bqm = oj.ChimeraModel(h=h, J=J, unit_num_L=2)
        self.assertTrue(bqm.validate_chimera())

        J = {(0, 4): -1, (5, 13): 1, (24, 8): 2,
             (18, 20): 1, (16, 0): 0.5, (19, 23): -2}
        h = {13: 2}
        chimera = oj.ChimeraModel(h, J, unit_num_L=2)
        self.assertEqual(chimera.to_index(1, 1, 1, unit_num_L=2), 25)

        self.assertTrue(chimera.validate_chimera())

    def test_ising_dict(self):
        Q = {(0, 4): -1.0, (6, 2): -3.0}
        bqm = oj.ChimeraModel(Q=Q, var_type='BINARY', unit_num_L=3)

    def test_king_graph(self):
        h = {}
        J = {(0, 1): -1.0, (1, 2): -3.0}
        king_interaction = [[0, 0, 1, 0, -1.0], [1, 0, 2, 0, -3.0]]

        king_graph = oj.KingGraph(machine_type="ASIC", h=h, J=J)
        correct_mat = np.array([[0, -1, 0, ], [-1, 0, -3], [0, -3, 0]])
        np.testing.assert_array_equal(
            king_graph.ising_interactions(), correct_mat.astype(np.float))
        np.testing.assert_array_equal(
            king_interaction, king_graph._ising_king_graph)

        king_graph = oj.KingGraph(
            machine_type="ASIC", king_graph=king_interaction)
        np.testing.assert_array_equal(
            king_interaction, king_graph._ising_king_graph)

        king_graph = oj.KingGraph(machine_type="ASIC", Q={
                                  (0, 1): -1}, var_type="BINARY")
        king_interaction = [[0, 0, 0, 0, -0.25],
                            [0, 0, 1, 0, -0.25], [1, 0, 1, 0, -0.25]]
        np.testing.assert_array_equal(
            king_interaction, king_graph._ising_king_graph)


if __name__ == '__main__':
    unittest.main()
