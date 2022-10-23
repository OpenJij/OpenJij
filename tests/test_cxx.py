from logging import getLogger, StreamHandler, INFO

import unittest
import numpy as np

#import openjij as oj
import openjij.cxxjij.graph as G
import openjij.cxxjij.system as S
import openjij.cxxjij.algorithm as A
import openjij.cxxjij.utility as U
import openjij.cxxjij.result as R
from scipy import sparse


class CXXTest(unittest.TestCase):

    def setUp(self):
        self.size = 8
        #dense graph
        self.dense = G.Dense(self.size)
        self.dense = self.gen_testcase(self.dense)
        #sparse graph
        self.sparse = G.Sparse(self.size)
        self.sparse = self.gen_testcase(self.sparse)
        #chimera graph
        #Note: make sure to use ChimeraGPU (not Chimera) when using GPU since the type between FloatType and GPUFloatType is in general different.
        self.chimera = G.ChimeraGPU(2,2)
        self.chimera = self.gen_chimera_testcase(self.chimera)

        self.seed_for_spin = 1234
        self.seed_for_mc = 5678
        

    def gen_testcase(self, J):
        J[0,0]=-0.1
        J[0,1]=-0.9
        J[0,2]=0.2
        J[0,3]=0.1
        J[0,4]=1.3
        J[0,5]=0.8
        J[0,6]=0.9
        J[0,7]=0.4
        J[1,1]=-0.7
        J[1,2]=-1.6
        J[1,3]=1.5
        J[1,4]=1.5
        J[1,5]=1.2
        J[1,6]=-1.5
        J[1,7]=-1.7
        J[2,2]=-0.6
        J[2,3]=1.2
        J[2,4]=-1.3
        J[2,5]=-0.5
        J[2,6]=-1.9
        J[2,7]=1.2
        J[3,3]=0.8
        J[3,4]=-0.5
        J[3,5]=-0.4
        J[3,6]=-1.8
        J[3,7]=-2.0
        J[4,4]=0.6
        J[4,5]=-2.0
        J[4,6]=-1.9
        J[4,7]=0.5
        J[5,5]=-1.8
        J[5,6]=-1.2
        J[5,7]=1.8
        J[6,6]=0.3
        J[6,7]=1.4
        J[7,7]=1.8
        self.true_groundstate = [-1, -1, 1, 1, 1, 1, 1, -1]
        return J

    def gen_testcase_polynomial(self, J):
        J[0, 1, 2, 3, 4] = 0.0686616367121328
        J[0, 2, 3, 4] = 0.0682112165613232
        J[2, 3, 4] = -0.1763027211493039
        J[0, 1, 3, 4] = -0.0907800090462850
        J[1, 3, 4] = 0.1318413458843757
        J[0, 3, 4] = 0.1316587643599703
        J[3, 4] = 0.1460080982070779
        J[4,] = -0.0171180762893237
        J[1, 2, 3] = 0.0137655628870602
        J[0, 2, 4] = 0.1211030013829714
        J[1,] = -0.1487502208910776
        J[0, 1, 2] = 0.0678984161788189
        J[0, 1, 2, 3] = 0.1655848090229992
        J[1, 2, 4] = -0.1628796758769616
        J[3,] = 0.1742156290818721
        J[0, 2, 3] = -0.1081691119002069
        J[1, 4] = 0.1756511179861042
        J[0, 1, 3] = 0.0098192651462946
        J[1, 3] = -0.0746905947645014
        J[0, 3] = 0.1385243673379363
        J[0, 4] = -0.0277205719092218
        J[0, 1, 4] = 0.1113556942155680
        J[0, 2] = -0.0413677095349563
        J[0, 1, 2, 4] = 0.0072610193576964
        J[2,] = -0.1055644094807323
        J[0, 1] = 0.1996162061861095
        J[2, 3] = -0.0226188424784269
        J[1, 2, 3, 4] = 0.0372262067253093
        J[0,] = 0.1730229445472662
        J[2, 4] = 0.0863882044144668
        J[1, 2] = -0.0448357038957756
        J[[]]=0.198873923292106
        self.true_energy = -1.3422641349549371
        return J

    def gen_chimera_testcase(self, J):
        J[0,0,0,G.ChimeraDir.IN_0or4] = +0.25
        J[0,0,0,G.ChimeraDir.IN_1or5] = +0.25
        J[0,0,0,G.ChimeraDir.IN_2or6] = +0.25
        J[0,0,0,G.ChimeraDir.IN_3or7] = +0.25
        J[0,0,1,G.ChimeraDir.IN_0or4] = +0.25
        J[0,0,1,G.ChimeraDir.IN_1or5] = +0.25
        J[0,0,1,G.ChimeraDir.IN_2or6] = +0.25
        J[0,0,1,G.ChimeraDir.IN_3or7] = +0.25
        J[0,0,2,G.ChimeraDir.IN_0or4] = +0.25
        J[0,0,2,G.ChimeraDir.IN_1or5] = +0.25
        J[0,0,2,G.ChimeraDir.IN_2or6] = +0.25
        J[0,0,2,G.ChimeraDir.IN_3or7] = +0.25
        J[0,0,3,G.ChimeraDir.IN_0or4] = +0.25
        J[0,0,3,G.ChimeraDir.IN_1or5] = +0.25
        J[0,0,3,G.ChimeraDir.IN_2or6] = +0.25
        J[0,0,3,G.ChimeraDir.IN_3or7] = +0.25

        J[0,1,0,G.ChimeraDir.IN_0or4] = +0.25
        J[0,1,0,G.ChimeraDir.IN_1or5] = +0.25
        J[0,1,0,G.ChimeraDir.IN_2or6] = +0.25
        J[0,1,0,G.ChimeraDir.IN_3or7] = +0.25
        J[0,1,1,G.ChimeraDir.IN_0or4] = +0.25
        J[0,1,1,G.ChimeraDir.IN_1or5] = +0.25
        J[0,1,1,G.ChimeraDir.IN_2or6] = +0.25
        J[0,1,1,G.ChimeraDir.IN_3or7] = +0.25
        J[0,1,2,G.ChimeraDir.IN_0or4] = +0.25
        J[0,1,2,G.ChimeraDir.IN_1or5] = +0.25
        J[0,1,2,G.ChimeraDir.IN_2or6] = +0.25
        J[0,1,2,G.ChimeraDir.IN_3or7] = +0.25
        J[0,1,3,G.ChimeraDir.IN_0or4] = +0.25
        J[0,1,3,G.ChimeraDir.IN_1or5] = +0.25
        J[0,1,3,G.ChimeraDir.IN_2or6] = +0.25
        J[0,1,3,G.ChimeraDir.IN_3or7] = +0.25

        J[1,0,0,G.ChimeraDir.IN_0or4] = +0.25
        J[1,0,0,G.ChimeraDir.IN_1or5] = +0.25
        J[1,0,0,G.ChimeraDir.IN_2or6] = +0.25
        J[1,0,0,G.ChimeraDir.IN_3or7] = +0.25
        J[1,0,1,G.ChimeraDir.IN_0or4] = +0.25
        J[1,0,1,G.ChimeraDir.IN_1or5] = +0.25
        J[1,0,1,G.ChimeraDir.IN_2or6] = +0.25
        J[1,0,1,G.ChimeraDir.IN_3or7] = +0.25
        J[1,0,2,G.ChimeraDir.IN_0or4] = +0.25
        J[1,0,2,G.ChimeraDir.IN_1or5] = +0.25
        J[1,0,2,G.ChimeraDir.IN_2or6] = +0.25
        J[1,0,2,G.ChimeraDir.IN_3or7] = +0.25
        J[1,0,3,G.ChimeraDir.IN_0or4] = +0.25
        J[1,0,3,G.ChimeraDir.IN_1or5] = +0.25
        J[1,0,3,G.ChimeraDir.IN_2or6] = +0.25
        J[1,0,3,G.ChimeraDir.IN_3or7] = +0.25

        J[1,1,0,G.ChimeraDir.IN_0or4] = +0.25
        J[1,1,0,G.ChimeraDir.IN_1or5] = +0.25
        J[1,1,0,G.ChimeraDir.IN_2or6] = +0.25
        J[1,1,0,G.ChimeraDir.IN_3or7] = +0.25
        J[1,1,1,G.ChimeraDir.IN_0or4] = +0.25
        J[1,1,1,G.ChimeraDir.IN_1or5] = +0.25
        J[1,1,1,G.ChimeraDir.IN_2or6] = +0.25
        J[1,1,1,G.ChimeraDir.IN_3or7] = +0.25
        J[1,1,2,G.ChimeraDir.IN_0or4] = +0.25
        J[1,1,2,G.ChimeraDir.IN_1or5] = +0.25
        J[1,1,2,G.ChimeraDir.IN_2or6] = +0.25
        J[1,1,2,G.ChimeraDir.IN_3or7] = +0.25
        J[1,1,3,G.ChimeraDir.IN_0or4] = +0.25
        J[1,1,3,G.ChimeraDir.IN_1or5] = +0.25
        J[1,1,3,G.ChimeraDir.IN_2or6] = +0.25
        J[1,1,3,G.ChimeraDir.IN_3or7] = +0.2

        J[0,0,0] = +1

        J[0,0,6,G.ChimeraDir.PLUS_C] = +1
        J[0,0,3,G.ChimeraDir.PLUS_R] = -1
        J[1,0,5,G.ChimeraDir.PLUS_C] = +1

        self.true_chimera_spin = [0] * J.size()
        
        self.true_chimera_spin[J.to_ind(0,0,0)] = -1
        self.true_chimera_spin[J.to_ind(0,0,1)] = -1
        self.true_chimera_spin[J.to_ind(0,0,2)] = -1
        self.true_chimera_spin[J.to_ind(0,0,3)] = -1
        self.true_chimera_spin[J.to_ind(0,0,4)] = +1
        self.true_chimera_spin[J.to_ind(0,0,5)] = +1
        self.true_chimera_spin[J.to_ind(0,0,6)] = +1
        self.true_chimera_spin[J.to_ind(0,0,7)] = +1

        self.true_chimera_spin[J.to_ind(0,1,0)] = +1
        self.true_chimera_spin[J.to_ind(0,1,1)] = +1
        self.true_chimera_spin[J.to_ind(0,1,2)] = +1
        self.true_chimera_spin[J.to_ind(0,1,3)] = +1
        self.true_chimera_spin[J.to_ind(0,1,4)] = -1
        self.true_chimera_spin[J.to_ind(0,1,5)] = -1
        self.true_chimera_spin[J.to_ind(0,1,6)] = -1
        self.true_chimera_spin[J.to_ind(0,1,7)] = -1

        self.true_chimera_spin[J.to_ind(1,0,0)] = -1
        self.true_chimera_spin[J.to_ind(1,0,1)] = -1
        self.true_chimera_spin[J.to_ind(1,0,2)] = -1
        self.true_chimera_spin[J.to_ind(1,0,3)] = -1
        self.true_chimera_spin[J.to_ind(1,0,4)] = +1
        self.true_chimera_spin[J.to_ind(1,0,5)] = +1
        self.true_chimera_spin[J.to_ind(1,0,6)] = +1
        self.true_chimera_spin[J.to_ind(1,0,7)] = +1

        self.true_chimera_spin[J.to_ind(1,1,0)] = +1
        self.true_chimera_spin[J.to_ind(1,1,1)] = +1
        self.true_chimera_spin[J.to_ind(1,1,2)] = +1
        self.true_chimera_spin[J.to_ind(1,1,3)] = +1
        self.true_chimera_spin[J.to_ind(1,1,4)] = -1
        self.true_chimera_spin[J.to_ind(1,1,5)] = -1
        self.true_chimera_spin[J.to_ind(1,1,6)] = -1
        self.true_chimera_spin[J.to_ind(1,1,7)] = -1

        return J

    
    def test_SingleSpinFlip_ClassicalIsing_Dense(self):

        #classial ising (dense)
        system = S.make_classical_ising(self.dense.gen_spin(self.seed_for_spin), self.dense)

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 100, 100)

        #anneal
        A.Algorithm_SingleSpinFlip_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertTrue(self.true_groundstate == result_spin)

    def test_SingleSpinFlip_ClassicalIsing_Sparse(self):

        #classial ising (sparse)
        system = S.make_classical_ising(self.sparse.gen_spin(self.seed_for_spin), self.sparse)

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 100, 100)

        #anneal
        A.Algorithm_SingleSpinFlip_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertTrue(self.true_groundstate == result_spin)

    def test_SingleSpinFlip_ClassicalIsing_CSRSparse(self):

        #classial ising (csr sparse)
        csr_sparse = G.CSRSparse(sparse.csr_matrix(np.triu(self.dense.get_interactions())))

        system = S.make_classical_ising(self.sparse.gen_spin(self.seed_for_spin), csr_sparse)

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 100, 100)

        #anneal
        A.Algorithm_SingleSpinFlip_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertTrue(self.true_groundstate == result_spin)

    def test_SingleSpinFlip_Polynomial_Spin(self):
        system_size = 5
        self.polynomial = G.Polynomial(system_size)
        self.polynomial = self.gen_testcase_polynomial(self.polynomial)
        #classial ising (Polynomial)
        system = S.make_classical_ising_polynomial(self.polynomial.gen_spin(), self.polynomial, "SPIN")

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 200, 200)

        #anneal
        A.Algorithm_SingleSpinFlip_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = system.variables

        #compare
        self.assertAlmostEqual(self.true_energy, self.polynomial.calc_energy(result_spin))

    def test_SingleSpinFlip_Polynomial_Binary(self):
        system_size = 5
        self.polynomial = G.Polynomial(system_size)
        self.polynomial[0]   = +1
        self.polynomial[0,1] = -1
        self.polynomial[0,2] = +1.5
        self.polynomial[0,3] = -1.6
        self.polynomial[0,4] = -1.7
        self.polynomial[1,3] = +2.3
        self.polynomial[1,4] = -0.3
        self.polynomial[2,3] = +3.4
        self.polynomial[2,4] = +3.7
        self.polynomial[3,4] = -0.8
        self.polynomial[0,1,2] = -0.5
        self.polynomial[1,2,3] = -1.0
        self.polynomial[2,3,4] = +0.9
        #classial ising (Polynomial)
        system = S.make_classical_ising_polynomial(self.polynomial.gen_binary(), self.polynomial, "BINARY")

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 200, 200)

        #anneal
        A.Algorithm_SingleSpinFlip_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = system.variables

        #compare
        self.assertAlmostEqual(-3.1, self.polynomial.calc_energy(result_spin))

    def test_SingleSpinFlip_KLocal_1(self):
        system_size = 5
        self.polynomial = G.Polynomial(system_size)
        self.polynomial[0]   = +1
        self.polynomial[0,1] = -1
        self.polynomial[0,2] = +1.5
        self.polynomial[0,3] = -1.6
        self.polynomial[0,4] = -1.7
        self.polynomial[1,3] = +2.3
        self.polynomial[1,4] = -0.3
        self.polynomial[2,3] = +3.4
        self.polynomial[2,4] = +3.7
        self.polynomial[3,4] = -0.8
        self.polynomial[0,1,2] = -0.5
        self.polynomial[1,2,3] = -1.0
        self.polynomial[2,3,4] = +0.9
        #classial ising (Polynomial)
        system = S.make_k_local_polynomial(self.polynomial.gen_binary(), self.polynomial)

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 200, 200)

        #anneal
        A.Algorithm_KLocal_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertAlmostEqual(-3.1, self.polynomial.calc_energy(result_spin))
    
    def test_SingleSpinFlip_KLocal_2(self):
        system_size = 30
        self.polynomial = G.Polynomial(system_size)
        self.polynomial[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] = -1
        #classial ising (Polynomial)
        system = S.make_k_local_polynomial(self.polynomial.gen_binary(), self.polynomial)

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 200, 200)

        #anneal
        A.Algorithm_KLocal_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertAlmostEqual(-1, self.polynomial.calc_energy(result_spin))

    def test_SingleSpinFlip_KLocal_3(self):
        system_size = 30
        self.polynomial = G.Polynomial(system_size)
        self.polynomial[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] = +1
        self.polynomial[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,   17,18,19,20,21,22,23,24,25,26,27,28,29] = -1
        #classial ising (Polynomial)
        system = S.make_k_local_polynomial(self.polynomial.gen_binary(), self.polynomial)

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 200, 200)

        #anneal
        A.Algorithm_KLocal_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertAlmostEqual(-1, self.polynomial.calc_energy(result_spin))

    def test_SingleSpinFlip_KLocal_4(self):
        system_size = 30
        self.polynomial = G.Polynomial(system_size)
        self.polynomial[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] = -1
        self.polynomial[0,1,2,3,4,5,6,7,8,  10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] = +1
        self.polynomial[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,   24,25,26,27,28,29] = +1
        self.polynomial[0,1,2,3,4,5,6,7,8,  10,11,12,13,14,15,16,17,18,19,20,21,22,   24,25,26,27,28,29] = -1
        #classial ising (Polynomial)
        system = S.make_k_local_polynomial(self.polynomial.gen_binary(), self.polynomial)

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 200, 200)

        #anneal
        A.Algorithm_KLocal_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertAlmostEqual(-1, self.polynomial.calc_energy(result_spin))
    
    def test_SingleSpinFlip_TransverseIsing_Dense(self):

        #transverse ising (dense)
        system = S.make_transverse_ising(self.dense.gen_spin(self.seed_for_spin), self.dense, 1.0, 10)

        #schedulelist
        schedule_list = U.make_transverse_field_schedule_list(10, 100, 100)

        #anneal
        A.Algorithm_SingleSpinFlip_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertTrue(self.true_groundstate == result_spin)

    def test_SingleSpinFlip_TransverseIsing_Sparse(self):

        #classial ising (sparse)
        system = S.make_transverse_ising(self.sparse.gen_spin(self.seed_for_spin), self.sparse, 1.0, 10)

        #schedulelist
        schedule_list = U.make_transverse_field_schedule_list(10, 100, 100)

        #anneal
        A.Algorithm_SingleSpinFlip_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertTrue(self.true_groundstate == result_spin)

    def test_SingleSpinFlip_TransverseIsing_CSRSparse(self):
        #classial ising (csr sparse)
        csr_sparse = G.CSRSparse(sparse.csr_matrix(np.triu(self.dense.get_interactions())))

        system = S.make_transverse_ising(self.sparse.gen_spin(self.seed_for_spin), csr_sparse, 1.0, 10)

        #schedulelist
        schedule_list = U.make_transverse_field_schedule_list(10, 100, 100)

        #anneal
        A.Algorithm_SingleSpinFlip_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertTrue(self.true_groundstate == result_spin)

    def test_SwendsenWang_ClassicalIsing_Sparse(self):

        #classial ising (sparse)
        system = S.make_classical_ising(self.sparse.gen_spin(self.seed_for_spin), self.sparse)

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 100, 2000)

        #anneal
        A.Algorithm_SwendsenWang_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertTrue(self.true_groundstate == result_spin)

    def test_SwendsenWang_ClassicalIsing_CSRSparse(self):
        #classial ising (csr sparse)
        csr_sparse = G.CSRSparse(sparse.csr_matrix(np.triu(self.dense.get_interactions())))

        system = S.make_classical_ising(self.sparse.gen_spin(self.seed_for_spin), csr_sparse)

        #schedulelist
        schedule_list = U.make_classical_schedule_list(0.1, 100.0, 100, 2000)

        #anneal
        A.Algorithm_SwendsenWang_run(system, self.seed_for_mc, schedule_list)

        #result spin
        result_spin = R.get_solution(system)

        #compare
        self.assertTrue(self.true_groundstate == result_spin)

    # currently disabled

    #def test_ContinuousTimeSwendsenWang_ContinuousTimeIsing_Sparse(self):

    #    #classial ising (sparse)
    #    system = S.make_continuous_time_ising(self.sparse.gen_spin(self.seed_for_spin), self.sparse, 1.0)

    #    #schedulelist (TODO: why is it so hard?)
    #    schedule_list = U.make_transverse_field_schedule_list(10, 500, 3000)

    #    #anneal
    #    A.Algorithm_ContinuousTimeSwendsenWang_run(system, self.seed_for_mc, schedule_list)

    #    #result spin
    #    result_spin = R.get_solution(system)

    #    #compare
    #    self.assertTrue(self.true_groundstate == result_spin)

# GPU Test is currently disabled.

    # def test_GPU_ChimeraTransverseGPU(self):

    #    #classial ising (sparse)
    #    system = S.make_chimera_transverse_gpu(self.chimera.gen_spin(self.seed_for_spin), self.chimera, 1.0, 10)

    #    #schedulelist
    #    schedule_list = U.make_transverse_field_schedule_list(10, 100, 100)

    #    #anneal
    #    A.Algorithm_GPU_run(system, self.seed_for_mc, schedule_list)

    #    #result spin
    #    result_spin = R.get_solution(system)

    #    #compare
    #    self.assertTrue(self.true_chimera_spin == result_spin)

    # def test_GPU_ChimeraClassicalGPU(self):

    #    #classial ising (sparse)
    #    system = S.make_chimera_classical_gpu(self.chimera.gen_spin(self.seed_for_spin), self.chimera)

    #    #schedulelist
    #    schedule_list = U.make_classical_schedule_list(0.1, 100.0, 100, 100)

    #    #anneal
    #    A.Algorithm_GPU_run(system, self.seed_for_mc, schedule_list)

    #    #result spin
    #    result_spin = R.get_solution(system)

    #    #compare
    #    self.assertTrue(self.true_chimera_spin == result_spin)

#class UtilsTest(unittest.TestCase):
#
#    def test_benchmark(self):
#        h = {0: 1}
#        J = {(0, 1):-1.0, (1,2): -1.0}
#
#        def solver(time_param, iteration):
#            sa_samp = oj.SASampler()
#            sa_samp.step_num = time_param 
#            sa_samp.iteration = iteration
#            return sa_samp.sample_ising(h, J)
#
#        # logger setting
#        logger = getLogger('openjij')
#        stream_handler = StreamHandler()
#        stream_handler.setLevel(INFO)
#        logger.addHandler(stream_handler)
#
#        ground_state = [-1, -1, -1]
#        ground_energy = oj.BinaryQuadraticModel(h, J).calc_energy(ground_state)
#        step_num_list = np.linspace(1, 5, 5, dtype=np.int)
#        bm_res = oj.benchmark([ground_state], ground_energy, solver, time_param_list=step_num_list)
#        self.assertTrue(set(bm_res) >= {'time', 'error', 'e_res', 'tts', 'tts_threshold_prob'})
#
#        self.assertEqual(len(bm_res) ,len(step_num_list))
#
#    def test_response_converter(self):
#        try:
#            from dimod.sampleset import SampleSet
#            import neal
#        except ImportError:
#            print(' skip')
#            return
#        
#        neal_sampler = neal.SimulatedAnnealingSampler()
#        Q = {(1,2):-1, (2,3):-1}
#        response = neal_sampler.sample_qubo(Q)
#        oj_res = oj.convert_response(response)
#
#class CXXTest(unittest.TestCase):
#    def setUp(self):
#        self.N = 10
#        self.dense = cj.graph.Dense(self.N)
#        for i in range(self.N):
#            for j in range(i+1, self.N):
#                self.dense[i, j] = -1
#    def test_cxx_sa(self):
#        sa = cj.system.ClassicalIsing(self.dense)
#        sa.simulated_annealing(beta_min=0.1, beta_max=10.0, step_length=10, step_num=10)
#        ground_spins = sa.get_spins()
#
#        sa.simulated_annealing(schedule=[[0.01, 20]])
#        spins = sa.get_spins()
#
#        self.assertNotEqual(ground_spins, spins)
#
#    def test_cxx_sqa(self):
#        # 1-d model
#        one_d = cj.graph.Dense(self.N)
#        for i in range(self.N):
#            one_d[i, (i+1)%self.N] = -1
#            one_d[i, i] = -1
#        sqa = cj.system.QuantumIsing(one_d, num_trotter_slices=5)
#        sqa.simulated_quantum_annealing(beta=1.0, gamma=2.0, step_length=10, step_num=10)
#        ground_spins = sqa.get_spins()
#
#        sqa.simulated_quantum_annealing(beta=1.0, gamma=2.0, schedule=[[0.5, 200]])
#        spins = sqa.get_spins()
#
#        self.assertNotEqual(ground_spins, spins)
#
#
#
#class ModelTest(unittest.TestCase):
#    def test_bqm(self):
#        h = {}
#        J = {(0,1): -1.0, (1,2): -3.0}
#        bqm = oj.BinaryQuadraticModel(h=h, J=J)
#
#        self.assertEqual(type(bqm.ising_interactions()), np.ndarray)
#        correct_mat = np.array([[0, -1, 0,],[-1, 0, -3],[0, -3, 0]])
#        np.testing.assert_array_equal(bqm.ising_interactions(), correct_mat.astype(np.float))
#
#    def test_chimera_converter(self):
#        h = {}
#        J = {(0,4): -1.0, (6,2): -3.0, (16, 0): 4}
#        chimera = oj.ChimeraModel(h=h, J=J, unit_num_L=2)
#        self.assertEqual(chimera.chimera_coordinate(4, unit_num_L=2), (0,0,4))
#        self.assertEqual(chimera.chimera_coordinate(12, unit_num_L=2), (0,1,4))
#        self.assertEqual(chimera.chimera_coordinate(16, unit_num_L=2), (1,0,0))
#        
#
#    def test_chimera(self):
#        h = {}
#        J = {(0,4): -1.0, (6,2): -3.0}
#        bqm = oj.ChimeraModel(h=h, J=J, unit_num_L=3)
#        self.assertTrue(bqm.validate_chimera())
#
#        J = {(0, 1): -1}
#        bqm = oj.ChimeraModel(h=h, J=J, unit_num_L=3)
#        with self.assertRaises(ValueError):
#            bqm.validate_chimera()
#        
#        J = {(4, 12): -1}
#        bqm = oj.ChimeraModel(h=h, J=J, unit_num_L=2)
#        self.assertTrue(bqm.validate_chimera())
#
#        J = {(0,4): -1, (5, 13):1, (24, 8):2, (18,20): 1, (16,0):0.5, (19, 23): -2}
#        h = {13: 2}
#        chimera = oj.ChimeraModel(h, J, unit_num_L=2)
#        self.assertEqual(chimera.to_index(1,1,1, unit_num_L=2), 25)
#
#        self.assertTrue(chimera.validate_chimera())
#
#
#
#    def test_ising_dict(self):
#        Q = {(0,4): -1.0, (6,2): -3.0}
#        bqm = oj.ChimeraModel(Q=Q, vartype='BINARY', unit_num_L=3)
#
#    def test_king_graph(self):
#        h = {}
#        J = {(0,1): -1.0, (1,2): -3.0}
#        king_interaction = [[0,0, 1,0, -1.0], [1,0, 2,0, -3.0]]
#
#        king_graph = oj.KingGraph(machine_type="ASIC", h=h, J=J)
#        correct_mat = np.array([[0, -1, 0,],[-1, 0, -3],[0, -3, 0]])
#        np.testing.assert_array_equal(king_graph.ising_interactions(), correct_mat.astype(np.float))
#        np.testing.assert_array_equal(king_interaction, king_graph._ising_king_graph)
#
#        king_graph = oj.KingGraph(machine_type="ASIC", king_graph=king_interaction)
#        np.testing.assert_array_equal(king_interaction, king_graph._ising_king_graph)
#
#
#        king_graph = oj.KingGraph(machine_type="ASIC", Q={(0,1): -1}, vartype="BINARY")
#        king_interaction = [[0, 0, 0, 0, -0.25], [0,0,1,0,-0.25], [1,0,1,0,-0.25]]
#        np.testing.assert_array_equal(king_interaction, king_graph._ising_king_graph)
#
#class TestChimeraGraph(unittest.TestCase):
#    def full_chimera_qubo(self, L):
#
#        left_side = [0,1,2,3]
#        right_side = [4,5,6,7]
#        to_ind = lambda r,c,i: 8*L*r + 8*c + i
#        Q = {}
#        # Set to -1 for all bonds in each chimera unit
#        for c in range(L):
#            for r in range(L):
#                for z_l in left_side:
#                    for z_r in right_side:
#                        Q[to_ind(r,c,z_l), to_ind(r,c,z_r)] = -1
#
#                        # linear term
#                        Q[to_ind(r,c,z_l), to_ind(r,c,z_l)] = -1
#                    #linear term
#                    Q[to_ind(r,c,z_r), to_ind(r,c,z_r)] = -1
#
#        # connect all chimera unit
#        # column direction
#        for c in range(L-1):
#            for r in range(L):
#                for z_r in right_side:
#                    Q[to_ind(r,c,z_r), to_ind(r,c+1,z_r)] = +0.49
#        # row direction
#        for r in range(L-1):
#            for c in range(L):
#                for z_l in left_side:
#                    Q[to_ind(r,c,z_l), to_ind(r+1,c,z_l)] = 0.49
#        return Q
#
#    def full_chimera_ising(self, L):
#        Q = self.full_chimera_qubo(L)
#        h, J = {}, {}
#        for (i, j), value in Q.items():
#            if i == j:
#                h[i] = value
#            else:
#                J[i, j] = value
#        return h, J
#
#    def test_chimera_validate(self):
#        L = 4
#        Q = self.full_chimera_qubo(L=L)
#        chimera = oj.ChimeraModel(Q=Q, unit_num_L=L, vartype='BINARY')
#
#        self.assertTrue(chimera._validate((0,0,0),(0,0,4),L))
#        self.assertFalse(chimera._validate((0,0,0),(96,0,0),L))
#
#
#
#    def test_chimera_connect(self):
#        Q = self.full_chimera_qubo(L=2)
#        chimera = oj.ChimeraModel(Q=Q, unit_num_L=2, vartype='BINARY')
#        self.assertTrue(chimera.validate_chimera())
#
#        Q = self.full_chimera_qubo(L=4)
#        chimera = oj.ChimeraModel(Q=Q, unit_num_L=4, vartype='BINARY')
#        self.assertTrue(chimera.validate_chimera())
#
#
if __name__ == '__main__':
    unittest.main()



