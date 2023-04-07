import unittest
import random
import openjij as oj
import cimod

def calculate_bpm_energy(polynomial, variables):
    energy = 0.0
    for (index, val) in polynomial.items():
        temp = 1
        for site in index:
            temp *= variables[site]
        energy += temp*val
    return energy

class HUBOTest(unittest.TestCase):

    def setUp(self):
        self.J_quad = {(0,): 1, (1,): -2, (0, 1): -1, (1, 2): -3, (2, 3): 0.5}
        self.spins = {0: 1, 1: -1, 2: 1, 3: 1}

        self.Q_quad = {(0,): 1, (1, 2): -1, (2, 0): -0.2, (1, 3): 3}
        self.binaries = {0: 0, 1: 1, 2: 1, 3: 0}

    def gen_testcase_polynomial(self):
        J = {}
        J[(0, 1, 2, 3, 4)] = 100
        J[(             )] = 50
        true_energy = -50
        return J, true_energy

    def test_SASampler_hubo_spin_1(self):
        K, true_energy = self.gen_testcase_polynomial()

        update_method_list = ["METROPOLIS", "HEAT_BATH"]
        random_number_engine_list = ["XORSHIFT", "MT", "MT_64"]
        temperature_schedule_list = ["GEOMETRIC", "LINEAR"]

        for update_method in update_method_list:
            for temperature_schedule in temperature_schedule_list:
                for random_number_engine in random_number_engine_list:
                    response = oj.SASampler().sample_hubo(
                        K, 
                        vartype="SPIN", 
                        seed=3,
                        updater=update_method,
                        temperature_schedule=temperature_schedule,
                        random_number_engine=random_number_engine)
                    self.assertAlmostEqual(true_energy, response.energies[0])

        bpm_oj = oj.BinaryPolynomialModel(K, "SPIN")
        response = oj.SASampler().sample_hubo(bpm_oj, seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])

        bpm_ci = cimod.BinaryPolynomialModel(K, "SPIN")
        response = oj.SASampler().sample_hubo(bpm_ci, seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])

    def test_SASampler_hubo_spin_2(self):
        K = {}
        K[0,]  = +1
        K[0,1] = -1
        K[0,2] = +1.5
        K[0,3] = -1.6
        K[0,4] = -1.7
        K[1,3] = +2.3
        K[1,4] = -0.3
        K[2,3] = +3.4
        K[2,4] = +3.7
        K[3,4] = -0.8
        K[0,1,2] = -0.5
        K[1,2,3] = -1.0
        K[2,3,4] = +0.9
        true_energy = -15.1

        update_method_list = ["METROPOLIS", "HEAT_BATH"]
        random_number_engine_list = ["XORSHIFT", "MT", "MT_64"]
        temperature_schedule_list = ["GEOMETRIC", "LINEAR"]

        for update_method in update_method_list:
            for temperature_schedule in temperature_schedule_list:
                for random_number_engine in random_number_engine_list:
                    response = oj.SASampler().sample_hubo(
                        K, 
                        vartype="SPIN", 
                        seed=1,
                        num_reads=10,
                        updater=update_method,
                        temperature_schedule=temperature_schedule,
                        random_number_engine=random_number_engine)
                    self.assertAlmostEqual(true_energy, min(response.energies))

        bpm_oj = oj.BinaryPolynomialModel(K, "SPIN")
        response = oj.SASampler().sample_hubo(bpm_oj, seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])

        bpm_ci = cimod.BinaryPolynomialModel(K, "SPIN")
        response = oj.SASampler().sample_hubo(bpm_ci, seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])
    
    def test_SASampler_hubo_binary_1(self):
        K = {}
        K[0,]  = +1
        K[0,1] = -1
        K[0,2] = +1.5
        K[0,3] = -1.6
        K[0,4] = -1.7
        K[1,3] = +2.3
        K[1,4] = -0.3
        K[2,3] = +3.4
        K[2,4] = +3.7
        K[3,4] = -0.8
        K[0,1,2] = -0.5
        K[1,2,3] = -1.0
        K[2,3,4] = +0.9
        true_energy = -3.1

        update_method_list = ["METROPOLIS", "HEAT_BATH"]
        random_number_engine_list = ["XORSHIFT", "MT", "MT_64"]
        temperature_schedule_list = ["GEOMETRIC", "LINEAR"]

        for update_method in update_method_list:
            for temperature_schedule in temperature_schedule_list:
                for random_number_engine in random_number_engine_list:
                    response = oj.SASampler().sample_hubo(
                        K, 
                        vartype="BINARY", 
                        seed = 3,
                        updater=update_method,
                        temperature_schedule=temperature_schedule,
                        random_number_engine=random_number_engine)
                    self.assertAlmostEqual(true_energy, response.energies[0])
                    
        response = oj.SASampler().sample_hubo(K, vartype="BINARY", updater = "single spin flip", seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])
        response = oj.SASampler().sample_hubo(K, vartype="BINARY", updater = "k-local", seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])

        bpm_oj = oj.BinaryPolynomialModel(K, "BINARY")
        response = oj.SASampler().sample_hubo(bpm_oj, updater = "single spin flip", seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])
        response = oj.SASampler().sample_hubo(bpm_oj, updater = "k-local", seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])
        response = oj.SASampler().sample_hubo(bpm_oj, seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])

        bpm_ci = cimod.BinaryPolynomialModel(K, "BINARY")
        response = oj.SASampler().sample_hubo(bpm_ci, updater = "single spin flip", seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])
        response = oj.SASampler().sample_hubo(bpm_ci, updater = "k-local", seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])
        response = oj.SASampler().sample_hubo(bpm_ci, seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])
    
    def test_SASampler_hubo_binary_2(self):
        K = {}
        K[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] = -1
        true_energy = -1
        response = oj.SASampler().sample_hubo(K, vartype="BINARY", seed = 3, updater="k-local")
        self.assertAlmostEqual(true_energy, response.energies[0])

        bpm_oj = oj.BinaryPolynomialModel(K, "BINARY")
        response = oj.SASampler().sample_hubo(bpm_oj, seed = 3, updater="k-local")
        self.assertAlmostEqual(true_energy, response.energies[0])

        bpm_ci = cimod.BinaryPolynomialModel(K, "BINARY")
        response = oj.SASampler().sample_hubo(bpm_ci, seed = 3, updater="k-local")
        self.assertAlmostEqual(true_energy, response.energies[0])

    def test_SASampler_hubo_binary_3(self):
        K = {}
        K[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] = +1
        K[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,   17,18,19,20,21,22,23,24,25,26,27,28,29] = -1
        true_energy = -1

        response = oj.SASampler().sample_hubo(K, vartype="BINARY", seed = 3, updater="k-local")
        self.assertAlmostEqual(true_energy, response.energies[0])

        bpm_oj = oj.BinaryPolynomialModel(K, "BINARY")
        response = oj.SASampler().sample_hubo(bpm_oj, seed = 3, updater="k-local")
        self.assertAlmostEqual(true_energy, response.energies[0])

        bpm_ci = cimod.BinaryPolynomialModel(K, "BINARY")
        response = oj.SASampler().sample_hubo(bpm_ci, seed = 3, updater="k-local")
        self.assertAlmostEqual(true_energy, response.energies[0])
    
    def test_SASampler_hubo_binary_4(self):
        K = {}
        K[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] = -1
        K[0,1,2,3,4,5,6,7,8,  10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29] = +1
        K[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,   24,25,26,27,28,29] = +1
        K[0,1,2,3,4,5,6,7,8,  10,11,12,13,14,15,16,17,18,19,20,21,22,   24,25,26,27,28,29] = -1
        true_energy = -1
        response = oj.SASampler().sample_hubo(K, vartype="BINARY", seed = 3, updater="k-local")
        self.assertAlmostEqual(true_energy, response.energies[0])  

        bpm_oj = oj.BinaryPolynomialModel(K, "BINARY")
        response = oj.SASampler().sample_hubo(bpm_oj, seed = 3, updater="k-local")
        self.assertAlmostEqual(true_energy, response.energies[0])

        bpm_ci = cimod.BinaryPolynomialModel(K, "BINARY")
        response = oj.SASampler().sample_hubo(bpm_ci, seed = 3, updater="k-local")
        self.assertAlmostEqual(true_energy, response.energies[0])
    
    def test_hubo_constructor(self):
        hubo_spin = oj.BinaryPolynomialModel(self.J_quad, oj.SPIN)
        self.assertEqual(hubo_spin.vartype, oj.SPIN)

        hubo_binary = oj.BinaryPolynomialModel(self.Q_quad, "BINARY")
        self.assertEqual(hubo_binary.vartype, oj.BINARY)

    def test_zero_interaction(self):
        sampler = oj.SASampler()
        response = sampler.sample_hubo({(1,2,3):0.0, (1,2):1}, "SPIN")
    

#BinaryPolynomialModel
class PolynomialModelTest(unittest.TestCase):
    def setUp(self):
        self.poly     = {(1,):1.0, (3,):3.0, (1,2):12.0, (1,3):13.0, (2,3,4):234.0, (3,5):35.0}
        self.spins    = {1:+1, 2:-1, 3:+1, 4:-1, 5:+1} 
        self.binaries = {1: 1, 2: 0, 3: 1, 4: 0, 5: 1}

        self.poly_str     = {("a",):1.0, ("c",):3.0, ("a","b"):12.0, ("a","c"):13.0, ("b","c","d"):234.0, ("c","e"):35.0}
        self.spins_str    = {"a":+1, "b":-1, "c":+1, "d":-1, "e":+1} 
        self.binaries_str = {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1}

        self.poly_tuple2     = {((1,1),):1.0, ((3,3),):3.0, ((1,1),(2,2)):12.0, ((1,1),(3,3)):13.0, ((2,2),(3,3),(4,4)):234.0, ((3,3),(5,5)):35.0}
        self.spins_tuple2    = {(1,1):+1, (2,2):-1, (3,3):+1, (4,4):-1, (5,5):+1} 
        self.binaries_tuple2 = {(1,1): 1, (2,2): 0, (3,3): 1, (4,4): 0, (5,5): 1}

        self.poly_tuple3     = {((1,1,1),):1.0, ((3,3,3),):3.0, ((1,1,1),(2,2,2)):12.0, ((1,1,1),(3,3,3)):13.0, ((2,2,2),(3,3,3),(4,4,4)):234.0, ((3,3,3),(5,5,5)):35.0}
        self.spins_tuple3    = {(1,1,1):+1, (2,2,2):-1, (3,3,3):+1, (4,4,4):-1, (5,5,5):+1} 
        self.binaries_tuple3 = {(1,1,1): 1, (2,2,2): 0, (3,3,3): 1, (4,4,4): 0, (5,5,5): 1}

        self.poly_tuple4     = {((1,1,1,1),):1.0, ((3,3,3,3),):3.0, ((1,1,1,1),(2,2,2,2)):12.0, ((1,1,1,1),(3,3,3,3)):13.0, ((2,2,2,2),(3,3,3,3),(4,4,4,4)):234.0, ((3,3,3,3),(5,5,5,5)):35.0}
        self.spins_tuple4    = {(1,1,1,1):+1, (2,2,2,2):-1, (3,3,3,3):+1, (4,4,4,4):-1, (5,5,5,5):+1} 
        self.binaries_tuple4 = {(1,1,1,1): 1, (2,2,2,2): 0, (3,3,3,3): 1, (4,4,4,4): 0, (5,5,5,5): 1}

    def state_test_bpm(self, bpm, poly: dict, vartype):
        self.assertEqual(bpm.vartype, vartype) #Check Vartype
        self.assertEqual(bpm.num_interactions, len(poly)) #Check the number of the interactions
        self.assertEqual(bpm.num_variables, len(set(j for i in poly.keys() for j in i))) #Check the number of the variables
        self.assertEqual(bpm.degree, max([len(i) for i in poly.keys()])) #Check the max degree of the interactions
        self.assertEqual(bpm.get_variables(), sorted(list(set(j for i in poly.keys() for j in i)))) #Check the variables
        self.assertAlmostEqual(bpm.get_offset(), poly[()] if tuple() in poly else 0.0) #Check the offset
        for k, v in bpm.get_polynomial().items():#Check the interactions
            self.assertAlmostEqual(v, poly[k])

        num = 0
        for i in sorted(list(set(j for i in poly.keys() for j in i))):
            self.assertEqual(bpm.get_variables_to_integers(i), num)
            self.assertEqual(bpm.has_variable(i), True)
            num += 1
        
         #Check the specific interactions 
        for index in poly.keys():
            self.assertAlmostEqual(bpm.get_polynomial(index)                           , poly[index])
            self.assertAlmostEqual(bpm.get_polynomial(random.sample(index, len(index))), poly[index])
            self.assertAlmostEqual(bpm.get_polynomial(list(index))                     , poly[index])
            self.assertAlmostEqual(bpm.get_polynomial(key = index)                     , poly[index])
            if tuple(index) != ():
                self.assertAlmostEqual(bpm.get_polynomial(*index), poly[index])
            else:
                self.assertAlmostEqual(bpm.get_polynomial(index), poly[index])

    def state_test_bpm_empty(self, bpm, vartype):
        self.assertEqual(bpm.vartype, vartype)
        self.assertEqual(bpm.num_interactions, 0)
        self.assertEqual(bpm.num_variables, 0)
        self.assertEqual(bpm.degree, 0)
        self.assertEqual(bpm.get_polynomial(), {})
        self.assertEqual(bpm.get_variables_to_integers(), {})
        self.assertEqual(bpm.get_variables(), [])
        self.assertAlmostEqual(bpm.get_offset(), 0.0)

    # Test BinaryPolynomialModel constructor
    def test_construction_bpm(self):
        self.state_test_bpm(oj.BinaryPolynomialModel(self.poly       , oj.SPIN), self.poly       , oj.SPIN)
        self.state_test_bpm(oj.BinaryPolynomialModel(self.poly_str   , oj.SPIN), self.poly_str   , oj.SPIN)
        self.state_test_bpm(oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN), self.poly_tuple2, oj.SPIN)
        self.state_test_bpm(oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN), self.poly_tuple3, oj.SPIN)
        self.state_test_bpm(oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN), self.poly_tuple4, oj.SPIN)

        self.state_test_bpm(oj.BinaryPolynomialModel(list(self.poly.keys())       , list(self.poly.values())       , oj.SPIN), self.poly       , oj.SPIN)
        self.state_test_bpm(oj.BinaryPolynomialModel(list(self.poly_str.keys())   , list(self.poly_str.values())   , oj.SPIN), self.poly_str   , oj.SPIN)
        self.state_test_bpm(oj.BinaryPolynomialModel(list(self.poly_tuple2.keys()), list(self.poly_tuple2.values()), oj.SPIN), self.poly_tuple2, oj.SPIN)
        self.state_test_bpm(oj.BinaryPolynomialModel(list(self.poly_tuple3.keys()), list(self.poly_tuple3.values()), oj.SPIN), self.poly_tuple3, oj.SPIN)
        self.state_test_bpm(oj.BinaryPolynomialModel(list(self.poly_tuple4.keys()), list(self.poly_tuple4.values()), oj.SPIN), self.poly_tuple4, oj.SPIN)

        bpm = oj.BinaryPolynomialModel({}, "SPIN")
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel([], [], "SPIN")
        self.state_test_bpm_empty(bpm, oj.SPIN)

    def test_add_interaction_bpm_basic(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm.add_interaction((-12345, -2, 897654321), 0.1234567)
        self.poly[tuple(sorted([-12345, -2, 897654321]))] = 0.1234567
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm.add_interaction(("åß∂ƒ©", "あいうえお", "ABCD"), -123)
        self.poly_str[tuple(sorted(["åß∂ƒ©", "あいうえお", "ABCD"]))] = -123
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm.add_interaction(((-11, 11), (22, 22)), -123)
        self.poly_tuple2[tuple(sorted([(-11, 11), (22, 22)]))] = -123
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm.add_interaction(((-11, 11, -321), (22, 22, -321)), -123)
        self.poly_tuple3[tuple(sorted([(-11, 11, -321), (22, 22, -321)]))] = -123
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm.add_interaction(((-11, 11, -321, 0), (22, 22, -321, 0)), -123)
        self.poly_tuple4[tuple(sorted([(-11, 11, -321, 0), (22, 22, -321, 0)]))] = -123
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_add_interaction_bpm_duplicate_value_1(self):
        bpm = oj.BinaryPolynomialModel({k: v/2 for k, v in self.poly.items()}, oj.SPIN)
        for k, v in self.poly.items():
            bpm.add_interaction(k, v/2)
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel({k: v/2 for k, v in self.poly_str.items()}, oj.SPIN)
        for k, v in self.poly_str.items():
            bpm.add_interaction(k, v/2)
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel({k: v/2 for k, v in self.poly_tuple2.items()}, oj.SPIN)
        for k, v in self.poly_tuple2.items():
            bpm.add_interaction(k, v/2)
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel({k: v/2 for k, v in self.poly_tuple3.items()}, oj.SPIN)
        for k, v in self.poly_tuple3.items():
            bpm.add_interaction(k, v/2)
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel({k: v/2 for k, v in self.poly_tuple4.items()}, oj.SPIN)
        for k, v in self.poly_tuple4.items():
            bpm.add_interaction(k, v/2)
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_add_interaction_bpm_duplicate_value_2(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        for k, v in self.poly.items():
            bpm.add_interaction(k, -v)
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        for k, v in self.poly_str.items():
            bpm.add_interaction(k, -v)
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        for k, v in self.poly_tuple2.items():
            bpm.add_interaction(k, -v)
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        for k, v in self.poly_tuple3.items():
            bpm.add_interaction(k, -v)
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        for k, v in self.poly_tuple4.items():
            bpm.add_interaction(k, -v)
        self.state_test_bpm_empty(bpm, oj.SPIN)


    def test_add_interactions_from_bpm_dict(self):
        bpm = oj.BinaryPolynomialModel(self.poly, "SPIN").empty("SPIN")
        bpm.add_interactions_from(self.poly)
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, "SPIN").empty("SPIN")
        bpm.add_interactions_from(self.poly_str)
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, "SPIN").empty("SPIN")
        bpm.add_interactions_from(self.poly_tuple2)
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, "SPIN").empty("SPIN")
        bpm.add_interactions_from(self.poly_tuple3)
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, "SPIN").empty("SPIN")
        bpm.add_interactions_from(self.poly_tuple4)
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_add_interactions_from_bpm_keyvalues(self):
        bpm = oj.BinaryPolynomialModel(self.poly, "SPIN").empty("SPIN")
        bpm.add_interactions_from(list(self.poly.keys()), list(self.poly.values()))
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, "SPIN").empty("SPIN")
        bpm.add_interactions_from(list(self.poly_str.keys()), list(self.poly_str.values()))
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, "SPIN").empty("SPIN")
        bpm.add_interactions_from(list(self.poly_tuple2.keys()), list(self.poly_tuple2.values()))
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, "SPIN").empty("SPIN")
        bpm.add_interactions_from(list(self.poly_tuple3.keys()), list(self.poly_tuple3.values()))
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, "SPIN").empty("SPIN")
        bpm.add_interactions_from(list(self.poly_tuple4.keys()), list(self.poly_tuple4.values()))
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_add_offset_bpm(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm.add_offset(3.0)
        self.assertAlmostEqual(bpm.get_offset(), 3.0)
        self.assertAlmostEqual(bpm.get_polynomial(()), 3.0)
        bpm.add_interaction((), 3.0)
        self.assertAlmostEqual(bpm.get_offset(), 6.0)
        self.assertAlmostEqual(bpm.get_polynomial(()), 6.0)
        bpm.add_offset(-6.0)
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm.add_offset(3.0)
        self.assertAlmostEqual(bpm.get_offset(), 3.0)
        self.assertAlmostEqual(bpm.get_polynomial(()), 3.0)
        bpm.add_interaction((), 3.0)
        self.assertAlmostEqual(bpm.get_offset(), 6.0)
        self.assertAlmostEqual(bpm.get_polynomial(()), 6.0)
        bpm.add_offset(-6.0)
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm.add_offset(3.0)
        self.assertAlmostEqual(bpm.get_offset(), 3.0)
        self.assertAlmostEqual(bpm.get_polynomial(()), 3.0)
        bpm.add_interaction((), 3.0)
        self.assertAlmostEqual(bpm.get_offset(), 6.0)
        self.assertAlmostEqual(bpm.get_polynomial(()), 6.0)
        bpm.add_offset(-6.0)
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm.add_offset(3.0)
        self.assertAlmostEqual(bpm.get_offset(), 3.0)
        self.assertAlmostEqual(bpm.get_polynomial(()), 3.0)
        bpm.add_interaction((), 3.0)
        self.assertAlmostEqual(bpm.get_offset(), 6.0)
        self.assertAlmostEqual(bpm.get_polynomial(()), 6.0)
        bpm.add_offset(-6.0)
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm.add_offset(3.0)
        self.assertAlmostEqual(bpm.get_offset(), 3.0)
        self.assertAlmostEqual(bpm.get_polynomial(()), 3.0)
        bpm.add_interaction((), 3.0)
        self.assertAlmostEqual(bpm.get_offset(), 6.0)
        self.assertAlmostEqual(bpm.get_polynomial(()), 6.0)
        bpm.add_offset(-6.0)
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_remove_interaction_bpm_basic(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm.add_interaction((11, 12, 14), -1.0)
        bpm.add_interaction((-7,)       , -2.0)
        bpm.add_interaction((2, 11)     , -3.0)
        bpm.add_interaction(()          , -4.0)
        self.assertAlmostEqual(bpm.get_polynomial(11, 14, 12), -1.0)
        self.assertAlmostEqual(bpm.get_polynomial((-7,))     , -2.0)
        self.assertAlmostEqual(bpm.get_polynomial([11, 2])   , -3.0)
        self.assertAlmostEqual(bpm.get_polynomial([])        , -4.0)
        bpm.remove_interaction((11, 12, 14))
        bpm.remove_interaction(-7)
        bpm.remove_interaction([2, 11])
        bpm.remove_interaction([])
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm.add_interaction(("œ∑®", "≈ç", "≥≤µ"), -1.0)
        bpm.add_interaction(("A",)              , -2.0)
        bpm.add_interaction(("¡", "∆")          , -3.0)
        bpm.add_interaction(()                  , -4.0)
        self.assertAlmostEqual(bpm.get_polynomial("œ∑®", "≈ç", "≥≤µ"), -1.0)
        self.assertAlmostEqual(bpm.get_polynomial(("A",))            , -2.0)
        self.assertAlmostEqual(bpm.get_polynomial(["¡", "∆"])        , -3.0)
        self.assertAlmostEqual(bpm.get_polynomial([])                , -4.0)
        bpm.remove_interaction(("œ∑®", "≈ç", "≥≤µ"))
        bpm.remove_interaction("A")
        bpm.remove_interaction(["¡", "∆"])
        bpm.remove_interaction([])
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm.add_interaction(((11,11), (12,12), (14,14)), -1.0)
        bpm.add_interaction(((-7,-7),)                 , -2.0)
        bpm.add_interaction(((2,2), (11,11))           , -3.0)
        bpm.add_interaction(()                         , -4.0)
        self.assertAlmostEqual(bpm.get_polynomial((11,11), (12,12), (14,14)), -1.0)
        self.assertAlmostEqual(bpm.get_polynomial(((-7,-7),))               , -2.0)
        self.assertAlmostEqual(bpm.get_polynomial([(2,2), (11,11)])         , -3.0)
        self.assertAlmostEqual(bpm.get_polynomial([])                       , -4.0)
        bpm.remove_interaction(((11,11), (12,12), (14,14)))
        bpm.remove_interaction((-7,-7))
        bpm.remove_interaction([(2,2), (11,11)])
        bpm.remove_interaction([])
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm.add_interaction(((11,11,11), (12,12,12), (14,14,14)), -1.0)
        bpm.add_interaction(((-7,-7,-7),)                       , -2.0)
        bpm.add_interaction(((2,2,2), (11,11,11))               , -3.0)
        bpm.add_interaction(()                                  , -4.0)
        self.assertAlmostEqual(bpm.get_polynomial((11,11,11), (12,12,12), (14,14,14)), -1.0)
        self.assertAlmostEqual(bpm.get_polynomial(((-7,-7,-7),))                     , -2.0)
        self.assertAlmostEqual(bpm.get_polynomial([(2,2,2), (11,11,11)])             , -3.0)
        self.assertAlmostEqual(bpm.get_polynomial([])                                , -4.0)
        bpm.remove_interaction(((11,11,11), (12,12,12), (14,14,14)))
        bpm.remove_interaction((-7,-7,-7))
        bpm.remove_interaction([(2,2,2), (11,11,11)])
        bpm.remove_interaction([])
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm.add_interaction(((11,11,11,123456789), (12,12,12,123456789), (14,14,14,123456789)), -1.0)
        bpm.add_interaction(((-7,-7,-7,123456789),)                       , -2.0)
        bpm.add_interaction(((2,2,2,123456789), (11,11,11,123456789))               , -3.0)
        bpm.add_interaction(()                                  , -4.0)
        self.assertAlmostEqual(bpm.get_polynomial((14,14,14,123456789), (11,11,11,123456789), (12,12,12,123456789)), -1.0)
        self.assertAlmostEqual(bpm.get_polynomial(((-7,-7,-7,123456789),))                     , -2.0)
        self.assertAlmostEqual(bpm.get_polynomial([(2,2,2,123456789), (11,11,11,123456789)])             , -3.0)
        self.assertAlmostEqual(bpm.get_polynomial([])                                , -4.0)
        bpm.remove_interaction(((11,11,11,123456789), (14,14,14,123456789), (12,12,12,123456789)))
        bpm.remove_interaction((-7,-7,-7,123456789))
        bpm.remove_interaction([(2,2,2,123456789), (11,11,11,123456789)])
        bpm.remove_interaction([])
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_remove_interaction_bpm_remove_all(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        for k in self.poly.keys():
            bpm.remove_interaction(k)
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        for k in self.poly_str.keys():
            bpm.remove_interaction(*k)
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        for k in self.poly_tuple2.keys():
            bpm.remove_interaction(list(k))
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        for k in self.poly_tuple3.keys():
            bpm.remove_interaction(k)
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        for k in self.poly_tuple4.keys():
            bpm.remove_interaction(*k)
        self.state_test_bpm_empty(bpm, oj.SPIN)

    def test_remove_interactions_from_bpm_basic(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        added_J = {(11, 12, 14): -1.0, (-7,): -2.0, (2, 11): -3.0, (): -4.0}
        bpm.add_interactions_from(added_J)
        self.assertAlmostEqual(bpm.get_polynomial(11, 14, 12), -1.0)
        self.assertAlmostEqual(bpm.get_polynomial((-7,))     , -2.0)
        self.assertAlmostEqual(bpm.get_polynomial([11, 2])   , -3.0)
        self.assertAlmostEqual(bpm.get_polynomial([])        , -4.0)
        bpm.remove_interactions_from(list(added_J))
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        added_J = {("œ∑®", "≈ç", "≥≤µ"): -1.0, ("A",): -2.0, ("¡", "∆"): -3.0, (): -4.0}
        bpm.add_interactions_from(added_J)
        self.assertAlmostEqual(bpm.get_polynomial("œ∑®", "≈ç", "≥≤µ"), -1.0)
        self.assertAlmostEqual(bpm.get_polynomial(("A",))            , -2.0)
        self.assertAlmostEqual(bpm.get_polynomial(["¡", "∆"])        , -3.0)
        self.assertAlmostEqual(bpm.get_polynomial([])                , -4.0)
        bpm.remove_interactions_from(list(added_J))
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        added_J = {((11,11), (12,12), (14,14)): -1.0, ((-7,-7),): -2.0, ((2,2), (11,11)): -3.0, (): -4.0}
        bpm.add_interactions_from(added_J)
        self.assertAlmostEqual(bpm.get_polynomial((11,11), (12,12), (14,14)), -1.0)
        self.assertAlmostEqual(bpm.get_polynomial(((-7,-7),))               , -2.0)
        self.assertAlmostEqual(bpm.get_polynomial([(2,2), (11,11)])         , -3.0)
        self.assertAlmostEqual(bpm.get_polynomial([])                       , -4.0)
        bpm.remove_interactions_from(list(added_J))
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        added_J = {((11,11,11), (12,12,12), (14,14,14)): -1.0, ((-7,-7,-7),): -2.0, ((2,2,2), (11,11,11)): -3.0, (): -4.0}
        bpm.add_interactions_from(added_J)
        self.assertAlmostEqual(bpm.get_polynomial((11,11,11), (12,12,12), (14,14,14)), -1.0)
        self.assertAlmostEqual(bpm.get_polynomial(((-7,-7,-7),))                     , -2.0)
        self.assertAlmostEqual(bpm.get_polynomial([(2,2,2), (11,11,11)])             , -3.0)
        self.assertAlmostEqual(bpm.get_polynomial([])                                , -4.0)
        bpm.remove_interactions_from(list(added_J))
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        added_J = {((11,11,11,123456789), (12,12,12,123456789), (14,14,14,123456789)): -1.0, ((-7,-7,-7,123456789),): -2.0, ((2,2,2,123456789), (11,11,11,123456789)): -3.0, (): -4.0}
        bpm.add_interactions_from(added_J)
        self.assertAlmostEqual(bpm.get_polynomial((14,14,14,123456789), (11,11,11,123456789), (12,12,12,123456789)), -1.0)
        self.assertAlmostEqual(bpm.get_polynomial(((-7,-7,-7,123456789),))                                         , -2.0)
        self.assertAlmostEqual(bpm.get_polynomial([(2,2,2,123456789), (11,11,11,123456789)])                       , -3.0)
        self.assertAlmostEqual(bpm.get_polynomial([])                                                              , -4.0)
        bpm.remove_interactions_from(list(added_J))
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_remove_interactions_from_bpm_remove_all(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm.remove_interactions_from(list(self.poly.keys()))
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm.remove_interactions_from(tuple(self.poly_str.keys()))
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm.remove_interactions_from(*list(self.poly_tuple2.keys()))
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm.remove_interactions_from(*tuple(self.poly_tuple3.keys()))
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm.remove_interactions_from(list(self.poly_tuple4.keys()))
        self.state_test_bpm_empty(bpm, oj.SPIN)

    def test_remove_offset_bpm(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm.add_offset(100)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.remove_offset()
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm.add_offset(100)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.remove_offset()
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm.add_offset(100)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.remove_offset()
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm.add_offset(100)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.remove_offset()
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm.add_offset(100)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.remove_offset()
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_energy_bpm(self):
        #Spin
        self.assertEqual(oj.BinaryPolynomialModel(self.poly       , oj.SPIN).energy(self.spins)       , calculate_bpm_energy(self.poly       , self.spins)       )
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_str   , oj.SPIN).energy(self.spins_str)   , calculate_bpm_energy(self.poly_str   , self.spins_str)   )
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN).energy(self.spins_tuple2), calculate_bpm_energy(self.poly_tuple2, self.spins_tuple2))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN).energy(self.spins_tuple3), calculate_bpm_energy(self.poly_tuple3, self.spins_tuple3))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN).energy(self.spins_tuple4), calculate_bpm_energy(self.poly_tuple4, self.spins_tuple4))

        self.assertEqual(oj.BinaryPolynomialModel(self.poly       , oj.SPIN).energy(list(self.spins.values()))       , calculate_bpm_energy(self.poly       , self.spins)       )
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_str   , oj.SPIN).energy(list(self.spins_str.values()))   , calculate_bpm_energy(self.poly_str   , self.spins_str)   )
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN).energy(list(self.spins_tuple2.values())), calculate_bpm_energy(self.poly_tuple2, self.spins_tuple2))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN).energy(list(self.spins_tuple3.values())), calculate_bpm_energy(self.poly_tuple3, self.spins_tuple3))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN).energy(list(self.spins_tuple4.values())), calculate_bpm_energy(self.poly_tuple4, self.spins_tuple4))

        #Binary
        self.assertEqual(oj.BinaryPolynomialModel(self.poly       , oj.BINARY).energy(self.binaries)       , calculate_bpm_energy(self.poly       , self.binaries)       )
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_str   , oj.BINARY).energy(self.binaries_str)   , calculate_bpm_energy(self.poly_str   , self.binaries_str)   )
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple2, oj.BINARY).energy(self.binaries_tuple2), calculate_bpm_energy(self.poly_tuple2, self.binaries_tuple2))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple3, oj.BINARY).energy(self.binaries_tuple3), calculate_bpm_energy(self.poly_tuple3, self.binaries_tuple3))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple4, oj.BINARY).energy(self.binaries_tuple4), calculate_bpm_energy(self.poly_tuple4, self.binaries_tuple4))

        self.assertEqual(oj.BinaryPolynomialModel(self.poly       , oj.BINARY).energy(list(self.binaries.values()))       , calculate_bpm_energy(self.poly       , self.binaries)       )
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_str   , oj.BINARY).energy(list(self.binaries_str.values()))   , calculate_bpm_energy(self.poly_str   , self.binaries_str)   )
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple2, oj.BINARY).energy(list(self.binaries_tuple2.values())), calculate_bpm_energy(self.poly_tuple2, self.binaries_tuple2))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple3, oj.BINARY).energy(list(self.binaries_tuple3.values())), calculate_bpm_energy(self.poly_tuple3, self.binaries_tuple3))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple4, oj.BINARY).energy(list(self.binaries_tuple4.values())), calculate_bpm_energy(self.poly_tuple4, self.binaries_tuple4))

    def test_energies_bpm(self):
        #Spin
        spins_list = [self.spins, self.spins, self.spins, self.spins]
        anser_list = [calculate_bpm_energy(self.poly, self.spins) for _ in range(4)]
        self.assertListEqual(oj.BinaryPolynomialModel(self.poly, oj.SPIN).energies(spins_list), anser_list)

        spins_list = [self.spins_str, self.spins_str, self.spins_str, self.spins_str]
        anser_list = [calculate_bpm_energy(self.poly_str, self.spins_str) for _ in range(4)]
        self.assertListEqual(oj.BinaryPolynomialModel(self.poly_str, oj.SPIN).energies(spins_list), anser_list)

        spins_list = [self.spins_tuple2, self.spins_tuple2, self.spins_tuple2, self.spins_tuple2]
        anser_list = [calculate_bpm_energy(self.poly_tuple2, self.spins_tuple2) for _ in range(4)]
        self.assertListEqual(oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN).energies(spins_list), anser_list)

        spins_list = [self.spins_tuple3, self.spins_tuple3, self.spins_tuple3, self.spins_tuple3]
        anser_list = [calculate_bpm_energy(self.poly_tuple3, self.spins_tuple3) for _ in range(4)]
        self.assertListEqual(oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN).energies(spins_list), anser_list)

        spins_list = [self.spins_tuple4, self.spins_tuple4, self.spins_tuple4, self.spins_tuple4]
        anser_list = [calculate_bpm_energy(self.poly_tuple4, self.spins_tuple4) for _ in range(4)]
        self.assertListEqual(oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN).energies(spins_list), anser_list)

        #Binary
        binaries_list = [self.binaries, self.binaries, self.binaries, self.binaries]
        anser_list = [calculate_bpm_energy(self.poly, self.binaries) for _ in range(4)]
        self.assertListEqual(oj.BinaryPolynomialModel(self.poly, oj.BINARY).energies(binaries_list), anser_list)

        binaries_list = [self.binaries_str, self.binaries_str, self.binaries_str, self.binaries_str]
        anser_list = [calculate_bpm_energy(self.poly_str, self.binaries_str) for _ in range(4)]
        self.assertListEqual(oj.BinaryPolynomialModel(self.poly_str, oj.BINARY).energies(binaries_list), anser_list)

        binaries_list = [self.binaries_tuple2, self.binaries_tuple2, self.binaries_tuple2, self.binaries_tuple2]
        anser_list = [calculate_bpm_energy(self.poly_tuple2, self.binaries_tuple2) for _ in range(4)]
        self.assertListEqual(oj.BinaryPolynomialModel(self.poly_tuple2, oj.BINARY).energies(binaries_list), anser_list)

        binaries_list = [self.binaries_tuple3, self.binaries_tuple3, self.binaries_tuple3, self.binaries_tuple3]
        anser_list = [calculate_bpm_energy(self.poly_tuple3, self.binaries_tuple3) for _ in range(4)]
        self.assertListEqual(oj.BinaryPolynomialModel(self.poly_tuple3, oj.BINARY).energies(binaries_list), anser_list)

        binaries_list = [self.binaries_tuple4, self.binaries_tuple4, self.binaries_tuple4, self.binaries_tuple4]
        anser_list = [calculate_bpm_energy(self.poly_tuple4, self.binaries_tuple4) for _ in range(4)]
        self.assertListEqual(oj.BinaryPolynomialModel(self.poly_tuple4, oj.BINARY).energies(binaries_list), anser_list)

    def test_scale_bpm_all_scaled(self):
        d = {}
        for k, v in self.poly.items():
            d[k] = 2*v
        bpm = oj.BinaryPolynomialModel(d, oj.SPIN)
        bpm.scale(0.5)
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        d = {}
        for k, v in self.poly_str.items():
            d[k] = 2*v
        bpm = oj.BinaryPolynomialModel(d, oj.SPIN)
        bpm.scale(0.5)
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        d = {}
        for k, v in self.poly_tuple2.items():
            d[k] = 2*v
        bpm = oj.BinaryPolynomialModel(d, oj.SPIN)
        bpm.scale(0.5)
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        d = {}
        for k, v in self.poly_tuple3.items():
            d[k] = 2*v
        bpm = oj.BinaryPolynomialModel(d, oj.SPIN)
        bpm.scale(0.5)
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        d = {}
        for k, v in self.poly_tuple4.items():
            d[k] = 2*v
        bpm = oj.BinaryPolynomialModel(d, oj.SPIN)
        bpm.scale(0.5)
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_scale_bpm_ignored_interaction(self):
        bpm = oj.BinaryPolynomialModel({k: 2*v for k, v in self.poly.items()}, oj.SPIN)
        bpm.scale(0.5, ((1, 2), [2, 3, 4]))
        self.assertAlmostEqual(bpm.get_polynomial(1, 2)   , 12.0*2 )
        self.assertAlmostEqual(bpm.get_polynomial(2, 3, 4), 234.0*2)
        bpm.add_interaction((1, 2)   , -12 )
        bpm.add_interaction([2, 3, 4], -234)
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel({k: 2*v for k, v in self.poly_str.items()}, oj.SPIN)
        bpm.scale(0.5, (("a", "b"), ["b", "c", "d"]))
        self.assertAlmostEqual(bpm.get_polynomial("a", "b")     , 12.0*2 )
        self.assertAlmostEqual(bpm.get_polynomial("b", "c", "d"), 234.0*2)
        bpm.add_interaction(("a", "b")     , -12)
        bpm.add_interaction(["b", "c", "d"], -234)
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel({k: 2*v for k, v in self.poly_tuple2.items()}, oj.SPIN)
        bpm.scale(0.5, (((1, 1), (2, 2)), [(2, 2), (3, 3), (4, 4)]))
        self.assertAlmostEqual(bpm.get_polynomial((2, 2), (1, 1))        , 12.0*2 )
        self.assertAlmostEqual(bpm.get_polynomial((4, 4), (3, 3), (2, 2)), 234.0*2)
        bpm.add_interaction([(2, 2), (1, 1)]        , -12 )
        bpm.add_interaction([(4, 4), (3, 3), (2, 2)], -234)
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel({k: 2*v for k, v in self.poly_tuple3.items()}, oj.SPIN)
        bpm.scale(0.5, (((1, 1, 1), (2, 2, 2)), [(2, 2, 2), (3, 3, 3), (4, 4, 4)]))
        self.assertAlmostEqual(bpm.get_polynomial((2, 2, 2), (1, 1, 1))           , 12.0*2 )
        self.assertAlmostEqual(bpm.get_polynomial((4, 4, 4), (3, 3, 3), (2, 2, 2)), 234.0*2)
        bpm.add_interaction([(2, 2, 2), (1, 1, 1)]           , -12 )
        bpm.add_interaction([(4, 4, 4), (3, 3, 3), (2, 2, 2)], -234)
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel({k: 2*v for k, v in self.poly_tuple4.items()}, oj.SPIN)
        bpm.scale(0.5, (((1, 1, 1, 1), (2, 2, 2, 2)), [(2, 2, 2, 2), (3, 3, 3, 3), (4, 4, 4, 4)]))
        self.assertAlmostEqual(bpm.get_polynomial((2, 2, 2, 2), (1, 1, 1, 1))              , 12.0*2 )
        self.assertAlmostEqual(bpm.get_polynomial((4, 4, 4, 4), (3, 3, 3, 3), (2, 2, 2, 2)), 234.0*2)
        bpm.add_interaction([(2, 2, 2, 2), (1, 1, 1, 1)]              , -12 )
        bpm.add_interaction([(4, 4, 4, 4), (3, 3, 3, 3), (2, 2, 2, 2)], -234)
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_scale_bpm_ignored_offset(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm.add_offset(100)
        bpm.scale(0.5, ((),))
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.scale(0.5, ignored_offset = True)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.scale(0.5, [()], True)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm.add_offset(100)
        bpm.scale(0.5, ((),))
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.scale(0.5, ignored_offset = True)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.scale(0.5, [()], True)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm.add_offset(100)
        bpm.scale(0.5, ((),))
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.scale(0.5, ignored_offset = True)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.scale(0.5, [()], True)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm.add_offset(100)
        bpm.scale(0.5, ((),))
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.scale(0.5, ignored_offset = True)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.scale(0.5, [()], True)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm.add_offset(100)
        bpm.scale(0.5, ((),))
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.scale(0.5, ignored_offset = True)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        bpm.scale(0.5, [()], True)
        self.assertAlmostEqual(bpm.get_polynomial(()), 100)
        
    def test_normalize_bpm_all_normalize(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)        
        bpm.normalize((-1, +1))
        bpm.scale(234.0)
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)        
        bpm.normalize((-1, +1))
        bpm.scale(234.0)
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)        
        bpm.normalize((-1, +1))
        bpm.scale(234.0)
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)        
        bpm.normalize((-1, +1))
        bpm.scale(234.0)
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)        
        bpm.normalize((-1, +1))
        bpm.scale(234.0)
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_normalize_bpm_ignored_interaction(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm.normalize((-1, 1), list(self.poly))
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm.normalize((-1, 1), list(self.poly_str))
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm.normalize((-1, 1), list(self.poly_tuple2))
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm.normalize((-1, 1), list(self.poly_tuple3))
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm.normalize((-1, 1), list(self.poly_tuple4))
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

    def test_serializable_bpm(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm_from = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.state_test_bpm(bpm_from, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm_from = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.state_test_bpm(bpm_from, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm_from = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.state_test_bpm(bpm_from, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm_from = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.state_test_bpm(bpm_from, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm_from = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.state_test_bpm(bpm_from, self.poly_tuple4, oj.SPIN)

    def test_serializable_bpm_empty(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm.clear()
        bpm_from = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.state_test_bpm_empty(bpm_from, oj.SPIN)
        self.assertEqual(bpm_from.index_type, bpm.index_type)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm.clear()
        bpm_from = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.state_test_bpm_empty(bpm_from, oj.SPIN)
        self.assertEqual(bpm_from.index_type, bpm.index_type)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm.clear()
        bpm_from = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.state_test_bpm_empty(bpm_from, oj.SPIN)
        self.assertEqual(bpm_from.index_type, bpm.index_type)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm.clear()
        bpm_from = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.state_test_bpm_empty(bpm_from, oj.SPIN)
        self.assertEqual(bpm_from.index_type, bpm.index_type)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm.clear()
        bpm_from = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.state_test_bpm_empty(bpm_from, oj.SPIN)
        self.assertEqual(bpm_from.index_type, bpm.index_type)

    def test_from_hubo_bpm_from_dict(self):
        bpm = oj.BinaryPolynomialModel.from_hubo(self.poly)
        self.state_test_bpm(bpm, self.poly, oj.BINARY)

        bpm = oj.BinaryPolynomialModel.from_hubo(self.poly_str)
        self.state_test_bpm(bpm, self.poly_str, oj.BINARY)

        bpm = oj.BinaryPolynomialModel.from_hubo(self.poly_tuple2)
        self.state_test_bpm(bpm, self.poly_tuple2, oj.BINARY)

        bpm = oj.BinaryPolynomialModel.from_hubo(self.poly_tuple3)
        self.state_test_bpm(bpm, self.poly_tuple3, oj.BINARY)

        bpm = oj.BinaryPolynomialModel.from_hubo(self.poly_tuple4)
        self.state_test_bpm(bpm, self.poly_tuple4, oj.BINARY)

        bpm = oj.BinaryPolynomialModel.from_hubo({})
        self.state_test_bpm_empty(bpm, oj.BINARY)

    def test_from_hubo_bpm_from_key_value(self):
        bpm = oj.BinaryPolynomialModel.from_hubo(list(self.poly.keys()), list(self.poly.values()))
        self.state_test_bpm(bpm, self.poly, oj.BINARY)

        bpm = oj.BinaryPolynomialModel.from_hubo(list(self.poly_str.keys()), list(self.poly_str.values()))
        self.state_test_bpm(bpm, self.poly_str, oj.BINARY)

        bpm = oj.BinaryPolynomialModel.from_hubo(list(self.poly_tuple2.keys()), list(self.poly_tuple2.values()))
        self.state_test_bpm(bpm, self.poly_tuple2, oj.BINARY)

        bpm = oj.BinaryPolynomialModel.from_hubo(list(self.poly_tuple3.keys()), list(self.poly_tuple3.values()))
        self.state_test_bpm(bpm, self.poly_tuple3, oj.BINARY)

        bpm = oj.BinaryPolynomialModel.from_hubo(list(self.poly_tuple4.keys()), list(self.poly_tuple4.values()))
        self.state_test_bpm(bpm, self.poly_tuple4, oj.BINARY)

        bpm = oj.BinaryPolynomialModel.from_hubo([], [])
        self.state_test_bpm_empty(bpm, oj.BINARY)

    def test_from_hising_bpm_from_dict(self):
        bpm = oj.BinaryPolynomialModel.from_hising(self.poly)
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel.from_hising(self.poly_str)
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel.from_hising(self.poly_tuple2)
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel.from_hising(self.poly_tuple3)
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel.from_hising(self.poly_tuple4)
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

        bpm = oj.BinaryPolynomialModel.from_hising({})
        self.state_test_bpm_empty(bpm, oj.SPIN)

    def test_from_hising_bpm_from_key_value(self):
        bpm = oj.BinaryPolynomialModel.from_hising(list(self.poly.keys()), list(self.poly.values()))
        self.state_test_bpm(bpm, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel.from_hising(list(self.poly_str.keys()), list(self.poly_str.values()))
        self.state_test_bpm(bpm, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel.from_hising(list(self.poly_tuple2.keys()), list(self.poly_tuple2.values()))
        self.state_test_bpm(bpm, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel.from_hising(list(self.poly_tuple3.keys()), list(self.poly_tuple3.values()))
        self.state_test_bpm(bpm, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel.from_hising(list(self.poly_tuple4.keys()), list(self.poly_tuple4.values()))
        self.state_test_bpm(bpm, self.poly_tuple4, oj.SPIN)

        bpm = oj.BinaryPolynomialModel.from_hising([], [])
        self.state_test_bpm_empty(bpm, oj.SPIN)

    def test_clear_bpm(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm.clear()
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm.clear()
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm.clear()
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm.clear()
        self.state_test_bpm_empty(bpm, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm.clear()
        self.state_test_bpm_empty(bpm, oj.SPIN)

    def test_to_hubo_bpm(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        J_hubo = bpm.to_hubo()
        bpm.change_vartype("BINARY")
        self.state_test_bpm(bpm, J_hubo, oj.BINARY)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        J_hubo = bpm.to_hubo()
        bpm.change_vartype("BINARY")
        self.state_test_bpm(bpm, J_hubo, oj.BINARY)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        J_hubo = bpm.to_hubo()
        bpm.change_vartype("BINARY")
        self.state_test_bpm(bpm, J_hubo, oj.BINARY)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        J_hubo = bpm.to_hubo()
        bpm.change_vartype("BINARY")
        self.state_test_bpm(bpm, J_hubo, oj.BINARY)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        J_hubo = bpm.to_hubo()
        bpm.change_vartype("BINARY")
        self.state_test_bpm(bpm, J_hubo, oj.BINARY)

    def test_to_hising_bpm(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.BINARY)
        J_ising = bpm.to_hising()
        bpm.change_vartype("SPIN")
        self.state_test_bpm(bpm, J_ising, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.BINARY)
        J_ising = bpm.to_hising()
        bpm.change_vartype("SPIN")
        self.state_test_bpm(bpm, J_ising, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.BINARY)
        J_ising = bpm.to_hising()
        bpm.change_vartype("SPIN")
        self.state_test_bpm(bpm, J_ising, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.BINARY)
        J_ising = bpm.to_hising()
        bpm.change_vartype("SPIN")
        self.state_test_bpm(bpm, J_ising, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.BINARY)
        J_ising = bpm.to_hising()
        bpm.change_vartype("SPIN")
        self.state_test_bpm(bpm, J_ising, oj.SPIN)

    def test_change_vartype_bpm_spin_binary_spin(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.SPIN)
        bpm_binary = bpm.change_vartype("BINARY", False)
        bpm_ising  = bpm_binary.change_vartype("SPIN", False)
        self.state_test_bpm(bpm_ising, self.poly, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.SPIN)
        bpm_binary = bpm.change_vartype("BINARY", False)
        bpm_ising  = bpm_binary.change_vartype("SPIN", False)
        self.state_test_bpm(bpm_ising, self.poly_str, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.SPIN)
        bpm_binary = bpm.change_vartype("BINARY", False)
        bpm_ising  = bpm_binary.change_vartype("SPIN", False)
        self.state_test_bpm(bpm_ising, self.poly_tuple2, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.SPIN)
        bpm_binary = bpm.change_vartype("BINARY", False)
        bpm_ising  = bpm_binary.change_vartype("SPIN", False)
        self.state_test_bpm(bpm_ising, self.poly_tuple3, oj.SPIN)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.SPIN)
        bpm_binary = bpm.change_vartype("BINARY", False)
        bpm_ising  = bpm_binary.change_vartype("SPIN", False)
        self.state_test_bpm(bpm_ising, self.poly_tuple4, oj.SPIN)

    def test_change_vartype_bpm_binary_spin_binary(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.BINARY)
        bpm.change_vartype("SPIN")
        bpm.change_vartype("BINARY")
        self.state_test_bpm(bpm, self.poly, oj.BINARY)

        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.BINARY)
        bpm.change_vartype("SPIN")
        bpm.change_vartype("BINARY")
        self.state_test_bpm(bpm, self.poly_str, oj.BINARY)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.BINARY)
        bpm.change_vartype("SPIN")
        bpm.change_vartype("BINARY")
        self.state_test_bpm(bpm, self.poly_tuple2, oj.BINARY)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.BINARY)
        bpm.change_vartype("SPIN")
        bpm.change_vartype("BINARY")
        self.state_test_bpm(bpm, self.poly_tuple3, oj.BINARY)

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.BINARY)
        bpm.change_vartype("SPIN")
        bpm.change_vartype("BINARY")
        self.state_test_bpm(bpm, self.poly_tuple4, oj.BINARY)

if __name__ == '__main__':
    unittest.main()
