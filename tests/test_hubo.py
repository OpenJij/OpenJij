import unittest
import openjij as oj

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
        response = sampler.sample_hubo(K, vartype="SPIN", seed = 3)
        self.assertAlmostEqual(true_energy, response.energies[0])

    def test_hubo_constructor(self):
        hubo_spin = oj.BinaryPolynomialModel(self.J_quad, oj.SPIN)
        self.assertEqual(hubo_spin.vartype, oj.SPIN)

        hubo_binary = oj.BinaryPolynomialModel(self.Q_quad, "BINARY")
        self.assertEqual(hubo_binary.vartype, oj.BINARY)

    def test_zero_interaction(self):
        sampler = oj.SASampler()
        response = sampler.sample_hubo({(1,2,3):0.0, (1,2):1})

"""
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

        self.poly_tuple4     = {(((1,1,1,1)),):1.0, (((3,3,3,3)),):3.0, (((1,1,1,1)),((2,2,2,2))):12.0, \
                                    (((1,1,1,1)),((3,3,3,3))):13.0, (((2,2,2,2)),((3,3,3,3)),((4,4,4,4))):234.0, (((3,3,3,3)),((5,5,5,5))):35.0}
        self.spins_tuple4    = {((1,1,1,1)):+1, ((2,2,2,2)):-1, ((3,3,3,3)):+1, ((4,4,4,4)):-1, ((5,5,5,5)):+1} 
        self.binaries_tuple4 = {((1,1,1,1)): 1, ((2,2,2,2)): 0, ((3,3,3,3)): 1, ((4,4,4,4)): 0, ((5,5,5,5)): 1}



    # Test BinaryPolynomialModel constructor
    def test_bpm_constructor(self):
        #IntegerType
        bpm = oj.BinaryPolynomialModel(self.poly)
        temp_poly = {(2,):0.0, (5,): 0.0, (4,): 0.0}
        self.assertEqual    (bpm.vartype        , oj.SPIN)  #vartype
        self.assertEqual    (bpm.get_length()   , 5)           #get_length()
        self.assertSetEqual (bpm.variables      , {1,2,3,4,5}) #variables
        self.assertSetEqual (bpm.get_variables(), {1,2,3,4,5}) #get_variables()
        self.assertDictEqual(bpm.polynomial      , {**self.poly, **temp_poly}) #polynomial
        self.assertDictEqual(bpm.get_polynomial(), {**self.poly, **temp_poly}) #get_polynomial()
        self.assertDictEqual(bpm.adj             , {1:{(1,2):12.0, (1,3):13.0}, 2:{(2,3,4):234.0}, 3:{(3,5):35.0}}) #adj
        self.assertDictEqual(bpm.get_adjacency() , {1:{(1,2):12.0, (1,3):13.0}, 2:{(2,3,4):234.0}, 3:{(3,5):35.0}}) #get_adjacency()

        #StringType
        bpm = oj.BinaryPolynomialModel(self.poly_str)
        temp_poly = {("b",):0.0, ("e",): 0.0, ("d",): 0.0}
        self.assertEqual    (bpm.vartype        , oj.SPIN)  #vartype
        self.assertEqual    (bpm.get_length()   , 5)           #get_length()
        self.assertSetEqual (bpm.variables      , {"a","b","c","d","e"}) #variables
        self.assertSetEqual (bpm.get_variables(), {"a","b","c","d","e"}) #get_variables()
        self.assertDictEqual(bpm.polynomial      , {**self.poly_str, **temp_poly}) #polynomial
        self.assertDictEqual(bpm.get_polynomial(), {**self.poly_str, **temp_poly}) #get_polynomial()
        self.assertDictEqual(bpm.adj             , {"a":{("a","b"):12.0, ("a","c"):13.0}, "b":{("b","c","d"):234.0}, "c":{("c","e"):35.0}}) #adj
        self.assertDictEqual(bpm.get_adjacency() , {"a":{("a","b"):12.0, ("a","c"):13.0}, "b":{("b","c","d"):234.0}, "c":{("c","e"):35.0}}) #get_adjacency()
        
        #IntegerTypeTuple2
        bpm = oj.BinaryPolynomialModel(self.poly_tuple2) 
        temp_poly = {((2,2),):0.0, ((5,5),): 0.0, ((4,4),): 0.0}
        self.assertEqual    (bpm.vartype        , oj.SPIN)  #vartype
        self.assertEqual    (bpm.get_length()   , 5)           #get_length()
        self.assertSetEqual (bpm.variables      , {(1,1),(2,2),(3,3),(4,4),(5,5)}) #variables
        self.assertSetEqual (bpm.get_variables(), {(1,1),(2,2),(3,3),(4,4),(5,5)}) #get_variables()
        self.assertDictEqual(bpm.polynomial      , {**self.poly_tuple2, **temp_poly}) #polynomial
        self.assertDictEqual(bpm.get_polynomial(), {**self.poly_tuple2, **temp_poly}) #get_polynomial()
        self.assertDictEqual(bpm.adj             , {(1,1):{((1,1),(2,2)):12.0, ((1,1),(3,3)):13.0}, (2,2):{((2,2),(3,3),(4,4)):234.0}, (3,3):{((3,3),(5,5)):35.0}}) #adj
        self.assertDictEqual(bpm.get_adjacency() , {(1,1):{((1,1),(2,2)):12.0, ((1,1),(3,3)):13.0}, (2,2):{((2,2),(3,3),(4,4)):234.0}, (3,3):{((3,3),(5,5)):35.0}}) #get_adjacency()

        #IntegerTypeTuple3
        bpm = oj.BinaryPolynomialModel(self.poly_tuple3) 
        temp_poly = {((2,2,2),):0.0, ((5,5,5),): 0.0, ((4,4,4),): 0.0}
        self.assertEqual    (bpm.vartype        , oj.SPIN)  #vartype
        self.assertEqual    (bpm.get_length()   , 5)           #get_length()
        self.assertSetEqual (bpm.variables      , {(1,1,1),(2,2,2),(3,3,3),(4,4,4),(5,5,5)}) #variables
        self.assertSetEqual (bpm.get_variables(), {(1,1,1),(2,2,2),(3,3,3),(4,4,4),(5,5,5)}) #get_variables()
        self.assertDictEqual(bpm.polynomial      , {**self.poly_tuple3, **temp_poly}) #polynomial
        self.assertDictEqual(bpm.get_polynomial(), {**self.poly_tuple3, **temp_poly}) #get_polynomial()
        self.assertDictEqual(bpm.adj             , {(1,1,1):{((1,1,1),(2,2,2)):12.0, ((1,1,1),(3,3,3)):13.0}, (2,2,2):{((2,2,2),(3,3,3),(4,4,4)):234.0}, (3,3,3):{((3,3,3),(5,5,5)):35.0}}) #adj
        self.assertDictEqual(bpm.get_adjacency() , {(1,1,1):{((1,1,1),(2,2,2)):12.0, ((1,1,1),(3,3,3)):13.0}, (2,2,2):{((2,2,2),(3,3,3),(4,4,4)):234.0}, (3,3,3):{((3,3,3),(5,5,5)):35.0}}) #get_adjacency()

        #StringTypeTuple4
        bpm = oj.BinaryPolynomialModel(self.poly_tuple4)
        temp_poly = {(((2,2,2,2)),):0.0, (((5,5,5,5)),): 0.0, (((4,4,4,4)),): 0.0}
        self.assertEqual    (bpm.vartype        , oj.SPIN)  #vartype
        self.assertEqual    (bpm.get_length()   , 5)           #get_length()
        self.assertSetEqual (bpm.variables      , {((1,1,1,1)),((2,2,2,2)),((3,3,3,3)),((4,4,4,4)),((5,5,5,5))}) #variables
        self.assertSetEqual (bpm.get_variables(), {((1,1,1,1)),((2,2,2,2)),((3,3,3,3)),((4,4,4,4)),((5,5,5,5))}) #get_variables()
        self.assertDictEqual(bpm.polynomial      , {**self.poly_tuple4, **temp_poly}) #polynomial
        self.assertDictEqual(bpm.get_polynomial(), {**self.poly_tuple4, **temp_poly}) #get_polynomial()
        self.assertDictEqual(bpm.adj             , {((1,1,1,1)):{(((1,1,1,1)),((2,2,2,2))):12.0, (((1,1,1,1)),((3,3,3,3))):13.0}, \
                                                    ((2,2,2,2)):{(((2,2,2,2)),((3,3,3,3)),((4,4,4,4))):234.0}, ((3,3,3,3)):{(((3,3,3,3)),((5,5,5,5))):35.0}}) #adj
        self.assertDictEqual(bpm.get_adjacency() , {((1,1,1,1)):{(((1,1,1,1)),((2,2,2,2))):12.0, (((1,1,1,1)),((3,3,3,3))):13.0}, \
                                                    ((2,2,2,2)):{(((2,2,2,2)),((3,3,3,3)),((4,4,4,4))):234.0}, ((3,3,3,3)):{(((3,3,3,3)),((5,5,5,5))):35.0}}) #get_adjacency()

    #Tese energy calculations
    def test_bpm_calc_energy(self):
        #Spin
        self.assertEqual(oj.BinaryPolynomialModel(self.poly).energy(self.spins)              , calculate_bpm_energy(self.poly, self.spins))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_str).energy(self.spins_str)      , calculate_bpm_energy(self.poly_str, self.spins_str))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple2).energy(self.spins_tuple2), calculate_bpm_energy(self.poly_tuple2, self.spins_tuple2))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple3).energy(self.spins_tuple3), calculate_bpm_energy(self.poly_tuple3, self.spins_tuple3))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple4).energy(self.spins_tuple4), calculate_bpm_energy(self.poly_tuple4, self.spins_tuple4))

        #Binary
        self.assertEqual(oj.BinaryPolynomialModel(self.poly, oj.BINARY).energy(self.binaries)        , calculate_bpm_energy(self.poly, self.binaries))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_str, oj.BINARY).energy(self.binaries_str), calculate_bpm_energy(self.poly_str, self.binaries_str))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple2, oj.BINARY).energy(self.binaries_tuple2), calculate_bpm_energy(self.poly_tuple2, self.binaries_tuple2))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple3, oj.BINARY).energy(self.binaries_tuple3), calculate_bpm_energy(self.poly_tuple3, self.binaries_tuple3))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple4, oj.BINARY).energy(self.binaries_tuple4), calculate_bpm_energy(self.poly_tuple4, self.binaries_tuple4))

        #PUBO == Ising
        temp_spins        = {**self.spins}
        temp_spins_str    = {**self.spins_str}
        temp_spins_tuple2 = {**self.spins_tuple2}
        temp_spins_tuple3 = {**self.spins_tuple3}
        temp_spins_tuple4 = {**self.spins_tuple4}

        self.assertEqual(oj.BinaryPolynomialModel(self.poly, oj.BINARY).energy(temp_spins, convert_sample=True), calculate_bpm_energy(self.poly, self.binaries))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_str, oj.BINARY).energy(temp_spins_str, convert_sample=True), calculate_bpm_energy(self.poly_str, self.binaries_str))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple2, oj.BINARY).energy(temp_spins_tuple2, convert_sample=True), calculate_bpm_energy(self.poly_tuple2, self.binaries_tuple2))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple3, oj.BINARY).energy(temp_spins_tuple3, convert_sample=True), calculate_bpm_energy(self.poly_tuple3, self.binaries_tuple3))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple4, oj.BINARY).energy(temp_spins_tuple4, convert_sample=True), calculate_bpm_energy(self.poly_tuple4, self.binaries_tuple4))

        #Ising == PUBO
        temp_binaries        = {**self.binaries}
        temp_binaries_str    = {**self.binaries_str}
        temp_binaries_tuple2 = {**self.binaries_tuple2}
        temp_binaries_tuple3 = {**self.binaries_tuple3}
        temp_binaries_tuple4 = {**self.binaries_tuple4}
        self.assertEqual(oj.BinaryPolynomialModel(self.poly       ).energy(temp_binaries, convert_sample=True),        calculate_bpm_energy(self.poly,        self.spins))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_str   ).energy(temp_binaries_str, convert_sample=True),    calculate_bpm_energy(self.poly_str,    self.spins_str))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple2).energy(temp_binaries_tuple2, convert_sample=True), calculate_bpm_energy(self.poly_tuple2, self.spins_tuple2))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple3).energy(temp_binaries_tuple3, convert_sample=True), calculate_bpm_energy(self.poly_tuple3, self.spins_tuple3))
        self.assertEqual(oj.BinaryPolynomialModel(self.poly_tuple4).energy(temp_binaries_tuple4, convert_sample=True), calculate_bpm_energy(self.poly_tuple4, self.spins_tuple4))

    # Test BinaryPolynomialModel from_pubo function
    def test_bpm_from_pubo(self):
        bpm = oj.BinaryPolynomialModel(self.poly, oj.BINARY)
        bpm_from_pubo = oj.BinaryPolynomialModel.from_pubo(self.poly)
        self.assertEqual    (bpm_from_pubo.vartype         , bpm.vartype)         #vartype
        self.assertEqual    (bpm_from_pubo.get_length()    , bpm.get_length())    #get_length()
        self.assertSetEqual (bpm_from_pubo.variables       , bpm.variables)       #variables
        self.assertSetEqual (bpm_from_pubo.get_variables() , bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm_from_pubo.polynomial      , bpm.polynomial)      #polynomial
        self.assertDictEqual(bpm_from_pubo.get_polynomial(), bpm.get_polynomial())#get_polynomial()
        self.assertDictEqual(bpm_from_pubo.adj             , bpm.adj)             #adj
        self.assertDictEqual(bpm_from_pubo.get_adjacency() , bpm.get_adjacency()) #get_adjacency()
        
        bpm = oj.BinaryPolynomialModel(self.poly_str, oj.BINARY)
        bpm_from_pubo = oj.BinaryPolynomialModel.from_pubo(self.poly_str)
        self.assertEqual    (bpm_from_pubo.vartype         , bpm.vartype)         #vartype
        self.assertEqual    (bpm_from_pubo.get_length()    , bpm.get_length())    #get_length()
        self.assertSetEqual (bpm_from_pubo.variables       , bpm.variables)       #variables
        self.assertSetEqual (bpm_from_pubo.get_variables() , bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm_from_pubo.polynomial      , bpm.polynomial)      #polynomial
        self.assertDictEqual(bpm_from_pubo.get_polynomial(), bpm.get_polynomial())#get_polynomial()
        self.assertDictEqual(bpm_from_pubo.adj             , bpm.adj)             #adj
        self.assertDictEqual(bpm_from_pubo.get_adjacency() , bpm.get_adjacency()) #get_adjacency()

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2, oj.BINARY)
        bpm_from_pubo = oj.BinaryPolynomialModel.from_pubo(self.poly_tuple2)
        self.assertEqual    (bpm_from_pubo.vartype         , bpm.vartype)         #vartype
        self.assertEqual    (bpm_from_pubo.get_length()    , bpm.get_length())    #get_length()
        self.assertSetEqual (bpm_from_pubo.variables       , bpm.variables)       #variables
        self.assertSetEqual (bpm_from_pubo.get_variables() , bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm_from_pubo.polynomial      , bpm.polynomial)      #polynomial
        self.assertDictEqual(bpm_from_pubo.get_polynomial(), bpm.get_polynomial())#get_polynomial()
        self.assertDictEqual(bpm_from_pubo.adj             , bpm.adj)             #adj
        self.assertDictEqual(bpm_from_pubo.get_adjacency() , bpm.get_adjacency()) #get_adjacency()

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3, oj.BINARY)
        bpm_from_pubo = oj.BinaryPolynomialModel.from_pubo(self.poly_tuple3)
        self.assertEqual    (bpm_from_pubo.vartype         , bpm.vartype)         #vartype
        self.assertEqual    (bpm_from_pubo.get_length()    , bpm.get_length())    #get_length()
        self.assertSetEqual (bpm_from_pubo.variables       , bpm.variables)       #variables
        self.assertSetEqual (bpm_from_pubo.get_variables() , bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm_from_pubo.polynomial      , bpm.polynomial)      #polynomial
        self.assertDictEqual(bpm_from_pubo.get_polynomial(), bpm.get_polynomial())#get_polynomial()
        self.assertDictEqual(bpm_from_pubo.adj             , bpm.adj)             #adj
        self.assertDictEqual(bpm_from_pubo.get_adjacency() , bpm.get_adjacency()) #get_adjacency()

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4, oj.BINARY)
        bpm_from_pubo = oj.BinaryPolynomialModel.from_pubo(self.poly_tuple4)
        self.assertEqual    (bpm_from_pubo.vartype         , bpm.vartype)         #vartype
        self.assertEqual    (bpm_from_pubo.get_length()    , bpm.get_length())    #get_length()
        self.assertSetEqual (bpm_from_pubo.variables       , bpm.variables)       #variables
        self.assertSetEqual (bpm_from_pubo.get_variables() , bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm_from_pubo.polynomial      , bpm.polynomial)      #polynomial
        self.assertDictEqual(bpm_from_pubo.get_polynomial(), bpm.get_polynomial())#get_polynomial()
        self.assertDictEqual(bpm_from_pubo.adj             , bpm.adj)             #adj
        self.assertDictEqual(bpm_from_pubo.get_adjacency() , bpm.get_adjacency()) #get_adjacency()

    # Test BinaryPolynomialModel from_ising function
    def test_bpm_from_ising(self):
        bpm = oj.BinaryPolynomialModel(self.poly)
        bpm_from_pubo = oj.BinaryPolynomialModel.from_ising(self.poly)
        self.assertEqual    (bpm_from_pubo.vartype         , bpm.vartype)         #vartype
        self.assertEqual    (bpm_from_pubo.get_length()    , bpm.get_length())    #get_length()
        self.assertSetEqual (bpm_from_pubo.variables       , bpm.variables)       #variables
        self.assertSetEqual (bpm_from_pubo.get_variables() , bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm_from_pubo.polynomial      , bpm.polynomial)      #polynomial
        self.assertDictEqual(bpm_from_pubo.get_polynomial(), bpm.get_polynomial())#get_polynomial()
        self.assertDictEqual(bpm_from_pubo.adj             , bpm.adj)             #adj
        self.assertDictEqual(bpm_from_pubo.get_adjacency() , bpm.get_adjacency()) #get_adjacency()
        
        bpm = oj.BinaryPolynomialModel(self.poly_str)
        bpm_from_pubo = oj.BinaryPolynomialModel.from_ising(self.poly_str)
        self.assertEqual    (bpm_from_pubo.vartype         , bpm.vartype)         #vartype
        self.assertEqual    (bpm_from_pubo.get_length()    , bpm.get_length())    #get_length()
        self.assertSetEqual (bpm_from_pubo.variables       , bpm.variables)       #variables
        self.assertSetEqual (bpm_from_pubo.get_variables() , bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm_from_pubo.polynomial      , bpm.polynomial)      #polynomial
        self.assertDictEqual(bpm_from_pubo.get_polynomial(), bpm.get_polynomial())#get_polynomial()
        self.assertDictEqual(bpm_from_pubo.adj             , bpm.adj)             #adj
        self.assertDictEqual(bpm_from_pubo.get_adjacency() , bpm.get_adjacency()) #get_adjacency()

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2)
        bpm_from_pubo = oj.BinaryPolynomialModel.from_ising(self.poly_tuple2)
        self.assertEqual    (bpm_from_pubo.vartype         , bpm.vartype)         #vartype
        self.assertEqual    (bpm_from_pubo.get_length()    , bpm.get_length())    #get_length()
        self.assertSetEqual (bpm_from_pubo.variables       , bpm.variables)       #variables
        self.assertSetEqual (bpm_from_pubo.get_variables() , bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm_from_pubo.polynomial      , bpm.polynomial)      #polynomial
        self.assertDictEqual(bpm_from_pubo.get_polynomial(), bpm.get_polynomial())#get_polynomial()
        self.assertDictEqual(bpm_from_pubo.adj             , bpm.adj)             #adj
        self.assertDictEqual(bpm_from_pubo.get_adjacency() , bpm.get_adjacency()) #get_adjacency()

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3)
        bpm_from_pubo = oj.BinaryPolynomialModel.from_ising(self.poly_tuple3)
        self.assertEqual    (bpm_from_pubo.vartype         , bpm.vartype)         #vartype
        self.assertEqual    (bpm_from_pubo.get_length()    , bpm.get_length())    #get_length()
        self.assertSetEqual (bpm_from_pubo.variables       , bpm.variables)       #variables
        self.assertSetEqual (bpm_from_pubo.get_variables() , bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm_from_pubo.polynomial      , bpm.polynomial)      #polynomial
        self.assertDictEqual(bpm_from_pubo.get_polynomial(), bpm.get_polynomial())#get_polynomial()
        self.assertDictEqual(bpm_from_pubo.adj             , bpm.adj)             #adj
        self.assertDictEqual(bpm_from_pubo.get_adjacency() , bpm.get_adjacency()) #get_adjacency()

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4)
        bpm_from_pubo = oj.BinaryPolynomialModel.from_ising(self.poly_tuple4)
        self.assertEqual    (bpm_from_pubo.vartype         , bpm.vartype)         #vartype
        self.assertEqual    (bpm_from_pubo.get_length()    , bpm.get_length())    #get_length()
        self.assertSetEqual (bpm_from_pubo.variables       , bpm.variables)       #variables
        self.assertSetEqual (bpm_from_pubo.get_variables() , bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm_from_pubo.polynomial      , bpm.polynomial)      #polynomial
        self.assertDictEqual(bpm_from_pubo.get_polynomial(), bpm.get_polynomial())#get_polynomial()
        self.assertDictEqual(bpm_from_pubo.adj             , bpm.adj)             #adj
        self.assertDictEqual(bpm_from_pubo.get_adjacency() , bpm.get_adjacency()) #get_adjacency()

    def test_bpm_serializable(self):
        bpm = oj.BinaryPolynomialModel(self.poly)
        decode_bpm = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.assertEqual    (bpm.vartype         , decode_bpm.vartype)  #vartype
        self.assertEqual    (bpm.get_length()    , decode_bpm.get_length())           #get_length()
        self.assertSetEqual (bpm.variables       , decode_bpm.variables) #variables
        self.assertSetEqual (bpm.get_variables() , decode_bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm.polynomial      , decode_bpm.polynomial) #polynomial
        self.assertDictEqual(bpm.get_polynomial(), decode_bpm.get_polynomial()) #get_polynomial()
        self.assertDictEqual(bpm.adj             , decode_bpm.adj ) #adj
        self.assertDictEqual(bpm.get_adjacency() , decode_bpm.get_adjacency()) #get_adjacency()

        bpm = oj.BinaryPolynomialModel(self.poly_str)
        decode_bpm = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.assertEqual    (bpm.vartype         , decode_bpm.vartype)  #vartype
        self.assertEqual    (bpm.get_length()    , decode_bpm.get_length())           #get_length()
        self.assertSetEqual (bpm.variables       , decode_bpm.variables) #variables
        self.assertSetEqual (bpm.get_variables() , decode_bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm.polynomial      , decode_bpm.polynomial) #polynomial
        self.assertDictEqual(bpm.get_polynomial(), decode_bpm.get_polynomial()) #get_polynomial()
        self.assertDictEqual(bpm.adj             , decode_bpm.adj ) #adj
        self.assertDictEqual(bpm.get_adjacency() , decode_bpm.get_adjacency()) #get_adjacency()

        bpm = oj.BinaryPolynomialModel(self.poly_tuple2)
        decode_bpm = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.assertEqual    (bpm.vartype         , decode_bpm.vartype)  #vartype
        self.assertEqual    (bpm.get_length()    , decode_bpm.get_length())           #get_length()
        self.assertSetEqual (bpm.variables       , decode_bpm.variables) #variables
        self.assertSetEqual (bpm.get_variables() , decode_bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm.polynomial      , decode_bpm.polynomial) #polynomial
        self.assertDictEqual(bpm.get_polynomial(), decode_bpm.get_polynomial()) #get_polynomial()
        self.assertDictEqual(bpm.adj             , decode_bpm.adj ) #adj
        self.assertDictEqual(bpm.get_adjacency() , decode_bpm.get_adjacency()) #get_adjacency()

        bpm = oj.BinaryPolynomialModel(self.poly_tuple3)
        decode_bpm = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.assertEqual    (bpm.vartype         , decode_bpm.vartype)  #vartype
        self.assertEqual    (bpm.get_length()    , decode_bpm.get_length())           #get_length()
        self.assertSetEqual (bpm.variables       , decode_bpm.variables) #variables
        self.assertSetEqual (bpm.get_variables() , decode_bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm.polynomial      , decode_bpm.polynomial) #polynomial
        self.assertDictEqual(bpm.get_polynomial(), decode_bpm.get_polynomial()) #get_polynomial()
        self.assertDictEqual(bpm.adj             , decode_bpm.adj ) #adj
        self.assertDictEqual(bpm.get_adjacency() , decode_bpm.get_adjacency()) #get_adjacency()

        bpm = oj.BinaryPolynomialModel(self.poly_tuple4)
        decode_bpm = oj.BinaryPolynomialModel.from_serializable(bpm.to_serializable())
        self.assertEqual    (bpm.vartype         , decode_bpm.vartype)  #vartype
        self.assertEqual    (bpm.get_length()    , decode_bpm.get_length())           #get_length()
        self.assertSetEqual (bpm.variables       , decode_bpm.variables) #variables
        self.assertSetEqual (bpm.get_variables() , decode_bpm.get_variables()) #get_variables()
        self.assertDictEqual(bpm.polynomial      , decode_bpm.polynomial) #polynomial
        self.assertDictEqual(bpm.get_polynomial(), decode_bpm.get_polynomial()) #get_polynomial()
        self.assertDictEqual(bpm.adj             , decode_bpm.adj ) #adj
        self.assertDictEqual(bpm.get_adjacency() , decode_bpm.get_adjacency()) #get_adjacency()
"""

if __name__ == '__main__':
    unittest.main()
