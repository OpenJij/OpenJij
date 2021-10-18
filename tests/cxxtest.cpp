// include Google Test
#include <gtest/gtest.h>
#include <gmock/gmock.h>
// include STL
#include <iostream>
#include <utility>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <algorithm>

// include OpenJij
#include <graph/all.hpp>
#include <system/all.hpp>
#include <updater/all.hpp>
#include <algorithm/all.hpp>
#include <result/all.hpp>
#include <utility/schedule_list.hpp>
#include <utility/union_find.hpp>
#include <utility/random.hpp>
#include <utility/gpu/memory.hpp>
#include <utility/gpu/cublas.hpp>

#include "polynomial_test.hpp"

// #####################################
// helper functions
// #####################################
/**
 * @brief generate interaction
 *
 * @return classical interaction which represents specific optimization problem
 */

static constexpr std::size_t num_system_size = 8;

#define TEST_CASE_INDEX 1

#include "./testcase.hpp"


static openjij::utility::ClassicalScheduleList generate_schedule_list(){
    return openjij::utility::make_classical_schedule_list(0.1, 100.0, 100, 100);
}

static openjij::utility::TransverseFieldScheduleList generate_tfm_schedule_list(){
    return openjij::utility::make_transverse_field_schedule_list(10, 100, 100);
}

// #####################################
// tests
// #####################################

#include <chrono>

//speed test
//TEST(Graph, speedtest){
//    using namespace openjij::graph;
//    std::size_t N = 5000;
//    auto begin = std::chrono::high_resolution_clock::now();
//    Dense<double> a(N);
//    for(std::size_t i=0; i<N; i++){
//        for(std::size_t j=i; j<N; j++){
//            a.J(i, j)  = 1;
//        }
//    }
//    auto end = std::chrono::high_resolution_clock::now();
//    std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << std::endl;
//
//    auto quad = cimod::Quadratic<size_t, double>();
//    auto lin = cimod::Linear<size_t, double>();
//    begin = std::chrono::high_resolution_clock::now();
//    for(std::size_t i=0; i<N; i++){
//        for(std::size_t j=i+1; j<N; j++){
//            quad[std::make_pair(i,j)] = 1;
//        }
//    }
//    for(std::size_t i=0; i<N; i++){
//        lin[i] = 1;
//    }
//    auto bqm = cimod::BinaryQuadraticModel<size_t, double>(lin, quad, 0, cimod::Vartype::BINARY);
//    end = std::chrono::high_resolution_clock::now();
//    std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << std::endl;
//}

TEST(Graph, DenseGraphCheck){
    using namespace openjij::graph;
    using namespace openjij;

    std::size_t N = 500;
    Dense<double> a(N);
    auto r = utility::Xorshift(1234);
    auto urd = std::uniform_real_distribution<>{-10, 10};
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            a.J(i, j)  = urd(r);
        }
    }

    r = utility::Xorshift(1234);

    // check if graph holds correct variables
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(a.J(i, j) , urd(r));
        }
    }

    r = utility::Xorshift(1234);

    // check if graph index is reversible (Jij = Jji)
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(a.J(j, i) , urd(r));
        }
    }
}

TEST(Graph, SparseGraphCheck){
    using namespace openjij::graph;
    using namespace openjij;

    std::size_t N = 500;
    Sparse<double> b(N, N-1);
    auto r = utility::Xorshift(1234);
    auto urd = std::uniform_real_distribution<>{-10, 10};
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            b.J(i, j) = urd(r);
        }
    }

    r = utility::Xorshift(1234);

    // check if graph holds correct variables
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            EXPECT_EQ(b.J(i, j) , urd(r));
        }
    }

    r = utility::Xorshift(1234);

    // check if graph index is reversible (Jij = Jji)
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            EXPECT_EQ(b.J(j, i) , urd(r));
        }
    }

    //check adj_nodes
    for(std::size_t i=0; i<N; i++){
        std::size_t tot = 0;
        for(auto&& elem : b.adj_nodes(i)){
            tot += elem;
        }
        EXPECT_EQ(tot, N*(N-1)/2 - i);
    }
    EXPECT_EQ(b.get_num_edges(), N-1);

    Sparse<double> c(N, N);
    r = utility::Xorshift(1234);
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            c.J(j, i) = urd(r);
        }
    }

    r = utility::Xorshift(1234);

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(c.J(i, j) , urd(r));
        }
    }

    r = utility::Xorshift(1234);

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(c.J(j, i) , urd(r));
        }
    }
    for(std::size_t i=0; i<N; i++){
        std::size_t tot = 0;
        for(auto&& elem : c.adj_nodes(i)){
            tot += elem;
        }
        EXPECT_EQ(tot, N*(N-1)/2);
    }
    EXPECT_EQ(c.get_num_edges(), N);
}

TEST(Graph, EnergyCheck){
    using namespace openjij::graph;
    std::size_t N = 500;

    Dense<double> b_d(N);
    Sparse<double> b(N, N-1);

    Spins spins(N, 1);
    Spins spins_neg(N, -1);
    auto random_engine = std::mt19937(1);
    Spins spins_r = b_d.gen_spin(random_engine);

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            b_d.J(i, j) = 1;
        }
    }

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            b.J(i, j) = 1;
        }
    }

    EXPECT_EQ(b_d.calc_energy(spins), (1./2) * (N*N - N));
    EXPECT_EQ(b_d.calc_energy(spins_neg), (1./2) * (N*N - N));
    EXPECT_EQ(b.calc_energy(spins), (1./2) * (N*N - N));
    EXPECT_EQ(b.calc_energy(spins_neg), (1./2) * (N*N - N));
    EXPECT_EQ(b_d.calc_energy(spins_r), b.calc_energy(spins_r));

    Dense<double> c_d(N);
    Sparse<double> c(N, N);

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            c_d.J(i, j) = 1;
        }
    }

    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            c.J(i, j) = 1;
        }
    }

    EXPECT_EQ(c_d.calc_energy(spins), (1./2) * (N*N + N));
    EXPECT_EQ(c_d.calc_energy(spins_neg), (1./2) * (N*N - 3*N));
    EXPECT_EQ(c.calc_energy(spins), (1./2) * (N*N + N));
    EXPECT_EQ(c.calc_energy(spins_neg), (1./2) * (N*N - 3*N));
    EXPECT_EQ(c_d.calc_energy(spins_r), c.calc_energy(spins_r));
}

//json tests
TEST(Graph, JSONTest){
    using namespace cimod;
    using namespace openjij;

    Linear<uint32_t, double> linear{ {0, 1.0}, {11, 3.0}};
    Quadratic<uint32_t, double> quadratic
    {
        {std::make_pair(0, 4), 12.0}, {std::make_pair(4, 12), 13.0}, {std::make_pair(6, 14), 14.0},
            {std::make_pair(3, 4), 23.0}, {std::make_pair(11, 12), 24.0},
            {std::make_pair(5, 13), 34.0}
    };
    double offset = 0.0;
    Vartype vartype = Vartype::SPIN;
    BinaryQuadraticModel<uint32_t, double, cimod::Sparse> bqm_k4(linear, quadratic, offset, vartype);
    auto s = graph::Chimera<double>(bqm_k4.to_serializable(), 1, 2);
    EXPECT_NEAR(s.J(0,1,3,graph::ChimeraDir::IN_0or4), 24, 1e-5);
    EXPECT_NEAR(s.J(0,0,4,graph::ChimeraDir::PLUS_C), 13, 1e-5);
    EXPECT_NEAR(s.J(0,1,6,graph::ChimeraDir::MINUS_C), 14, 1e-5);
    EXPECT_NEAR(s.J(0,0,4,graph::ChimeraDir::IN_3or7), 23, 1e-5);
    EXPECT_NEAR(s.J(0,0,3,graph::ChimeraDir::IN_0or4), 23, 1e-5);
    EXPECT_NEAR(s.J(0,1,5,graph::ChimeraDir::MINUS_C), 34, 1e-5);
    EXPECT_NEAR(s.h(0,1,3), 3, 1e-5);
}

//ClassicalIsing tests

TEST(ClassicalIsing, GenerateTheSameEigenObject){
    using namespace openjij;
    graph::Dense<double> d(4);
    graph::Sparse<double> s(4);
    d.J(2,3) = s.J(2,3) = 4;
    d.J(1,0) = s.J(1,0) = -2;
    d.J(1,1) = s.J(1,1) = 5;
    d.J(2,2) = s.J(2,2) = 10;

    auto engine_for_spin = std::mt19937(1);
    auto cl_dense = system::make_classical_ising(d.gen_spin(engine_for_spin), d);
    auto cl_sparse = system::make_classical_ising(s.gen_spin(engine_for_spin), s);
    Eigen::MatrixXd m1 = cl_dense.interaction;
    //convert from sparse to dense
    Eigen::MatrixXd m2 = cl_sparse.interaction;
    EXPECT_EQ(m1, m2);
}

//TODO: macro?
//SingleSpinFlip tests


TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Dense) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); 
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Sparse) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction);
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_Dense) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense<double>>();
    auto engine_for_spin = std::mt19937(1);
    std::size_t num_trotter_slices = 10;

    //generate random trotter spins
    system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = interaction.gen_spin(engine_for_spin);
    }

    auto transverse_ising = system::make_transverse_ising(init_trotter_spins, interaction, 1.0);

    auto transverse_ising2 = system::make_transverse_ising(interaction.gen_spin(engine_for_spin), interaction, 1.0, 10);
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(transverse_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(transverse_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_Sparse) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse<double>>();
    auto engine_for_spin = std::mt19937(1);
    std::size_t num_trotter_slices = 10;

    //generate random trotter spins
    system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = interaction.gen_spin(engine_for_spin);
    }

    auto transverse_ising = system::make_transverse_ising(init_trotter_spins, interaction, 1.0); //gamma = 1.0
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(transverse_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(transverse_ising));
}



///-------------------------------------------------------------------------------------------
///-------------------------Graph Class Tests: Polynomiall-------------------------
///-------------------------------------------------------------------------------------------
TEST(PolyGraph, ConstructorCimodDenseInt) {
   TestPolyGraphConstructorCimodDense<openjij::graph::Index, double>(GeneratePolynomialInteractionsDenseInt<double>());
}
TEST(PolyGraph, ConstructorCimodDenseString) {
   TestPolyGraphConstructorCimodDense<std::string, double>(GeneratePolynomialInteractionsDenseString<double>());
}
TEST(PolyGraph, ConstructorCimodDenseTuple2) {
   TestPolyGraphConstructorCimodDense<std::tuple<openjij::graph::Index, openjij::graph::Index>, double>(GeneratePolynomialInteractionsDenseTuple2<double>());
}
TEST(PolyGraph, ConstructorCimodSparseInt1) {
   TestPolyGraphConstructorCimodSparse<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt1<double>());
}
TEST(PolyGraph, ConstructorCimodSparseInt2) {
   TestPolyGraphConstructorCimodSparse<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt2<double>());
}
TEST(PolyGraph, ConstructorCimodSparseInt3) {
   TestPolyGraphConstructorCimodSparse<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt3<double>());
}

TEST(PolyGraph, AddInteractions1) {
   const openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(   {}    ) = +0.1  ;
   poly_graph.J(   {0}   ) = -0.5  ;
   poly_graph.J(   {1}   ) = +1.0  ;
   poly_graph.J(   {2}   ) = -2.0  ;
   poly_graph.J( {0, 1}  ) = +10.0 ;
   poly_graph.J( {0, 2}  ) = -20.0 ;
   poly_graph.J( {1, 2}  ) = +21.0 ;
   poly_graph.J({0, 1, 2}) = -120.0;
   
   TestPolyGraphDense(poly_graph);
}

TEST(PolyGraph, AddInteractions2) {
   const openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(       ) = +0.1  /2.0;
   poly_graph.J(   0   ) = -0.5  /2.0;
   poly_graph.J(   1   ) = +1.0  /2.0;
   poly_graph.J(   2   ) = -2.0  /2.0;
   poly_graph.J( 0, 1  ) = +10.0 /2.0;
   poly_graph.J( 0, 2  ) = -20.0 /2.0;
   poly_graph.J( 1, 2  ) = +21.0 /2.0;
   poly_graph.J(0, 1, 2) = -120.0/2.0;
   
   poly_graph.J(       ) += +0.1  /2.0;
   poly_graph.J(   0   ) += -0.5  /2.0;
   poly_graph.J(   1   ) += +1.0  /2.0;
   poly_graph.J(   2   ) += -2.0  /2.0;
   poly_graph.J( 0, 1  ) += +10.0 /2.0;
   poly_graph.J( 0, 2  ) += -20.0 /2.0;
   poly_graph.J( 1, 2  ) += +21.0 /2.0;
   poly_graph.J(0, 1, 2) += -120.0/2.0;
   
   TestPolyGraphDense(poly_graph);
}

TEST(PolyGraph, AddInteractions3) {
   const openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(       ) = +0.1  ;
   poly_graph.J(   0   ) = -0.5  ;
   poly_graph.J(   1   ) = +1.0  ;
   poly_graph.J(   2   ) = -2.0  ;
   poly_graph.J( 0, 1  ) = +10.0 ;
   poly_graph.J( 0, 2  ) = -20.0 ;
   poly_graph.J( 1, 2  ) = +21.0 ;
   poly_graph.J(0, 1, 2) = -120.0;
   
   TestPolyGraphDense(poly_graph);
}

TEST(PolyGraph, AddInteractions4) {
   const openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(       ) = +999999;
   poly_graph.J(   0   ) = +999999;
   poly_graph.J(   1   ) = +999999;
   poly_graph.J(   2   ) = +999999;
   poly_graph.J( 0, 1  ) = +999999;
   poly_graph.J( 0, 2  ) = +999999;
   poly_graph.J( 1, 2  ) = +999999;
   poly_graph.J(0, 1, 2) = +999999;
   
   poly_graph.J(       ) = +0.1  ;
   poly_graph.J(   0   ) = -0.5  ;
   poly_graph.J(   1   ) = +1.0  ;
   poly_graph.J(   2   ) = -2.0  ;
   poly_graph.J( 0, 1  ) = +10.0 ;
   poly_graph.J( 0, 2  ) = -20.0 ;
   poly_graph.J( 1, 2  ) = +21.0 ;
   poly_graph.J(0, 1, 2) = -120.0;
   
   TestPolyGraphDense(poly_graph);
}

///---------------------------------------------------------------------------------------------------------------
///-------------------------System Class Tests: ClassicalIsingPolynomial-------------------------
///---------------------------------------------------------------------------------------------------------------
TEST(PolySystemCIP, ConstructorCimodDenseInt) {
   TestCIPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsDenseInt<double>(), cimod::Vartype::SPIN, "Dense");
   TestCIPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsDenseInt<double>(), cimod::Vartype::BINARY, "Dense");
}
TEST(PolySystemCIP, ConstructorCimodDenseString) {
   TestCIPConstructorCimod<std::string, double>(GeneratePolynomialInteractionsDenseString<double>(), cimod::Vartype::SPIN, "Dense");
   TestCIPConstructorCimod<std::string, double>(GeneratePolynomialInteractionsDenseString<double>(), cimod::Vartype::BINARY, "Dense");
}
TEST(PolySystemCIP, ConstructorCimodDenseTuple2) {
   TestCIPConstructorCimod<std::tuple<openjij::graph::Index, openjij::graph::Index>, double>(GeneratePolynomialInteractionsDenseTuple2<double>(), cimod::Vartype::SPIN, "Dense");
   TestCIPConstructorCimod<std::tuple<openjij::graph::Index, openjij::graph::Index>, double>(GeneratePolynomialInteractionsDenseTuple2<double>(), cimod::Vartype::BINARY, "Dense");
}
TEST(PolySystemCIP, ConstructorCimodSparseInt1) {
   TestCIPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt1<double>(), cimod::Vartype::SPIN, "Sparse");
   TestCIPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt1<double>(), cimod::Vartype::BINARY, "Sparse");
}
TEST(PolySystemCIP, ConstructorCimodSparseInt2) {
   TestCIPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt2<double>(), cimod::Vartype::SPIN, "Sparse");
   TestCIPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt2<double>(), cimod::Vartype::BINARY, "Sparse");
}
TEST(PolySystemCIP, ConstructorCimodSparseInt3) {
   TestCIPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt3<double>(), cimod::Vartype::SPIN, "Sparse");
   TestCIPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt3<double>(), cimod::Vartype::BINARY, "Sparse");
}
TEST(PolySystemCIP, ConstructorGraphDenseInt) {
   TestCIPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsDenseInt<double>(), cimod::Vartype::SPIN, "Dense");
   TestCIPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsDenseInt<double>(), cimod::Vartype::BINARY, "Dense");
}
TEST(PolySystemCIP, ConstructorGraphSparseInt1) {
   TestCIPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt1<double>(), cimod::Vartype::SPIN, "Sparse");
   TestCIPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt1<double>(), cimod::Vartype::BINARY, "Sparse");
}
TEST(PolySystemCIP, ConstructorGraphSparseInt2) {
   TestCIPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt2<double>(), cimod::Vartype::SPIN, "Sparse");
   TestCIPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt2<double>(), cimod::Vartype::BINARY, "Sparse");
}

///---------------------------------------------------------------------------------------------------------------
///-------------------------System Class Tests: KLocalPolynomial-------------------------
///---------------------------------------------------------------------------------------------------------------
TEST(PolySystemKLP, ConstructorCimodDenseInt) {
   TestKLPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsDenseInt<double>(), "Dense");
}
TEST(PolySystemKLP, ConstructorCimodDenseString) {
   TestKLPConstructorCimod<std::string, double>(GeneratePolynomialInteractionsDenseString<double>(), "Dense");
}
TEST(PolySystemKLP, ConstructorCimodDenseTuple2) {
   TestKLPConstructorCimod<std::tuple<openjij::graph::Index, openjij::graph::Index>, double>(GeneratePolynomialInteractionsDenseTuple2<double>(), "Dense");
}
TEST(PolySystemKLP, ConstructorCimodSparseInt1) {
   TestKLPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt1<double>(), "Sparse");
}
TEST(PolySystemKLP, ConstructorCimodSparseInt2) {
   TestKLPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt2<double>(), "Sparse");
}
TEST(PolySystemKLP, ConstructorCimodSparseInt3) {
   TestKLPConstructorCimod<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt3<double>(), "Sparse");
}
TEST(PolySystemKLP, ConstructorGraphDenseInt) {
   TestKLPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsDenseInt<double>(), "Dense");
}
TEST(PolySystemKLP, ConstructorGraphSparseInt1) {
   TestKLPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt1<double>(), "Sparse");
}
TEST(PolySystemKLP, ConstructorGraphSparseInt2) {
   TestKLPConstructorGraph<openjij::graph::Index, double>(GeneratePolynomialInteractionsSparseInt2<double>(), "Sparse");
}

TEST(PolyUpdater, SingleSpinFlipSPIN) {
   
   //Check the polynomial updater work properly by comparing the result of the quadratic updater
   const int seed = 1;
   const int system_size = 9;
   
   //generate classical sparse system
   auto engin_for_interaction = std::mt19937(seed);
   auto urd = std::uniform_real_distribution<>(-1.0/system_size, 1.0/system_size);
   auto interaction = openjij::graph::Sparse<double>(system_size);
   auto poly_graph  = openjij::graph::Polynomial<double>(system_size);
   for (int i = 0; i < system_size; ++i) {
      for (int j = i + 1; j < system_size; ++j) {
         interaction.J(i, j) = urd(engin_for_interaction);
         poly_graph.J(i, j)  = interaction.J(i, j);
      }
   }
   auto engine_for_spin = std::mt19937(seed);
   const auto spin = interaction.gen_spin(engine_for_spin);
   auto classical_ising      = openjij::system::make_classical_ising(spin, interaction);
   auto classical_ising_poly = openjij::system::make_classical_ising_polynomial(spin, poly_graph, "SPIN");
   
   auto random_numder_engine = std::mt19937(seed);
   const auto schedule_list  = generate_schedule_list();
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising     , random_numder_engine, schedule_list);
   auto random_numder_engine_a = std::mt19937(seed);
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_poly, random_numder_engine_a, schedule_list);
   const auto result_spin      = openjij::result::get_solution(classical_ising);
   const auto result_spin_poly = openjij::result::get_solution(classical_ising_poly);

   //Check both equal
   EXPECT_EQ(result_spin_poly.size(), result_spin.size());
   for (std::size_t i = 0; i < result_spin_poly.size(); ++i) {
      EXPECT_EQ(result_spin_poly[i], result_spin[i]);
   }
   
   EXPECT_DOUBLE_EQ(poly_graph.energy(result_spin_poly), interaction.calc_energy(result_spin));
}

TEST(PolyUpdater, SingleSpinFlipBINARY) {
   
   const int seed = 1;
   const int system_size = 9;
   
   cimod::BinaryPolynomialModel<int, double> cimod_bpm({}, cimod::Vartype::BINARY);
   
   auto engin_for_interaction = std::mt19937(seed);
   auto urd = std::uniform_real_distribution<>(-1.0/system_size, 1.0/system_size);
   
   for (int i = 0; i < system_size; ++i) {
      for (int j = i + 1; j < system_size; ++j) {
         cimod_bpm.AddInteraction({i, j}, urd(engin_for_interaction));
      }
   }
   
   auto interaction_binary = openjij::graph::Polynomial<double>(cimod_bpm.ToSerializable());
   cimod_bpm.ChangeVartype(cimod::Vartype::SPIN);
   auto interaction_spin   = openjij::graph::Sparse<double>(system_size);
   
   for (const auto &it: cimod_bpm.GetPolynomial()) {
      if (it.first.size() == 2) {
         interaction_spin.J(it.first[0], it.first[1]) = it.second;
      }
      else if (it.first.size() == 1) {
         interaction_spin.h(it.first[0]) = it.second;
      }
   }
   
   const auto offset = cimod_bpm.GetOffset();
   
   auto engine_for_spin   = std::mt19937(seed);
   auto engine_for_binary = std::mt19937(seed);
   const auto spin   = interaction_spin.gen_spin(engine_for_spin);
   const auto binary = interaction_binary.gen_binary(engine_for_binary);

   auto classical_ising      = openjij::system::make_classical_ising(spin, interaction_spin);
   auto classical_ising_poly = openjij::system::make_classical_ising_polynomial(binary, interaction_binary, "BINARY");
   
   auto random_numder_engine = std::mt19937(seed);
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising     , random_numder_engine, generate_schedule_list());
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_poly, random_numder_engine, generate_schedule_list());
   const auto result_spin   = openjij::result::get_solution(classical_ising);
   const auto result_binary = openjij::result::get_solution(classical_ising_poly);

   //Check both equal
   EXPECT_EQ(result_binary.size(), result_spin.size());
   for (std::size_t i = 0; i < result_binary.size(); ++i) {
      EXPECT_EQ(result_binary[i], (result_spin[i] + 1)/2);
   }
   
   EXPECT_DOUBLE_EQ(interaction_binary.energy(result_binary), interaction_spin.calc_energy(result_spin) + offset);
}


TEST(PolyUpdater, KLocal1) {
   const int seed = 1;
   openjij::graph::Index num_spins = 30;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}) = -1;
   
   auto engine_for_binary = std::mt19937(seed);
   
   openjij::graph::Binaries binary = poly_graph.gen_binary(engine_for_binary);
   
   auto poly_system = openjij::system::make_k_local_polynomial(binary, poly_graph);
   
   auto random_numder_engine = std::mt19937(seed);
   
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, generate_schedule_list());
   
   const auto result_binary_poly = openjij::result::get_solution(poly_system);
   
   for (const auto &binary: result_binary_poly) {
      EXPECT_EQ(binary, 1);
   }
   
}

TEST(PolyUpdater, KLocal2) {
   const int seed = 1;
   openjij::graph::Index num_spins = 30;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}) = +1;
   poly_graph.J({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,   17,18,19,20,21,22,23,24,25,26,27,28,29}) = -1;

   auto engine_for_binary = std::mt19937(seed);
   
   openjij::graph::Binaries binary = poly_graph.gen_binary(engine_for_binary);
   
   auto poly_system = openjij::system::make_k_local_polynomial(binary, poly_graph);
      
   auto random_numder_engine = std::mt19937(seed);
   
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, generate_schedule_list());
   
   const auto result_binary_poly = openjij::result::get_solution(poly_system);
   
   for (std::size_t i = 0; i < result_binary_poly.size(); ++i) {
      if (i == 16) {
         EXPECT_EQ(result_binary_poly[i], 0);
      }
      else {
         EXPECT_EQ(result_binary_poly[i], 1);
      }
   }
}

TEST(PolyUpdater, KLocal3) {
   
   const int seed = 1;
   openjij::graph::Index num_spins = 30;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29) = -1;
   poly_graph.J(0,1,2,3,4,5,6,7,8,  10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29) = +1;
   poly_graph.J(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,   24,25,26,27,28,29) = +1;
   poly_graph.J(0,1,2,3,4,5,6,7,8,  10,11,12,13,14,15,16,17,18,19,20,21,22,   24,25,26,27,28,29) = -1;

      
   auto engine_for_binary = std::mt19937(seed);
   
   openjij::graph::Binaries binary = poly_graph.gen_binary(engine_for_binary);
   
   auto poly_system = openjij::system::make_k_local_polynomial(binary, poly_graph);
      
   auto random_numder_engine = std::mt19937(seed);
      
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, generate_schedule_list());
   
   const auto result_binary_poly = openjij::result::get_solution(poly_system);
   
   for (std::size_t i = 0; i < result_binary_poly.size(); ++i) {
      if (i == 9 || i == 23) {
         EXPECT_EQ(result_binary_poly[i], 0);
      }
      else {
         EXPECT_EQ(result_binary_poly[i], 1);
      }
   }

}

//swendsen-wang test
TEST(SwendsenWang, FindTrueGroundState_ClassicalIsing_Sparse_OneDimensionalIsing) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = [](){
        auto interaction = graph::Sparse<double>(num_system_size);
        interaction.J(0,1) = -1;
        interaction.J(1,2) = -1;
        interaction.J(2,3) = -1;
        interaction.J(3,4) = -1;
        interaction.J(4,5) = +1;
        interaction.J(5,6) = +1;
        interaction.J(6,7) = +1;
        interaction.h(0) = +1;
        return interaction;
    }();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); //default: no eigen implementation

    auto random_number_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SwendsenWang>::run(classical_ising, random_number_engine, schedule_list);

    EXPECT_EQ(openjij::graph::Spins({-1, -1, -1, -1, -1, +1, -1, +1}), result::get_solution(classical_ising));
}

TEST(SwendsenWang, FindTrueGroundState_ClassicalIsing_Sparse) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse<double>>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); //with eigen implementation

    auto random_numder_engine = std::mt19937(1);

    //in general swendsen wang is not efficient in simulating frustrated systems. We need more annealing time.
    const auto schedule_list = openjij::utility::make_classical_schedule_list(0.01, 100.0, 100, 3000);

    algorithm::Algorithm<updater::SwendsenWang>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}


// Continuous time Swendsen-Wang test
TEST(ContinuousTimeSwendsenWang, Place_Cuts) {
    using namespace openjij;
    using TimeType = typename system::ContinuousTimeIsing<graph::Sparse<double>>::TimeType;
    using CutPoint = typename system::ContinuousTimeIsing<graph::Sparse<double>>::CutPoint;

    std::vector<CutPoint> timeline;
    timeline.emplace_back(1.0, 1);
    timeline.emplace_back(2.0, 2);
    timeline.emplace_back(3.0, 2);
    timeline.emplace_back(4.0, 3);
    timeline.emplace_back(5.0, 3);
    timeline.emplace_back(6.0, 4);
    timeline.emplace_back(7.0, 4);

    std::vector<TimeType> cuts { 0.5, 1.5, 3.5, 4.5, 5.5, 7.5, 8.5 };
    timeline = updater::ContinuousTimeSwendsenWang<system::ContinuousTimeIsing<graph::Sparse<double>>>::create_timeline(timeline, cuts);

    std::vector<CutPoint> correct_timeline;
    correct_timeline.emplace_back(0.5, 4);
    correct_timeline.emplace_back(1.0, 1);
    correct_timeline.emplace_back(1.5, 1);
    correct_timeline.emplace_back(2.0, 2);
    correct_timeline.emplace_back(3.5, 2);
    correct_timeline.emplace_back(4.0, 3);
    correct_timeline.emplace_back(4.5, 3);
    correct_timeline.emplace_back(5.5, 3);
    correct_timeline.emplace_back(6.0, 4);
    correct_timeline.emplace_back(7.5, 4);
    correct_timeline.emplace_back(8.5, 4);

    EXPECT_EQ(timeline, correct_timeline);
}

TEST(ContinuousTimeSwendsenWang, Place_Cuts_Special_Case) {
    using namespace openjij;
    using TimeType = typename system::ContinuousTimeIsing<graph::Sparse<double>>::TimeType;
    using CutPoint = typename system::ContinuousTimeIsing<graph::Sparse<double>>::CutPoint;

    std::vector<CutPoint> timeline { {1.0, 1}, {2.0, 1} };

    std::vector<TimeType> cuts { };
    timeline = updater::ContinuousTimeSwendsenWang<system::ContinuousTimeIsing<graph::Sparse<double>>>::create_timeline(timeline, cuts);
    std::vector<CutPoint> correct_timeline { {1.0, 1} };

    EXPECT_EQ(timeline, correct_timeline);
}

/*************** currently disabled *************

TEST(ContinuousTimeSwendsenWang, FindTrueGroundState_ContinuousTimeIsing_Sparse_OneDimensionalIsing) {
    using namespace openjij;

    const auto interaction = [](){
        auto interaction = graph::Sparse<double>(num_system_size);
        interaction.J(0,1) = -1;
        interaction.J(1,2) = -1;
        interaction.J(2,3) = -1;
        interaction.J(3,4) = -1;
        interaction.J(4,5) = +1;
        interaction.J(5,6) = +1;
        interaction.J(6,7) = +1;
        interaction.h(0) = +1;
        return interaction;
    }();

    auto engine_for_spin = std::mt19937(1);
    const auto spins = interaction.gen_spin(engine_for_spin);

    auto ising = system::make_continuous_time_ising(spins, interaction, 1.0);

    //auto random_numder_engine = std::mt19937(1);
    //const auto schedule_list = utility::make_transverse_field_schedule_list(10, 100, 100);

    //algorithm::Algorithm<updater::ContinuousTimeSwendsenWang>::run(ising, random_numder_engine, schedule_list);

    //EXPECT_EQ(openjij::graph::Spins({-1, -1, -1, -1, -1, +1, -1, +1}), result::get_solution(ising));
}

TEST(ContinuousTimeSwendsenWang, FindTrueGroundState_ContinuousTimeIsing_Sparse) {
    using namespace openjij;

    const auto interaction = generate_interaction<graph::Sparse<double>>();
    auto engine_for_spin = std::mt19937(1);

    const auto spins = interaction.gen_spin(engine_for_spin);

    auto ising = system::make_continuous_time_ising(spins, interaction, 1.0);

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = utility::make_transverse_field_schedule_list(10, 100, 3000);

    algorithm::Algorithm<updater::ContinuousTimeSwendsenWang>::run(ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(ising));
}

**********************************/

TEST(RESULT, GetSolutionFromTrotter){
    auto graph = openjij::graph::Dense<float>(4);
    graph.J(1, 1) = -1.0;
    graph.J(0, 1) = -1.0;
    graph.J(1, 2) = -1.0;
    graph.J(2, 3) = 1.0;

    auto r = openjij::utility::Xorshift(1234);
    int num_trotter_slices = 4;
    openjij::system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = graph.gen_spin(r);
    }

    init_trotter_spins[0] = openjij::graph::Spins({1, 1, 1, -1});

    auto q_sys = openjij::system::make_transverse_ising(init_trotter_spins, graph, 1.0);
    // get_solution get minimum energy state
    auto solution = openjij::result::get_solution(q_sys);
    EXPECT_EQ(solution, init_trotter_spins[0]);
}

TEST(RESULT, GetSolutionFromChimera){
    auto graph = openjij::graph::Chimera<float>(1,1);
    graph.h(0, 0, 0) = 1.0;
    graph.J(0, 0, 0, openjij::graph::ChimeraDir::IN_0or4) = -1.0;
    graph.J(0, 0, 4, openjij::graph::ChimeraDir::IN_1or5) = -1.0;
    graph.J(0, 0, 2, openjij::graph::ChimeraDir::IN_2or6) = -1.0;
    graph.J(0, 0, 6, openjij::graph::ChimeraDir::IN_3or7) = -1.0;
    graph.J(0, 0, 3, openjij::graph::ChimeraDir::IN_3or7) = -1.0;

    auto r = openjij::utility::Xorshift(1234);
    int num_trotter_slices = 4;
    openjij::system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = graph.gen_spin(r);
    }

    init_trotter_spins[0] = openjij::graph::Spins({-1,-1,-1,-1,-1,-1,-1,-1});

    auto q_sys = openjij::system::make_transverse_ising(init_trotter_spins, static_cast<openjij::graph::Sparse<float>>(graph), 1.0);
    // get_solution get minimum energy state
    auto solution = openjij::result::get_solution(q_sys);
    EXPECT_EQ(solution, init_trotter_spins[0]);
}


//gpu test

#ifdef USE_CUDA

TEST(GPU, glIdxConsistencyCheck_Chimera) {
    using namespace openjij;

    system::ChimeraInfo info{134,175,231};

    size_t a = 0;

    for(size_t t=0; t<info.trotters; t++){
        for(size_t r=0; r<info.rows; r++){
            for(size_t c=0; c<info.cols; c++){
                for(size_t i=0; i<info.chimera_unitsize; i++){
                    EXPECT_EQ(a, system::chimera_cuda::glIdx(info,r,c,i,t));
                    a++;
                }
            }
        }
    }
}

TEST(GPU, FindTrueGroundState_ChimeraTransverseGPU) {
    using namespace openjij;

    //generate classical chimera system
    const auto interaction = generate_chimera_interaction<float>();
    auto engine_for_spin = std::mt19937(1253);
    std::size_t num_trotter_slices = 1;
    system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = interaction.gen_spin(engine_for_spin);
    }

    auto chimera_quantum_gpu = system::make_chimera_transverse_gpu<1,1,1>(init_trotter_spins, interaction, 1.0); 

    auto random_number_engine = utility::cuda::CurandWrapper<float, CURAND_RNG_PSEUDO_XORWOW>(12356);

    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::GPU>::run(chimera_quantum_gpu, random_number_engine, schedule_list);

    graph::Spins res = result::get_solution(chimera_quantum_gpu);
    
    EXPECT_EQ(get_true_chimera_groundstate(interaction), result::get_solution(chimera_quantum_gpu));
}

TEST(GPU, FindTrueGroundState_ChimeraClassicalGPU) {
    using namespace openjij;

    //generate classical chimera system
    const auto interaction = generate_chimera_interaction<float>();
    auto engine_for_spin = std::mt19937(1264);
    const auto spin = interaction.gen_spin(engine_for_spin);

    auto chimera_classical_gpu = system::make_chimera_classical_gpu<1,1>(spin, interaction); 

    auto random_number_engine = utility::cuda::CurandWrapper<float, CURAND_RNG_PSEUDO_XORWOW>(12356);

    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::GPU>::run(chimera_classical_gpu, random_number_engine, schedule_list);

    graph::Spins res = result::get_solution(chimera_classical_gpu);
    
    EXPECT_EQ(get_true_chimera_groundstate(interaction), result::get_solution(chimera_classical_gpu));
}

#endif

//utility test

TEST(Eigen, CopyFromVectorToEigenMatrix) {
    using namespace openjij;
    
    std::size_t N = 500;
    auto spins = graph::Spins(N);
    graph::Dense<double> a(N);
    auto r = utility::Xorshift(1234);
    //auto uid = std::uniform_int_distribution<>{0, 1};

    spins = a.gen_spin(r);

    Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
    vec = utility::gen_vector_from_std_vector<double>(spins);

    for(std::size_t i=0; i<N; i++){
        EXPECT_EQ(vec(i), spins[i]);
    }

    EXPECT_EQ(vec(N), 1);
}

TEST(Eigen, CopyFromTrotterSpinToEigenMatrix) {
    using namespace openjij;
    
    std::size_t N = 500;
    std::size_t num_trot = 10;

    auto trotter_spins = system::TrotterSpins(num_trot);
    graph::Dense<double> a(N);
    auto r = utility::Xorshift(1234);

    for(auto& spins : trotter_spins){
        spins = a.gen_spin(r);
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
    mat = utility::gen_matrix_from_trotter_spins<double>(trotter_spins);

    //initialize spin
    for(size_t j=0; j<num_trot; j++){
        for(size_t i=0; i<N; i++){
            EXPECT_EQ(mat(i,j), trotter_spins[j][i]); //ith spin in jth trotter slice
        }
    }

    //dummy spins
    for(size_t j=0; j<num_trot; j++){
        EXPECT_EQ(mat(N,j), 1);
    }
}

TEST(Eigen, CopyFromGraphToEigenMatrix) {
    using namespace openjij;

    std::size_t N = 500;
    graph::Dense<double> a(N);
    
    //generate dense matrix
    auto r = utility::Xorshift(1234);
    auto urd = std::uniform_real_distribution<>{-10, 10};
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            a.J(i, j)  = urd(r);
        }
    }
    
    //copy to Eigen Matrix
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
    mat = utility::gen_matrix_from_graph(a);

    //interaction
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            EXPECT_EQ(mat(i,j), a.J(i,j));
            EXPECT_EQ(mat(j,i), a.J(j,i));
        }
    }

    //local field
    for(std::size_t i=0; i<N; i++){
        EXPECT_EQ(mat(i,N), a.h(i));
        EXPECT_EQ(mat(N,i), a.h(i));
    }

    EXPECT_EQ(mat(N,N), 1);

    graph::Sparse<double> b(N);
    
    //generate sparse matrix
    r = utility::Xorshift(1234);
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            b.J(i, j)  = urd(r);
        }
    }
    
    //copy to Eigen SparseMatrix
    Eigen::SparseMatrix<double> mat_s(N+1, N+1);
    mat_s = utility::gen_matrix_from_graph(b);

    //interaction
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            EXPECT_EQ(mat_s.coeff(i,j), a.J(i,j));
            EXPECT_EQ(mat_s.coeff(j,i), a.J(j,i));
        }
    }

    //local field
    for(std::size_t i=0; i<N; i++){
        EXPECT_EQ(mat_s.coeff(i,N), a.h(i));
        EXPECT_EQ(mat_s.coeff(N,i), a.h(i));
    }

    EXPECT_EQ(mat_s.coeff(N,N), 1);
}

TEST(UnionFind, UniteSevenNodesToMakeThreeSets) {
    auto union_find = openjij::utility::UnionFind(7);

    union_find.unite_sets(0,1);
    union_find.unite_sets(1,4);
    union_find.unite_sets(3,5);
    union_find.unite_sets(5,6);

    auto expect = std::vector<decltype(union_find)::Node>{1,1,2,5,1,5,5};
    for (std::size_t node = 0; node < 7; ++node) {
        EXPECT_EQ(union_find.find_set(node), expect[node]);
    }
}

TEST(UnionFind, EachNodeIsInEachClusterByDefault) {
    auto union_find = openjij::utility::UnionFind(7);

    auto expect = std::vector<decltype(union_find)::Node>{0,1,2,3,4,5,6};
    for (std::size_t node = 0; node < 7; ++node) {
        EXPECT_EQ(union_find.find_set(node), expect[node]);
    }
}

TEST(UnionFind, ConnectingEachNodeAndAllAdjacentNodesResultsInOneSet) {
    auto union_find = openjij::utility::UnionFind(7);

    for (std::size_t node = 0; node < 6; ++node) {
        union_find.unite_sets(node, node+1);
    }

    auto expect = std::vector<decltype(union_find)::Node>{1,1,1,1,1,1,1};
    for (std::size_t node = 0; node < 7; ++node) {
        EXPECT_EQ(union_find.find_set(node), expect[node]);
    }
}

#ifdef USE_CUDA

TEST(GPUUtil, UniqueDevPtrTest){
    constexpr std::size_t SIZE = 10000;
    using namespace openjij;
    auto urd = std::uniform_real_distribution<float>{-10, 10};
    auto r = utility::Xorshift(1234);
    std::vector<float> input(SIZE);
    std::vector<float> output(SIZE);
    for(std::size_t i=0; i<SIZE; i++){
        input[i] = urd(r);
    }
    auto gpu_mem = utility::cuda::make_dev_unique<float[]>(SIZE);
    HANDLE_ERROR_CUDA(cudaMemcpy(gpu_mem.get(), input.data(), SIZE*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR_CUDA(cudaMemcpy(output.data(), gpu_mem.get(), SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_EQ(input, output);

}

TEST(GPUUtil, UniqueHostPtrTest){
    constexpr std::size_t SIZE = 10000;
    using namespace openjij;
    auto urd = std::uniform_real_distribution<float>{-10, 10};
    auto r = utility::Xorshift(1234);
    auto input = utility::cuda::make_host_unique<float[]>(SIZE);
    auto output = utility::cuda::make_host_unique<float[]>(SIZE);
    for(std::size_t i=0; i<SIZE; i++){
        input[i] = urd(r);
    }
    auto gpu_mem = utility::cuda::make_dev_unique<float[]>(SIZE);
    HANDLE_ERROR_CUDA(cudaMemcpy(gpu_mem.get(), input.get(), SIZE*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR_CUDA(cudaMemcpy(output.get(), gpu_mem.get(), SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    for(std::size_t i=0; i<SIZE; i++){
        EXPECT_EQ(input[i], output[i]);
    }
}

TEST(GPUUtil, CurandWrapperTest){
    constexpr std::size_t SIZE = 10000;
    using namespace openjij;
    auto wrap = utility::cuda::CurandWrapper<float, CURAND_RNG_PSEUDO_XORWOW>(1234);
    auto output = utility::cuda::make_host_unique<float[]>(SIZE);
    auto gpu_mem = utility::cuda::make_dev_unique<float[]>(SIZE);
    wrap.generate_uniform(SIZE, gpu_mem);
    HANDLE_ERROR_CUDA(cudaMemcpy(output.get(), gpu_mem.get(), SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_TRUE(0 <= output[0] && output[0] <= 1);
    for(std::size_t i=1; i<SIZE; i++){
        EXPECT_NE(output[i-1], output[i]);
        EXPECT_TRUE(0 <= output[i] && output[i] <= 1);
    }
}
TEST(GPUUtil, CuBLASWrapperTest){
    using namespace openjij;
    constexpr std::size_t M = 1000;
    constexpr std::size_t K = 205;
    constexpr std::size_t N = 6;

    //Note: matrix in cuBLAS in column major
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_A(M, K);

    host_A = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Random(M, K);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_B(K, N);

    host_B = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>::Random(K, N);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_C_answer(M, N);
    host_C_answer = host_A * host_B;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> host_C(M, N);

    //copy to gpu
    auto A = utility::cuda::make_dev_unique<float[]>(M*K);
    auto B = utility::cuda::make_dev_unique<float[]>(K*N);
    auto C = utility::cuda::make_dev_unique<float[]>(M*N);

    HANDLE_ERROR_CUDA(cudaMemcpy(A.get(), host_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR_CUDA(cudaMemcpy(B.get(), host_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));

    auto cublas = utility::cuda::CuBLASWrapper();

    //matrix product
    cublas.matmul(M, K, N, A, B, C);

    HANDLE_ERROR_CUDA(cudaMemcpy(host_C.data(), C.get(), M*N*sizeof(float), cudaMemcpyDeviceToHost));
    for(std::size_t i=0; i<M; i++){
        for(std::size_t j=0; j<N; j++){
            EXPECT_NEAR(host_C(i,j), host_C_answer(i,j), 1e-5);
        }
    }

    //Iamax test
    const size_t SIZE = 10000;
    auto r = utility::Xorshift(23456678);
    auto urd = std::uniform_real_distribution<float>{0, 10};
    std::vector<float> host_vec(SIZE);
    auto device_vec = utility::cuda::make_dev_unique<float[]>(SIZE);
    for(auto&& elem : host_vec){
        elem = urd(r);
    }

    HANDLE_ERROR_CUDA(cudaMemcpy(device_vec.get(), host_vec.data(), SIZE*sizeof(float), cudaMemcpyHostToDevice));

    //index
    int host_idx;
    int dev_idx;
    auto device_idx = utility::cuda::make_dev_unique<int[]>(1);

    //calc maxind (host)
    host_idx = std::distance(host_vec.begin(), std::max_element(host_vec.begin(), host_vec.end()));
    //calc maxind (device)
    cublas.absmax_val_index(SIZE, device_vec, device_idx);
    HANDLE_ERROR_CUDA(cudaMemcpy(&dev_idx, device_idx.get(), 1*sizeof(int), cudaMemcpyDeviceToHost));
    //NOTE: max_val will return 1-indexed value!!
    EXPECT_EQ(host_idx, dev_idx-1);

    //dot product test
    std::vector<float> host_vec2(SIZE);
    auto device_vec2 = utility::cuda::make_dev_unique<float[]>(SIZE);
    urd = std::uniform_real_distribution<float>{0, 1};
    for(auto&& elem : host_vec2){
        elem = urd(r);
    }
    HANDLE_ERROR_CUDA(cudaMemcpy(device_vec2.get(), host_vec2.data(), SIZE*sizeof(float), cudaMemcpyHostToDevice));
    float dotprod = 0;
    float dev_dotprod;
    auto device_dotprod = utility::cuda::make_dev_unique<float[]>(1);
    for(std::size_t i=0; i<SIZE; i++){
        dotprod += host_vec[i]*host_vec2[i];
    }
    cublas.dot(SIZE, device_vec, device_vec2, device_dotprod);
    HANDLE_ERROR_CUDA(cudaMemcpy(&dev_dotprod, device_dotprod.get(), 1*sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_NEAR(dev_dotprod/1000.0, dotprod/1000.0, 1e-4);
}

#endif

