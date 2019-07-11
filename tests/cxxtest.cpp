// include Google Test
#include <gtest/gtest.h>
#include <gmock/gmock.h> // include STL
#include <iostream>
#include <utility>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

// include OpenJij
#include <graph/all.hpp>
#include <system/all.hpp>
#include <updater/all.hpp>
#include <algorithm/algorithm.hpp>
#include <result/get_solution.hpp>
#include <utility/schedule_list.hpp>
#include <utility/union_find.hpp>
#include <utility/random.hpp>

// #####################################
// helper functions
// #####################################
/**
 * @brief generate interaction
 *
 * @return classical interaction which represents specific optimization problem
 */

static constexpr std::size_t num_system_size = 8;

//GraphType -> Dense or Sparse
template<template<class> class GraphType>
static GraphType<double> generate_interaction() {
    auto interaction = GraphType<double>(num_system_size);
    interaction.J(0,0)=-0.1;
    interaction.J(0,1)=-0.9;
    interaction.J(0,2)=0.2;
    interaction.J(0,3)=0.1;
    interaction.J(0,4)=1.3;
    interaction.J(0,5)=0.8;
    interaction.J(0,6)=0.9;
    interaction.J(0,7)=0.4;
    interaction.J(1,1)=-0.7;
    interaction.J(1,2)=-1.6;
    interaction.J(1,3)=1.5;
    interaction.J(1,4)=1.5;
    interaction.J(1,5)=1.2;
    interaction.J(1,6)=-1.5;
    interaction.J(1,7)=-1.7;
    interaction.J(2,2)=-0.6;
    interaction.J(2,3)=1.2;
    interaction.J(2,4)=-1.3;
    interaction.J(2,5)=-0.5;
    interaction.J(2,6)=-1.9;
    interaction.J(2,7)=1.2;
    interaction.J(3,3)=0.8;
    interaction.J(3,4)=-0.5;
    interaction.J(3,5)=-0.4;
    interaction.J(3,6)=-1.8;
    interaction.J(3,7)=-2.0;
    interaction.J(4,4)=0.6;
    interaction.J(4,5)=-2.0;
    interaction.J(4,6)=-1.9;
    interaction.J(4,7)=0.5;
    interaction.J(5,5)=-1.8;
    interaction.J(5,6)=-1.2;
    interaction.J(5,7)=1.8;
    interaction.J(6,6)=0.3;
    interaction.J(6,7)=1.4;
    interaction.J(7,7)=1.8;
    return interaction;
}

static openjij::graph::Spins get_true_groundstate(){
    return openjij::graph::Spins({-1, -1, 1, 1, 1, 1, 1, -1});
}

static openjij::utility::ClassicalScheduleList generate_schedule_list(){
    return openjij::utility::make_classical_schedule_list(0.1, 100.0, 100, 100);
}

static openjij::utility::TransverseFieldScheduleList generate_tfm_schedule_list(){
    return openjij::utility::make_transverse_field_schedule_list(10, 100, 100);
}

// #####################################
// tests
// #####################################

//graph tests

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
    auto cl_dense = system::make_classical_ising<true>(d.gen_spin(engine_for_spin), d);
    auto cl_sparse = system::make_classical_ising<true>(s.gen_spin(engine_for_spin), s);
    Eigen::MatrixXd m1 = cl_dense.interaction;
    //convert from sparse to dense
    Eigen::MatrixXd m2 = cl_sparse.interaction;
    EXPECT_EQ(m1, m2);
}

//TODO: macro?
//SingleSpinFlip tests

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Dense_NoEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); //default: no eigen implementation

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Sparse_NoEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); //default: no eigen implementation

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Dense_WithEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising<true>(spin, interaction); //Eigen implementation enabled
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_ClassicalIsing_Sparse_WithEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising<true>(spin, interaction); //Eigen implementation enabled
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_Dense_NoEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense>();
    auto engine_for_spin = std::mt19937(1);
    std::size_t num_trotter_slices = 10;

    //generate random trotter spins
    system::TrotterSpins init_trotter_spins(num_trotter_slices);
    for(auto& spins : init_trotter_spins){
        spins = interaction.gen_spin(engine_for_spin);
    }

    auto transverse_ising = system::make_transverse_ising(init_trotter_spins, interaction, 1.0);
    
    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_tfm_schedule_list();

    algorithm::Algorithm<updater::SingleSpinFlip>::run(transverse_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(transverse_ising));
}

TEST(SingleSpinFlip, FindTrueGroundState_TransverseIsing_Sparse_NoEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Sparse>();
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

//swendsen-wang test

TEST(SwendsenWang, FindTrueGroundState_ClassicalIsing_Dense_NoEigenImpl) {
    using namespace openjij;

    //generate classical dense system
    const auto interaction = generate_interaction<graph::Dense>();
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = system::make_classical_ising(spin, interaction); //default: no eigen implementation

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list = generate_schedule_list();

    algorithm::Algorithm<updater::SwendsenWang>::run(classical_ising, random_numder_engine, schedule_list);

    EXPECT_EQ(get_true_groundstate(), result::get_solution(classical_ising));
}

//utility test

TEST(Eigen, CopyFromVectorToEigenMatrix) {
    using namespace openjij;
    
    std::size_t N = 500;
    auto spins = graph::Spins(N);
    graph::Dense<double> a(N);
    auto r = utility::Xorshift(1234);
    auto uid = std::uniform_int_distribution<>{0, 1};

    spins = a.gen_spin(r);

    Eigen::Matrix<double, Eigen::Dynamic, 1> vec(N+1);
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

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat(N+1, num_trot);
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
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat(N+1, N+1);
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

