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

static constexpr std::size_t num_system_size = 7;

//GraphType -> Dense or Sparse
template<template<class> class GraphType>
static GraphType<double> generate_interaction() {
    auto interaction = GraphType<double>(num_system_size);
    interaction.J(0,0) = -0.9999999999999991;
    interaction.J(0,1) = 1.500000000000003;
    interaction.J(0,2) = 1.500000000000003;
    interaction.J(0,3) = -1.3999999999999995;
    interaction.J(0,4) = -1.4999999999999996;
    interaction.J(0,5) = 1.300000000000003;
    interaction.J(0,6) = -1.1999999999999993;
    interaction.J(1,1) = -0.2999999999999985;
    interaction.J(1,2) = -0.2999999999999985;
    interaction.J(1,3) = -1.7999999999999998;
    interaction.J(1,4) = 1.500000000000003;
    interaction.J(1,5) = 1.400000000000003;
    interaction.J(1,6) = -0.9999999999999991;
    interaction.J(2,2) = -0.2999999999999985;
    interaction.J(2,3) = -2.0;
    interaction.J(2,4) = 1.7763568394002505e-15;
    interaction.J(2,5) = 0.40000000000000213;
    interaction.J(2,6) = -1.6999999999999997;
    interaction.J(3,3) = -1.7999999999999998;
    interaction.J(3,4) = -0.1999999999999984;
    interaction.J(3,5) = -1.6999999999999997;
    interaction.J(3,6) = 1.7000000000000033;
    interaction.J(4,4) = -0.49999999999999867;
    interaction.J(4,5) = 1.300000000000003;
    interaction.J(4,6) = 1.400000000000003;
    interaction.J(5,5) = -0.49999999999999867;
    interaction.J(5,6) = 1.400000000000003;
    interaction.J(6,6) = 1.1000000000000028;
    return interaction;
}

static openjij::graph::Spins get_true_groundstate(){
    return openjij::graph::Spins({1, -1, -1, 1, 1, 1, -1});
}

static openjij::utility::ClassicalScheduleList generate_schedule_list(){
    return openjij::utility::make_classical_schedule_list(0.1, 100.0, 100, 100);
}
// #####################################


// #####################################
// tests
// #####################################

//graph tests

TEST(Graph, DenseGraphCheck){
    using namespace openjij::graph;
    std::size_t N = 500;
    Dense<double> a(N);
    double s = 0;
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            a.J(i, j)  = s;
            s+=1./N;
        }
    }
    s = 0;
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(a.J(i, j) , s);
            s+=1./N;
        }
    }
    s = 0;
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(a.J(j, i) , s);
            s+=1./N;
        }
    }
}

TEST(Graph, SparseGraphCheck){
    using namespace openjij::graph;
    std::size_t N = 500;
    Sparse<double> b(N, N-1);
    double s = 0;
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            b.J(i, j) = s;
            s+=1./N;
        }
    }
    s = 0;
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            EXPECT_EQ(b.J(i, j) , s);
            s+=1./N;
        }
    }
    s = 0;
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i+1; j<N; j++){
            EXPECT_EQ(b.J(j, i) , s);
            s+=1./N;
        }
    }
    for(std::size_t i=0; i<N; i++){
        std::size_t tot = 0;
        for(auto&& elem : b.adj_nodes(i)){
            tot += elem;
        }
        EXPECT_EQ(tot, N*(N-1)/2 - i);
    }
    EXPECT_EQ(b.get_num_edges(), N-1);

    Sparse<double> c(N, N);
    s = 0;
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            c.J(j, i) = s;
            s+=1./N;
        }
    }
    s = 0;
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(c.J(i, j) , s);
            s+=1./N;
        }
    }
    s = 0;
    for(std::size_t i=0; i<N; i++){
        for(std::size_t j=i; j<N; j++){
            EXPECT_EQ(c.J(j, i) , s);
            s+=1./N;
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
//ClassicalIsing_Dense_SingleSpinFlip tests

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

