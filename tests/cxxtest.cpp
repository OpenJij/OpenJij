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
#include <system/classical_ising.hpp>
#include <system/quantum_ising.hpp>
#include <algorithm/algorithm.hpp>
#include <updater/single_spin_flip.hpp>
#include <updater/swendsen_wang.hpp>
#include <utility/schedule_list.hpp>
#include <utility/union_find.hpp>
#include <graph/all.hpp>

// #####################################
// helper functions
// #####################################
/**
 * @brief generate interaction
 *
 * @param system_size
 *
 * @return classical interaction
 */
openjij::graph::Dense<double> generate_sa_interaction(std::size_t system_size) {
    auto interaction = openjij::graph::Dense<double>(system_size);
    for (std::size_t row = 0; row < system_size; ++row) {
        for (std::size_t col = row+1; col < system_size; ++col) {
            interaction.J(row, col) = -1;
        }
    }
    return interaction;
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


TEST(ClassicalIsing_SingleSpinFlip, StateAtLowTemperatureIsNotEqualToStateAtHighTemperature) {
    constexpr auto N = 10;
    const auto interaction = generate_sa_interaction(N);
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = openjij::system::ClassicalIsing(spin, interaction);

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list1 = openjij::utility::make_classical_schedule_list(0.1, 10.0, 10, 10);

    openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list1);
    const auto target = classical_ising.spin;

    const auto schedule_list2 = [](){
        auto schedule_list = openjij::utility::ClassicalScheduleList(1);
        schedule_list[0].one_mc_step = 20;
        schedule_list[0].updater_parameter = openjij::utility::ClassicalUpdaterParameter(0.01);

        return schedule_list;
    }();
    openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list2);
    const auto expect = classical_ising.spin;

    EXPECT_NE(target, expect);
}

TEST(ClassicalIsing_SingleSpinFlip, StateAtLowTemperatureIsEqualToStateAtLowTemperature) {
    constexpr auto N = 10;
    const auto interaction = generate_sa_interaction(N);
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = openjij::system::ClassicalIsing(spin, interaction);

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list1 = openjij::utility::make_classical_schedule_list(0.1, 100.0, 100, 10);
    openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list1);
    const auto target = classical_ising.spin;

    const auto schedule_list2 = [](){
        auto schedule_list = openjij::utility::ClassicalScheduleList(1);
        schedule_list[0].one_mc_step = 1000;
        schedule_list[0].updater_parameter = openjij::utility::ClassicalUpdaterParameter(100.0);

        return schedule_list;
    }();
    openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list2);
    const auto expect = classical_ising.spin;

    EXPECT_EQ(target, expect);
}

TEST(ClassicalIsing_SwendsenWang, StateAtLowTemperatureIsNotEqualToStateAtHighTemperature) {
    constexpr auto N = 10;
    const auto interaction = generate_sa_interaction(N);
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = openjij::system::ClassicalIsing(spin, interaction);

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list1 = openjij::utility::make_classical_schedule_list(0.1, 10.0, 10, 10);

    openjij::algorithm::Algorithm<openjij::updater::SwendsenWang>::run(classical_ising, random_numder_engine, schedule_list1);
    const auto target = classical_ising.spin;

    const auto schedule_list2 = [](){
        auto schedule_list = openjij::utility::ClassicalScheduleList(1);
        schedule_list[0].one_mc_step = 20;
        schedule_list[0].updater_parameter = openjij::utility::ClassicalUpdaterParameter(0.01);

        return schedule_list;
    }();
    openjij::algorithm::Algorithm<openjij::updater::SwendsenWang>::run(classical_ising, random_numder_engine, schedule_list2);
    const auto expect = classical_ising.spin;

    EXPECT_NE(target, expect);
}

TEST(ClassicalIsing_SwendsenWang, StateAtLowTemperatureIsEqualToStateAtLowTemperature) {
    constexpr auto N = 10;
    const auto interaction = generate_sa_interaction(N);
    auto engine_for_spin = std::mt19937(1);
    const auto spin = interaction.gen_spin(engine_for_spin);
    auto classical_ising = openjij::system::ClassicalIsing(spin, interaction);

    auto random_numder_engine = std::mt19937(1);
    const auto schedule_list1 = openjij::utility::make_classical_schedule_list(0.1, 100.0, 100, 10);
    openjij::algorithm::Algorithm<openjij::updater::SwendsenWang>::run(classical_ising, random_numder_engine, schedule_list1);
    const auto target = classical_ising.spin;

    const auto schedule_list2 = [](){
        auto schedule_list = openjij::utility::ClassicalScheduleList(1);
        schedule_list[0].one_mc_step = 1000;
        schedule_list[0].updater_parameter = openjij::utility::ClassicalUpdaterParameter(100.0);

        return schedule_list;
    }();
    openjij::algorithm::Algorithm<openjij::updater::SwendsenWang>::run(classical_ising, random_numder_engine, schedule_list2);
    const auto expect = classical_ising.spin;

    EXPECT_EQ(target, expect);
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
