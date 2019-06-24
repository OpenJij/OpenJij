// include Google Test
#include <gtest/gtest.h>
#include <gmock/gmock.h>

// include STL
#include <iostream>
#include <utility>
#include <numeric>
#include <tuple>
#include <vector>

// include OpenJij
#include <system/classical_ising.hpp>
#include <system/quantum_ising.hpp>
#include <algorithm/algorithm.hpp>
#include <updater/single_spin_flip.hpp>
#include <utility/schedule_list.hpp>
#include <graph/graph.h>
#include <graph/dense.h>

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
TEST(ClassicalIsing_SingleSpinFlip, StateAtLowTemperatureIsNotEqualToStateAtHighTemperature) {
    constexpr auto N = 10;
    const auto interaction = generate_sa_interaction(N);
    const auto spin = interaction.gen_spin();
    auto classical_ising = openjij::system::ClassicalIsing(spin, interaction);

    const auto schedule_list1 = openjij::utility::make_schedule_list(0.1, 10.0, 10, 10);
    openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, schedule_list1);
    const auto target = classical_ising.spin;

    const auto schedule_list2 = openjij::utility::ScheduleList({{20, 0.01}});
    openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, schedule_list2);
    const auto expect = classical_ising.spin;

    EXPECT_NE(target, expect);
}

TEST(ClassicalIsing_SingleSpinFlip, StateAtLowTemperatureIsEqualToStateAtLowTemperature) {
    constexpr auto N = 10;
    const auto interaction = generate_sa_interaction(N);
    const auto spin = interaction.gen_spin();
    auto classical_ising = openjij::system::ClassicalIsing(spin, interaction);

    const auto schedule_list1 = openjij::utility::make_schedule_list(0.1, 100.0, 10, 10);
    openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, schedule_list1);
    const auto target = classical_ising.spin;

    const auto schedule_list2 = openjij::utility::ScheduleList({{100, 100.0}});
    openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, schedule_list2);
    const auto expect = classical_ising.spin;

    EXPECT_EQ(target, expect);
}
