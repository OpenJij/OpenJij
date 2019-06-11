#include "classical_ising.hpp"
#include "quantum_ising.hpp"
#include "algorithm.hpp"
#include "single_spin_flip.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>
#include <utility>
#include <numeric>
#include <tuple>
#include <vector>

using ::testing::ElementsAre;
using ::testing::_;

TEST(hoge, fuga) {
    auto ci = openjij::system::ClassicalIsing();
    openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(ci, std::vector<double>{1., 2., 3.});

    auto qi = openjij::system::QuantumIsing();
    openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(qi, std::vector<double>{1., 2., 3.});
}
