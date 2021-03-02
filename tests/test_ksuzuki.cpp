//
//  test_ksuzuki.cpp
//  OpenJijXcode
//
//  Created by 鈴木浩平 on 2021/03/01.
//


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

static constexpr std::size_t num_system_size = 8;
#define TEST_CASE_INDEX 1
#include "./testcase.hpp"
static openjij::utility::ClassicalScheduleList generate_schedule_list() {
    return openjij::utility::make_classical_schedule_list(0.1, 100.0, 100, 100);
}

//BinaryPolynomialModel
TEST(BPM, test) {
   /*
   cimod::Polynomial<std::string, double> polynomial_str {
      //linear biases
      {{"a"}, 1.0}, {{"b"}, 2.0}, {{"c"}, 3.0}, {{"d"}, 4.0},
      //quadratic biases
      {{"a", "b"}, 12.0}, {{"a", "c"}, 13.0}, {{"a", "d"}, 14.0},
      {{"b", "c"}, 23.0}, {{"b", "d"}, 24.0},
      {{"c", "d"}, 34.0},
      //polynomial biases
      {{"a", "b", "c"}, 123.0}, {{"a", "b", "d"}, 124.0}, {{"a", "c", "d"}, 134.0},
      {{"b", "c", "d"}, 234.0},
      {{"a", "b", "c", "d"}, 1234.0}
   };
      
   cimod::BinaryPolynomialModel<std::string, double> cimod_bpm(polynomial_str, cimod::Vartype::SPIN);
   openjij::graph::Polynomial<double> bpm(cimod_bpm.to_serializable());
      */
   const auto interaction = generate_polynomial_interaction<openjij::graph::Polynomial<double>>();
   auto engine_for_spin = std::mt19937(1);
   const auto spin = interaction.gen_spin(engine_for_spin);
   auto classical_ising_polynomial = openjij::system::make_classical_ising_polynomial(spin, interaction);
   
   auto random_numder_engine = std::mt19937(1);
   const auto schedule_list = generate_schedule_list();
   
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_polynomial, random_numder_engine, schedule_list);
   
   for (const auto &it: openjij::result::get_solution(classical_ising_polynomial)) {
      std::cout << it << std::endl;
   }
   
   
   
   
}
