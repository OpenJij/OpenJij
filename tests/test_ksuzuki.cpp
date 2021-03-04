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
    return openjij::utility::make_classical_schedule_list(10.0, 100.0, 100, 100);
}

static openjij::utility::ClassicalScheduleList generate_schedule_list(double beta, int mc_step) {
   auto list = openjij::utility::ClassicalScheduleList(1);
   list[0].one_mc_step = mc_step;
   list[0].updater_parameter = openjij::utility::ClassicalUpdaterParameter(beta);
   return list;
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
   
   /*
   const auto interaction = generate_polynomial_interaction<openjij::graph::Polynomial<double>>();
   auto engine_for_spin = std::mt19937(1);
   auto spin = interaction.gen_spin(engine_for_spin);
   
   spin[0] = 1;
   spin[1] = 1;
   spin[2] = -1;
   spin[3] = -1;
   spin[4] = 1;
   spin[5] = 1;
   spin[6] = -1;
   spin[7] = -1;
   
   for (std::size_t index = 0; index < num_system_size; ++index) {
      printf("Spin[%ld] = %d\n", index, spin[index]);
   }
   
   auto classical_ising_polynomial = openjij::system::make_classical_ising_polynomial(spin, interaction);
   
   auto random_numder_engine = std::mt19937(1);
   const auto schedule_list = generate_schedule_list(100, 1);
   
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_polynomial, random_numder_engine, schedule_list);
   
   for (const auto &it: openjij::result::get_solution(classical_ising_polynomial)) {
      std::cout << it << std::endl;
   }
   */
   
   
   
}

TEST(BPM, speed_quad) {
   
   auto system_size = 20;
   auto interaction = openjij::graph::Sparse<double>(system_size);
   
   auto begin = std::chrono::high_resolution_clock::now();
   for (auto i = 0; i < system_size; ++i) {
      for (auto j = i + 1; j < system_size; ++j) {
         interaction.J(i,j) = +1;
      }
   }
   auto end = std::chrono::high_resolution_clock::now();
   std::cout << "time1-1: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << std::endl;
   
   auto engine_for_spin = std::mt19937(1);
   const auto spin = interaction.gen_spin(engine_for_spin);
   
   begin = std::chrono::high_resolution_clock::now();
   auto classical_ising = openjij::system::make_classical_ising(spin, interaction);
   end = std::chrono::high_resolution_clock::now();
   std::cout << "time1-2: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << std::endl;
   
   auto random_numder_engine = std::mt19937(1);
   const auto schedule_list = generate_schedule_list(10, 10000);

   begin = std::chrono::high_resolution_clock::now();
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);
   end = std::chrono::high_resolution_clock::now();
   std::cout << "time1-3: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << std::endl;
   
   for (const auto &it: openjij::result::get_solution(classical_ising)) {
      std::cout << it << std::endl;
   }
       
}

TEST(BPM, speed_poly) {
   
   auto system_size = 20;
   auto interaction = openjij::graph::Polynomial<double>(system_size);
   
    
   auto begin = std::chrono::high_resolution_clock::now();
   for (auto i = 0; i < system_size; ++i) {
      for (auto j = i + 1; j < system_size; ++j) {
         interaction.J(i,j) = +1;
      }
   }
   auto end = std::chrono::high_resolution_clock::now();
   std::cout << "time2-1: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << std::endl;
   
   auto engine_for_spin = std::mt19937(1);
   const auto spin = interaction.gen_spin(engine_for_spin);
   
   begin = std::chrono::high_resolution_clock::now();
   auto classical_ising_polynomial = openjij::system::make_classical_ising_polynomial(spin, interaction);
   end = std::chrono::high_resolution_clock::now();
   std::cout << "time2-2: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << std::endl;
   
   auto random_numder_engine = std::mt19937(1);
   const auto schedule_list = generate_schedule_list(10, 10000);

   begin = std::chrono::high_resolution_clock::now();
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_polynomial, random_numder_engine, schedule_list);
   end = std::chrono::high_resolution_clock::now();
   std::cout << "time2-3: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << std::endl;
   
   for (const auto &it: openjij::result::get_solution(classical_ising_polynomial)) {
      std::cout << it << std::endl;
   }
      
}
 
