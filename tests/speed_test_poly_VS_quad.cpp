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
#include "./testcase.hpp"

static openjij::utility::ClassicalScheduleList generate_schedule_list(double beta, int mc_step) {
   auto list = openjij::utility::ClassicalScheduleList(1);
   list[0].one_mc_step = mc_step;
   list[0].updater_parameter = openjij::utility::ClassicalUpdaterParameter(beta);
   return list;
}

TEST(SpeedTestPolyVsQuad, QuadraticSparseInteractions) {
   
   //Polynomial
   std::cout << "##Polynomial" << std::endl;
   for (auto total_loop = 100000; total_loop <= 1000000; total_loop += 100000) {
      int system_size = 2000;
      auto engine_for_interaction = std::mt19937(1);
      std::uniform_int_distribution<> rand(0, system_size - 1);
      
      auto interaction = openjij::graph::Polynomial<double>(system_size);
      std::cout << total_loop;
      auto begin = std::chrono::high_resolution_clock::now();
            
      for (auto loop = 0; loop < total_loop; ++loop) {
         std::size_t i = rand(engine_for_interaction);
         std::size_t j = rand(engine_for_interaction);
         interaction.J(std::min(i,j),std::max(i,j)) = +1;
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
            
      auto engine_for_spin = std::mt19937(1);
      const auto spin = interaction.gen_spin(engine_for_spin);
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising_polynomial = openjij::system::make_classical_ising_polynomial(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(1, 10000);
      
      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_polynomial, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      std::cout << "  " << interaction.calc_energy(openjij::result::get_solution(classical_ising_polynomial)) << std::endl;
   }

   //Quadratic Dense
   std::cout << "##Quadratic Dense" << std::endl;
   for (auto total_loop = 100000; total_loop <= 1000000; total_loop += 100000) {
      int system_size = 2000;
      auto engine_for_interaction = std::mt19937(1);
      std::uniform_int_distribution<> rand(0, system_size - 1);
      
      auto interaction = openjij::graph::Dense<double>(system_size);
      std::cout << total_loop;
      auto begin = std::chrono::high_resolution_clock::now();
            
      for (auto loop = 0; loop < total_loop; ++loop) {
         std::size_t i = rand(engine_for_interaction);
         std::size_t j = rand(engine_for_interaction);
         interaction.J(std::min(i,j),std::max(i,j)) = +1;
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
            
      auto engine_for_spin = std::mt19937(1);
      const auto spin = interaction.gen_spin(engine_for_spin);
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising = openjij::system::make_classical_ising(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(1, 10000);
      
      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      std::cout << "  " << interaction.calc_energy(openjij::result::get_solution(classical_ising)) << std::endl;
   }
   
   //Quadratic Sparse
   std::cout << "##Quadratic Sparse" << std::endl;
   for (auto total_loop = 100000; total_loop <= 1000000; total_loop += 100000) {
      int system_size = 2000;
      auto engine_for_interaction = std::mt19937(1);
      std::uniform_int_distribution<> rand(0, system_size - 1);
      
      auto interaction = openjij::graph::Sparse<double>(system_size);
      std::cout << total_loop;
      auto begin = std::chrono::high_resolution_clock::now();
      
      for (auto loop = 0; loop < total_loop; ++loop) {
         std::size_t i = rand(engine_for_interaction);
         std::size_t j = rand(engine_for_interaction);
         interaction.J(std::min(i,j),std::max(i,j)) = +1;
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
         
      auto engine_for_spin = std::mt19937(1);
      const auto spin = interaction.gen_spin(engine_for_spin);
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising = openjij::system::make_classical_ising(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(1, 10000);

      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      std::cout << "  " << interaction.calc_energy(openjij::result::get_solution(classical_ising)) << std::endl;
   }
   
}
 

TEST(SpeedTestPolyVsQuad, QuadraticDenseInteractions) {
   
   std::cout << "##Polynomial" << std::endl;
   for (auto system_size = 100; system_size <= 1000; system_size += 100) {
      auto interaction = openjij::graph::Polynomial<double>(system_size);
      std::cout << system_size;
      auto begin = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < system_size; ++i) {
         for (auto j = i + 1; j < system_size; ++j) {
            interaction.J(i,j) = +1;
         }
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto engine_for_spin = std::mt19937(1);
      const auto spin = interaction.gen_spin(engine_for_spin);
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising = openjij::system::make_classical_ising_polynomial(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(1, 1000);
      
      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      std::cout << "  " << interaction.calc_energy(openjij::result::get_solution(classical_ising)) << std::endl;
   }
   
   std::cout << "##Quadratic Dense" << std::endl;
   for (auto system_size = 100; system_size <= 1000; system_size += 100) {
      auto interaction = openjij::graph::Dense<double>(system_size);
      std::cout << system_size;
      auto begin = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < system_size; ++i) {
         for (auto j = i + 1; j < system_size; ++j) {
            interaction.J(i,j) = +1;
         }
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto engine_for_spin = std::mt19937(1);
      const auto spin = interaction.gen_spin(engine_for_spin);
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising = openjij::system::make_classical_ising(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(1, 1000);
      
      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      std::cout << "  " << interaction.calc_energy(openjij::result::get_solution(classical_ising)) << std::endl;
   }
   
   std::cout << "##Quadratic Sparse" << std::endl;
   for (auto system_size = 100; system_size <= 1000; system_size += 100) {
      auto interaction = openjij::graph::Sparse<double>(system_size);
      std::cout << system_size;
      auto begin = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < system_size; ++i) {
         for (auto j = i + 1; j < system_size; ++j) {
            interaction.J(i,j) = +1;
         }
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto engine_for_spin = std::mt19937(1);
      const auto spin = interaction.gen_spin(engine_for_spin);
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising = openjij::system::make_classical_ising(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(1, 1000);

      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      std::cout << "  " << interaction.calc_energy(openjij::result::get_solution(classical_ising)) << std::endl;
   }
 
}

TEST(SpeedTestPoly, FullyConnectedDense) {
   
   std::cout << "##Polynomial" << std::endl;
   for (auto system_size = 5; system_size <= 17; system_size += 1) {
      auto interaction = openjij::graph::Polynomial<double>(system_size);
      std::cout << system_size;
      auto begin = std::chrono::high_resolution_clock::now();
      std::vector<openjij::graph::Index> temp_vec(system_size);
      for (int i = 0; i < system_size; ++i) {
         temp_vec[i] = i;
      }
      for (auto &it: PolynomialGenerateCombinations(temp_vec)) {
         interaction.J(it) = +1;
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto engine_for_spin = std::mt19937(1);
      const auto spin = interaction.gen_spin(engine_for_spin);
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising = openjij::system::make_classical_ising_polynomial(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(1, 1000);
      
      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      std::cout << "  " << interaction.calc_energy(openjij::result::get_solution(classical_ising)) << std::endl;
   }
   
}
 
