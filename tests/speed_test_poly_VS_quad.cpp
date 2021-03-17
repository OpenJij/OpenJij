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

static constexpr std::size_t num_system_size = 4;
#define TEST_CASE_INDEX 1
#include "./testcase.hpp"
static openjij::utility::ClassicalScheduleList generate_schedule_list() {
    return openjij::utility::make_classical_schedule_list(1, 1000.0, 100, 100);
}

static openjij::utility::ClassicalScheduleList generate_schedule_list(double beta, int mc_step) {
   auto list = openjij::utility::ClassicalScheduleList(1);
   list[0].one_mc_step = mc_step;
   list[0].updater_parameter = openjij::utility::ClassicalUpdaterParameter(beta);
   return list;
}
/*
int main() {
   
   auto system_size = 2000;
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
   return 0;
}
*/
std::vector<openjij::graph::Spin> GetSpinState(std::size_t basis, std::size_t system_size);
//BinaryPolynomialModel
TEST(BPM, test) {
   /*
   const auto interaction = generate_polynomial_interaction<openjij::graph::Polynomial<double>>();
   auto engine_for_spin = std::mt19937(1);
   auto spin = interaction.gen_spin(engine_for_spin);
   
   spin[0] = -1;
   spin[1] = -1;
   spin[2] = -1;
   spin[3] = -1;

   auto classical_ising_polynomial = openjij::system::make_classical_ising_polynomial(spin, interaction);
   
   auto random_numder_engine = std::mt19937(1);
   const auto schedule_list = generate_schedule_list(100, 1);
   //const auto schedule_list = generate_schedule_list();
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_polynomial, random_numder_engine, schedule_list);

   for (const auto &it: GetExactGroundState(interaction, num_system_size)) {
      std::cout << it << std::endl;
   }
   printf("DDDDDD\n");
   for (const auto &it: openjij::result::get_solution(classical_ising_polynomial)) {
      std::cout << it << std::endl;
   }
   
   printf("dd\n");
   std::cout << interaction.CalclateEnergy(openjij::result::get_solution(classical_ising_polynomial)) << std::endl;
   printf("dddd\n");
    */
}

/*
TEST(Polynomial, PolynomialDense) {
   std::size_t system_size = 3;
   auto interaction = openjij::graph::Polynomial<double>(system_size);
   auto engine_for_interaction = std::mt19937(3);
   std::uniform_int_distribution<> rand(-1, 1);
   for (std::size_t num_interactions = 1; num_interactions <= system_size; ++num_interactions) {
      std::vector<std::unordered_set<std::size_t>> inter = Combination(system_size, num_interactions);
      for (const auto &it: inter) {
         interaction.J(it) = rand(engine_for_interaction);
      }
   }
   
   auto engine_for_spin = std::mt19937(1);
   const auto spin = interaction.gen_spin(engine_for_spin);
   auto classical_ising_polynomial = openjij::system::make_classical_ising_polynomial(spin, interaction);
   auto random_numder_engine = std::mt19937(1);
   const auto schedule_list = generate_schedule_list(100,10000);
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_polynomial, random_numder_engine, schedule_list);
   printf("%lf, %lf\n",interaction.CalclateEnergy(openjij::result::get_solution(classical_ising_polynomial)), PolynomialExactGroundStateEnergy(interaction, system_size));
}
*/
/*
TEST(BPM, speed_poly_quad_sparse_interactions) {
   
   for (auto total_loop = 10000; total_loop <= 50000; total_loop += 5000) {
      
      int system_size = 1000;
      double sparse = 0.1;
      auto engine_for_interaction = std::mt19937(3);
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
      
      //std::cout << "  " << interaction.GetInteractions().size();
      
      auto engine_for_spin = std::mt19937(1);
      const auto spin = interaction.gen_spin(engine_for_spin);
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising_polynomial = openjij::system::make_classical_ising_polynomial(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(0.1, 10000);
      
      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_polynomial, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      printf("  %lf\n",interaction.CalclateEnergy(openjij::result::get_solution(classical_ising_polynomial)));
      
   }
   
   //Quadratic Dense
   for (auto total_loop = 10000; total_loop <= 50000; total_loop += 5000) {
      
      int system_size = 1000;
      double sparse = 0.1;
      auto engine_for_interaction = std::mt19937(3);
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
      
      //std::cout << "  " << interaction.get_interactions().size();
      
      auto engine_for_spin = std::mt19937(1);
      const auto spin = interaction.gen_spin(engine_for_spin);
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising = openjij::system::make_classical_ising(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(0.1, 10000);
      
      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      printf("  %lf\n", interaction.calc_energy(openjij::result::get_solution(classical_ising)));
      
   }
   
   //Quadratic Sparse
   //auto system_size = 2000;
   for (auto total_loop = 10000; total_loop <= 50000; total_loop += 5000) {
      int system_size = 1000;
      double sparse = 0.1;
      auto engine_for_interaction = std::mt19937(3);
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
      
      //std::cout << "  " << interaction.get_num_edges();
      
      auto engine_for_spin = std::mt19937(1);
      const auto spin = interaction.gen_spin(engine_for_spin);
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising = openjij::system::make_classical_ising(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(0.1, 10000);

      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      printf("  %lf\n", interaction.calc_energy(openjij::result::get_solution(classical_ising)));
   }
   
}
 
/*
TEST(BPM, speed_quad_dense) {
   
   //auto system_size = 2000;
   for (auto system_size = 100; system_size <= 2000; system_size += 100) {
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
      printf("  %lf\n", interaction.calc_energy(openjij::result::get_solution(classical_ising)));
   }
 
}
 
*/

TEST(BPM, speed_quad_sparse) {
   
   //auto system_size = 2000;
   for (auto system_size = 100; system_size <= 500; system_size += 100) {
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
      printf("  %lf\n", interaction.calc_energy(openjij::result::get_solution(classical_ising)));
   }
    
}
 

TEST(BPM, speed_poly_quad_dense_intetactions) {
   
   //auto system_size = 2000;
   for (auto system_size = 100; system_size <= 500; system_size += 100) {
      auto interaction = openjij::graph::Polynomial<double>(system_size, "SPIN");
      std::cout << system_size;
      auto begin = std::chrono::high_resolution_clock::now();
      for (auto i = 0; i < system_size; ++i) {
         for (auto j = i + 1; j < system_size; ++j) {
            for (auto k = j + 1; k < system_size; ++k) {
               interaction.J(i,j,k) = +1;
            }
         }
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      auto engine_for_spin = std::mt19937(1);
      auto spin = interaction.gen_spin(engine_for_spin);
      
  
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_ising_polynomial = openjij::system::make_classical_ising_polynomial(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      auto random_numder_engine = std::mt19937(1);
      const auto schedule_list = generate_schedule_list(1, 1000);
      
      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_polynomial, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      printf("  %lf\n",interaction.CalclateEnergy(openjij::result::get_solution(classical_ising_polynomial)));

      
      
      
      std::cout << system_size;
      begin = std::chrono::high_resolution_clock::now();
      interaction.SPIN_to_BINARY();
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      engine_for_spin = std::mt19937(1);
      spin = interaction.gen_spin(engine_for_spin);
      for (auto &&it: spin) {
         if (it == -1) {
            it = 0;
         }
      }
      
      begin = std::chrono::high_resolution_clock::now();
      auto classical_binary_polynomial = openjij::system::make_classical_ising_polynomial(spin, interaction);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      random_numder_engine = std::mt19937(1);
      begin = std::chrono::high_resolution_clock::now();
      openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_binary_polynomial, random_numder_engine, schedule_list);
      end = std::chrono::high_resolution_clock::now();
      std::cout << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
      
      printf("  %lf\n",interaction.CalclateEnergy(openjij::result::get_solution(classical_binary_polynomial)));

      
      
   }
}


