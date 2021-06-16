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
#include <chrono>

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

static openjij::utility::ClassicalScheduleList generate_schedule_list(double beta, int mc_step) {
   auto list = openjij::utility::ClassicalScheduleList(1);
   list[0].one_mc_step = mc_step;
   list[0].updater_parameter = openjij::utility::ClassicalUpdaterParameter(beta);
   return list;
}

TEST(KLocal, test0) {
   
   openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins, cimod::Vartype::BINARY);
   
   poly_graph.J(   {0}   ) = -2;//0
   poly_graph.J( {0, 1}  ) = -1;//1
   poly_graph.J( {0, 2}  ) = +1;//2
   poly_graph.J({0, 1, 2}) = +3;//3
   poly_graph.J(   {1}   ) = -2;//4
   poly_graph.J( {1, 2}  ) = +1.5;//5
   poly_graph.J(   {2}   ) = -6;//6
   
   openjij::graph::Spins spin = {0, 0, 1};
   
   auto poly_system = openjij::system::make_k_local_polynomial(spin, poly_graph);
   
   poly_system.print_adj();
   
   const int seed = 1;
   
   auto random_numder_engine = std::mt19937(seed);
   const auto schedule_list = generate_schedule_list(100, 1);
   poly_system.print_adj();
   poly_system.print_dE();
   poly_system.print_zero_count();
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);
   poly_system.print_dE();
   poly_system.print_zero_count();
   //openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);

   //poly_system.PrintInfo();
   
   const auto result_spin_poly = openjij::result::get_solution(poly_system);
   for (std::size_t i = 0; i < result_spin_poly.size(); ++i) {
      printf("Result_spin[%ld]=%d\n", i , result_spin_poly[i]);
   }
   
   
}

TEST(KLocal, test1) {
   
   openjij::graph::Index num_spins = 10;
   openjij::graph::Polynomial<double> poly_graph(num_spins, cimod::Vartype::BINARY);
   
   poly_graph.J({0,1,2,3,4,5,6,7,8,9}) = -1;
   
   openjij::graph::Spins spin = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
   
   auto poly_system = openjij::system::make_k_local_polynomial(spin, poly_graph);
   
   //poly_system.PrintInfo();
   
   const int seed = 1;
   
   auto random_numder_engine = std::mt19937(seed);
   const auto schedule_list = generate_schedule_list(100, 1);
   
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);
   
   //openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);

   //poly_system.PrintInfo();
   
   const auto result_spin_poly = openjij::result::get_solution(poly_system);
   for (std::size_t i = 0; i < result_spin_poly.size(); ++i) {
      printf("Result_spin[%ld]=%d\n", i , result_spin_poly[i]);
   }
   
}

TEST(KLocal, test2) {
   
   openjij::graph::Index num_spins = 10;
   openjij::graph::Polynomial<double> poly_graph(num_spins, cimod::Vartype::BINARY);
   
   poly_graph.J({0,1,2,3,4,5,6,7,8,9}) = +1;
   poly_graph.J({0,1,2,3,4,5,7,8,9}) = -2;

   
   openjij::graph::Spins spin = {0, 0, 1, 1, 0, 0, 1, 1, 0, 0};
   
   auto poly_system = openjij::system::make_k_local_polynomial(spin, poly_graph);
   
   //poly_system.PrintInfo();
   
   const int seed = 1;
   
   auto random_numder_engine = std::mt19937(seed);
   const auto schedule_list = generate_schedule_list(100, 3);
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);
   poly_system.print_dE();

   //openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);

   //poly_system.PrintInfo();
   
   const auto result_spin_poly = openjij::result::get_solution(poly_system);
   for (std::size_t i = 0; i < result_spin_poly.size(); ++i) {
      printf("Result_spin[%ld]=%d\n", i , result_spin_poly[i]);
   }
   
}

TEST(KLocal, test3) {
   
   const int seed = 1;
   openjij::graph::Index num_spins = 100;
   openjij::graph::Polynomial<double> poly_graph(num_spins, cimod::Vartype::BINARY);
   const int specific_index = 50;

   std::vector<openjij::graph::Index> index;
   for (int i = 0; i < num_spins; ++i) {
      index.push_back(i);
   }
   
   poly_graph.J(index) = +1;
   index.erase(index.begin() + specific_index);
   poly_graph.J(index) = -2;

   auto engine_for_binary_poly = std::mt19937(seed);
   openjij::graph::Spins spin = poly_graph.gen_binary(engine_for_binary_poly);
   
   auto poly_system = openjij::system::make_k_local_polynomial(spin, poly_graph);
   
   //poly_system.PrintInfo();
   
   
   auto random_numder_engine = std::mt19937(seed);
   const auto schedule_list = generate_schedule_list(100, 3);
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);
   poly_system.print_dE();

   //openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);

   //poly_system.PrintInfo();
   
   const auto result_spin_poly = openjij::result::get_solution(poly_system);
   for (std::size_t i = 0; i < result_spin_poly.size(); ++i) {
      printf("Result_spin[%ld]=%d\n", i , result_spin_poly[i]);
   }
   
}

TEST(KLocal, test4) {
   
   const int seed = 1;
   openjij::graph::Index num_spins = 100;
   openjij::graph::Polynomial<double> poly_graph(num_spins, cimod::Vartype::BINARY);
   const std::vector<int> specific_index = {10,20,30,40,50,60,70,80,90};

   std::vector<openjij::graph::Index> index;
   for (int i = 0; i < num_spins; ++i) {
      index.push_back(i);
   }
   
   poly_graph.J(index) = +1;
   int count = 0;
   for (const auto &i: specific_index) {
      index.erase(index.begin() + i - count);
      count++;
   }
   poly_graph.J(index) = -2;

   auto engine_for_binary_poly = std::mt19937(seed);
   openjij::graph::Spins spin = poly_graph.gen_binary(engine_for_binary_poly);
   
   auto poly_system = openjij::system::make_k_local_polynomial(spin, poly_graph);
   
   //poly_system.PrintInfo();
   
   
   auto random_numder_engine = std::mt19937(seed);
   const auto schedule_list = generate_schedule_list(100, 3);
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);
   poly_system.print_dE();

   //openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);

   //poly_system.PrintInfo();
   
   const auto result_spin_poly = openjij::result::get_solution(poly_system);
   for (std::size_t i = 0; i < result_spin_poly.size(); ++i) {
      printf("Result_spin[%ld]=%d\n", i , result_spin_poly[i]);
   }
   printf("energy=%lf\n", poly_graph.calc_energy(result_spin_poly));
   
}

