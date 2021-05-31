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

TEST(KLocal, Constructor) {
   
   openjij::graph::Index num_spins = 3;
   openjij::graph::Polynomial<double> poly_graph(num_spins, cimod::Vartype::BINARY);
   
   poly_graph.J(   {0}   ) = -1;//0
   poly_graph.J( {0, 1}  ) = -1;//1
   poly_graph.J( {0, 2}  ) = -1;//2
   poly_graph.J({0, 1, 2}) = -1;//3
   poly_graph.J(   {1}   ) = -1;//4
   poly_graph.J( {1, 2}  ) = -1;//5
   poly_graph.J(   {2}   ) = -1;//6
   
   openjij::graph::Spins spin = {1, 0, 1};
   
   auto poly_system = openjij::system::make_k_local_polynomial(spin, poly_graph);
   
   poly_system.PrintInfo();
   
   const int seed = 1;
   
   auto random_numder_engine = std::mt19937(seed);
   const auto schedule_list = generate_schedule_list(100, 10);
   
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, schedule_list);
      
   poly_system.PrintInfo();
   
   
}

