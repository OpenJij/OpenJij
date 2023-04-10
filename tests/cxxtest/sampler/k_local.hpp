//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once


namespace openjij {
namespace test {

TEST(PolyUpdater, KLocal1) {
   const int seed = 1;
   openjij::graph::Index num_spins = 30;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}) = -1;
   
   auto engine_for_binary = std::mt19937(seed);
   
   openjij::graph::Binaries binary = poly_graph.gen_binary(engine_for_binary);
   
   auto poly_system = openjij::system::make_k_local_polynomial(binary, poly_graph);
   
   auto random_numder_engine = std::mt19937(seed);
   
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, generate_schedule_list());
   
   const auto result_binary_poly = openjij::result::get_solution(poly_system);
   
   for (const auto &binary: result_binary_poly) {
      EXPECT_EQ(binary, 1);
   }
   
}

TEST(PolyUpdater, KLocal2) {
   const int seed = 1;
   openjij::graph::Index num_spins = 30;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}) = +1;
   poly_graph.J({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,   17,18,19,20,21,22,23,24,25,26,27,28,29}) = -1;

   auto engine_for_binary = std::mt19937(seed);
   
   openjij::graph::Binaries binary = poly_graph.gen_binary(engine_for_binary);
   
   auto poly_system = openjij::system::make_k_local_polynomial(binary, poly_graph);
      
   auto random_numder_engine = std::mt19937(seed);
   
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, generate_schedule_list());
   
   const auto result_binary_poly = openjij::result::get_solution(poly_system);
   
   for (std::size_t i = 0; i < result_binary_poly.size(); ++i) {
      if (i == 16) {
         EXPECT_EQ(result_binary_poly[i], 0);
      }
      else {
         EXPECT_EQ(result_binary_poly[i], 1);
      }
   }
}

TEST(PolyUpdater, KLocal3) {
   
   const int seed = 1;
   openjij::graph::Index num_spins = 30;
   openjij::graph::Polynomial<double> poly_graph(num_spins);
   
   poly_graph.J(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29) = -1;
   poly_graph.J(0,1,2,3,4,5,6,7,8,  10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29) = +1;
   poly_graph.J(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,   24,25,26,27,28,29) = +1;
   poly_graph.J(0,1,2,3,4,5,6,7,8,  10,11,12,13,14,15,16,17,18,19,20,21,22,   24,25,26,27,28,29) = -1;

      
   auto engine_for_binary = std::mt19937(seed);
   
   openjij::graph::Binaries binary = poly_graph.gen_binary(engine_for_binary);
   
   auto poly_system = openjij::system::make_k_local_polynomial(binary, poly_graph);
      
   auto random_numder_engine = std::mt19937(seed);
      
   openjij::algorithm::Algorithm<openjij::updater::KLocal>::run(poly_system, random_numder_engine, generate_schedule_list());
   
   const auto result_binary_poly = openjij::result::get_solution(poly_system);
   
   for (std::size_t i = 0; i < result_binary_poly.size(); ++i) {
      if (i == 9 || i == 23) {
         EXPECT_EQ(result_binary_poly[i], 0);
      }
      else {
         EXPECT_EQ(result_binary_poly[i], 1);
      }
   }

}



}
}
