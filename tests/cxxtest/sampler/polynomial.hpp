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

TEST(PolyUpdater, SingleSpinFlipSPIN) {
   
   //Check the polynomial updater work properly by comparing the result of the quadratic updater
   const int seed = 1;
   const int system_size = 9;
   
   //generate classical sparse system
   auto engin_for_interaction = std::mt19937(seed);
   auto urd = std::uniform_real_distribution<>(-1.0/system_size, 1.0/system_size);
   auto interaction = openjij::graph::Sparse<double>(system_size);
   auto poly_graph  = openjij::graph::Polynomial<double>(system_size);
   for (int i = 0; i < system_size; ++i) {
      for (int j = i + 1; j < system_size; ++j) {
         interaction.J(i, j) = urd(engin_for_interaction);
         poly_graph.J(i, j)  = interaction.J(i, j);
      }
   }
   auto engine_for_spin = std::mt19937(seed);
   const auto spin = interaction.gen_spin(engine_for_spin);
   auto classical_ising      = openjij::system::make_classical_ising(spin, interaction);
   auto classical_ising_poly = openjij::system::make_classical_ising_polynomial(spin, poly_graph, "SPIN");
   
   auto random_numder_engine = std::mt19937(seed);
   const auto schedule_list  = generate_schedule_list();
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising     , random_numder_engine, schedule_list);
   auto random_numder_engine_a = std::mt19937(seed);
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_poly, random_numder_engine_a, schedule_list);
   const auto result_spin      = openjij::result::get_solution(classical_ising);
   const auto result_spin_poly = openjij::result::get_solution(classical_ising_poly);

   //Check both equal
   EXPECT_EQ(result_spin_poly.size(), result_spin.size());
   for (std::size_t i = 0; i < result_spin_poly.size(); ++i) {
      EXPECT_EQ(result_spin_poly[i], result_spin[i]);
   }
   
   EXPECT_DOUBLE_EQ(poly_graph.energy(result_spin_poly), interaction.calc_energy(result_spin));
}

TEST(PolyUpdater, SingleSpinFlipBINARY) {
   
   const int seed = 1;
   const int system_size = 9;
   
   cimod::BinaryPolynomialModel<int, double> cimod_bpm({}, cimod::Vartype::BINARY);
   
   auto engin_for_interaction = std::mt19937(seed);
   auto urd = std::uniform_real_distribution<>(-1.0/system_size, 1.0/system_size);
   
   for (int i = 0; i < system_size; ++i) {
      for (int j = i + 1; j < system_size; ++j) {
         cimod_bpm.AddInteraction({i, j}, urd(engin_for_interaction));
      }
   }
   
   auto interaction_binary = openjij::graph::Polynomial<double>(cimod_bpm.ToSerializable());
   cimod_bpm.ChangeVartype(cimod::Vartype::SPIN);
   auto interaction_spin   = openjij::graph::Sparse<double>(system_size);
   
   for (const auto &it: cimod_bpm.GetPolynomial()) {
      if (it.first.size() == 2) {
         interaction_spin.J(it.first[0], it.first[1]) = it.second;
      }
      else if (it.first.size() == 1) {
         interaction_spin.h(it.first[0]) = it.second;
      }
   }
   
   const auto offset = cimod_bpm.GetOffset();
   
   auto engine_for_spin   = std::mt19937(seed);
   auto engine_for_binary = std::mt19937(seed);
   const auto spin   = interaction_spin.gen_spin(engine_for_spin);
   const auto binary = interaction_binary.gen_binary(engine_for_binary);

   auto classical_ising      = openjij::system::make_classical_ising(spin, interaction_spin);
   auto classical_ising_poly = openjij::system::make_classical_ising_polynomial(binary, interaction_binary, "BINARY");
   
   auto random_numder_engine = std::mt19937(seed);
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising     , random_numder_engine, generate_schedule_list());
   openjij::algorithm::Algorithm<openjij::updater::SingleSpinFlip>::run(classical_ising_poly, random_numder_engine, generate_schedule_list());
   const auto result_spin   = openjij::result::get_solution(classical_ising);
   const auto result_binary = openjij::result::get_solution(classical_ising_poly);

   //Check both equal
   EXPECT_EQ(result_binary.size(), result_spin.size());
   for (std::size_t i = 0; i < result_binary.size(); ++i) {
      EXPECT_EQ(result_binary[i], (result_spin[i] + 1)/2);
   }
   
   EXPECT_DOUBLE_EQ(interaction_binary.energy(result_binary), interaction_spin.calc_energy(result_spin) + offset);
}



}
}
