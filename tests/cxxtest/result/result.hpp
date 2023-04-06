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

TEST(RESULT, GetSolutionFromTrotter){
   auto graph = openjij::graph::Dense<float>(4);
   graph.J(1, 1) = -1.0;
   graph.J(0, 1) = -1.0;
   graph.J(1, 2) = -1.0;
   graph.J(2, 3) = 1.0;
   
   auto r = openjij::utility::Xorshift(1234);
   int num_trotter_slices = 4;
   openjij::system::TrotterSpins init_trotter_spins(num_trotter_slices);
   for(auto& spins : init_trotter_spins){
      spins = graph.gen_spin(r);
   }
   
   init_trotter_spins[0] = openjij::graph::Spins({1, 1, 1, -1});
   
   auto q_sys = openjij::system::make_transverse_ising(init_trotter_spins, graph, 1.0);
   // get_solution get minimum energy state
   auto solution = openjij::result::get_solution(q_sys);
   EXPECT_EQ(solution, init_trotter_spins[0]);
}

TEST(RESULT, GetSolutionFromChimera){
   auto graph = openjij::graph::Chimera<float>(1,1);
   graph.h(0, 0, 0) = 1.0;
   graph.J(0, 0, 0, openjij::graph::ChimeraDir::IN_0or4) = -1.0;
   graph.J(0, 0, 4, openjij::graph::ChimeraDir::IN_1or5) = -1.0;
   graph.J(0, 0, 2, openjij::graph::ChimeraDir::IN_2or6) = -1.0;
   graph.J(0, 0, 6, openjij::graph::ChimeraDir::IN_3or7) = -1.0;
   graph.J(0, 0, 3, openjij::graph::ChimeraDir::IN_3or7) = -1.0;
   
   auto r = openjij::utility::Xorshift(1234);
   int num_trotter_slices = 4;
   openjij::system::TrotterSpins init_trotter_spins(num_trotter_slices);
   for(auto& spins : init_trotter_spins){
      spins = graph.gen_spin(r);
   }
   
   init_trotter_spins[0] = openjij::graph::Spins({-1,-1,-1,-1,-1,-1,-1,-1});
   
   auto q_sys = openjij::system::make_transverse_ising(init_trotter_spins, static_cast<openjij::graph::Sparse<float>>(graph), 1.0);
   // get_solution get minimum energy state
   auto solution = openjij::result::get_solution(q_sys);
   EXPECT_EQ(solution, init_trotter_spins[0]);
}


}
}
