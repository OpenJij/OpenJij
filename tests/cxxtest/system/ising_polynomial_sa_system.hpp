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

TEST(System, IsingPolynomialSystemConstructor) {
   using Tup = utility::AnyTupleType;
   using IPM = graph::IsingPolynomialModel<double>;
   
   std::vector<std::vector<typename IPM::IndexType>> key_list = {
      {1, 1, 2, 3, 4},
      {"a", "a"},
      {Tup{2, "b"}, Tup{2, "b"}},
      {1, 2},
      {"a", 1},
      {Tup{2, "b"}, Tup{2, "a"}}
   };
      
   std::vector<double> value_list = {
      +4.0,
      +2.0,
      +3.0,
      -1.0,
      -1.5,
      -2.5
   };
   
   const auto ipm = IPM{key_list, value_list};
   
   EXPECT_NO_THROW((system::SASystem<IPM, std::mt19937>{ipm, 1}));
   EXPECT_NO_THROW((system::SASystem<IPM, std::mt19937_64>{ipm, 1}));
   EXPECT_NO_THROW((system::SASystem<IPM, utility::Xorshift>{ipm, 1}));
}


TEST(System, IsingPolynomialSystemSeed) {
   using Tup = utility::AnyTupleType;
   using IPM = graph::IsingPolynomialModel<double>;
   
   std::vector<std::vector<typename IPM::IndexType>> key_list = {
      {1, 1},
      {"a", "a"},
      {Tup{2, "b"}, Tup{2, "b"}},
      {1, 2},
      {"a", 1},
      {Tup{2, "b"}, Tup{2, "a"}}
   };
      
   std::vector<double> value_list = {
      +4.0,
      +2.0,
      +3.0,
      -1.0,
      -1.5,
      -2.5
   };
   
   const auto ipm = IPM{key_list, value_list};
   auto sa_system = system::SASystem<IPM, std::mt19937>{ipm, 1};
   
   EXPECT_EQ(sa_system.GetSystemSize(), 5);
   EXPECT_EQ(sa_system.ExtractSample().size(), 5);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 5);
   EXPECT_NO_THROW(sa_system.Flip(0));
   EXPECT_NO_THROW(sa_system.Flip(1));
   EXPECT_NO_THROW(sa_system.Flip(2));
   EXPECT_NO_THROW(sa_system.Flip(3));
   EXPECT_NO_THROW(sa_system.Flip(4));
   
}


TEST(System, IsingPolynomialSystemExplicitLinearQuad) {
   using IPM = graph::IsingPolynomialModel<double>;
         
   std::vector<std::vector<typename IPM::IndexType>> key_list = {
      {1, 1, 1},
      {2, 2, 2},
      {3, 3, 3},
      {1, 2},
      {1, 3},
      {2, 3}
   };
      
   std::vector<double> value_list = {
      +1.0,
      +2.0,
      +3.0,
      +3.0,
      -1.0,
      -1.5
   };
   
   const auto bpm = IPM{key_list, value_list};
   auto sa_system = system::SASystem<IPM, std::mt19937>{bpm, 1};
   
   sa_system.SetSample({+1, +1, +1});
   EXPECT_EQ(sa_system.GetSystemSize(), 3);
   EXPECT_EQ(sa_system.ExtractSample().size(), 3);
   EXPECT_EQ(sa_system.ExtractSample().at(0), 1);
   EXPECT_EQ(sa_system.ExtractSample().at(1), 1);
   EXPECT_EQ(sa_system.ExtractSample().at(2), 1);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), -6.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), -7.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -1.0);
   sa_system.Flip(0);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), +6.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), +5.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -5.0);
   sa_system.Flip(0);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), -6.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), -7.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -1.0);
   sa_system.Flip(1);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), +6.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), +7.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -7.0);
   sa_system.Flip(1);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), -6.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), -7.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -1.0);
   sa_system.Flip(2);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), -10.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), -13.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), +1.0);
   sa_system.Flip(2);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), -6.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), -7.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -1.0);
}

TEST(System, IsingPolynomialSystemExplicitLinearPoly) {
   using IPM = graph::IsingPolynomialModel<double>;
         
   std::vector<std::vector<typename IPM::IndexType>> key_list = {
      {0, 1, 2},
      {0, 1, 2}
   };
      
   std::vector<double> value_list = {
      +2.0,
      +2.0
   };
   
   const auto bpm = IPM{key_list, value_list};
   auto sa_system = system::SASystem<IPM, std::mt19937>{bpm, 1};
   
   sa_system.SetSample({+1, +1, +1});
   EXPECT_EQ(sa_system.GetSystemSize(), 3);
   EXPECT_EQ(sa_system.ExtractSample().size(), 3);
   EXPECT_EQ(sa_system.ExtractSample().at(0), 1);
   EXPECT_EQ(sa_system.ExtractSample().at(1), 1);
   EXPECT_EQ(sa_system.ExtractSample().at(2), 1);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), -8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), -8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -8.0);
   sa_system.Flip(0);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), +8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), +8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), +8.0);
   sa_system.Flip(0);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), -8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), -8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -8.0);
   sa_system.Flip(1);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), +8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), +8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), +8.0);
   sa_system.Flip(1);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), -8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), -8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -8.0);
   sa_system.Flip(2);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), +8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), +8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), +8.0);
   sa_system.Flip(2);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), -8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), -8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -8.0);
}

TEST(System, IsingPolynomialSystemExplicitLinearPolyPart2) {
   using IPM = graph::IsingPolynomialModel<double>;
         
   std::vector<std::vector<typename IPM::IndexType>> key_list = {
      {0, 1, 2},
      {0, 1, 2}
   };
      
   std::vector<double> value_list = {
      +2.0,
      +2.0
   };
   
   const auto bpm = IPM{key_list, value_list};
   auto sa_system = system::SASystem<IPM, std::mt19937>{bpm, 1};
   
   sa_system.SetSample({+1, -1, +1});
   EXPECT_EQ(sa_system.GetSystemSize(), 3);
   EXPECT_EQ(sa_system.ExtractSample().size(), 3);
   EXPECT_EQ(sa_system.ExtractSample().at(0), 1);
   EXPECT_EQ(sa_system.ExtractSample().at(1), -1);
   EXPECT_EQ(sa_system.ExtractSample().at(2), 1);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), +8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), +8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), +8.0);
   sa_system.Flip(2);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), -8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), -8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), -8.0);
   sa_system.Flip(2);
   EXPECT_EQ(sa_system.GetBaseEnergyDifference().size(), 3);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(0), +8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(1), +8.0);
   EXPECT_DOUBLE_EQ(sa_system.GetEnergyDifference(2), +8.0);
}




}
}
