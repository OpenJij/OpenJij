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


TEST(Graph, BinaryPolynomialModelQuad) {
   
   using FloatType = double;
   using Tup = utility::AnyTupleType;
   using BPM = graph::BinaryPolynomialModel<FloatType>;
   
   const std::vector<std::vector<typename BPM::IndexType>> key_list = {
      {1, 1},
      {"a", "a"},
      {Tup{2, "b"}, Tup{2, "b"}},
      {1, 2},
      {"a", 1},
      {Tup{2, "b"}, Tup{2, "a"}}
   };
      
   const std::vector<FloatType> value_list = {
      +4.0,
      +2.0,
      +3.0,
      -1.0,
      -1.5,
      -2.5
   };
   
   const auto bpm = BPM{key_list, value_list};
   
   EXPECT_EQ(bpm.GetDegree(), 2);
   EXPECT_EQ(bpm.GetSystemSize(), 5);
   EXPECT_EQ(bpm.GetIndexList(),
             (std::vector<BPM::IndexType>{1, 2, "a", Tup{2, "a"}, Tup{2, "b"}}));
   EXPECT_EQ(bpm.GetIndexMap().size(), 5);
   EXPECT_EQ(bpm.GetIndexMap().at(1)          , 0);
   EXPECT_EQ(bpm.GetIndexMap().at(2)          , 1);
   EXPECT_EQ(bpm.GetIndexMap().at("a")        , 2);
   EXPECT_EQ(bpm.GetIndexMap().at(Tup{2, "a"}), 3);
   EXPECT_EQ(bpm.GetIndexMap().at(Tup{2, "b"}), 4);
   
   EXPECT_EQ(bpm.GetKeyValueList().size(), 6);
   EXPECT_EQ(bpm.GetKeyValueList().at(0).first.size(), 1);
   EXPECT_EQ(bpm.GetKeyValueList().at(1).first.size(), 2);
   EXPECT_EQ(bpm.GetKeyValueList().at(2).first.size(), 2);
   EXPECT_EQ(bpm.GetKeyValueList().at(3).first.size(), 1);
   EXPECT_EQ(bpm.GetKeyValueList().at(4).first.size(), 2);
   EXPECT_EQ(bpm.GetKeyValueList().at(5).first.size(), 1);
   
   EXPECT_DOUBLE_EQ(bpm.GetKeyValueList().at(0).second, +4.0);
   EXPECT_DOUBLE_EQ(bpm.GetKeyValueList().at(1).second, -1.0);
   EXPECT_DOUBLE_EQ(bpm.GetKeyValueList().at(2).second, -1.5);
   EXPECT_DOUBLE_EQ(bpm.GetKeyValueList().at(3).second, +2.0);
   EXPECT_DOUBLE_EQ(bpm.GetKeyValueList().at(4).second, -2.5);
   EXPECT_DOUBLE_EQ(bpm.GetKeyValueList().at(5).second, +3.0);
   
   EXPECT_EQ(bpm.GetKeyValueList().at(0).first.at(0), 0);
   EXPECT_EQ(bpm.GetKeyValueList().at(1).first.at(0), 0);
   EXPECT_EQ(bpm.GetKeyValueList().at(1).first.at(1), 1);
   EXPECT_EQ(bpm.GetKeyValueList().at(2).first.at(0), 0);
   EXPECT_EQ(bpm.GetKeyValueList().at(2).first.at(1), 2);
   EXPECT_EQ(bpm.GetKeyValueList().at(3).first.at(0), 2);
   EXPECT_EQ(bpm.GetKeyValueList().at(4).first.at(0), 3);
   EXPECT_EQ(bpm.GetKeyValueList().at(4).first.at(1), 4);
   EXPECT_EQ(bpm.GetKeyValueList().at(5).first.at(0), 4);
   
   EXPECT_EQ(bpm.GetAdjacencyList().size(), 5);
   EXPECT_EQ(bpm.GetAdjacencyList().at(0).size(), 3);
   EXPECT_EQ(bpm.GetAdjacencyList().at(0).at(0), 0);
   EXPECT_EQ(bpm.GetAdjacencyList().at(0).at(1), 1);
   EXPECT_EQ(bpm.GetAdjacencyList().at(0).at(2), 2);
   EXPECT_EQ(bpm.GetAdjacencyList().at(1).size(), 1);
   EXPECT_EQ(bpm.GetAdjacencyList().at(1).at(0), 1);
   EXPECT_EQ(bpm.GetAdjacencyList().at(2).size(), 2);
   EXPECT_EQ(bpm.GetAdjacencyList().at(2).at(0), 2);
   EXPECT_EQ(bpm.GetAdjacencyList().at(2).at(1), 3);
   EXPECT_EQ(bpm.GetAdjacencyList().at(3).size(), 1);
   EXPECT_EQ(bpm.GetAdjacencyList().at(3).at(0), 4);
   EXPECT_EQ(bpm.GetAdjacencyList().at(4).size(), 2);
   EXPECT_EQ(bpm.GetAdjacencyList().at(4).at(0), 4);
   EXPECT_EQ(bpm.GetAdjacencyList().at(4).at(1), 5);

   EXPECT_DOUBLE_EQ(bpm.GetEstimatedMinEnergyDifference(), 1.0);
   EXPECT_DOUBLE_EQ(bpm.GetEstimatedMaxEnergyDifference(), 6.5);
   
   EXPECT_DOUBLE_EQ(bpm.CalculateEnergy((std::vector<std::int8_t>{1,1,1,1,1})), 4.0);
   EXPECT_DOUBLE_EQ(bpm.CalculateEnergy((std::vector<std::int8_t>{0,0,0,0,0})), 0.0);
   
}

TEST(Graph, BinaryPolynomialModelPoly) {
   
   using FloatType = double;
   using Tup = utility::AnyTupleType;
   using BPM = graph::BinaryPolynomialModel<FloatType>;
   
   const std::vector<std::vector<typename BPM::IndexType>> key_list = {
      {1,2,2,"a","a"}
   };
      
   const std::vector<FloatType> value_list = {
      +4.0
   };
   
   const auto bpm = BPM{key_list, value_list};
   
   EXPECT_EQ(bpm.GetDegree(), 3);
   EXPECT_EQ(bpm.GetSystemSize(), 3);
   EXPECT_EQ(bpm.GetIndexList(),
             (std::vector<BPM::IndexType>{1, 2, "a"}));
   EXPECT_EQ(bpm.GetIndexMap().size(), 3);
   EXPECT_EQ(bpm.GetIndexMap().at(1)          , 0);
   EXPECT_EQ(bpm.GetIndexMap().at(2)          , 1);
   EXPECT_EQ(bpm.GetIndexMap().at("a")        , 2);
   
   EXPECT_EQ(bpm.GetKeyValueList().size(), 1);
   EXPECT_EQ(bpm.GetKeyValueList().at(0).first.size(), 3);
   EXPECT_DOUBLE_EQ(bpm.GetKeyValueList().at(0).second, 4.0);
   EXPECT_EQ(bpm.GetKeyValueList().at(0).first.at(0), 0);
   EXPECT_EQ(bpm.GetKeyValueList().at(0).first.at(1), 1);
   EXPECT_EQ(bpm.GetKeyValueList().at(0).first.at(2), 2);
   
   EXPECT_EQ(bpm.GetAdjacencyList().size(), 3);
   EXPECT_EQ(bpm.GetAdjacencyList().at(0).size(), 1);
   EXPECT_EQ(bpm.GetAdjacencyList().at(0).at(0), 0);
   EXPECT_EQ(bpm.GetAdjacencyList().at(1).size(), 1);
   EXPECT_EQ(bpm.GetAdjacencyList().at(1).at(0), 0);
   EXPECT_EQ(bpm.GetAdjacencyList().at(2).size(), 1);
   EXPECT_EQ(bpm.GetAdjacencyList().at(2).at(0), 0);
   
   EXPECT_DOUBLE_EQ(bpm.GetEstimatedMinEnergyDifference(), 4.0);
   EXPECT_DOUBLE_EQ(bpm.GetEstimatedMaxEnergyDifference(), 4.0);
   
   EXPECT_DOUBLE_EQ(bpm.CalculateEnergy((std::vector<std::int8_t>{1,1,1})), 4.0);
   EXPECT_DOUBLE_EQ(bpm.CalculateEnergy((std::vector<std::int8_t>{0,0,0})), 0.0);
}


}
}
