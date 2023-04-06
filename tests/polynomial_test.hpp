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

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <openjij/graph/all.hpp>
#include <openjij/system/all.hpp>
#include <openjij/updater/all.hpp>
#include <openjij/algorithm/all.hpp>
#include <openjij/result/all.hpp>

std::vector<openjij::graph::Spin> PolynomialGetSpinState(std::size_t basis, const std::size_t system_size, const cimod::Vartype &vartype) {
   std::vector<openjij::graph::Spin> spins(system_size);
   for (std::size_t i = 0; i < system_size; ++i) {
      if (basis%2 == 0) {
         (vartype == cimod::Vartype::SPIN) ? spins[i] = -1 : spins[i] = 0;
      }
      else {
         spins[i] = +1;
      }
      basis /= 2;
   }
   return spins;
}

std::vector<std::vector<openjij::graph::Index>> PolynomialGenerateCombinations(const std::vector<openjij::graph::Index> &vec_in) {
   const std::size_t loop = static_cast<std::size_t>(std::pow(2, vec_in.size()));
   const std::size_t num  = vec_in.size();
   std::vector<std::vector<openjij::graph::Index>> vec_out(loop);
   for (std::size_t i = 0; i < loop; ++i) {
      std::bitset<64> bs(i);
      for (std::size_t j = 0; j < num; ++j) {
         if (bs[j]) {
            vec_out[i].push_back(vec_in[j]);
         }
      }
   }
   return vec_out;
}

template<typename FloatType>
std::unordered_map<std::vector<openjij::graph::Index>, FloatType, cimod::vector_hash>
PolynomialSpinToBinary(const std::unordered_map<std::vector<openjij::graph::Index>, FloatType, cimod::vector_hash> &J_in) {
   std::unordered_map<std::vector<openjij::graph::Index>, FloatType, cimod::vector_hash> J_out;
   for (const auto &it: J_in) {
      const auto &index_list = PolynomialGenerateCombinations(it.first);
      for (const auto &index: index_list) {
         FloatType sign = ((it.first.size() - index.size())%2 == 0) ? 1.0 : -1.0;
         J_out[index] += it.second*pow(2.0, index.size())*sign;
      }
   }
   return J_out;
}

template<typename FloatType>
std::unordered_map<std::vector<openjij::graph::Index>, FloatType, cimod::vector_hash>
PolynomialBinaryToSpin(const std::unordered_map<std::vector<openjij::graph::Index>, FloatType, cimod::vector_hash> &J_in) {
   std::unordered_map<std::vector<openjij::graph::Index>, FloatType, cimod::vector_hash> J_out;
   for (const auto &it: J_in) {
      FloatType coeef = std::pow(2.0, -static_cast<int64_t>(it.first.size()));
      const auto &index_list = PolynomialGenerateCombinations(it.first);
      for (const auto &index: index_list) {
         J_out[index] += it.second*coeef;
      }
   }
   return J_out;
}

template<typename FloatType>
FloatType PolynomialExactGroundStateEnergy(openjij::graph::Polynomial<FloatType> &polynomial, const cimod::Vartype &vartype) {
   const std::size_t system_size = polynomial.size();
   const std::size_t loop = std::pow(2, system_size);
   FloatType min_energy = DBL_MAX;
   for (std::size_t i = 0; i < loop; ++i) {
      std::vector<openjij::graph::Spin> temp_spin = PolynomialGetSpinState(i, system_size, vartype);
      FloatType temp_energy = 0.0;
      for (const auto &it: polynomial.get_polynomial()) {
         openjij::graph::Spin temp_spin_multiple = 1;
         for (const auto &index: it.first) {
            temp_spin_multiple *= temp_spin[index];
         }
         temp_energy += temp_spin_multiple*it.second;
      }
      if (min_energy > temp_energy) {
         min_energy = temp_energy;
      }
   }
   return min_energy;
}

template<typename ValueType>
bool EqualVector(const std::vector<ValueType> &vec1, const std::vector<ValueType> &vec2, const double threshold = 0.0) {
   if (vec1.size() != vec2.size()) {
      return false;
   }
   bool flag = true;
   for (std::size_t i = 0; i < vec1.size(); ++i) {
      if (std::fabs(vec1[i] - vec2[i]) > threshold) {
         flag = false;
         break;
      }
   }
   return flag;
}

template<typename ValueType>
bool ContainValue(const ValueType val, const std::vector<ValueType> &vec, const double threshold = 0.0) {
   bool flag = false;
   for (const auto &it: vec) {
      if (std::abs(it - val) <= threshold) {
         flag = true;
         break;
      }
   }
   return flag;
}

template<typename ValueType>
bool ContainVector(const std::vector<ValueType> &vec, const std::vector<std::vector<ValueType>> &vec_vec, const double threshold = 0.0) {
   bool flag = false;
   for (const auto &it: vec_vec) {
      if (EqualVector(vec, it, threshold)) {
         flag = true;
         break;
      }
   }
   return flag;
}

template<typename FloatType>
cimod::Polynomial<openjij::graph::Index, FloatType> GeneratePolynomialInteractionsDenseInt() {
   return cimod::Polynomial<openjij::graph::Index, FloatType> {
      {{}, +0.1 },
      {{0}, -0.5 }, {{1}, +1.0 }, {{2}, -2.0 },
      {{0, 1}, +10.0}, {{0, 2}, -20.0}, {{1, 2}, +21.0},
      {{0, 1, 2}, -120}
   };
}

template<typename FloatType>
cimod::Polynomial<openjij::graph::Index, FloatType> GeneratePolynomialInteractionsSparseInt1() {
   return cimod::Polynomial<openjij::graph::Index, FloatType> {
      {{2}, -2.0},
      {{0, 1}, +10.0}, {{1, 2}, +21.0},
      {{0, 1, 2}, -120}
   };
}

template<typename FloatType>
cimod::Polynomial<openjij::graph::Index, FloatType> GeneratePolynomialInteractionsSparseInt2() {
   return cimod::Polynomial<openjij::graph::Index, FloatType> {
      {{}, +0.0},
      {{0}, -0.0}, {{1}, +0.0}, {{2}, -2.0},
      {{0, 1}, +10.0}, {{0, 2}, -0.0}, {{1, 2}, +21.0},
      {{0, 1, 2}, -120}
   };
}

template<typename FloatType>
cimod::Polynomial<openjij::graph::Index, FloatType> GeneratePolynomialInteractionsSparseInt3() {
   return cimod::Polynomial<openjij::graph::Index, FloatType> {
      {{12}, -2.0},
      {{10, 11}, +10.0}, {{11, 12}, +21.0},
      {{10, 11, 12}, -120}
   };
}

template<typename FloatType>
cimod::Polynomial<std::string, FloatType> GeneratePolynomialInteractionsDenseString() {
   return cimod::Polynomial<std::string, FloatType> {
      {{}, +0.1},
      {{"a"}, -0.5 }, {{"b"}, +1.0 }, {{"c"}, -2.0 },
      {{"a", "b"}, +10.0}, {{"a", "c"}, -20.0}, {{"b", "c"}, +21.0},
      {{"a", "b", "c"}, -120}
   };
}

template<typename FloatType>
cimod::Polynomial<std::tuple<openjij::graph::Index, openjij::graph::Index>, FloatType> GeneratePolynomialInteractionsDenseTuple2() {
   return cimod::Polynomial<std::tuple<openjij::graph::Index, openjij::graph::Index>, FloatType> {
      {{}, +0.1 },
      {{{0, 0}}, -0.5 }, {{{1, 1}}, +1.0 }, {{{2, 2}}, -2.0 },
      {{{0, 0}, {1, 1}}, +10.0}, {{{0, 0}, {2, 2}}, -20.0}, {{{1, 1}, {2, 2}}, +21.0},
      {{{0, 0}, {1, 1}, {2, 2}}, -120}
   };
}

template<typename FloatType>
void TestPolyGraphDense(const openjij::graph::Polynomial<FloatType> &poly_graph) {
   EXPECT_EQ(poly_graph.size()                , 3);
   EXPECT_EQ(poly_graph.get_num_interactions(), 8);
   EXPECT_EQ(poly_graph.get_values().size()   , 8);
   EXPECT_EQ(poly_graph.get_keys().size()     , 8);
   
   EXPECT_DOUBLE_EQ(poly_graph.J(   {}    ), +0.1  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   {0}   ), -0.5  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   {1}   ), +1.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   {2}   ), -2.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 1}  ), +10.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 2}  ), -20.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( {1, 2}  ), +21.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J({0, 1, 2}), -120.0);
   
   EXPECT_DOUBLE_EQ(poly_graph.J( {1, 0}  ), +10.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( {2, 0}  ), -20.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( {2, 1}  ), +21.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J({0, 2, 1}), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J({1, 2, 0}), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J({1, 0, 2}), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J({2, 1, 0}), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J({2, 0, 1}), -120.0);
   
   EXPECT_DOUBLE_EQ(poly_graph.J(       ), +0.1  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   0   ), -0.5  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   1   ), +1.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   2   ), -2.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J( 0, 1  ), +10.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( 0, 2  ), -20.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( 1, 2  ), +21.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J(0, 1, 2), -120.0);
   
   EXPECT_DOUBLE_EQ(poly_graph.J( 1, 0  ), +10.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( 2, 0  ), -20.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( 2, 1  ), +21.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J(0, 2, 1), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J(1, 2, 0), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J(1, 0, 2), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J(2, 1, 0), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J(2, 0, 1), -120.0);
   
   EXPECT_THROW(poly_graph.J(  500  ), std::runtime_error);
   EXPECT_THROW(poly_graph.J( 0, 0  ), std::runtime_error);
   EXPECT_THROW(poly_graph.J(2, 1, 1), std::runtime_error);
   
   EXPECT_THROW(poly_graph.J(  {500}  ), std::runtime_error);
   EXPECT_THROW(poly_graph.J( {0, 0}  ), std::runtime_error);
   EXPECT_THROW(poly_graph.J({2, 1, 1}), std::runtime_error);
   
   std::random_device rnd;
   std::mt19937 mt(rnd());
   const auto spins    = poly_graph.gen_spin(mt);
   const auto binaries = poly_graph.gen_binary(mt);
   
   FloatType energy_spins    = 0.0;
   FloatType energy_binaries = 0.0;

   energy_spins += poly_graph.J();
   energy_spins += spins[0]*poly_graph.J(0);
   energy_spins += spins[1]*poly_graph.J(1);
   energy_spins += spins[2]*poly_graph.J(2);
   energy_spins += spins[0]*spins[1]*poly_graph.J(0,1);
   energy_spins += spins[0]*spins[2]*poly_graph.J(0,2);
   energy_spins += spins[1]*spins[2]*poly_graph.J(1,2);
   energy_spins += spins[0]*spins[1]*spins[2]*poly_graph.J(0,1,2);
   
   energy_binaries += poly_graph.J();
   energy_binaries += binaries[0]*poly_graph.J(0);
   energy_binaries += binaries[1]*poly_graph.J(1);
   energy_binaries += binaries[2]*poly_graph.J(2);
   energy_binaries += binaries[0]*binaries[1]*poly_graph.J(0,1);
   energy_binaries += binaries[0]*binaries[2]*poly_graph.J(0,2);
   energy_binaries += binaries[1]*binaries[2]*poly_graph.J(1,2);
   energy_binaries += binaries[0]*binaries[1]*binaries[2]*poly_graph.J(0,1,2);
   
   EXPECT_DOUBLE_EQ(poly_graph.energy(spins)   , energy_spins);
   EXPECT_DOUBLE_EQ(poly_graph.energy(binaries), energy_binaries);

}

template<typename FloatType>
void TestPolyGraphSparse(const openjij::graph::Polynomial<FloatType> &poly_graph) {
   EXPECT_EQ(poly_graph.size()                , 3);
   EXPECT_EQ(poly_graph.get_num_interactions(), 4);
   EXPECT_EQ(poly_graph.get_values().size()   , 4);
   EXPECT_EQ(poly_graph.get_keys().size()     , 4);
   
   EXPECT_DOUBLE_EQ(poly_graph.J(   {}    ), +0.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   {0}   ), +0.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   {1}   ), +0.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   {2}   ), -2.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 1}  ), +10.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( {0, 2}  ), +0.00 );
   EXPECT_DOUBLE_EQ(poly_graph.J( {1, 2}  ), +21.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J({0, 1, 2}), -120.0);
   
   EXPECT_DOUBLE_EQ(poly_graph.J( {1, 0}  ), +10.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( {2, 0}  ), +0.00 );
   EXPECT_DOUBLE_EQ(poly_graph.J( {2, 1}  ), +21.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J({0, 2, 1}), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J({1, 2, 0}), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J({1, 0, 2}), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J({2, 1, 0}), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J({2, 0, 1}), -120.0);
   
   EXPECT_DOUBLE_EQ(poly_graph.J(       ), +0.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   0   ), +0.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   1   ), +0.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J(   2   ), -2.0  );
   EXPECT_DOUBLE_EQ(poly_graph.J( 0, 1  ), +10.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( 0, 2  ), +0.00 );
   EXPECT_DOUBLE_EQ(poly_graph.J( 1, 2  ), +21.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J(0, 1, 2), -120.0);
   
   EXPECT_DOUBLE_EQ(poly_graph.J( 1, 0  ), +10.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J( 2, 0  ), +0.00 );
   EXPECT_DOUBLE_EQ(poly_graph.J( 2, 1  ), +21.0 );
   EXPECT_DOUBLE_EQ(poly_graph.J(0, 2, 1), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J(1, 2, 0), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J(1, 0, 2), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J(2, 1, 0), -120.0);
   EXPECT_DOUBLE_EQ(poly_graph.J(2, 0, 1), -120.0);
   
   EXPECT_THROW(poly_graph.J(  500  ), std::runtime_error);
   EXPECT_THROW(poly_graph.J( 0, 0  ), std::runtime_error);
   EXPECT_THROW(poly_graph.J(2, 1, 1), std::runtime_error);
   
   EXPECT_THROW(poly_graph.J(  {500}  ), std::runtime_error);
   EXPECT_THROW(poly_graph.J( {0, 0}  ), std::runtime_error);
   EXPECT_THROW(poly_graph.J({2, 1, 1}), std::runtime_error);
   
   std::random_device rnd;
   std::mt19937 mt(rnd());
   const auto spins    = poly_graph.gen_spin(mt);
   const auto binaries = poly_graph.gen_binary(mt);
   
   FloatType energy_spins    = 0.0;
   FloatType energy_binaries = 0.0;

   energy_spins += spins[2]*poly_graph.J(2);
   energy_spins += spins[0]*spins[1]*poly_graph.J(0,1);
   energy_spins += spins[1]*spins[2]*poly_graph.J(1,2);
   energy_spins += spins[0]*spins[1]*spins[2]*poly_graph.J(0,1,2);
   
   energy_binaries += binaries[2]*poly_graph.J(2);
   energy_binaries += binaries[0]*binaries[1]*poly_graph.J(0,1);
   energy_binaries += binaries[1]*binaries[2]*poly_graph.J(1,2);
   energy_binaries += binaries[0]*binaries[1]*binaries[2]*poly_graph.J(0,1,2);
   
   EXPECT_DOUBLE_EQ(poly_graph.energy(spins)   , energy_spins);
   EXPECT_DOUBLE_EQ(poly_graph.energy(binaries), energy_binaries);

}

template<typename IndexType, typename FloatType>
void TestPolyGraphConstructorCimodDense(const cimod::Polynomial<IndexType, FloatType> &polynomial) {
   cimod::BinaryPolynomialModel<IndexType, FloatType> bpm_cimod(polynomial, cimod::Vartype::SPIN);
   openjij::graph::Polynomial<FloatType> poly_graph(bpm_cimod.ToSerializable());
   TestPolyGraphDense(poly_graph);
}

template<typename IndexType, typename FloatType>
void TestPolyGraphConstructorCimodSparse(const cimod::Polynomial<IndexType, FloatType> &polynomial) {
   cimod::BinaryPolynomialModel<IndexType, FloatType> bpm_cimod(polynomial, cimod::Vartype::SPIN);
   openjij::graph::Polynomial<FloatType> poly_graph(bpm_cimod.ToSerializable());
   TestPolyGraphSparse(poly_graph);
}

template<typename FloatType>
void TestCIPSystemDense(const openjij::system::ClassicalIsingPolynomial<openjij::graph::Polynomial<FloatType>> &cip_system) {
   
   const int system_size = 3;
   
   EXPECT_EQ(cip_system.num_variables, system_size);
      
   EXPECT_EQ(cip_system.get_adj().at(0).size(), 4);
   EXPECT_EQ(cip_system.get_adj().at(1).size(), 4);
   EXPECT_EQ(cip_system.get_adj().at(2).size(), 4);
   
   std::vector<std::vector<std::vector<openjij::graph::Index>>> adj_key(system_size);
   
   for (int i = 0; i < system_size; ++i) {
      for (const auto &index_key: cip_system.get_adj().at(i)) {
         adj_key[i].push_back(cip_system.get_keys().at(index_key));
      }
   }

   EXPECT_TRUE(ContainVector<openjij::graph::Index>({   0   }, adj_key[0]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 0, 1  }, adj_key[0]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 0, 2  }, adj_key[0]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[0]));
   
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({   1   }, adj_key[1]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 0, 1  }, adj_key[1]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 1, 2  }, adj_key[1]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[1]));

   EXPECT_TRUE(ContainVector<openjij::graph::Index>({   2   }, adj_key[2]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 0, 2  }, adj_key[2]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 1, 2  }, adj_key[2]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[2]));
   
   const auto init_spins = cip_system.variables;
   
   cimod::Polynomial<openjij::graph::Index, FloatType> polynomial;
   
   for (std::size_t i = 0; i < cip_system.get_values().size(); ++i) {
      polynomial[cip_system.get_keys().at(i)] = cip_system.get_values().at(i);
   }
   
   const int s0 = init_spins[0];
   const int s1 = init_spins[1];
   const int s2 = init_spins[2];
   
   if (cip_system.vartype == cimod::Vartype::SPIN) {
      const double dE0 = -2*s0*(polynomial.at({0}) + s1*polynomial.at({0, 1}) + s2*polynomial.at({0, 2}) + s1*s2*polynomial.at({0, 1, 2}));
      const double dE1 = -2*s1*(polynomial.at({1}) + s0*polynomial.at({0, 1}) + s2*polynomial.at({1, 2}) + s0*s2*polynomial.at({0, 1, 2}));
      const double dE2 = -2*s2*(polynomial.at({2}) + s0*polynomial.at({0, 2}) + s1*polynomial.at({1, 2}) + s0*s1*polynomial.at({0, 1, 2}));

      EXPECT_DOUBLE_EQ(dE0, cip_system.dE(0));
      EXPECT_DOUBLE_EQ(dE1, cip_system.dE(1));
      EXPECT_DOUBLE_EQ(dE2, cip_system.dE(2));
      
      const double abs_dE0 = 2*(std::abs(polynomial.at({0})) + std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({0, 2})) + std::abs(polynomial.at({0, 1, 2})));
      const double abs_dE1 = 2*(std::abs(polynomial.at({1})) + std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2})));
      const double abs_dE2 = 2*(std::abs(polynomial.at({2})) + std::abs(polynomial.at({0, 2})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2})));

      EXPECT_DOUBLE_EQ(cip_system.get_max_effective_dE(), std::max({abs_dE0    , abs_dE1    , abs_dE2})    );
      EXPECT_DOUBLE_EQ(cip_system.get_min_effective_dE(), std::abs(*std::min_element(cip_system.get_values().begin(), cip_system.get_values().end(), [](const auto a, const auto b) {
         return std::abs(a) < std::abs(b);
      })));
   }
   else if (cip_system.vartype == cimod::Vartype::BINARY) {
      const double dE0 = (-2*s0 + 1)*(polynomial.at({0}) + s1*polynomial.at({0, 1}) + s2*polynomial.at({0, 2}) + s1*s2*polynomial.at({0, 1, 2}));
      const double dE1 = (-2*s1 + 1)*(polynomial.at({1}) + s0*polynomial.at({0, 1}) + s2*polynomial.at({1, 2}) + s0*s2*polynomial.at({0, 1, 2}));
      const double dE2 = (-2*s2 + 1)*(polynomial.at({2}) + s0*polynomial.at({0, 2}) + s1*polynomial.at({1, 2}) + s0*s1*polynomial.at({0, 1, 2}));

      EXPECT_DOUBLE_EQ(dE0, cip_system.dE(0));
      EXPECT_DOUBLE_EQ(dE1, cip_system.dE(1));
      EXPECT_DOUBLE_EQ(dE2, cip_system.dE(2));
      
      const double abs_dE0 = std::abs(polynomial.at({0})) + std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({0, 2})) + std::abs(polynomial.at({0, 1, 2}));
      const double abs_dE1 = std::abs(polynomial.at({1})) + std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2}));
      const double abs_dE2 = std::abs(polynomial.at({2})) + std::abs(polynomial.at({0, 2})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2}));

      EXPECT_DOUBLE_EQ(cip_system.get_max_effective_dE(), std::max({abs_dE0    , abs_dE1    , abs_dE2})    );
      EXPECT_DOUBLE_EQ(cip_system.get_min_effective_dE(), std::abs(*std::min_element(cip_system.get_values().begin(), cip_system.get_values().end(), [](const auto a, const auto b) {
         return std::abs(a) < std::abs(b);
      })));
   }
   else {
      throw std::runtime_error("Unknown vartype detected");
   }
   for (std::size_t i = 0; i < cip_system.get_active_variables().size(); ++i) {
      EXPECT_EQ(cip_system.get_active_variables().at(i), i);
   }
}

template<typename FloatType>
void TestCIPSystemSparse(const openjij::system::ClassicalIsingPolynomial<openjij::graph::Polynomial<FloatType>> &cip_system) {
   
   const int system_size = 3;
   
   EXPECT_EQ(cip_system.num_variables, system_size);
      
   EXPECT_EQ(cip_system.get_adj().at(0).size(), 2);
   EXPECT_EQ(cip_system.get_adj().at(1).size(), 3);
   EXPECT_EQ(cip_system.get_adj().at(2).size(), 3);
   
   std::vector<std::vector<std::vector<openjij::graph::Index>>> adj_key(system_size);
   
   for (int i = 0; i < system_size; ++i) {
      for (const auto &index_key: cip_system.get_adj().at(i)) {
         adj_key[i].push_back(cip_system.get_keys().at(index_key));
      }
   }

   EXPECT_FALSE(ContainVector<openjij::graph::Index>({   0   }, adj_key[0]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({ 0, 1  }, adj_key[0]));
   EXPECT_FALSE(ContainVector<openjij::graph::Index>({ 0, 2  }, adj_key[0]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[0]));
   
   EXPECT_FALSE(ContainVector<openjij::graph::Index>({   1   }, adj_key[1]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({ 0, 1  }, adj_key[1]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({ 1, 2  }, adj_key[1]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[1]));

   EXPECT_TRUE (ContainVector<openjij::graph::Index>({   2   }, adj_key[2]));
   EXPECT_FALSE(ContainVector<openjij::graph::Index>({ 0, 2  }, adj_key[2]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({ 1, 2  }, adj_key[2]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[2]));
   
   const auto init_spins = cip_system.variables;
   
   cimod::Polynomial<openjij::graph::Index, FloatType> polynomial;
   
   for (std::size_t i = 0; i < cip_system.get_values().size(); ++i) {
      polynomial[cip_system.get_keys().at(i)] = cip_system.get_values().at(i);
   }
   
   const int s0 = init_spins[0];
   const int s1 = init_spins[1];
   const int s2 = init_spins[2];
   
   if (cip_system.vartype == cimod::Vartype::SPIN) {
      const double dE0 = -2*s0*(s1*polynomial.at({0, 1}) + s1*s2*polynomial.at({0, 1, 2}));
      const double dE1 = -2*s1*(s0*polynomial.at({0, 1}) + s2*polynomial.at({1, 2}) + s0*s2*polynomial.at({0, 1, 2}));
      const double dE2 = -2*s2*(polynomial.at({2}) + s1*polynomial.at({1, 2}) + s0*s1*polynomial.at({0, 1, 2}));

      EXPECT_DOUBLE_EQ(dE0, cip_system.dE(0));
      EXPECT_DOUBLE_EQ(dE1, cip_system.dE(1));
      EXPECT_DOUBLE_EQ(dE2, cip_system.dE(2));
      
      const double abs_dE0 = 2*(std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({0, 1, 2})));
      const double abs_dE1 = 2*(std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2})));
      const double abs_dE2 = 2*(std::abs(polynomial.at({2})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2})));

      EXPECT_DOUBLE_EQ(cip_system.get_max_effective_dE(), std::max({abs_dE0    , abs_dE1    , abs_dE2})    );
      EXPECT_DOUBLE_EQ(cip_system.get_min_effective_dE(), std::abs(*std::min_element(cip_system.get_values().begin(), cip_system.get_values().end(), [](const auto a, const auto b) {
         return std::abs(a) < std::abs(b);
      })));
   }
   else if (cip_system.vartype == cimod::Vartype::BINARY) {
      const double dE0 = (-2*s0 + 1)*(s1*polynomial.at({0, 1}) + s1*s2*polynomial.at({0, 1, 2}));
      const double dE1 = (-2*s1 + 1)*(s0*polynomial.at({0, 1}) + s2*polynomial.at({1, 2}) + s0*s2*polynomial.at({0, 1, 2}));
      const double dE2 = (-2*s2 + 1)*(polynomial.at({2}) + s1*polynomial.at({1, 2}) + s0*s1*polynomial.at({0, 1, 2}));

      EXPECT_DOUBLE_EQ(dE0, cip_system.dE(0));
      EXPECT_DOUBLE_EQ(dE1, cip_system.dE(1));
      EXPECT_DOUBLE_EQ(dE2, cip_system.dE(2));
      
      const double abs_dE0 = std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({0, 1, 2}));
      const double abs_dE1 = std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2}));
      const double abs_dE2 = std::abs(polynomial.at({2})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2}));

      EXPECT_DOUBLE_EQ(cip_system.get_max_effective_dE(), std::max({abs_dE0    , abs_dE1    , abs_dE2})    );
      EXPECT_DOUBLE_EQ(cip_system.get_min_effective_dE(), std::abs(*std::min_element(cip_system.get_values().begin(), cip_system.get_values().end(), [](const auto a, const auto b) {
         return std::abs(a) < std::abs(b);
      })));
   }
   else {
      throw std::runtime_error("Unknown vartype detected");
   }
   for (std::size_t i = 0; i < cip_system.get_active_variables().size(); ++i) {
      EXPECT_EQ(cip_system.get_active_variables().at(i), i);
   }
}

template<typename IndexType, typename FloatType>
void TestCIPConstructorCimod(const cimod::Polynomial<IndexType, FloatType> &polynomial, cimod::Vartype vartype, std::string type) {
   cimod::BinaryPolynomialModel<IndexType, FloatType> bpm_cimod(polynomial, vartype);
   std::random_device rnd;
   std::mt19937 mt(rnd());
   openjij::graph::Spins init_spins_1;
   openjij::graph::Spins init_spins_2;
   if (vartype == cimod::Vartype::SPIN) {
      init_spins_1 = openjij::graph::Polynomial<FloatType>(3).gen_spin(mt);
      init_spins_2 = openjij::graph::Polynomial<FloatType>(3).gen_spin(mt);
   }
   else {
      init_spins_1 = openjij::graph::Polynomial<FloatType>(3).gen_binary(mt);
      init_spins_2 = openjij::graph::Polynomial<FloatType>(3).gen_binary(mt);
   }
   auto system = openjij::system::make_classical_ising_polynomial(init_spins_1, bpm_cimod.ToSerializable());
   if (type == "Dense") {
      TestCIPSystemDense(system);
      system.reset_variables(init_spins_2);
      TestCIPSystemDense(system);
   }
   else if (type == "Sparse") {
      TestCIPSystemSparse(system);
      system.reset_variables(init_spins_2);
      TestCIPSystemSparse(system);
   }
   else {
      throw std::runtime_error("Unknown type");
   }
}


template<typename IndexType, typename FloatType>
void TestCIPConstructorGraph(const cimod::Polynomial<IndexType, FloatType> &polynomial, cimod::Vartype vartype, std::string type) {
   cimod::BinaryPolynomialModel<IndexType, FloatType> bpm_cimod(polynomial, vartype);
   openjij::graph::Polynomial<FloatType> poly_graph(3);
   for (const auto &it: bpm_cimod.GetPolynomial()) {
      poly_graph.J(it.first) = it.second;
   }
   std::random_device rnd;
   std::mt19937 mt(rnd());
   openjij::graph::Spins init_spins_1;
   openjij::graph::Spins init_spins_2;
   if (vartype == cimod::Vartype::SPIN) {
      init_spins_1 = openjij::graph::Polynomial<FloatType>(3).gen_spin(mt);
      init_spins_2 = openjij::graph::Polynomial<FloatType>(3).gen_spin(mt);
   }
   else {
      init_spins_1 = openjij::graph::Polynomial<FloatType>(3).gen_binary(mt);
      init_spins_2 = openjij::graph::Polynomial<FloatType>(3).gen_binary(mt);
   }
   auto system = openjij::system::make_classical_ising_polynomial(init_spins_1, poly_graph, vartype);
   if (type == "Dense") {
      TestCIPSystemDense(system);
      system.reset_variables(init_spins_2);
      TestCIPSystemDense(system);
   }
   else if (type == "Sparse") {
      TestCIPSystemSparse(system);
      system.reset_variables(init_spins_2);
      TestCIPSystemSparse(system);
   }
   else {
      throw std::runtime_error("Unknown type");
   }
}

template<typename FloatType>
void TestKLPSystemDense(const openjij::system::KLocalPolynomial<openjij::graph::Polynomial<FloatType>> &klp_system) {
   
   const int system_size = 3;
   
   EXPECT_EQ(klp_system.num_binaries, system_size);
      
   EXPECT_EQ(klp_system.get_adj().at(0).size(), 4);
   EXPECT_EQ(klp_system.get_adj().at(1).size(), 4);
   EXPECT_EQ(klp_system.get_adj().at(2).size(), 4);
   
   std::vector<std::vector<std::vector<openjij::graph::Index>>> adj_key(system_size);
   
   for (int i = 0; i < system_size; ++i) {
      for (const auto &index_key: klp_system.get_adj().at(i)) {
         adj_key[i].push_back(klp_system.get_keys().at(index_key));
      }
   }

   EXPECT_TRUE(ContainVector<openjij::graph::Index>({   0   }, adj_key[0]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 0, 1  }, adj_key[0]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 0, 2  }, adj_key[0]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[0]));
   
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({   1   }, adj_key[1]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 0, 1  }, adj_key[1]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 1, 2  }, adj_key[1]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[1]));

   EXPECT_TRUE(ContainVector<openjij::graph::Index>({   2   }, adj_key[2]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 0, 2  }, adj_key[2]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({ 1, 2  }, adj_key[2]));
   EXPECT_TRUE(ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[2]));
   
   const auto init_spins = klp_system.binaries;
   
   cimod::Polynomial<openjij::graph::Index, FloatType> polynomial;
   
   for (std::size_t i = 0; i < klp_system.get_values().size(); ++i) {
      polynomial[klp_system.get_keys().at(i)] = klp_system.get_values().at(i);
   }
   
   const int s0 = init_spins[0];
   const int s1 = init_spins[1];
   const int s2 = init_spins[2];

   if (klp_system.vartype == cimod::Vartype::BINARY) {
      const double dE0 = (-2*s0 + 1)*(polynomial.at({0}) + s1*polynomial.at({0, 1}) + s2*polynomial.at({0, 2}) + s1*s2*polynomial.at({0, 1, 2}));
      const double dE1 = (-2*s1 + 1)*(polynomial.at({1}) + s0*polynomial.at({0, 1}) + s2*polynomial.at({1, 2}) + s0*s2*polynomial.at({0, 1, 2}));
      const double dE2 = (-2*s2 + 1)*(polynomial.at({2}) + s0*polynomial.at({0, 2}) + s1*polynomial.at({1, 2}) + s0*s1*polynomial.at({0, 1, 2}));

      EXPECT_DOUBLE_EQ(dE0, klp_system.dE_single(0));
      EXPECT_DOUBLE_EQ(dE1, klp_system.dE_single(1));
      EXPECT_DOUBLE_EQ(dE2, klp_system.dE_single(2));
      
      const double abs_dE0 = std::abs(polynomial.at({0})) + std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({0, 2})) + std::abs(polynomial.at({0, 1, 2}));
      const double abs_dE1 = std::abs(polynomial.at({1})) + std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2}));
      const double abs_dE2 = std::abs(polynomial.at({2})) + std::abs(polynomial.at({0, 2})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2}));

      EXPECT_DOUBLE_EQ(klp_system.get_max_effective_dE(), std::max({abs_dE0    , abs_dE1    , abs_dE2})    );
      EXPECT_DOUBLE_EQ(klp_system.get_min_effective_dE(), std::abs(*std::min_element(klp_system.get_values().begin(), klp_system.get_values().end(), [](const auto a, const auto b) {
         return std::abs(a) < std::abs(b);
      })));
   }
   else {
      throw std::runtime_error("Unknown vartype detected");
   }
   for (std::size_t i = 0; i < klp_system.get_active_binaries().size(); ++i) {
      EXPECT_EQ(klp_system.get_active_binaries().at(i), i);
   }
}

template<typename FloatType>
void TestKLPSystemSparse(const openjij::system::KLocalPolynomial<openjij::graph::Polynomial<FloatType>> &klp_system) {
   
   const int system_size = 3;
   
   EXPECT_EQ(klp_system.num_binaries, system_size);
      
   EXPECT_EQ(klp_system.get_adj().at(0).size(), 2);
   EXPECT_EQ(klp_system.get_adj().at(1).size(), 3);
   EXPECT_EQ(klp_system.get_adj().at(2).size(), 3);
   
   std::vector<std::vector<std::vector<openjij::graph::Index>>> adj_key(system_size);
   
   for (int i = 0; i < system_size; ++i) {
      for (const auto &index_key: klp_system.get_adj().at(i)) {
         adj_key[i].push_back(klp_system.get_keys().at(index_key));
      }
   }

   EXPECT_FALSE(ContainVector<openjij::graph::Index>({   0   }, adj_key[0]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({ 0, 1  }, adj_key[0]));
   EXPECT_FALSE(ContainVector<openjij::graph::Index>({ 0, 2  }, adj_key[0]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[0]));
   
   EXPECT_FALSE(ContainVector<openjij::graph::Index>({   1   }, adj_key[1]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({ 0, 1  }, adj_key[1]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({ 1, 2  }, adj_key[1]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[1]));

   EXPECT_TRUE (ContainVector<openjij::graph::Index>({   2   }, adj_key[2]));
   EXPECT_FALSE(ContainVector<openjij::graph::Index>({ 0, 2  }, adj_key[2]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({ 1, 2  }, adj_key[2]));
   EXPECT_TRUE (ContainVector<openjij::graph::Index>({0, 1, 2}, adj_key[2]));
   
   const auto init_spins = klp_system.binaries;
   
   cimod::Polynomial<openjij::graph::Index, FloatType> polynomial;
   
   for (std::size_t i = 0; i < klp_system.get_values().size(); ++i) {
      polynomial[klp_system.get_keys().at(i)] = klp_system.get_values().at(i);
   }
   
   const int s0 = init_spins[0];
   const int s1 = init_spins[1];
   const int s2 = init_spins[2];
   
   if (klp_system.vartype == cimod::Vartype::BINARY) {
      const double dE0 = (-2*s0 + 1)*(s1*polynomial.at({0, 1}) + s1*s2*polynomial.at({0, 1, 2}));
      const double dE1 = (-2*s1 + 1)*(s0*polynomial.at({0, 1}) + s2*polynomial.at({1, 2}) + s0*s2*polynomial.at({0, 1, 2}));
      const double dE2 = (-2*s2 + 1)*(polynomial.at({2}) + s1*polynomial.at({1, 2}) + s0*s1*polynomial.at({0, 1, 2}));

      EXPECT_DOUBLE_EQ(dE0, klp_system.dE_single(0));
      EXPECT_DOUBLE_EQ(dE1, klp_system.dE_single(1));
      EXPECT_DOUBLE_EQ(dE2, klp_system.dE_single(2));
      
      const double abs_dE0 = std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({0, 1, 2}));
      const double abs_dE1 = std::abs(polynomial.at({0, 1})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2}));
      const double abs_dE2 = std::abs(polynomial.at({2})) + std::abs(polynomial.at({1, 2})) + std::abs(polynomial.at({0, 1, 2}));

      EXPECT_DOUBLE_EQ(klp_system.get_max_effective_dE(), std::max({abs_dE0    , abs_dE1    , abs_dE2})    );
      EXPECT_DOUBLE_EQ(klp_system.get_min_effective_dE(), std::abs(*std::min_element(klp_system.get_values().begin(), klp_system.get_values().end(), [](const auto a, const auto b) {
         return std::abs(a) < std::abs(b);
      })));
   }
   else {
      throw std::runtime_error("Unknown vartype detected");
   }
   for (std::size_t i = 0; i < klp_system.get_active_binaries().size(); ++i) {
      EXPECT_EQ(klp_system.get_active_binaries().at(i), i);
   }
}


template<typename IndexType, typename FloatType>
void TestKLPConstructorCimod(const cimod::Polynomial<IndexType, FloatType> &polynomial, std::string type) {
   cimod::BinaryPolynomialModel<IndexType, FloatType> bpm_cimod(polynomial, cimod::Vartype::BINARY);
   std::random_device rnd;
   std::mt19937 mt(rnd());
   openjij::graph::Binaries init_spins = openjij::graph::Polynomial<FloatType>(3).gen_binary(mt);
   
   auto system = openjij::system::make_k_local_polynomial(init_spins, bpm_cimod.ToSerializable());
   if (type == "Dense") {
      TestKLPSystemDense(system);
      system.reset_binaries(openjij::graph::Polynomial<FloatType>(3).gen_binary(mt));
      TestKLPSystemDense(system);
   }
   else if (type == "Sparse") {
      TestKLPSystemSparse(system);
      system.reset_binaries(openjij::graph::Polynomial<FloatType>(3).gen_binary(mt));
      TestKLPSystemSparse(system);
   }
   else {
      throw std::runtime_error("Unknown type");
   }
}


template<typename IndexType, typename FloatType>
void TestKLPConstructorGraph(const cimod::Polynomial<IndexType, FloatType> &polynomial, std::string type) {
   cimod::BinaryPolynomialModel<IndexType, FloatType> bpm_cimod(polynomial, cimod::Vartype::BINARY);
   openjij::graph::Polynomial<FloatType> poly_graph(3);
   for (const auto &it: bpm_cimod.GetPolynomial()) {
      poly_graph.J(it.first) = it.second;
   }
   std::random_device rnd;
   std::mt19937 mt(rnd());
   openjij::graph::Binaries init_spins = poly_graph.gen_binary(mt);
   
   auto system = openjij::system::make_k_local_polynomial(init_spins, poly_graph);
   if (type == "Dense") {
      TestKLPSystemDense(system);
      system.reset_binaries(openjij::graph::Polynomial<FloatType>(3).gen_binary(mt));
      TestKLPSystemDense(system);
   }
   else if (type == "Sparse") {
      TestKLPSystemSparse(system);
      system.reset_binaries(openjij::graph::Polynomial<FloatType>(3).gen_binary(mt));
      TestKLPSystemSparse(system);
   }
   else {
      throw std::runtime_error("Unknown type");
   }
}

