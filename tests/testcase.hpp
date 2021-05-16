//    Copyright 2019 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef OPENJIJ_TEST_TESTCASE_HPP__
#define OPENJIJ_TEST_TESTCASE_HPP__

#include <graph/all.hpp>
#include <iostream>
#include <bitset>

#if TEST_CASE_INDEX == 1
//GraphType -> Dense or Sparse
template<typename GraphType>
GraphType generate_interaction() {
    auto interaction = GraphType(num_system_size);
    interaction.J(0,0)=-0.1;
    interaction.J(0,1)=-0.9;
    interaction.J(0,2)=0.2;
    interaction.J(0,3)=0.1;
    interaction.J(0,4)=1.3;
    interaction.J(0,5)=0.8;
    interaction.J(0,6)=0.9;
    interaction.J(0,7)=0.4;
    interaction.J(1,1)=-0.7;
    interaction.J(1,2)=-1.6;
    interaction.J(1,3)=1.5;
    interaction.J(1,4)=1.5;
    interaction.J(1,5)=1.2;
    interaction.J(1,6)=-1.5;
    interaction.J(1,7)=-1.7;
    interaction.J(2,2)=-0.6;
    interaction.J(2,3)=1.2;
    interaction.J(2,4)=-1.3;
    interaction.J(2,5)=-0.5;
    interaction.J(2,6)=-1.9;
    interaction.J(2,7)=1.2;
    interaction.J(3,3)=0.8;
    interaction.J(3,4)=-0.5;
    interaction.J(3,5)=-0.4;
    interaction.J(3,6)=-1.8;
    interaction.J(3,7)=-2.0;
    interaction.J(4,4)=0.6;
    interaction.J(4,5)=-2.0;
    interaction.J(4,6)=-1.9;
    interaction.J(4,7)=0.5;
    interaction.J(5,5)=-1.8;
    interaction.J(5,6)=-1.2;
    interaction.J(5,7)=1.8;
    interaction.J(6,6)=0.3;
    interaction.J(6,7)=1.4;
    interaction.J(7,7)=1.8;
    return interaction;
}

openjij::graph::Spins get_true_groundstate(){
    return openjij::graph::Spins({-1, -1, 1, 1, 1, 1, 1, -1});
}

#elif TEST_CASE_INDEX == 2
template<typename GraphType>
GraphType generate_interaction() {
    auto interaction = GraphType(num_system_size);
    interaction.J(0,0)=2.8;
    interaction.J(0,1)=2.5;
    interaction.J(0,2)=-0.2;
    interaction.J(0,3)=-1.6;
    interaction.J(0,4)=-0.8;
    interaction.J(0,5)=0.1;
    interaction.J(0,6)=-1.0;
    interaction.J(0,7)=-1.0;
    interaction.J(1,1)=2.4;
    interaction.J(1,2)=2.6;
    interaction.J(1,3)=2.9;
    interaction.J(1,4)=2.1;
    interaction.J(1,5)=0.2;
    interaction.J(1,6)=1.0;
    interaction.J(1,7)=1.4;
    interaction.J(2,2)=0.6;
    interaction.J(2,3)=-3.0;
    interaction.J(2,4)=2.2;
    interaction.J(2,5)=1.2;
    interaction.J(2,6)=0.6;
    interaction.J(2,7)=1.5;
    interaction.J(3,3)=-0.5;
    interaction.J(3,4)=-1.8;
    interaction.J(3,5)=-0.7;
    interaction.J(3,6)=0.6;
    interaction.J(3,7)=1.4;
    interaction.J(4,4)=-0.8;
    interaction.J(4,5)=-2.2;
    interaction.J(4,6)=-1.8;
    interaction.J(4,7)=0.1;
    interaction.J(5,5)=-1.8;
    interaction.J(5,6)=0.1;
    interaction.J(5,7)=-1.1;
    interaction.J(6,6)=-1.8;
    interaction.J(6,7)=2.0;
    interaction.J(7,7)=0.9;
    return interaction;
}

openjij::graph::Spins get_true_groundstate(){
    return openjij::graph::Spins({1, -1, 1, 1, 1, 1, 1, -1});
}
#elif TEST_CASE_INDEX == 3
//WARNING: Hard Instance
template<typename GraphType>
GraphType generate_interaction() {
    auto interaction = GraphType(num_system_size);
    interaction.J(0,0)=2.7;
    interaction.J(0,1)=-0.6;
    interaction.J(0,2)=-2.6;
    interaction.J(0,3)=2.0;
    interaction.J(0,4)=-3.0;
    interaction.J(0,5)=-2.6;
    interaction.J(0,6)=1.5;
    interaction.J(0,7)=1.5;
    interaction.J(1,1)=1.5;
    interaction.J(1,2)=-1.6;
    interaction.J(1,3)=2.9;
    interaction.J(1,4)=1.7;
    interaction.J(1,5)=-2.4;
    interaction.J(1,6)=2.6;
    interaction.J(1,7)=-1.4;
    interaction.J(2,2)=-1.3;
    interaction.J(2,3)=-1.1;
    interaction.J(2,4)=-0.1;
    interaction.J(2,5)=-1.8;
    interaction.J(2,6)=0.3;
    interaction.J(2,7)=-2.4;
    interaction.J(3,3)=-0.3;
    interaction.J(3,4)=0.4;
    interaction.J(3,5)=-0.8;
    interaction.J(3,6)=-2.4;
    interaction.J(3,7)=-1.5;
    interaction.J(4,4)=-0.3;
    interaction.J(4,5)=-0.6;
    interaction.J(4,6)=-0.6;
    interaction.J(4,7)=0.7;
    interaction.J(5,5)=0.2;
    interaction.J(5,6)=1.8;
    interaction.J(5,7)=-1.2;
    interaction.J(6,6)=1.6;
    interaction.J(6,7)=-1.1;
    interaction.J(7,7)=-0.3;
    return interaction;
}

openjij::graph::Spins get_true_groundstate(){
    return openjij::graph::Spins({-1, -1, -1, 1, -1, -1, 1, 1});
}
#elif TEST_CASE_INDEX == 4
template<typename GraphType>
GraphType generate_interaction() {
    auto interaction = GraphType(num_system_size);
    interaction.J(0,0)=-1.6;
    interaction.J(0,1)=-1.8;
    interaction.J(0,2)=1.3;
    interaction.J(0,3)=0.7;
    interaction.J(0,4)=-0.6;
    interaction.J(0,5)=1.6;
    interaction.J(0,6)=-2.7;
    interaction.J(0,7)=-0.7;
    interaction.J(1,1)=1.0;
    interaction.J(1,2)=-1.0;
    interaction.J(1,3)=-2.3;
    interaction.J(1,4)=-2.4;
    interaction.J(1,5)=0.6;
    interaction.J(1,6)=-0.1;
    interaction.J(1,7)=-2.1;
    interaction.J(2,2)=-0.9;
    interaction.J(2,3)=-1.0;
    interaction.J(2,4)=1.0;
    interaction.J(2,5)=0.5;
    interaction.J(2,6)=1.4;
    interaction.J(2,7)=2.7;
    interaction.J(3,3)=2.7;
    interaction.J(3,4)=0.6;
    interaction.J(3,5)=2.9;
    interaction.J(3,6)=-2.6;
    interaction.J(3,7)=1.8;
    interaction.J(4,4)=0.6;
    interaction.J(4,5)=0.6;
    interaction.J(4,6)=1.9;
    interaction.J(4,7)=-2.6;
    interaction.J(5,5)=0.1;
    interaction.J(5,6)=-2.0;
    interaction.J(5,7)=-2.0;
    interaction.J(6,6)=-1.1;
    interaction.J(6,7)=0.4;
    interaction.J(7,7)=1.6;
    return interaction;
}

openjij::graph::Spins get_true_groundstate(){
    return openjij::graph::Spins({-1, -1, 1, -1, -1, 1, -1, -1});
}

#elif TEST_CASE_INDEX == 5
template<typename GraphType>
GraphType generate_interaction() {
    auto interaction = GraphType(num_system_size);
    interaction.J(0,0)=-0.30;
    interaction.J(0,1)=-1.16;
    interaction.J(0,2)=0.05;
    interaction.J(0,3)=2.08;
    interaction.J(0,4)=0.38;
    interaction.J(0,5)=2.05;
    interaction.J(0,6)=-2.31;
    interaction.J(0,7)=-1.19;
    interaction.J(1,1)=-0.01;
    interaction.J(1,2)=-1.25;
    interaction.J(1,3)=-2.57;
    interaction.J(1,4)=-0.90;
    interaction.J(1,5)=-0.90;
    interaction.J(1,6)=-2.27;
    interaction.J(1,7)=-1.04;
    interaction.J(2,2)=-0.98;
    interaction.J(2,3)=2.65;
    interaction.J(2,4)=2.45;
    interaction.J(2,5)=2.65;
    interaction.J(2,6)=2.87;
    interaction.J(2,7)=2.30;
    interaction.J(3,3)=-2.70;
    interaction.J(3,4)=1.82;
    interaction.J(3,5)=-0.91;
    interaction.J(3,6)=1.99;
    interaction.J(3,7)=-0.16;
    interaction.J(4,4)=1.51;
    interaction.J(4,5)=2.79;
    interaction.J(4,6)=-2.87;
    interaction.J(4,7)=2.55;
    interaction.J(5,5)=-0.67;
    interaction.J(5,6)=-2.75;
    interaction.J(5,7)=-2.07;
    interaction.J(6,6)=1.41;
    interaction.J(6,7)=-2.27;
    interaction.J(7,7)=1.08;
    return interaction;
}

openjij::graph::Spins get_true_groundstate(){
    return openjij::graph::Spins({-1, -1, 1, 1, -1, -1, -1, -1});
}

#endif

//chimera graph

template<typename FloatType>
openjij::graph::Chimera<FloatType> generate_chimera_interaction() {
    auto interaction = openjij::graph::Chimera<FloatType>(2,2);
    interaction.J(0,0,0,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(0,0,0,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(0,0,0,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(0,0,0,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(0,0,1,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(0,0,1,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(0,0,1,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(0,0,1,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(0,0,2,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(0,0,2,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(0,0,2,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(0,0,2,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(0,0,3,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(0,0,3,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(0,0,3,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(0,0,3,openjij::graph::ChimeraDir::IN_3or7) = +0.25;

    interaction.J(0,1,0,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(0,1,0,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(0,1,0,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(0,1,0,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(0,1,1,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(0,1,1,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(0,1,1,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(0,1,1,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(0,1,2,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(0,1,2,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(0,1,2,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(0,1,2,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(0,1,3,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(0,1,3,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(0,1,3,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(0,1,3,openjij::graph::ChimeraDir::IN_3or7) = +0.25;

    interaction.J(1,0,0,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(1,0,0,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(1,0,0,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(1,0,0,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(1,0,1,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(1,0,1,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(1,0,1,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(1,0,1,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(1,0,2,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(1,0,2,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(1,0,2,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(1,0,2,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(1,0,3,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(1,0,3,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(1,0,3,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(1,0,3,openjij::graph::ChimeraDir::IN_3or7) = +0.25;

    interaction.J(1,1,0,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(1,1,0,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(1,1,0,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(1,1,0,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(1,1,1,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(1,1,1,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(1,1,1,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(1,1,1,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(1,1,2,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(1,1,2,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(1,1,2,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(1,1,2,openjij::graph::ChimeraDir::IN_3or7) = +0.25;
    interaction.J(1,1,3,openjij::graph::ChimeraDir::IN_0or4) = +0.25;
    interaction.J(1,1,3,openjij::graph::ChimeraDir::IN_1or5) = +0.25;
    interaction.J(1,1,3,openjij::graph::ChimeraDir::IN_2or6) = +0.25;
    interaction.J(1,1,3,openjij::graph::ChimeraDir::IN_3or7) = +0.25;

    interaction.h(0,0,0) = +1;

    interaction.J(0,0,6,openjij::graph::ChimeraDir::PLUS_C) = +1;
    interaction.J(0,0,3,openjij::graph::ChimeraDir::PLUS_R) = -1;
    interaction.J(1,0,5,openjij::graph::ChimeraDir::PLUS_C) = +1;


    return interaction;
}

template<typename FloatType> 
openjij::graph::Spins get_true_chimera_groundstate(const openjij::graph::Chimera<FloatType> &interaction){
    openjij::graph::Spins ret_spin(interaction.get_num_spins());
    ret_spin[interaction.to_ind(0,0,0)] = -1;
    ret_spin[interaction.to_ind(0,0,1)] = -1;
    ret_spin[interaction.to_ind(0,0,2)] = -1;
    ret_spin[interaction.to_ind(0,0,3)] = -1;
    ret_spin[interaction.to_ind(0,0,4)] = +1;
    ret_spin[interaction.to_ind(0,0,5)] = +1;
    ret_spin[interaction.to_ind(0,0,6)] = +1;
    ret_spin[interaction.to_ind(0,0,7)] = +1;

    ret_spin[interaction.to_ind(0,1,0)] = +1;
    ret_spin[interaction.to_ind(0,1,1)] = +1;
    ret_spin[interaction.to_ind(0,1,2)] = +1;
    ret_spin[interaction.to_ind(0,1,3)] = +1;
    ret_spin[interaction.to_ind(0,1,4)] = -1;
    ret_spin[interaction.to_ind(0,1,5)] = -1;
    ret_spin[interaction.to_ind(0,1,6)] = -1;
    ret_spin[interaction.to_ind(0,1,7)] = -1;

    ret_spin[interaction.to_ind(1,0,0)] = -1;
    ret_spin[interaction.to_ind(1,0,1)] = -1;
    ret_spin[interaction.to_ind(1,0,2)] = -1;
    ret_spin[interaction.to_ind(1,0,3)] = -1;
    ret_spin[interaction.to_ind(1,0,4)] = +1;
    ret_spin[interaction.to_ind(1,0,5)] = +1;
    ret_spin[interaction.to_ind(1,0,6)] = +1;
    ret_spin[interaction.to_ind(1,0,7)] = +1;

    ret_spin[interaction.to_ind(1,1,0)] = +1;
    ret_spin[interaction.to_ind(1,1,1)] = +1;
    ret_spin[interaction.to_ind(1,1,2)] = +1;
    ret_spin[interaction.to_ind(1,1,3)] = +1;
    ret_spin[interaction.to_ind(1,1,4)] = -1;
    ret_spin[interaction.to_ind(1,1,5)] = -1;
    ret_spin[interaction.to_ind(1,1,6)] = -1;
    ret_spin[interaction.to_ind(1,1,7)] = -1;

    return ret_spin;
}

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
   const std::size_t loop = std::pow(2, vec_in.size());
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


#endif
