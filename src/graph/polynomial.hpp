//    Copyright 2021 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

//! @file polynomial.hpp
//! @brief Graph class to represent polynomial unconstrained binary model or Ising model with polynomial interactions.
//! @date 2021-03-11
//! @copyright Copyright (c) Jij Inc. 2021

#ifndef polynomial_hpp
#define polynomial_hpp

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <graph/json/parse.hpp>
#include <graph/graph.hpp>
#include "cimod/src/hash.hpp"

namespace openjij {
namespace graph {

template<typename FloatType>
class Polynomial: public Graph {
   static_assert(std::is_floating_point<FloatType>::value, "FloatType must be floating-point type.");
   
public:
   
   //! @brief Floating-point type
   using value_type = FloatType;
   
   Polynomial(const std::size_t num_variables, const cimod::Vartype &vartype): Graph(num_variables), vartype_(vartype) {}
   
   explicit Polynomial(const std::size_t num_variables): Graph(num_variables), vartype_(cimod::Vartype::SPIN) {}
   
   explicit Polynomial(const nlohmann::json &j): Polynomial(json_parse_polynomial<FloatType>(j)) {}
   
   explicit Polynomial(nlohmann::json &j): Polynomial(json_parse_polynomial<FloatType>(j)) {}
      
   explicit Polynomial(const cimod::BinaryPolynomialModel<Index, FloatType> &bpm):
   Graph(bpm.get_num_variables()), poly_key_list_(bpm._get_keys()), poly_value_list_(bpm._get_values()), poly_key_inv_(bpm.GetKeysInv()), vartype_(bpm.get_vartype()) {
      if (poly_key_list_.size() != poly_value_list_.size() || poly_key_list_.size() != poly_key_inv_.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      for (const auto &it: bpm.GetVariables()) {
         UpdateMaxVariable(it);
      }
   }

   
   FloatType &J(std::vector<Index> &key) {
      std::sort(key.begin(), key.end());
      CheckKeyValid(key);
      if (poly_key_inv_.count(key) == 0) {
         poly_key_inv_[key] = poly_value_list_.size();
         poly_key_list_.push_back(key);
         poly_value_list_.push_back(0.0);
      }
      return poly_value_list_[poly_key_inv_[key]];
   }
      
   const FloatType &J(std::vector<Index> &key) const {
      std::sort(key.begin(), key.end());
      CheckKeyValid(key);
      if (poly_key_inv_.count(key) == 0) {
         poly_key_inv_[key] = poly_value_list_.size();
         poly_key_list_.push_back(key);
         poly_value_list_.push_back(0.0);
      }
      return poly_value_list_[poly_key_inv_[key]];
   }
   
   FloatType &J(const std::vector<Index> &key) {
      std::vector<Index> copied_key = key;
      return J(copied_key);
   }
      
   const FloatType &J(const std::vector<Index> &key) const {
      std::vector<Index> copied_key = key;
      return J(copied_key);
   }
   
   template<typename... Args>
   FloatType &J(Args... args) {
      std::vector<Index> copied_key{(Index)args...};
      return J(copied_key);
   }
   
   template<typename... Args>
   const FloatType &J(Args... args) const {
      std::vector<Index> copied_key{(Index)args...};
      return J(copied_key);
   }
   
   cimod::Polynomial<Index, FloatType> get_polynomial() const {
      cimod::Polynomial<Index, FloatType> poly_map;
      for (std::size_t i = 0; i < poly_key_list_.size(); ++i) {
         poly_map[poly_key_list_[i]] = poly_value_list_[i];
      }
      return poly_map;
   }
   
   const cimod::PolynomialKeyList<Index> &get_keys() const {
      return poly_key_list_;
   }
   
   const cimod::PolynomialValueList<FloatType> &get_values() const {
      return poly_value_list_;
   }
   
   cimod::Vartype get_vartype() const {
      return vartype_;
   }
   
   Index get_max_variable() const {
      return max_variable_;
   }
   
   FloatType calc_energy(const Spins& spins, bool omp_flag = true) const {
      if(spins.size() != Graph::size()){
         throw std::out_of_range("Out of range in calc_energy in Polynomial graph.");
      }
      
      FloatType energy = 0.0;
      std::size_t num_interactions = poly_key_list_.size();
      
      if (omp_flag) {
#pragma omp parallel for reduction (+: energy)
         for (std::size_t i = 0; i < num_interactions; ++i) {
            Spin spin_multiple = 1;
            for (const auto &index: poly_key_list_[i]) {
               spin_multiple *= spins[index];
               if (spin_multiple == 0.0) {
                  break;
               }
            }
            energy += spin_multiple*poly_value_list_[i];
         }
      }
      else {
         for (std::size_t i = 0; i < num_interactions; ++i) {
            Spin spin_multiple = 1;
            for (const auto &index: poly_key_list_[i]) {
               spin_multiple *= spins[index];
               if (spin_multiple == 0.0) {
                  break;
               }
            }
            energy += spin_multiple*poly_value_list_[i];
         }
      }
      return energy;
   }
   
private:
   cimod::PolynomialKeyList<Index> poly_key_list_;
   
   cimod::PolynomialValueList<FloatType> poly_value_list_;
   
   std::unordered_map<std::vector<Index>, std::size_t, cimod::vector_hash> poly_key_inv_;
   
   cimod::Vartype vartype_ = cimod::Vartype::NONE;
   
   Index max_variable_ = 0;
   
   void CheckKeyValid(const std::vector<Index> &key) const {
      if (0 < key.size()) {
         //key is assumed to be sorted
         for (std::size_t i = 0; i < key.size() - 1; ++i) {
            if (key[i] == key[i + 1]) {
               throw std::runtime_error("No self-loops allowed");
            }
         }
      }
      if (key.size() > Graph::size()) {
         std::stringstream ss;
         ss << "Too small system size. ";
         ss << "The degree of the input polynomial interaction is " << key.size();
         ss << ". But the system size is " << Graph::size() << std::string("\n");
         throw std::runtime_error(ss.str());
      }
   }
   
   void UpdateMaxVariable(Index variable) {
      if (max_variable_ < variable) {
         max_variable_ = variable;
      }
   }
   

   
};
 
} //graph
} //openjij


#endif /* polynomial_hpp */
