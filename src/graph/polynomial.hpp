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
#include <utility/vectorhash.hpp>

namespace openjij {
namespace graph {


//! @brief Polynomial graph class, which can treat many-body interactions
//! The Hamiltonian is like
//! \f[
//! H=\sum_{i \neq j} Q_{ij} x_i x_j +  \sum_{i \neq j \neq k} Q_{ijk} x_i x_j x_k + \ldots
//! \f]
//! Note here that \f$ x_i \in \{0, 1\} \f$ or \f$ x_i \in \{-1, +1\} \f$.
//! @tparam FloatType floating-point type
template<typename FloatType>
class Polynomial: public Graph {
   static_assert(std::is_floating_point<FloatType>::value, "FloatType must be floating-point type.");
   
public:
   //! @brief interaction type
   using Interactions = std::unordered_map<std::vector<Index>, FloatType, utility::VectorHash>;
   
   //! @brief float type
   using value_type   = FloatType;
      
   Polynomial(const std::size_t num_spins, const std::string vartype): Graph(num_spins), vartype_(vartype) {
      if (vartype != "SPIN" && vartype != "BINARY") {
         std::stringstream ss;
         ss << "The input vartype = " << vartype;
         ss << " is unknown vartype.";
         ss << "Ising or Binary is allowed\n";
         std::runtime_error(ss.str());
      }
   }
   
   Polynomial(const std::size_t num_spins): Graph(num_spins), vartype_("SPIN") {}
   
   Polynomial(const json &j): Polynomial(json_parse_polynomial<FloatType>(j)) {}
   
   Polynomial(const cimod::BinaryPolynomialModel<Index, FloatType> &bpm): Graph(bpm.length()) {
      if (bpm.get_vartype() == cimod::Vartype::SPIN) {
         vartype_ = "SPIN";
      }
      else if (bpm.get_vartype() == cimod::Vartype::BINARY) {
         vartype_ = "BINARY";
      }
      else {
         std::runtime_error("Unknown vartype is detected in cimod\n");
      }
      for (const auto &it: bpm.get_polynomial()) {
         auto temp = it.first;
         std::sort(temp.begin(), temp.end());
         UpdateMaxVariable(temp);
         J_[temp] += it.second;
      }
   }
      
   FloatType &J(const std::unordered_set<Index> &index_set) {
      if (index_set.size() > Graph::size()) {
         std::stringstream ss;
         ss << "Too small system size. ";
         ss << "The degree of the input polynomial interaction is " << index_set.size();
         ss << ". But the system size is" << Graph::size() << std::string("\n");
         std::runtime_error(ss.str());
      }
      std::vector<Index> index(index_set.begin(), index_set.end());
      std::sort(index.begin(), index.end());
      UpdateMaxVariable(index);
      return J_[index];
   }
   
   template<typename... Args>
   FloatType &J(Args... args) {
      std::unordered_set<Index> index_set{(Index)args...};
      return J(index_set);
   }
   
   const FloatType &J(const std::unordered_set<Index> &index_set) const {
      std::vector<Index> index(index_set.begin(), index_set.end());
      std::sort(index.begin(), index.end());
      return J_.at(index);
   }
   
   template<typename... Args>
   const FloatType &J(Args... args) const {
      std::unordered_set<Index> index_set{(Index)args...};
      return J(index_set);
   }
   
   const Interactions &GetInteractions() const {
      return J_;
   }
   
   const Index &GetMaxVariable() const {
      return max_variable_;
   }
   
   const std::string &GetVartype() const {
      return vartype_;
   }
   
   bool isIsing() const {
      if (vartype_ == "SPIN") {
         return true;
      }
      else {
         return false;
      }
   }
   
   FloatType CalculateEnergy(const Spins& spins) const {
      if(spins.size() != Graph::size()){
         std::out_of_range("Out of range in CalclateEnergy in Polynomial graph.");
      }
      FloatType energy = 0.0;
      for (const auto &it: J_) {
         Spin temp_spin_multiple = 1;
         for (const auto &index: it.first) {
            temp_spin_multiple *= spins[index];
         }
         energy += temp_spin_multiple*it.second;
      }
      return energy;
   }
   
private:
   Interactions J_;
   Index max_variable_ = 0;
   std::string vartype_ = "SPIN";
   
   void UpdateMaxVariable(const std::vector<Index> &index) {
      if (max_variable_ < index[index.size() - 1]) {
         max_variable_ = index[index.size() - 1];
      }
   }
   
};
}
}


#endif /* polynomial_hpp */