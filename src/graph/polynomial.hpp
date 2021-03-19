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
//! @author Kohei Suzuki
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
   
   //! @brief floating-point type
   using value_type = FloatType;
      
   //! @brief Constructor of Polynomial class to initialize variables and vartype.
   //! @param num_variables std::size_t
   //! @param vartype openjij::graph::Vartype
   Polynomial(const std::size_t num_variables, const Vartype &vartype): Graph(num_variables), vartype_(vartype) {}
   
   //! @brief Constructor of Polynomial class to initialize variables. Note that vartype is initialized with openjij::graph::Vartype::SPIN.
   //! @param num_variables std::size_t
   Polynomial(const std::size_t num_variables): Graph(num_variables), vartype_(Vartype::SPIN) {}
   
   //! @brief Constructor of Polynomial class to initialize num_variables, vartype, and interactions from json by using a delegating constructor.
   //! @param j JSON object
   Polynomial(const json &j): Polynomial(json_parse_polynomial<FloatType>(j)) {}
   
   //! @brief Constructor of Polynomial class to initialize num_variables, vartype, and interactions from cimod.
   //! @param bpm cimod::BinaryPolynomialModel object
   Polynomial(const cimod::BinaryPolynomialModel<Index, FloatType> &bpm): Graph(bpm.length()) {
      if (bpm.get_vartype() == cimod::Vartype::SPIN) {
         vartype_ = Vartype::SPIN;
      }
      else if (bpm.get_vartype() == cimod::Vartype::BINARY) {
         vartype_ = Vartype::BINARY;
      }
      else {
         std::stringstream ss;
         ss << "Unknown vartype detected in " << __func__ << std::endl;
         throw std::runtime_error(ss.str());
      }
      for (const auto &it: bpm.get_polynomial()) {
         auto temp = it.first;
         std::sort(temp.begin(), temp.end());
         UpdateMaxVariable(temp);
         J_[temp] += it.second;
      }
   }
    
   //! @brief Access the interaction corresponding to the input argument "std::vector<Index>&& index" (rvalue references) to set an interaction.
   //! @details The input argument is sorted and duplicate values are removed.
   //! @param index std::vector<Index>&&
   //! @return The interaction corresponding to "std::vector<Index>&& index", i.e., J[index]
   FloatType &J(std::vector<Index>&& index) {
      std::sort(index.begin(), index.end());
      index.erase(std::unique(index.begin(), index.end()), index.end());
      if (index.size() > Graph::size()) {
         std::stringstream ss;
         ss << "Too small system size. ";
         ss << "The degree of the input polynomial interaction is " << index.size();
         ss << ". But the system size is" << Graph::size() << std::string("\n");
         throw std::runtime_error(ss.str());
      }
      UpdateMaxVariable(index);
      return J_[index];
   }
   
   //! @brief Access the interaction corresponding to the input argument "std::vector<Index>& index" (lvalue references) to set an interaction.
   //! @details The input argument is sorted and duplicate values are removed.
   //! @param index std::vector<Index>&
   //! @return The interaction corresponding to "std::vector<Index>& index", i.e., J[index]
   FloatType &J(std::vector<Index>& index) {
      std::sort(index.begin(), index.end());
      index.erase(std::unique(index.begin(), index.end()), index.end());
      if (index.size() > Graph::size()) {
         std::stringstream ss;
         ss << "Too small system size. ";
         ss << "The degree of the input polynomial interaction is " << index.size();
         ss << ". But the system size is" << Graph::size() << std::string("\n");
         throw std::runtime_error(ss.str());
      }
      UpdateMaxVariable(index);
      return J_[index];
   }
   
   FloatType &J(const std::vector<Index>& index) {
      std::vector<Index> temp_index = index;
      return J(temp_index);
   }
   
   //! @brief Access the interaction corresponding to the input variables "args" (variadic templates) to set an interaction. Note that "args" is converted to std::vector<Index>.
   //! @details Note that "args" is converted to std::vector<Index>, which is sorted and whose duplicate values are removed.
   //! @param args parameter pack
   //! @return The interaction corresponding to "args", i.e., J[args]
   template<typename... Args>
   FloatType &J(Args... args) {
      return J(std::vector<Index>{(Index)args...});
   }
   
   //! @brief Return the interaction corresponding to the input argument "std::vector<Index> &&index".
   //! @details The input argument is sorted and duplicate values are removed. This function does not change the interaction.
   //! @param args parameter pack
   //! @return The interaction corresponding to "std::vector<Index>& index", i.e., J[index]
   FloatType J(std::vector<Index> &&index) const {
      std::sort(index.begin(), index.end());
      index.erase(std::unique(index.begin(), index.end()), index.end());
      return J_.at(index);
   }
   
   //! @brief Return the interaction corresponding to the input argument "std::vector<Index> &index".
   //! @details The input argument is sorted and duplicate values are removed. This function does not change the interaction.
   //! @param index std::vector<Index>&
   //! @return The interaction corresponding to "std::vector<Index>& index", i.e., J[index]
   const FloatType &J(std::vector<Index> &index) const {
      std::sort(index.begin(), index.end());
      index.erase(std::unique(index.begin(), index.end()), index.end());
      return J_.at(index);
   }
   
   const FloatType &J(const std::vector<Index> &index) const {
      std::vector<Index> temp_index = index;
      return J(temp_index);
   }
   
   template<typename... Args>
   const FloatType &J(Args... args) const {
      std::vector<Index> index{(Index)args...};
      std::sort(index.begin(), index.end());
      index.erase(std::unique(index.begin(), index.end()), index.end());
      return J_.at(index);
   }
   
   const Interactions &GetInteractions() const {
      return J_;
   }
   
   const Index &GetMaxVariable() const {
      return max_variable_;
   }
   
   const Vartype &GetVartype() const {
      return vartype_;
   }
   
   FloatType CalculateEnergy(const Spins& spins) const {
      if(spins.size() != Graph::size()){
         throw std::out_of_range("Out of range in CalclateEnergy in Polynomial graph.");
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
   Vartype vartype_    = Vartype::SPIN;
   
   void UpdateMaxVariable(const std::vector<Index> &index) {
      if (index.size() == 0) {
         return;
      }
      if (max_variable_ < index[index.size() - 1]) {
         max_variable_ = index[index.size() - 1];
      }
   }
   
};
}
}


#endif /* polynomial_hpp */
