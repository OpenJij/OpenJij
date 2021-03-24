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
#include <utility/vectorhash.hpp>

namespace openjij {
namespace graph {


//! @brief Polynomial graph class, which can treat many-body interactions.
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
   //! @brief Interaction type
   using Interactions = std::unordered_map<std::vector<Index>, FloatType, utility::VectorHash>;
   
   //! @brief Floating-point type
   using value_type = FloatType;
      
   //! @brief Constructor of Polynomial class to initialize variables and vartype.
   //! @param num_variables std::size_t
   //! @param vartype cimod::Vartype
   Polynomial(const std::size_t num_variables, const cimod::Vartype &vartype): Graph(num_variables), vartype_(vartype) {}
   
   //! @brief Constructor of Polynomial class to initialize variables. Note that vartype is initialized with the cimod::Vartype::SPIN.
   //! @param num_variables std::size_t
   Polynomial(const std::size_t num_variables): Graph(num_variables), vartype_(cimod::Vartype::SPIN) {}
   
   //! @brief Constructor of Polynomial class to initialize num_variables, vartype, and interactions from json by using a delegating constructor.
   //! @param j JSON object
   Polynomial(const json &j): Polynomial(json_parse_polynomial<FloatType>(j)) {}
   
   //! @brief Constructor of Polynomial class to initialize num_variables, vartype, and interactions from cimod.
   //! @param bpm cimod::BinaryPolynomialModel object
   Polynomial(const cimod::BinaryPolynomialModel<Index, FloatType> &bpm): Graph(bpm.length()), vartype_(bpm.get_vartype()) {
      for (const auto &it: bpm.get_polynomial()) {
         if (std::abs(it.second) > 0.0) {
            auto temp = it.first;
            std::sort(temp.begin(), temp.end());
            J_[temp] += it.second;
         }
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
         ss << ". But the system size is " << Graph::size() << std::string("\n");
         throw std::runtime_error(ss.str());
      }
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
         ss << ". But the system size is " << Graph::size() << std::string("\n");
         throw std::runtime_error(ss.str());
      }
      return J_[index];
   }
   
   //! @brief Access the interaction corresponding to the input argument "const std::vector<Index>& index" (lvalue references) to set an interaction.
   //! @details The temporary object is generated from the input argument and is passed to "FloatType &J(std::vector<Index>&& index)".
   //! @param index const std::vector<Index>&
   //! @return The interaction corresponding to "const std::vector<Index>& index", i.e., J[index]
   FloatType &J(const std::vector<Index>& index) {
      return J(std::vector<Index>{index});//Access J(std::vector<Index>&& index)
   }
   
   //! @brief Access the interaction corresponding to the input argument "args" (parameter pack) to set an interaction.
   //! @details The temporary object is generated from the input argument "args" and is passed to "FloatType &J(std::vector<Index>&& index)".
   //! @param args parameter pack
   //! @return The interaction corresponding to "args", i.e., J[args]
   template<typename... Args>
   FloatType &J(Args... args) {
      return J(std::vector<Index>{(Index)args...});//Access J(std::vector<Index>&& index)
   }
   
   //! @brief Return the interaction corresponding to the input argument "std::vector<Index> &&index" (rvalue references).
   //! @details The input argument is sorted and duplicate values are removed. This function does not change the interaction.
   //! @param index std::vector<Index>&&
   //! @return The interaction corresponding to "std::vector<Index>&& index", i.e., J.at(index)
   FloatType J(std::vector<Index> &&index) const {
      std::sort(index.begin(), index.end());
      index.erase(std::unique(index.begin(), index.end()), index.end());
      return J_.at(index);
   }
   
   //! @brief Return the interaction corresponding to the input argument "std::vector<Index> &index" (lvalue references).
   //! @details The input argument is sorted and duplicate values are removed. This function does not change the interaction.
   //! @param index std::vector<Index>&
   //! @return The interaction corresponding to "std::vector<Index>& index", i.e., J.at(index)
   const FloatType &J(std::vector<Index> &index) const {
      std::sort(index.begin(), index.end());
      index.erase(std::unique(index.begin(), index.end()), index.end());
      return J_.at(index);
   }
   
   //! @brief Return the interaction corresponding to the input argument "const std::vector<Index> &index".
   //! @details The temporary object is generated from the input argument and is passed to "FloatType &J(std::vector<Index>&& index) const".
   //! @param index const std::vector<Index>&
   //! @return The interaction corresponding to "const std::vector<Index>& index", i.e., J.at(index)
   const FloatType &J(const std::vector<Index> &index) const {
      return J(std::vector<Index>{index});//Access J(std::vector<Index>&& index) const
   }
   
   //! @brief Return the interaction corresponding to the input argument "args" (parameter pack).
   //! @details The temporary object is generated from the input argument "args" and is passed to "FloatType &J(std::vector<Index>&& index) const".
   //! @param args parameter pack
   //! @return The interaction corresponding to "args", i.e., J[args]
   template<typename... Args>
   const FloatType &J(Args... args) const {
      return J(std::vector<Index>{(Index)args...});//Access J(std::vector<Index>&& index) const
   }
   
   //! @brief Return the interactions.
   //! @return The interactions
   const Interactions &GetInteractions() const {
      return J_;
   }
   
   //! @brief Return the vartype
   //! @return The vartype
   const cimod::Vartype &GetVartype() const {
      return vartype_;
   }
   
   //! @brief Change vartype
   //! @param vartype cimod::Vartype
   void ChangeVartype(const cimod::Vartype &vartype) {
      vartype_ = vartype;
   }
   
   //! @brief Return the total energy corresponding to the input variables, Spins or Binaries.
   //! @param spins const Spins& or const Binaries& (both are the same type)
   //! @return The total energy
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
   //! @brief Polynomial interactions as std::unordered_map<std::vector<Index>, FloatType, utility::VectorHash>
   Interactions J_;
   
   //! @brief The model's type. SPIN or BINARY
   cimod::Vartype vartype_ = cimod::Vartype::NONE;
   
};
}
}


#endif /* polynomial_hpp */
