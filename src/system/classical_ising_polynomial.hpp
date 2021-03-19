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

#ifndef classical_ising_polynomial_hpp
#define classical_ising_polynomial_hpp

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <graph/all.hpp>

namespace openjij {
namespace system {

//! @brief ClassicalIsingPolynomial structure, which is a system for classical Ising models with poynomial interactions and polynomial unconstrained binary optimization (PUBO) models
//! @tparam GraphType type of graph
template<typename GraphType>
struct ClassicalIsingPolynomial;

//! @brief ClassicalIsingPolynomial structure
template<typename FloatType>
struct ClassicalIsingPolynomial<graph::Polynomial<FloatType>> {
   
   //! @brief Interaction type, which is the same as the one defined in graph/polynomial.hpp
   using Interactions = std::unordered_map<std::vector<graph::Index>, FloatType, utility::VectorHash>;
   
   //! @brief system type
   using system_type  = classical_system;
   
   //! @brief Constructor of ClassicalIsingPolynomial
   //! @param init_spins graph::Spins&. The initial spin/binary configurations.
   //! @param init_interaction graph::Polynomial<FloatType>& (Polynomial graph class). The initial interacrtions.
   ClassicalIsingPolynomial(const graph::Spins &init_spins, graph::Polynomial<FloatType> &init_interaction):
   num_spins(init_interaction.get_num_spins()), spin(init_spins), vartype(init_interaction.GetVartype()) {
      
      //Check if the number of the initial spins/binaries is equal to num_spins defined from the Polynomial graph class.
      assert(init_spins.size() == num_spins);
      
      //Check if the initial spin/binary configurations are valid for vartype defined from the Polynomial graph class.
      CheckVariables();
      
      //Set max_variable_, which is the max index of the std::vector representing the initial spin/binary configurations.
      SetMaxVariable(init_interaction.GetInteractions());
      
      //Receive the interactions by rvalue references from the Polynomial graph class.
      //If the "max_variable_" is larger than or equal to num_spins,
      //which means that the index of the spin/binary variables does not start at zero, relabel interactions.
      //"const_cast" is used here to have access to the interactions through operator[]. Note that "init_interaction" should not be changed.
      auto &&interaction = (GetMaxVariable() < num_spins) ?
      const_cast<Interactions&>(init_interaction.GetInteractions()) :
      RelabelInteractions(const_cast<Interactions&>(init_interaction.GetInteractions()));
      
      //"interacted_spins" stores the set of spins/binaries corresponding to the interactions,
      //i.e., the keys of "Interactions" (std::unordered_map<std::vector<graph::Index>, FloatType, utility::VectorHash>)
      std::vector<std::vector<graph::Index>> interacted_spins;
      interacted_spins.reserve(interaction.size());
      for (const auto &it: interaction) {
         interacted_spins.push_back(it.first);
      }
      
      //nameless function to sort "interacted_spins"
      auto comp = [](const auto &lhs, const auto &rhs) {
         if (lhs.size() != rhs.size()) {
            return lhs.size() < rhs.size();
         }
         else {
            if (lhs[0] != rhs[0]) {
               return lhs[0] < rhs[0];
            }
            else {
               return lhs[1] < rhs[1];
            }
         }
      };
      
      //Sort "interacted_spins" for continuous memory access
      std::sort(interacted_spins.begin(), interacted_spins.end(), comp);
      
      //Set "connected_J_term_index" and "J_term_". If "vartype" is cimod::SPIN, "zero_count_binary_" is also set.
      SetJTerm(interaction, interacted_spins);
      
      //Set "crs_row" and "crs_col". If "vartype" is cimod::SPIN, "crs_val_p_spin" is also set,
      //else if "vartype" is cimod::BINARY, "crs_val_binary" and "zero_count_p_binary" are also set.
      SetUpdateMatrix(interacted_spins);
      
      //Set "dE"
      SetdE();
   }
   
   //! @brief Set "max_variable_". If an interaction is zero, corresponding variable is ignored.
   //! @param interaction const Interactions&
   void SetMaxVariable(const Interactions &interaction) {
      for (const auto &it: interaction) {
         if (it.first.size() != 0 && std::abs(it.second) > 0.0 && max_variable_ < it.first.back()) {
            max_variable_ = it.first.back();
         }
      }
   }
   
   //! @brief Return "max_variable_"
   //! @return The max variable (the max index of spins/binaries)
   graph::Index GetMaxVariable() const {
      return max_variable_;
   }
   
   //! @brief Check if spin/binary configurations are valid for "vartype"
   void CheckVariables() {
      if (vartype == cimod::Vartype::SPIN) {
         for (std::size_t i = 0; i < spin.size(); ++i) {
            if (spin[i] != -1 && spin[i] != 1) {
               std::stringstream ss;
               ss << "The variable at " << i << " is " << spin[i] << ".\n";
               ss << "But the spin variable must be -1 or +1.\n";
               throw std::runtime_error(ss.str());
            }
         }
      }
      else if (vartype == cimod::Vartype::BINARY) {
         for (std::size_t i = 0; i < spin.size(); ++i) {
            if (spin[i] != 0 && spin[i] != 1) {
               std::stringstream ss;
               ss << "The variable at " << i << " is " << spin[i] << ".\n";
               ss << "But the binary variable must be 0 or 1.\n";
               throw std::runtime_error(ss.str());
            }
         }
      }
      else {
         std::stringstream ss;
         ss << "Unknown vartype detected in " << __func__ << std::endl;
         throw std::runtime_error(ss.str());
      }
   }
   
   //! @brief Set delta E (dE), which is used to determine whether to flip the spin/binary or not.
   void SetdE() {
      dE.resize(num_spins);
      if (vartype == cimod::Vartype::SPIN) {
//#pragma omp parallel for //Maybe OK but afraid so comment out for now
         for (graph::Index i = 0; i < num_spins; ++i) {
            FloatType temp_energy = 0.0;
            for (const auto &it: connected_J_term_index[i]) {
               temp_energy += J_term_[it];
            }
            dE[i] = -2*temp_energy;
         }
      }
      else if (vartype == cimod::Vartype::BINARY) {
//#pragma omp parallel for //Maybe OK but afraid so comment out for now
         for (graph::Index i = 0; i < num_spins; ++i) {
            FloatType temp_energy = 0.0;
            auto temp_spin = spin[i];
            for (const auto &it: connected_J_term_index[i]) {
               temp_energy += J_term_[it]*Sign(temp_spin)*ZeroOrOne(temp_spin, zero_count_binary_[it]);
            }
            dE[i] = temp_energy;
         }
      }
      else {
         std::stringstream ss;
         ss << "Unknown vartype detected in " << __func__ << std::endl;
         throw std::runtime_error(ss.str());
      }
   }
   
   //! @brief Relabel interactions
   //! @details For example, if the interaction is only J[2,3,7] = -1.0 for 3 sites system, this function relabel the interaction as J[0,1,2] = -1.0
   //! @param input_interactions Interactions&
   //! @return Relabeld interactions
   Interactions RelabelInteractions(Interactions &input_interactions) {
      //Extract variables from the keys of Interactions (std::unordered_map<std::vector<graph::Index>, FloatType, utility::VectorHash>)
      std::unordered_set<graph::Index> variable_set;
      for (const auto &it: input_interactions) {
         for (const auto &index: it.first) {
            variable_set.emplace(index);
         }
      }
      
      //Convert std::unordered_set to std::vector
      std::vector<graph::Index> variables(variable_set.begin(), variable_set.end());
      
      //Sort extracted variables
      std::sort(variables.begin(), variables.end());
      
      //Relabel interactions
      std::unordered_map<graph::Index, graph::Index> relabeld_variables;
      for (std::size_t i = 0; i < variables.size(); ++i) {
         relabeld_variables[variables[i]] = i;
      }
      
      Interactions relabeled_interactions;
      
      for (const auto &it: input_interactions) {
         std::vector<graph::Index> temp_vec(it.first.size());
         for (std::size_t i = 0; i < it.first.size(); ++i) {
            temp_vec[i] = relabeld_variables[it.first[i]];
         }
         relabeled_interactions[temp_vec] = it.second;
      }
      return relabeled_interactions;
   }
   
   //! @brief Return -1 or +1 in accordance with the input binary
   //! @param binary graph::Binary
   //! @return -1 if binary is odd number, otherwise +1
   int Sign(graph::Binary binary) {
      return (binary%2 == 0) ? 1 : -1;
   }
   
   //! @brief Return 0 or 1 in accordance with the input binary and zero_count
   //! @param binary graph::Binary
   //! @param zero_count std::size_t
   //! @return 1 if zero_count == 1 - binary, otherwise 0
   int ZeroOrOne(graph::Binary binary, std::size_t zero_count) {
      return (zero_count == 1 - binary) ? 1 : 0;
   }
   
   //! @brief Return 0 or 1 in accordance with the input binaries and zero_count
   //! @param binary1 graph::Binary
   //! @param binary2 graph::Binary
   //! @param zero_count std::size_t
   //! @return 1 if zero_count == 2 - binary1 - binary2, otherwise 0
   int ZeroOrOne(graph::Binary binary1, graph::Binary binary2, std::size_t zero_count) {
      return (zero_count == 2 - binary1 - binary2) ? 1 : 0;
   }
   
   //! @brief Set "connected_J_term_index" and "J_term_". If "vartype" is cimod::BINARY, "zero_count_binary_" is also set.
   //! @param interaction Interactions&
   //! @param interacted_spins std::vector<std::vector<graph::Index>>&
   void SetJTerm(Interactions &interaction, std::vector<std::vector<graph::Index>> &interacted_spins) {
      connected_J_term_index.resize(num_spins);
      J_term_.resize(interacted_spins.size());
      if (vartype == cimod::Vartype::SPIN) {
         for (std::size_t i = 0; i < interacted_spins.size(); ++i) {
            graph::Spin temp_spin_multiple = 1;
            for (const auto &index_spin: interacted_spins[i]) {
               temp_spin_multiple *= spin[index_spin];
               connected_J_term_index[index_spin].push_back(i);
            }
            J_term_[i] = interaction[interacted_spins[i]]*temp_spin_multiple;
         }
      }
      else if (vartype == cimod::Vartype::BINARY) {
         zero_count_binary_.resize(interacted_spins.size());
         for (std::size_t i = 0; i < interacted_spins.size(); ++i) {
            std::size_t temp_zero_count = 0;
            uint32_t    temp_spin_multiple = 1;
            for (const auto &index_spin: interacted_spins[i]) {
               temp_spin_multiple *= spin[index_spin];
               connected_J_term_index[index_spin].push_back(i);
               if (spin[index_spin] == 0) {
                  temp_zero_count++;
               }
            }
            J_term_[i] = interaction[interacted_spins[i]];
            zero_count_binary_[i] = temp_zero_count;
         }
      }
      else {
         std::stringstream ss;
         ss << "Unknown vartype detected in " << __func__ << std::endl;
         throw std::runtime_error(ss.str());
      }
   }
   
   //! @brief Set "crs_row" and "crs_col". If "vartype" is cimod::SPIN, "crs_val_p_spin" is also set, else if "vartype" is cimod::BINARY, "crs_val_binary" and "zero_count_p_binary" are also set.
   //! @details These std::vector constitute a sparse matrix (Compressed Row Storage), which is used to update "dE".
   //! @param interacted_spins std::vector<std::vector<graph::Index>>&
   void SetUpdateMatrix(std::vector<std::vector<graph::Index>> &interacted_spins) {
      crs_col.clear();
      crs_row.clear();
      crs_row.push_back(0);
      if (vartype == cimod::Vartype::SPIN) {
         crs_val_p_spin.clear();
         for (graph::Index index_spin_row = 0; index_spin_row < num_spins; ++index_spin_row) {
            std::unordered_set<graph::Index>     temp_set;
            std::vector<std::vector<FloatType*>> temp_val(num_spins);
            for (const auto &index_spin_term: connected_J_term_index[index_spin_row]) {
               for (const auto &index_spin_col: interacted_spins[index_spin_term]) {
                  if (index_spin_row != index_spin_col) {
                     temp_set.emplace(index_spin_col);
                     temp_val[index_spin_col].push_back(&J_term_[index_spin_term]);
                  }
               }
            }
            std::vector<std::size_t> temp_vec(temp_set.begin(), temp_set.end());
            std::sort(temp_vec.begin(), temp_vec.end());
            for (const auto &it: temp_vec) {
               for (const auto &it2: temp_val[it]) {
                  crs_col.push_back(it);
                  crs_val_p_spin.push_back(it2);
               }
            }
            crs_row.push_back(crs_col.size());
         }
      }
      else if (vartype == cimod::Vartype::BINARY) {
         crs_val_binary.clear();
         zero_count_p_binary.clear();
         for (graph::Index index_spin_row = 0; index_spin_row < num_spins; ++index_spin_row) {
            std::unordered_set<graph::Index>    temp_set;
            std::vector<std::vector<FloatType>> temp_val(num_spins);
            std::vector<std::vector<std::size_t*>> temp_zero(num_spins);
            for (const auto &index_spin_term: connected_J_term_index[index_spin_row]) {
               for (const auto &index_spin_col: interacted_spins[index_spin_term]) {
                  if (index_spin_row != index_spin_col) {
                     temp_set.emplace(index_spin_col);
                     temp_val[index_spin_col].push_back(J_term_[index_spin_term]);
                     temp_zero[index_spin_col].push_back(&zero_count_binary_[index_spin_term]);
                  }
               }
            }
            std::vector<std::size_t> temp_vec(temp_set.begin(), temp_set.end());
            std::sort(temp_vec.begin(), temp_vec.end());
            for (const auto &it: temp_vec) {
               for (std::size_t i = 0; i < temp_val[it].size(); ++i) {
                  crs_col.push_back(it);
                  crs_val_binary.push_back(temp_val[it][i]);
                  zero_count_p_binary.push_back(temp_zero[it][i]);
               }
            }
            crs_row.push_back(crs_col.size());
         }
      }
      else {
         std::stringstream ss;
         ss << "Unknown vartype detected in " << __func__ << std::endl;
         throw std::runtime_error(ss.str());
      }
   }
   
   //! @brief Return vartype
   //! @return vartype
   const cimod::Vartype &GetVartype() const {
      return vartype;
   }
   
   //! @brief Flip J_term_. This function is used only when" vartype" is cimod::Vartype::SPIN
   //! @param index const std::size_t
   void FlipJTerm(const std::size_t index) {
      J_term_[index] *= -1;
   }
   
   //! @brief Update "zero_count_binary_" and "spin".  This function is used only when" vartype" is cimod::Vartype::BINARY
   //! @param index const std::size_t
   void UpdateZeroCountBinaryAndSpin(const std::size_t index) {
      if (spin[index] == 0) {
         spin[index] = 1;
         for (const auto &index_interaction: connected_J_term_index[index]) {
            zero_count_binary_[index_interaction]--;
         }
      }
      else {
         spin[index] = 0;
         for (const auto &index_interaction: connected_J_term_index[index]) {
            zero_count_binary_[index_interaction]++;
         }
      }
   }
   
   //! @brief Return the interactions stored in "J_term_"
   //! @return Interactions
   const std::vector<FloatType> &GetJTerm() const {
      return J_term_;
   }
   
   //! @brief Return "zero_count_binary_", which stores the information about the number of the variables taking zero
   //! @return zero_count_binary_
   const std::vector<std::size_t> &GetZeroCountBinary() const {
      return zero_count_binary_;
   }
   
   //! @brief Spin/binary configurations
   graph::Spins spin;
   
   //! @brief The number of spins/binaries
   const graph::Index num_spins;
   
   //! @brief The model's type. cimod::vartype::SPIN or  cimod::vartype::BINARY
   const cimod::Vartype vartype;
   
   //! @brief Store the information about the indices of "J_term_" and "zero_count_binary_" in accordance with the index of spin/binary
   std::vector<std::vector<std::size_t>> connected_J_term_index;
   
   //! @brief Store the information about the energy difference when flipping a spin/binary
   std::vector<FloatType>    dE;
   
   //! @brief Row of a sparse matrix (Compressed Row Storage) to update "dE".
   std::vector<std::size_t>  crs_row;
   
   //! @brief Column of a sparse matrix (Compressed Row Storage) to update "dE".
   std::vector<graph::Index> crs_col;
   
   //! @brief Value of a sparse matrix (Compressed Row Storage) to update "dE".
   //! @details This is used only for spin variables, and stores the pointers for "J_term".
   std::vector<FloatType*>   crs_val_p_spin;
   
   //! @brief Value of a sparse matrix (Compressed Row Storage) to update "dE".
   //! @details This is used only for binary variables.
   std::vector<FloatType>    crs_val_binary;
   
   //! @brief Value of a sparse matrix (Compressed Row Storage) to update "dE".
   //! @details This is used only for binary variables and stores the pointers for "zero_count_binary_", which stores the information about the number of variables takeing zero.
   std::vector<std::size_t*> zero_count_p_binary;//used only for binary variables
   
private:
   
   //! @brief The value of the interactions.
   //! @details Note that the pointers of this std::vector is stored in "crs_val_p_spin" or "crs_val_binary". Do not change "J_term_".
   std::vector<FloatType> J_term_;
   
   //! @brief Store the information about the number of variables takeing zero.
   //! @details This is used only for binary variables. Note that the pointers of this std::vector is stored in "zero_count_p_binary". Do not change "zero_count_binary_".
   std::vector<std::size_t> zero_count_binary_;
   
   //! @brief The max variable (the max index of spin/binary configulations)
   graph::Index max_variable_ = 0;
};

//! @brief Helper function for ClassicalIsingPolynomial constructor
//! @tparam GraphType
//! @param init_spin const graph::Spins&. The initial spins/binaries.
//! @param init_interaction GraphType&. The initial interactions.
template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins &init_spin, GraphType &init_interaction) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction);
}



} //namespace system
} //namespace openjij



#endif /* classical_ising_polynomial_hpp */
