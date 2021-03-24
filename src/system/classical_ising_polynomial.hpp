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
   ClassicalIsingPolynomial(const graph::Spins &init_spins, const graph::Polynomial<FloatType> &init_interaction):
   num_spins(init_interaction.get_num_spins()), spin(init_spins), vartype(init_interaction.GetVartype()) {
      
      //Check if the number of the initial spins/binaries is equal to num_spins defined from the Polynomial graph class.
      assert(init_spins.size() == num_spins);
      
      //Check if the initial spin/binary configurations are valid for vartype defined from the Polynomial graph class.
      CheckVariables();
      
      //Set max_variable_, which is the max index of the std::vector representing the initial spin/binary configurations.
      SetMaxVariable(init_interaction.GetInteractions());
      
      //Receive the interactions from the Polynomial graph class.
      //If the "max_variable_" is larger than or equal to num_spins,
      //which means that the index of the spin/binary variables does not start at zero, relabel interactions.
      //"const_cast" is used here to have access to the interactions through operator[]. Note that "init_interaction" should not be changed.
      auto &&interaction = (GetMaxVariable() < num_spins) ?
      const_cast<Interactions&>(init_interaction.GetInteractions()) :
      RelabelInteractions(const_cast<Interactions&>(init_interaction.GetInteractions()));
      
      //Set "interacted_spins_", which stores the set of spins/binaries corresponding to the interactions.
      SetInteractedSpins(interaction);
      
      //Set "connected_J_term_index_" and "J_term_".
      //If "vartype" is cimod::SPIN, "sign_" is also set, else if "vartype" is cimod::BINARY, "zero_count_" is also set.
      SetJTerm(interaction);
      
      //Set "crs_row", "crs_col", and "crs_val". If "vartype" is cimod::SPIN, "crs_sign_p" is also set,
      //else if "vartype" is cimod::BINARY, "crs_zero_count_p" is also set.
      SetUpdateMatrix(interacted_spins_);
      
      //Set "dE"
      SetdE();
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
      return (zero_count == static_cast<std::size_t>(1 - binary)) ? 1 : 0;
   }
   
   //! @brief Return 0 or 1 in accordance with the input binaries and zero_count
   //! @param binary1 graph::Binary
   //! @param binary2 graph::Binary
   //! @param zero_count std::size_t
   //! @return 1 if zero_count == 2 - binary1 - binary2, otherwise 0
   int ZeroOrOne(graph::Binary binary1, graph::Binary binary2, std::size_t zero_count) {
      return (zero_count == static_cast<std::size_t>(2 - binary1 - binary2)) ? 1 : 0;
   }
   
   //! @brief Set "crs_row", "crs_col", "crs_val", and "crs_sign_p" (spin variable cases) or "crs_zero_count_p" (binary variable cases).
   //! @details These std::vector constitute a sparse matrix (Compressed Row Storage), which is used to update "dE".
   //! @param interacted_spins std::vector<std::vector<graph::Index>>&
   void SetUpdateMatrix(std::vector<std::vector<graph::Index>> &interacted_spins) {
      crs_col.clear();
      crs_row.clear();
      crs_val.clear();
      crs_row.push_back(0);
      if (vartype == cimod::Vartype::SPIN) {
         crs_sign_p.clear();
         for (graph::Index index_spin_row = 0; index_spin_row < num_spins; ++index_spin_row) {
            std::vector<graph::Index> temp_vec;
            std::vector<std::vector<FloatType>> temp_vec_val(num_spins);
            std::vector<std::vector<int8_t*>>  temp_vec_sign(num_spins);
            for (const auto &index_spin_term: connected_J_term_index_[index_spin_row]) {
               auto temp_J_term = J_term_[index_spin_term];
               auto temp_sign_p = &sign_[index_spin_term];
               for (const auto &index_spin_col: interacted_spins[index_spin_term]) {
                  if (index_spin_row != index_spin_col) {
                     temp_vec.push_back(index_spin_col);
                     temp_vec_val[index_spin_col].push_back(4*temp_J_term);
                     temp_vec_sign[index_spin_col].push_back(temp_sign_p);
                  }
               }
            }
            std::sort(temp_vec.begin(), temp_vec.end());
            temp_vec.erase(std::unique(temp_vec.begin(), temp_vec.end()), temp_vec.end());
            for (const auto &it: temp_vec) {
               for (std::size_t i = 0; i < temp_vec_val[it].size(); ++i) {
                  crs_col.push_back(it);
                  crs_val.push_back(temp_vec_val[it][i]);
                  crs_sign_p.push_back(temp_vec_sign[it][i]);
               }
            }
            crs_row.push_back(crs_col.size());
         }
      }
      else if (vartype == cimod::Vartype::BINARY) {
         crs_zero_count_p.clear();
         for (graph::Index index_spin_row = 0; index_spin_row < num_spins; ++index_spin_row) {
            std::vector<graph::Index> temp_vec;
            std::vector<std::vector<FloatType>> temp_vec_val(num_spins);
            std::vector<std::vector<uint64_t*>> temp_vec_zero_count(num_spins);
            for (const auto &index_spin_term: connected_J_term_index_[index_spin_row]) {
               auto temp_J_term       = J_term_[index_spin_term];
               auto temp_zero_count_p = &zero_count_[index_spin_term];
               for (const auto &index_spin_col: interacted_spins[index_spin_term]) {
                  if (index_spin_row != index_spin_col) {
                     temp_vec.push_back(index_spin_col);
                     temp_vec_val[index_spin_col].push_back(temp_J_term);
                     temp_vec_zero_count[index_spin_col].push_back(temp_zero_count_p);
                  }
               }
            }
            std::sort(temp_vec.begin(), temp_vec.end());
            temp_vec.erase(std::unique(temp_vec.begin(), temp_vec.end()), temp_vec.end());
            for (const auto &it: temp_vec) {
               for (std::size_t i = 0; i < temp_vec_val[it].size(); ++i) {
                  crs_col.push_back(it);
                  crs_val.push_back(temp_vec_val[it][i]);
                  crs_zero_count_p.push_back(temp_vec_zero_count[it][i]);
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
   
   //! @brief Update "zero_count_" and "spin".  This function is used only when" vartype" is cimod::Vartype::BINARY
   //! @param index const std::size_t
   inline void UpdateZeroCountAndSpin(const std::size_t index) {
      if (spin[index] == 0) {
         spin[index] = 1;
         for (const auto &index_interaction: connected_J_term_index_[index]) {
            zero_count_[index_interaction]--;
         }
      }
      else {
         spin[index] = 0;
         for (const auto &index_interaction: connected_J_term_index_[index]) {
            zero_count_[index_interaction]++;
         }
      }
   }
   
   //! @brief Update "sign_" and "spin".  This function is used only when" vartype" is cimod::Vartype::SPIN
   //! @param index const std::size_t
   inline void UpdateSignAndSpin(const std::size_t index) {
      spin[index] *= -1;
      for (const auto &index_interaction: connected_J_term_index_[index]) {
         sign_[index_interaction] *= -1;
      }
   }
   
   void ResetSpins(const graph::Spins& init_spin) {
      spin = init_spin;
      CheckVariables();
      SetdE();
   }
   
   //! @brief Return the interactions stored in "J_term_"
   //! @return Interactions
   const std::vector<FloatType> &GetJTerm() const {
      return J_term_;
   }
   
   //! @brief Return the interactes spins/binaries stored in "interacted_spins_"
   //! @return Interacted spins
   const std::vector<std::vector<graph::Index>> &GetInteractedSpins() const {
      return interacted_spins_;
   }
   
   //! @brief Return "connected_J_term_index_"
   //! @return "connected_J_term_index_"
   const std::vector<std::vector<graph::Index>> &GetConnectedJTermIndex() const {
      return connected_J_term_index_;
   }
   
   //! @brief Spin/binary configurations
   graph::Spins spin;
   
   //! @brief The number of spins/binaries
   const graph::Index num_spins;
   
   //! @brief The model's type. cimod::vartype::SPIN or  cimod::vartype::BINARY
   const cimod::Vartype vartype;
   
   //! @brief Store the information about the energy difference when flipping a spin/binary
   std::vector<FloatType> dE;
   
   //! @brief Row of a sparse matrix (Compressed Row Storage) to update "dE".
   std::vector<std::size_t> crs_row;
   
   //! @brief Column of a sparse matrix (Compressed Row Storage) to update "dE".
   std::vector<graph::Index> crs_col;
   
   //! @brief Value of a sparse matrix (Compressed Row Storage) to update "dE".
   std::vector<FloatType> crs_val;
   
   //! @brief Value of a sparse matrix (Compressed Row Storage) to update "dE".
   //! @details Note that this is used only for binary variable cases. This stores the pointers for "sign_", which stores the information about the sign of variables.
   std::vector<int8_t*> crs_sign_p;
   
   //! @brief Value of a sparse matrix (Compressed Row Storage) to update "dE".
   //! @details Note that this is used only for binary variable cases. This stores the pointers for "zero_count_", which stores the information about the number of variables takeing zero.
   std::vector<uint64_t*> crs_zero_count_p;

   
private:
   //! @brief Stores the set of spins/binaries corresponding to the interactions.
   //! @details The keys of "Interactions" (std::unordered_map<std::vector<graph::Index>, FloatType, utility::VectorHash>)
   std::vector<std::vector<graph::Index>> interacted_spins_;
   
   //! @brief Store the values of the interactions.
   //! @details The values of "Interactions" (std::unordered_map<std::vector<graph::Index>, FloatType, utility::VectorHash>)
   std::vector<FloatType> J_term_;
   
   //! @brief Store the information about the indices of "J_term_".
   std::vector<std::vector<std::size_t>> connected_J_term_index_;
   
   //! @brief Store the information about the sign of variables.
   //! @details Note that this is used only for spin variable cases, and the pointers of this std::vector is stored in "crs_sign_p". Do not change this std::vector.
   std::vector<int8_t> sign_;
   
   //! @brief Store the information about the number of variables takeing zero.
   //! @details Note that this is used only for binary variable cases, and the pointers of this std::vector is stored in "crs_zero_count_p". Do not change this std::vector.
   std::vector<uint64_t> zero_count_;
   
   //! @brief The max variable (the max index of spin/binary configulations)
   graph::Index max_variable_ = 0;
   
   //! @brief Set "max_variable_". If an interaction is zero, corresponding variable is ignored.
   //! @param interaction const Interactions&
   void SetMaxVariable(const Interactions &interaction) {
      for (const auto &it: interaction) {
         if (it.first.size() > 0 && std::abs(it.second) > 0.0 && max_variable_ < it.first.back()) {
            max_variable_ = it.first.back();
         }
      }
   }
   
   //! @brief Set "interacted_spins_"
   void SetInteractedSpins(const Interactions &interaction) {
      interacted_spins_.clear();
      interacted_spins_.reserve(interaction.size());
      for (const auto &it: interaction) {
         if (it.first.size() > 0 && std::abs(it.second) > 0.0) {
            interacted_spins_.push_back(it.first);
         }
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
      std::sort(interacted_spins_.begin(), interacted_spins_.end(), comp);
   }
   
   //! @brief Set "connected_J_term_index" and "J_term_". If "vartype" is cimod::SPIN, "sign_" is also set, else If "vartype" is cimod::BINARY, "zero_count_binary_" is also set.
   //! @param interaction Interactions&
   void SetJTerm(Interactions &interaction) {
      connected_J_term_index_.resize(num_spins);
      J_term_.resize(interacted_spins_.size());
      if (vartype == cimod::Vartype::SPIN) {
         sign_.resize(interacted_spins_.size());
         for (std::size_t i = 0; i < interacted_spins_.size(); ++i) {
            graph::Spin temp_spin_multiple = 1;
            if (interacted_spins_[i].size() > 0) {
               for (const auto &index_spin: interacted_spins_[i]) {
                  temp_spin_multiple *= spin[index_spin];
                  connected_J_term_index_[index_spin].push_back(i);
               }
            }
            J_term_[i] = interaction[interacted_spins_[i]];
            sign_[i]   = static_cast<int8_t>(temp_spin_multiple);
         }
      }
      else if (vartype == cimod::Vartype::BINARY) {
         zero_count_.resize(interacted_spins_.size());
         for (std::size_t i = 0; i < interacted_spins_.size(); ++i) {
            uint64_t      temp_zero_count = 0;
            graph::Binary temp_binary_multiple = 1;
            if (interacted_spins_[i].size() > 0) {
               for (const auto &index_binary: interacted_spins_[i]) {
                  temp_binary_multiple *= spin[index_binary];
                  connected_J_term_index_[index_binary].push_back(i);
                  if (spin[index_binary] == 0) {
                     temp_zero_count++;
                  }
               }
            }
            J_term_[i]     = interaction[interacted_spins_[i]];
            zero_count_[i] = temp_zero_count;
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
   
   //! @brief Set delta E (dE), which is used to determine whether to flip the spin/binary or not.
   void SetdE() {
      dE.resize(num_spins);
      if (vartype == cimod::Vartype::SPIN) {
//#pragma omp parallel for //Maybe OK but afraid so comment out for now
         for (graph::Index i = 0; i < num_spins; ++i) {
            FloatType temp_energy = 0.0;
            for (const auto &it: connected_J_term_index_[i]) {
               temp_energy += J_term_[it]*sign_[it];
            }
            dE[i] = -2*temp_energy;
         }
      }
      else if (vartype == cimod::Vartype::BINARY) {
//#pragma omp parallel for //Maybe OK but afraid so comment out for now
         for (graph::Index i = 0; i < num_spins; ++i) {
            FloatType temp_energy = 0.0;
            auto temp_spin = spin[i];
            for (const auto &it: connected_J_term_index_[i]) {
               temp_energy += J_term_[it]*Sign(temp_spin)*ZeroOrOne(temp_spin, zero_count_[it]);
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
   
};

//! @brief Helper function for ClassicalIsingPolynomial constructor
//! @tparam GraphType
//! @param init_spin const graph::Spins&. The initial spins/binaries.
//! @param init_interaction GraphType&. The initial interactions.
template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins &init_spin, const GraphType &init_interaction) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction);
}



} //namespace system
} //namespace openjij



#endif /* classical_ising_polynomial_hpp */
