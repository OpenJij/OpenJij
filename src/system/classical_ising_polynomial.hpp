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

template<typename GraphType>
class ClassicalIsingPolynomial;

template<typename FloatType>
class ClassicalIsingPolynomial<graph::Polynomial<FloatType>> {
   
public:
   
   using system_type = classical_system;

   ClassicalIsingPolynomial(const graph::Spins &initial_spins,
                            const graph::Polynomial<FloatType> &poly_graph): num_spins(poly_graph.size()), vartype_(poly_graph.get_vartype()), spin(initial_spins) {
            
      if (poly_graph.get_max_variable() != num_spins - 1) {
         std::vector<graph::Index> sorted_variables;
         for (const auto &it: poly_graph.get_keys()) {
            for (const auto &index: it) {
               sorted_variables.push_back(index);
            }
         }
         std::sort(sorted_variables.begin(), sorted_variables.end());
         sorted_variables.erase(std::unique(sorted_variables.begin(), sorted_variables.end()), sorted_variables.end());
         poly_key_list_   = RelabelKeys(poly_graph.get_keys(), sorted_variables);
         poly_value_list_ = poly_graph.get_values();
      }
      else {
         poly_key_list_   = poly_graph.get_keys();
         poly_value_list_ = poly_graph.get_values();
      }
      RemoveZeroValues(poly_key_list_, poly_value_list_);
      num_interactions_ = poly_key_list_.size();
      CheckInitialConditions();
      SetPolynomialIndex();
      SetUpdateMatrix();
      reset_dE();
   }
   
   //! @brief Return vartype
   //! @return vartype
   cimod::Vartype get_vartype() const {
      return vartype_;
   }
   
   //! @brief Return "max_dE"
   //! @return "max_dE"
   FloatType get_max_dE() const {
      return max_dE_;
   }
   
   //! @brief Return "min_dE"
   //! @return "min_dE"
   FloatType get_min_dE() const {
      return min_dE_;
   }
   
   const cimod::PolynomialKeyList<graph::Index> &get_keys() const {
      return poly_key_list_;
   }
   
   const cimod::PolynomialValueList<FloatType> &get_values() const {
      return poly_value_list_;
   }
   
   const std::vector<std::vector<std::size_t>> &get_connected_J_term_index() const {
      return connected_J_term_index_;
   }
   
   const std::vector<std::size_t> &get_crs_row() const {
      return crs_row_;
   }
   
   const std::vector<graph::Index> &get_crs_col() const {
      return crs_col_;
   }
   
   const std::vector<FloatType> &get_crs_val() const {
      return crs_val_;
   }
   
   const std::vector<int8_t*> &get_crs_sign_p() const {
      return crs_sign_p_;
   }
   
   const std::vector<std::size_t*> &get_crs_zero_count_p() const {
      return crs_zero_count_p_;
   }
   
   
   void reset_dE() {
      dE.resize(num_spins);
      max_dE_ = std::abs(poly_value_list_[0]);//Initialize
      min_dE_ = std::abs(poly_value_list_[0]);//Initialize
      if (vartype_ == cimod::Vartype::SPIN) {
         for (graph::Index i = 0; i < num_spins; ++i) {
            FloatType temp_energy = 0.0;
            FloatType temp_abs    = 0.0;
            bool flag = false;
            for (const auto &it: connected_J_term_index_[i]) {
               temp_energy += poly_value_list_[it]*spin_sign_poly_[it];
               temp_abs    += std::abs(poly_value_list_[it]);
            }
            dE[i] = -2*temp_energy;
            if (flag && max_dE_ < 2*temp_abs) {
               max_dE_ = 2*temp_abs;
            }
            if (flag && min_dE_ > 2*temp_abs) {
               min_dE_ = 2*temp_abs;
            }
         }
      }
      else if (vartype_ == cimod::Vartype::BINARY) {
         for (graph::Index i = 0; i < num_spins; ++i) {
            FloatType temp_energy = 0.0;
            FloatType temp_abs    = 0.0;
            bool flag = false;
            auto temp_spin = spin[i];
            for (const auto &it: connected_J_term_index_[i]) {
               temp_energy += poly_value_list_[it]*Sign(temp_spin)*ZeroOrOne(temp_spin, binary_zero_count_poly_[it]);
               temp_abs    += std::abs(poly_value_list_[it]);
               flag = true;
            }
            dE[i] = temp_energy;
            if (flag && max_dE_ < temp_abs) {
               max_dE_ = temp_abs;
            }
            if (flag && min_dE_ > temp_abs) {
               min_dE_ = temp_abs;
            }
         }
      }
      else {
         std::stringstream ss;
         ss << "Unknown vartype detected in " << __func__ << std::endl;
         throw std::runtime_error(ss.str());
      }
   }
   
   //! @brief Update "zero_count_" and "spin".  This function is used only when" vartype" is cimod::Vartype::BINARY
   //! @param index const std::size_t
   inline void update_zero_count_and_binary(const std::size_t index) {
      if (spin[index] == 0) {
         spin[index] = 1;
         for (const auto &index_interaction: connected_J_term_index_[index]) {
            binary_zero_count_poly_[index_interaction]--;
         }
      }
      else {
         spin[index] = 0;
         for (const auto &index_interaction: connected_J_term_index_[index]) {
            binary_zero_count_poly_[index_interaction]++;
         }
      }
   }
   
   //! @brief Update "sign_" and "spin".  This function is used only when" vartype" is cimod::Vartype::SPIN
   //! @param index const std::size_t
   inline void update_sign_and_spin(const std::size_t index) {
      spin[index] *= -1;
      for (const auto &index_interaction: connected_J_term_index_[index]) {
         spin_sign_poly_[index_interaction] *= -1;
      }
   }
   
   void update_dE_for_spin(std::size_t index) {
      dE[index] *= -1;
      const std::size_t begin = crs_row_[index];
      const std::size_t end   = crs_row_[index + 1];
      for (std::size_t i = begin; i < end; ++i) {
         dE[crs_col_[i]] += crs_val_[i]*(*crs_sign_p_[i]);
      }
   }
   
   void update_dE_for_binary(std::size_t index) {
      dE[index] *= -1;
      const std::size_t begin = crs_row_[index];
      const std::size_t end   = crs_row_[index + 1];
      for (std::size_t i = begin; i < end; ++i) {
         graph::Index col = crs_col_[i];
         dE[col] += Sign(spin[col] + spin[index])*(crs_val_[i])*ZeroOrOne(spin[index], spin[col], *crs_zero_count_p_[i]);
      }
   }
   
   const std::size_t num_spins;
   std::vector<FloatType> dE;
   graph::Spins spin;
      
private:
   const cimod::Vartype vartype_;
   std::size_t num_interactions_;
   std::vector<std::size_t>  crs_row_;
   std::vector<graph::Index> crs_col_;
   std::vector<FloatType>    crs_val_;
   std::vector<int8_t*>      crs_sign_p_;
   std::vector<std::size_t*> crs_zero_count_p_;
   cimod::PolynomialKeyList<graph::Index> poly_key_list_;
   cimod::PolynomialValueList<FloatType>  poly_value_list_;
   std::vector<std::vector<std::size_t>>        connected_J_term_index_;
   std::vector<int8_t>      spin_sign_poly_;
   std::vector<std::size_t> binary_zero_count_poly_;
   FloatType max_dE_;
   FloatType min_dE_;
   
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
   
   
   
   void SetUpdateMatrix() {
      crs_col_.clear();
      crs_row_.clear();
      crs_val_.clear();
      crs_row_.push_back(0);
      if (vartype_ == cimod::Vartype::SPIN) {
         crs_sign_p_.clear();
         for (std::size_t row = 0; row < num_spins; ++row) {
            std::vector<graph::Index> temp_vec;
            std::vector<std::vector<FloatType>> temp_vec_val (num_spins);
            std::vector<std::vector<int8_t*>>   temp_vec_sign(num_spins);
            for (const auto &index: connected_J_term_index_[row]) {
               auto temp_J_term = poly_value_list_[index];
               auto temp_sign_p = &spin_sign_poly_[index];
               for (const auto &col: poly_key_list_[index]) {
                  if (row != col) {
                     temp_vec.push_back(col);
                     temp_vec_val [col].push_back(4*temp_J_term);
                     temp_vec_sign[col].push_back(temp_sign_p);
                  }
               }
            }
            std::sort(temp_vec.begin(), temp_vec.end());
            temp_vec.erase(std::unique(temp_vec.begin(), temp_vec.end()), temp_vec.end());
            for (const auto &it: temp_vec) {
               for (std::size_t i = 0; i < temp_vec_val[it].size(); ++i) {
                  crs_col_.push_back(it);
                  crs_val_.push_back(temp_vec_val[it][i]);
                  crs_sign_p_.push_back(temp_vec_sign[it][i]);
               }
            }
            crs_row_.push_back(crs_col_.size());
         }
      }
      else if (vartype_ == cimod::Vartype::BINARY) {
         crs_zero_count_p_.clear();
         for (std::size_t row = 0; row < num_spins; ++row) {
            std::vector<graph::Index> temp_vec;
            std::vector<std::vector<FloatType>>    temp_vec_val(num_spins);
            std::vector<std::vector<std::size_t*>> temp_vec_zero_count(num_spins);
            for (const auto &index: connected_J_term_index_[row]) {
               auto temp_J_term       = poly_value_list_[index];
               auto temp_zero_count_p = &binary_zero_count_poly_[index];
               for (const auto &col: poly_key_list_[index]) {
                  if (row != col) {
                     temp_vec.push_back(col);
                     temp_vec_val[col].push_back(temp_J_term);
                     temp_vec_zero_count[col].push_back(temp_zero_count_p);
                  }
               }
            }
            std::sort(temp_vec.begin(), temp_vec.end());
            temp_vec.erase(std::unique(temp_vec.begin(), temp_vec.end()), temp_vec.end());
            for (const auto &it: temp_vec) {
               for (std::size_t i = 0; i < temp_vec_val[it].size(); ++i) {
                  crs_col_.push_back(it);
                  crs_val_.push_back(temp_vec_val[it][i]);
                  crs_zero_count_p_.push_back(temp_vec_zero_count[it][i]);
               }
            }
            crs_row_.push_back(crs_col_.size());
         }
      }
      else {
         throw std::runtime_error("Unknown vartype detected");
      }
   }
   
   void SetPolynomialIndex() {
      connected_J_term_index_.resize(num_spins);
      if (vartype_ == cimod::Vartype::SPIN) {
         spin_sign_poly_.resize(num_interactions_);
         for (std::size_t i = 0; i < num_interactions_; ++i) {
            graph::Spin spin_multiple = 1;
            for (const auto &it: poly_key_list_[i]) {
               spin_multiple *= spin[it];
               connected_J_term_index_[it].push_back(i);
            }
            spin_sign_poly_[i] = static_cast<int8_t>(spin_multiple);
         }
      }
      else if (vartype_ == cimod::Vartype::BINARY) {
         binary_zero_count_poly_.resize(num_interactions_);
         for (std::size_t i = 0; i < num_interactions_; ++i) {
            std::size_t   zero_count = 0;
            graph::Binary binary_multiple = 1;
            for (const auto &it: poly_key_list_[i]) {
               binary_multiple *= spin[it];
               connected_J_term_index_[it].push_back(i);
               if (spin[it] == 0) {
                  zero_count++;
               }
            }
            binary_zero_count_poly_[i] = zero_count;
         }
      }
      else {
         throw std::runtime_error("Unknown vartype detected");
      }
   }
   
   void RemoveZeroValues(cimod::PolynomialKeyList<graph::Index> &poly_key_list, cimod::PolynomialValueList<FloatType> &poly_value_list) {
      std::vector<std::size_t> index_to_be_removed;
      for (std::size_t i = 0; i < poly_key_list.size(); ++i) {
         if (poly_value_list[i] == 0.0) {
            index_to_be_removed.push_back(i);
         }
      }
      
      for (const auto &it: index_to_be_removed) {
         std::swap(poly_key_list[it], poly_key_list.back());
         poly_key_list.pop_back();
         std::swap(poly_value_list[it], poly_value_list.back());
         poly_value_list.pop_back();
      }
      
   }
   
   void CheckInitialConditions() const {
      if (spin.size() != num_spins) {
         throw std::runtime_error("The number of variables is not equal to the size of the initial spin");
      }
      if (poly_key_list_.size() != poly_value_list_.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      if (poly_key_list_.size() <= 0) {
         throw std::runtime_error("The number of interactions is 0");
      }
      if (vartype_ == cimod::Vartype::SPIN) {
         for (std::size_t i = 0; i < spin.size(); ++i) {
            if (spin[i] != -1 && spin[i] != 1) {
               std::stringstream ss;
               ss << "The variable at " << i << " is " << spin[i] << ".\n";
               ss << "But the spin variable must be -1 or +1.\n";
               throw std::runtime_error(ss.str());
            }
         }
      }
      else if (vartype_ == cimod::Vartype::BINARY) {
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
         throw std::runtime_error("Unknown vartype detected");
      }
   }
   
   cimod::PolynomialKeyList<graph::Index> RelabelKeys(const cimod::PolynomialKeyList<graph::Index> &poly_key_list, const std::vector<graph::Index> &sorted_variables) {
      std::unordered_map<graph::Index, graph::Index> relabeld_variables;
      for (std::size_t i = 0; i < sorted_variables.size(); ++i) {
         relabeld_variables[sorted_variables[i]] = i;
      }
      
      cimod::PolynomialKeyList<graph::Index> relabeld_poly_key_list(poly_key_list.size());
      
#pragma omp parallel for
      for (std::size_t i = 0; i < poly_key_list.size(); ++i) {
         std::vector<graph::Index> temp;
         for (const auto &it: poly_key_list[i]) {
            temp.push_back(relabeld_variables[it]);
         }
         relabeld_poly_key_list[i] = temp;
      }
      return relabeld_poly_key_list;
   }
   
};

//! @brief Helper function for ClassicalIsingPolynomial constructor
//! @tparam GraphType
//! @param init_spin const graph::Spins&. The initial spin/binaries.
//! @param init_interaction GraphType&. The initial interactions.
template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins &init_spin, const GraphType &init_interaction) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction);
}


} //namespace system
} //namespace openjij



#endif /* classical_ising_polynomial_hpp */
