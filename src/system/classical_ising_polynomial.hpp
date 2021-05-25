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
#include <graph/all.hpp>
#include <nlohmann/json.hpp>
#include <graph/json/parse.hpp>

namespace openjij {
namespace system {

//! @brief ClassicalIsingPolynomial structure, which is a system for classical Ising models with poynomial interactions and higher ordere unconstrained binary optimization (HUBO) models
//! @tparam GraphType type of graph
template<typename GraphType>
class ClassicalIsingPolynomial;

//! @brief ClassicalIsingPolynomial class
template<typename FloatType>
class ClassicalIsingPolynomial<graph::Polynomial<FloatType>> {
   
public:
   
   //! @brief system type
   using system_type = classical_system;

   //! @brief Constructor of ClassicalIsingPolynomial
   //! @param init_spins graph::Spins&. The initial spin/binary configurations.
   //! @param poly_graph graph::Polynomial<FloatType>& (Polynomial graph class). The initial interacrtions.
   ClassicalIsingPolynomial(const graph::Spins &initial_spins, const graph::Polynomial<FloatType> &poly_graph): num_spins(poly_graph.size()), vartype_(poly_graph.get_vartype()), spin(initial_spins) {
      SetPolyKeysAndValues(poly_graph.get_keys(), poly_graph.get_values(), (poly_graph.get_max_variable() != num_spins - 1));
      CheckInitialConditions();
      SetPolynomialIndex();
      SetUpdateMatrix();
      reset_dE();
   }
   
   //! @brief Constructor of ClassicalIsingPolynomial
   //! @param init_spins graph::Spins&. The initial spin/binary configurations.
   //! @param j const nlohmann::json object
   ClassicalIsingPolynomial(const graph::Spins &initial_spins, const nlohmann::json &j):num_spins(initial_spins.size()), spin(initial_spins), vartype_(j.at("vartype") == "SPIN" ? cimod::Vartype::SPIN : cimod::Vartype::BINARY) {
      const auto &v_k_v = graph::json_parse_polynomial<FloatType>(j);
      const auto &poly_key_list   = std::get<1>(v_k_v);
      const auto &poly_value_list = std::get<2>(v_k_v);
      
      if (j.at("vartype") != "SPIN" && j.at("vartype") != "BINARY" ) {
         throw std::runtime_error("Unknown vartype detected");
      }
      
      if (poly_key_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      
      poly_key_list_.resize(poly_key_list.size());
      poly_value_list_.resize(poly_value_list.size());
      
#pragma omp parallel for
      for (int64_t i = 0; i < (int64_t)poly_key_list.size(); ++i) {
         poly_key_list_[i]   = poly_key_list[i];
         poly_value_list_[i] = poly_value_list[i];
      }
      num_interactions_ = poly_key_list_.size();
      CheckInitialConditions();
      SetPolynomialIndex();
      SetUpdateMatrix();
      reset_dE();
   }
   
   //! @brief Constructor of ClassicalIsingPolynomial
   //! @param init_spins graph::Spins&. The initial spin/binary configurations.
   //! @param bpm const cimod::BinaryPolynomialModel<graph::Index, FloatType> object
   ClassicalIsingPolynomial(const graph::Spins &initial_spins, const cimod::BinaryPolynomialModel<graph::Index, FloatType> &bpm): num_spins(bpm.GetVariables().size()), vartype_(bpm.get_vartype()), spin(initial_spins) {
      poly_key_list_.resize(bpm._get_keys().size());
      poly_value_list_.resize(bpm._get_values().size());
      
#pragma omp parallel for
      for (int64_t i = 0; i < (int64_t)bpm._get_keys().size(); ++i) {
         poly_key_list_[i]   = bpm._get_keys()[i];
         poly_value_list_[i] = bpm._get_values()[i];
      }
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
   
   //! @brief Get the PolynomialKeyList object.
   //! @return PolynomialKeyList object as std::vector<std::vector>>.
   const cimod::PolynomialKeyList<graph::Index> &get_keys() const {
      return poly_key_list_;
   }
   
   //! @brief Get the PolynomialValueList object.
   //! @return PolynomialValueList object as std::vector.
   const cimod::PolynomialValueList<FloatType> &get_values() const {
      return poly_value_list_;
   }
   
   //! @brief Return "connected_J_term_index_"
   //! @return "connected_J_term_index_"
   const std::vector<std::vector<std::size_t>> &get_connected_J_term_index() const {
      return connected_J_term_index_;
   }
   
   //! @brief Return "crs_row_"
   //! @return "crs_row_"
   const std::vector<std::size_t> &get_crs_row() const {
      return crs_row_;
   }
   
   //! @brief Return "crs_col_"
   //! @return "crs_col_"
   const std::vector<graph::Index> &get_crs_col() const {
      return crs_col_;
   }
   
   //! @brief Return "crs_val_"
   //! @return "crs_val_"
   const std::vector<FloatType> &get_crs_val() const {
      return crs_val_;
   }
   
   //! @brief Return "crs_sign_p_"
   //! @return "crs_sign_p_"
   const std::vector<int8_t*> &get_crs_sign_p() const {
      return crs_sign_p_;
   }
   
   //! @brief Return "crs_zero_count_p_"
   //! @return "crs_zero_count_p_"
   const std::vector<std::size_t*> &get_crs_zero_count_p() const {
      return crs_zero_count_p_;
   }
   
   //! @brief Set delta E (dE), which is used to determine whether to flip the spin/binary or not.
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
   
   //! @brief Update "binary_zero_count_poly_" and "spin".  This function is used only when" vartype" is cimod::Vartype::BINARY
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
   
   //! @brief Update "spin_sign_poly_" and "spin".  This function is used only when" vartype" is cimod::Vartype::SPIN
   //! @param index const std::size_t
   inline void update_sign_and_spin(const std::size_t index) {
      spin[index] *= -1;
      for (const auto &index_interaction: connected_J_term_index_[index]) {
         spin_sign_poly_[index_interaction] *= -1;
      }
   }
   
   //! @brief Update delta E (dE) and spin for the vartype being "SPIN" case
   //! @param index std::size_t
   void update_dE_for_spin(std::size_t index) {
      dE[index] *= -1;
      const std::size_t begin = crs_row_[index];
      const std::size_t end   = crs_row_[index + 1];
      for (std::size_t i = begin; i < end; ++i) {
         dE[crs_col_[i]] += crs_val_[i]*(*crs_sign_p_[i]);
      }
   }
   
   //! @brief Update delta E (dE) and binary for the vartype being "BINARY" case
   //! @param index std::size_t
   void update_dE_for_binary(std::size_t index) {
      dE[index] *= -1;
      const std::size_t begin = crs_row_[index];
      const std::size_t end   = crs_row_[index + 1];
      for (std::size_t i = begin; i < end; ++i) {
         graph::Index col = crs_col_[i];
         dE[col] += Sign(spin[col] + spin[index])*(crs_val_[i])*ZeroOrOne(spin[index], spin[col], *crs_zero_count_p_[i]);
      }
   }
   
   //! @brief Reset spin/binary configurations
   //! @param init_spins const graph::Spins&
   void reset_spins(const graph::Spins& init_spins) {
      assert(init_spins.size() == num_spins);
      CheckInitialConditions();
      
      if (vartype_ == cimod::Vartype::SPIN) {
         for (std::size_t index = 0; index < init_spins.size(); ++index) {
            if (spin[index] != init_spins[index]) {
               update_dE_for_spin(index);
               update_sign_and_spin(index);
            }
         }
      }
      else if (vartype_ == cimod::Vartype::BINARY) {
         for (std::size_t index = 0; index < init_spins.size(); ++index) {
            if (spin[index] != init_spins[index]) {
               update_dE_for_binary(index);
               update_zero_count_and_binary(index);
            }
         }
      }
      else {
         std::stringstream ss;
         ss << "Unknown vartype detected in " << __func__ << std::endl;
         throw std::runtime_error(ss.str());
      }
      
      for (std::size_t index = 0; index < init_spins.size(); ++index) {
         assert(spin[index] == init_spins[index]);
      }
      
   }
   
   //! @brief The number of spins/binaries
   const std::size_t num_spins;
   
   //! @brief The model's type. cimod::vartype::SPIN or  cimod::vartype::BINARY
   const cimod::Vartype vartype_;
   
   //! @brief Store the information about the energy difference when flipping a spin/binary
   std::vector<FloatType> dE;
   
   //! @brief Spin/binary configurations
   graph::Spins spin;
      
private:
   //! @brief The number of the interactions
   std::size_t num_interactions_;
   
   //! @brief Row of a sparse matrix (Compressed Row Storage) to update "dE".
   std::vector<std::size_t>  crs_row_;
   
   //! @brief Column of a sparse matrix (Compressed Row Storage) to update "dE".
   std::vector<graph::Index> crs_col_;
   
   //! @brief Value of a sparse matrix (Compressed Row Storage) to update "dE".
   std::vector<FloatType>    crs_val_;
   
   //! @brief Value of a sparse matrix (Compressed Row Storage) to update "dE".
   //! @details Note that this is used only for binary variable cases. This stores the pointers for "spin_sign_poly_", which stores the information about the sign of variables.
   std::vector<int8_t*>      crs_sign_p_;
   
   //! @brief Value of a sparse matrix (Compressed Row Storage) to update "dE".
   //! @details Note that this is used only for binary variable cases. This stores the pointers for "binary_zero_count_poly_", which stores the information about the number of variables takeing zero.
   std::vector<std::size_t*> crs_zero_count_p_;
   
   //! @brief The list of the indices of the polynomial interactions (namely, the list of keys of the polynomial interactions as std::unordered_map) as std::vector<std::vector>>.
   cimod::PolynomialKeyList<graph::Index> poly_key_list_;
   
   //! @brief The list of the values of the polynomial interactions (namely, the list of values of the polynomial interactions as std::unordered_map) as std::vector.
   cimod::PolynomialValueList<FloatType>  poly_value_list_;
   
   //! @brief Store the information about the indices of "poly_value_list_".
   std::vector<std::vector<std::size_t>>  connected_J_term_index_;
   
   //! @brief Store the information about the sign of variables.
   //! @details Note that this is used only for spin variable cases, and the pointers of this std::vector is stored in "crs_sign_p". Do not change this std::vector.
   std::vector<int8_t>      spin_sign_poly_;
   
   //! @brief Store the information about the number of variables takeing zero.
   //! @details Note that this is used only for binary variable cases, and the pointers of this std::vector is stored in "crs_zero_count_p". Do not change this std::vector.
   std::vector<std::size_t> binary_zero_count_poly_;
   
   //! @brief The maximal energy difference between any two neighboring solutions
   FloatType max_dE_;
   
   //! @brief The minimal energy difference between any two neighboring solutions
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
   
   
   //! @brief Set "crs_row", "crs_col", "crs_val", and "crs_sign_p" (spin variable cases) or "crs_zero_count_p" (binary variable cases).
   //! @details These std::vector constitute a sparse matrix (Compressed Row Storage), which is used to update "dE".
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
   
   //! @brief Set "connected_J_term_index". If "vartype" is cimod::SPIN, "spin_sign_poly_" is also set, else If "vartype" is cimod::BINARY, "binary_zero_count_poly_" is also set.
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
   
   //! @brief Set "poly_key_list_" and "poly_value_list_".
   //! @param input_keys
   //! @param input_values
   //! @param relabel_flag When true, the variables are relabeld to the non-negative integers.
   void SetPolyKeysAndValues(const cimod::PolynomialKeyList<graph::Index> &input_keys, const cimod::PolynomialValueList<FloatType> &input_values, const bool relabel_flag) {
      if (input_keys.size() != input_values.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      
      poly_key_list_  .clear();
      poly_value_list_.clear();
      
      if (relabel_flag) {
         std::unordered_set<graph::Index> variables;
         for (std::size_t i = 0; i < input_keys.size(); ++i) {
            if (input_values[i] != 0.0) {
               for (const auto &it: input_keys[i]) {
                  variables.emplace(it);
               }
            }
         }
         std::vector<graph::Index> sorted_variables(variables.begin(), variables.end());
         std::sort(sorted_variables.begin(), sorted_variables.end());
         
         std::unordered_map<graph::Index, graph::Index> relabeld_variables;
         for (std::size_t i = 0; i < sorted_variables.size(); ++i) {
            relabeld_variables[sorted_variables[i]] = i;
         }

         for (std::size_t i = 0; i < input_keys.size(); ++i) {
            if (input_values[i] != 0) {
               std::vector<graph::Index> temp;
               for (const auto &it: input_keys[i]) {
                  temp.push_back(relabeld_variables[it]);
               }
               poly_key_list_.push_back(temp);
               poly_value_list_.push_back(input_values[i]);
            }
         }
      }
      else {
         for (std::size_t i = 0; i < input_keys.size(); ++i) {
            if (input_values[i] != 0) {
               poly_key_list_.push_back(input_keys[i]);
               poly_value_list_.push_back(input_values[i]);
            }
         }
      }
      num_interactions_ = poly_key_list_.size();
   }
   
   //! @brief Check the input initial conditions are valid
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
   
};

//! @brief Helper function for ClassicalIsingPolynomial constructor
//! @tparam GraphType
//! @param init_spin const graph::Spins&. The initial spin/binaries.
//! @param init_interaction GraphType&. The initial interactions.
template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins &init_spin, const GraphType &init_interaction) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction);
}

//! @brief Helper function for ClassicalIsingPolynomial constructor by using nlohmann::json object
//! @tparam FloatType
//! @param init_spin const graph::Spins&. The initial spin/binaries.
//! @param init_obj nlohmann::json&
auto make_classical_ising_polynomial(const graph::Spins &init_spin, const nlohmann::json &init_obj) {
   return ClassicalIsingPolynomial<graph::Polynomial<double>>(init_spin, init_obj);
}


} //namespace system
} //namespace openjij



#endif /* classical_ising_polynomial_hpp */
