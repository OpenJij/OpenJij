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

   //! @brief The number of binaries/binaries
   const int64_t num_variables;
   
   graph::Spins variables;
   
   const cimod::Vartype vartype;
   
   //! @brief Constructor of ClassicalIsingPolynomial
   //! @param init_spins graph::Spins&. The initial spin/binary configurations.
   //! @param poly_graph graph::Polynomial<FloatType>& (Polynomial graph class). The initial interacrtions.
   ClassicalIsingPolynomial(const graph::Spins &init_variables, const graph::Polynomial<FloatType> &poly_graph, const cimod::Vartype init_vartype): num_variables(poly_graph.size()), variables(init_variables), vartype(init_vartype) {
      SetInteractions(poly_graph);
      SetAdj();
      ResetZeroCount();
      ResetSignKey();
      reset_dE();
   }
   
   //! @brief Constructor of ClassicalIsingPolynomial
   //! @param init_spins graph::Spins&. The initial spin/binary configurations.
   //! @param poly_graph graph::Polynomial<FloatType>& (Polynomial graph class). The initial interacrtions.
   ClassicalIsingPolynomial(const graph::Spins &init_variables, const graph::Polynomial<FloatType> &poly_graph, const std::string init_vartype): num_variables(poly_graph.size()), variables(init_variables), vartype(ConvertVartype(init_vartype)) {
      SetInteractions(poly_graph);
      SetAdj();
      ResetZeroCount();
      ResetSignKey();
      reset_dE();
   }
   
   //! @brief Constructor of ClassicalIsingPolynomial
   //! @param init_spins graph::Spins&. The initial spin/binary configurations.
   //! @param j const nlohmann::json object
   ClassicalIsingPolynomial(const graph::Spins &init_variables, const nlohmann::json &j):num_variables(init_variables.size()), variables(init_variables), vartype(j.at("vartype") == "SPIN" ? cimod::Vartype::SPIN : cimod::Vartype::BINARY) {
      const auto &v_k_v = graph::json_parse_polynomial<FloatType>(j);
      const auto &poly_key_list   = std::get<1>(v_k_v);
      const auto &poly_value_list = std::get<2>(v_k_v);
      
      if (poly_key_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      
      num_interactions_ = static_cast<int64_t>(poly_key_list.size());
      
      poly_key_list_.resize(num_interactions_);
      poly_value_list_.resize(num_interactions_);
      
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         poly_key_list_[i]   = poly_key_list[i];
         poly_value_list_[i] = poly_value_list[i];
      }
      
      active_variables_.resize(num_variables);
      std::iota(active_variables_.begin(),active_variables_.end(), 0);
      
      SetAdj();
      ResetZeroCount();
      ResetSignKey();
      reset_dE();
   }
   
   void reset_variables(const graph::Spins &init_variables) {
      if (init_variables.size() != variables.size()) {
         throw std::runtime_error("The size of initial spins/binaries does not equal to system size");
      }
      
      if (vartype == cimod::Vartype::SPIN) {
         for (const auto &index_variable: active_variables_) {
            if (variables[index_variable] != init_variables[index_variable]) {
               update_spin_system(index_variable);
            }
            if (variables[index_variable] != init_variables[index_variable]) {
               std::stringstream ss;
               ss << "Unknown error detected in " << __func__;
               throw std::runtime_error(ss.str());
            }
         }
      }
      else if (vartype == cimod::Vartype::BINARY) {
         for (const auto &index_variable: active_variables_) {
            if (variables[index_variable] != init_variables[index_variable]) {
               update_binary_system(index_variable);
            }
            if (variables[index_variable] != init_variables[index_variable]) {
               std::stringstream ss;
               ss << "Unknown error detected in " << __func__;
               throw std::runtime_error(ss.str());
            }
         }
      }
      else {
         throw std::runtime_error("Unknown vartype detected");
      }
   }
   
   void reset_dE() {
      dE_.clear();
      dE_.resize(num_variables);
      
      max_abs_dE_ = std::abs(poly_value_list_.front());
      min_abs_dE_ = std::abs(poly_value_list_.front());
      
      if (vartype == cimod::Vartype::SPIN) {
         for (const auto &index_binary: active_variables_) {
            FloatType val     = 0.0;
            FloatType abs_val = 0.0;
            bool flag = false;
            for (const auto &index_key: adj_[index_binary]) {
               val     += poly_value_list_[index_key]*sign_key_[index_key];
               abs_val += std::abs(poly_value_list_[index_key]);
               flag = true;
            }
            dE_[index_binary] = -2*val;
            
            if (flag && max_abs_dE_ < 2*abs_val) {
               max_abs_dE_ = 2*abs_val;
            }
            if (flag && min_abs_dE_ > 2*abs_val) {
               min_abs_dE_ = 2*abs_val;
            }
         }
      }
      else if (vartype == cimod::Vartype::BINARY) {
         for (const auto &index_binary: active_variables_) {
            FloatType val     = 0.0;
            FloatType abs_val = 0.0;
            bool flag = false;
            const graph::Binary binary = variables[index_binary];
            for (const auto &index_key: adj_[index_binary]) {
               if (zero_count_[index_key] + binary == 1) {
                  val     += poly_value_list_[index_key];
                  abs_val += std::abs(poly_value_list_[index_key]);
                  flag = true;
               }
            }
            dE_[index_binary] = (-2*binary + 1)*val;
            
            if (flag && max_abs_dE_ < abs_val) {
               max_abs_dE_ = abs_val;
            }
            if (flag && min_abs_dE_ > abs_val) {
               min_abs_dE_ = abs_val;
            }
         }
      }
      else {
         throw std::runtime_error("Unknown vartype detected");
      }
   }
   
   void update_spin_system(const graph::Index index_update_variable) {
      
   }
   
   void update_binary_system(const graph::Index index_update_binary) {
      const graph::Binary update_binary = variables[index_update_binary];
      const int coeef = -2*update_binary + 1;
      const int count = +2*update_binary - 1;
      for (const auto &index_key: adj_[index_update_binary]) {
         FloatType val = poly_value_list_[index_key];
         for (const auto &index_binary: poly_key_list_[index_key]) {
            const graph::Binary binary = variables[index_binary];
            if (zero_count_[index_key] + update_binary + binary == 2 && index_binary != index_update_binary) {
               dE_[index_binary] += coeef*(-2*binary + 1)*val;
            }
         }
         zero_count_[index_key] += count;
      }
      dE_[index_update_binary] *= -1;
      variables[index_update_binary] = 1 - variables[index_update_binary];
   }
 
      
private:
   int64_t num_interactions_;
   
   std::vector<FloatType> dE_;
   
   std::vector<int64_t> zero_count_;
   
   std::vector<int8_t>  sign_key_;
   
   std::vector<std::vector<graph::Index>> adj_;
   
   cimod::PolynomialKeyList<graph::Index> poly_key_list_;
   
   cimod::PolynomialValueList<FloatType>  poly_value_list_;
   
   std::vector<graph::Index> active_variables_;
   
   FloatType max_abs_dE_;
   
   FloatType min_abs_dE_;
   
   void SetAdj() {
      adj_.clear();
      adj_.resize(num_variables);
      for (int64_t i = 0; i < num_interactions_; ++i) {
         for (const auto &index: poly_key_list_[i]) {
            adj_[index].push_back(i);
         }
      }
   }
   
   void SetInteractions(const graph::Polynomial<FloatType> &poly_graph) {
      const auto &poly_key_list   = poly_graph.get_keys();
      const auto &poly_value_list = poly_graph.get_values();
      
      if (poly_key_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }

      std::unordered_set<graph::Index> active_variable_set;
      
      poly_key_list_.clear();
      poly_value_list_.clear();
      
      for (std::size_t i = 0; i < poly_key_list.size(); ++i) {
         if (poly_value_list[i] != 0.0) {
            poly_key_list_.push_back(poly_key_list[i]);
            poly_value_list_.push_back(poly_value_list[i]);
            for (const auto &it: poly_key_list[i]) {
               active_variable_set.emplace(it);
            }
         }
      }
      num_interactions_ = static_cast<int64_t>(poly_key_list_.size());
      active_variables_ = std::vector<graph::Index>(active_variable_set.begin(), active_variable_set.end());
      std::sort(active_variables_.begin(), active_variables_.end());
   }
   
   void ResetZeroCount() {
      if (vartype != cimod::Vartype::BINARY) {
         return;
      }
      zero_count_.resize(num_interactions_);
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         int64_t zero_count = 0;
         for (const auto &index: poly_key_list_[i]) {
            if (variables[index] == 0) {
               zero_count++;
            }
         }
         zero_count_[i] = zero_count;
      }
   }
   
   void ResetSignKey() {
      if (vartype != cimod::Vartype::SPIN) {
         return;
      }
      sign_key_.resize(num_interactions_);
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         int8_t sign_key = 1;
         for (const auto &index: poly_key_list_[i]) {
            sign_key *= variables[index];
         }
         sign_key_[i] = sign_key;
      }
   }
   
   cimod::Vartype ConvertVartype(const std::string vartype) const {
      if (vartype == "SPIN") {
         return cimod::Vartype::SPIN;
      }
      else if (vartype == "BINARY") {
         return cimod::Vartype::BINARY;
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
auto make_classical_ising_polynomial(const graph::Spins &init_spin, const GraphType &init_interaction, const cimod::Vartype vartype) {
   return ClassicalIsingPolynomial<GraphType>(init_spin, init_interaction, vartype);
}

template<typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins &init_spin, const GraphType &init_interaction, const std::string vartype) {
   return ClassicalIsingPolynomial<graph::Polynomial<double>>(init_spin, init_interaction, vartype);
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
