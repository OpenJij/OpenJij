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

#ifndef k_local_polynomial_hpp
#define k_local_polynomial_hpp

#include <graph/all.hpp>
#include <nlohmann/json.hpp>
#include <graph/json/parse.hpp>

namespace openjij {
namespace system {

//! @brief KLocalPolynomial class, which is a system for higher ordere unconstrained binary optimization (HUBO) models with vartype being "BINARY"
//! @tparam GraphType type of graph
template<class GraphType>
class KLocalPolynomial;

//! @brief KLocalPolynomial class
template<typename FloatType>
class KLocalPolynomial<graph::Polynomial<FloatType>> {
   
   
public:
   
   using system_type = classical_system;
   
   const std::size_t num_spins;
      
   graph::Spins spin;
   
   KLocalPolynomial(const graph::Spins &initial_spins, const graph::Polynomial<FloatType> &poly_graph): num_spins(initial_spins.size()) {
      if (poly_graph.get_vartype() != cimod::Vartype::BINARY) {
         throw std::runtime_error("Only Binary variables are supported");
      }
      
      vartype_ = poly_graph.get_vartype();
      num_interactions_ = poly_graph.get_keys().size();
      
      const auto &poly_key_list   = poly_graph.get_keys();
      const auto &poly_value_list = poly_graph.get_values();
      
      //TO DO
      //Check Input Interactions
      
      if (poly_key_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      
      poly_key_list_.resize(num_interactions_);
      poly_value_list_.resize(num_interactions_);
      
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         poly_key_list_[i]   = poly_key_list[i];
         poly_value_list_[i] = poly_value_list[i];
      }
      
      SetAdj();
      reset_spins(initial_spins);
   }
   
   KLocalPolynomial(const graph::Spins &initial_spins, const nlohmann::json &j) :num_spins(initial_spins.size()) {
      
      vartype_ = ConvertVartype(j.at("vartype"));
      
      const auto &v_k_v = graph::json_parse_polynomial<FloatType>(j);
      const auto &poly_key_list   = std::get<1>(v_k_v);
      const auto &poly_value_list = std::get<2>(v_k_v);
      
      if (poly_key_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      
      num_interactions_ = poly_key_list.size();
      
      poly_key_list_.resize(num_interactions_);
      poly_value_list_.resize(num_interactions_);
      
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         poly_key_list_[i]   = poly_key_list[i];
         poly_value_list_[i] = poly_value_list[i];
      }
      
      SetAdj();
      reset_spins(initial_spins);
   }
   
   void reset_spins(const graph::Spins &init_spin) {
      spin = init_spin;
      ResetZeroCount();
      reset_dE();
   }
   
   void reset_dE() {
      dE_.clear();
      dE_.resize(num_spins);
      
#pragma omp parallel for
      for (int64_t index_binary = 0; index_binary < num_spins; ++index_binary) {
         FloatType val = 0.0;
         const graph::Binary binary = spin[index_binary];
         for (const auto &index_key: adj_[index_binary]) {
            if (zero_count_[index_key] + binary == 1) {
               val += poly_value_list_[index_key];
            }
         }
         dE_[index_binary] = (-2*binary + 1)*val;
      }
   }
   
   inline FloatType dE_single(const graph::Index index_binary) const {
      return dE_[index_binary];
   }
   
   FloatType dE_k_local(const std::size_t index_key) const {
      FloatType dE = 0.0;
      for (const auto &index_binary: poly_key_list_[index_key]) {
         if (spin[index_binary] == 0) {
            dE += dE_[index_binary];
            
         }
         
         
      }
      return dE;
   }
   
   void update_system_single(const graph::Index index_update_binary) {
      const graph::Binary update_binary = spin[index_update_binary];
      const int coeef = -2*update_binary + 1;
      const int count = +2*update_binary - 1;
      for (const auto &index_key: adj_[index_update_binary]) {
         FloatType val = poly_value_list_[index_key];
         for (const auto &index_binary: poly_key_list_[index_key]) {
            const graph::Binary binary = spin[index_binary];
            if (zero_count_[index_key] + update_binary + binary == 2 && index_binary != index_update_binary) {
               dE_[index_binary] += coeef*(-2*binary + 1)*val;
            }
         }
         zero_count_[index_key] += count;
      }
      dE_[index_update_binary] *= -1;
      spin[index_update_binary] = 1 - spin[index_update_binary];
   }
   
   void print_dE() const {
      for (std::size_t i = 0; i < dE_.size(); ++i) {
         printf("dE[%2ld]=%+.15lf\n", i, dE_[i]);
      }
   }
   
   void print_zero_count() const {
      for (std::size_t i = 0; i < num_interactions_; ++i) {
         printf("zero_count[");
         for (const auto &index_binary: poly_key_list_[i]) {
            printf("%ld, ", index_binary);
         }
         printf("]=%lld\n", zero_count_[i]);
      }
   }
   
   void print_adj() const {
      for (std::size_t i = 0; i < num_spins; ++i) {
         printf("adj[%ld]=", i);
         for (const auto &index_key: adj_[i]) {
            printf("%ld, ", index_key);
         }
         printf("\n");
      }
   }
   
private:
   cimod::Vartype vartype_;
   
   int64_t num_interactions_;
   
   std::vector<FloatType> dE_;
   
   std::vector<FloatType> virtual_dE_;
   
   graph::Spins virtual_spin_;
   
   std::vector<std::size_t> virtual_update_count_;
   
   std::vector<int64_t> virtual_zero_count_;
   
   std::vector<int64_t> zero_count_;
   
   std::vector<std::vector<graph::Index>> adj_;
   
   cimod::PolynomialKeyList<graph::Index> poly_key_list_;
   
   cimod::PolynomialValueList<FloatType>  poly_value_list_;
   
   
   cimod::Vartype ConvertVartype(const std::string str) const {
      if (str == "BINARY") {
         return cimod::Vartype::BINARY;
      }
      else if (str == "SPIN") {
         throw std::runtime_error("SPIN variables are not supported");
      }
      else {
         throw std::runtime_error("Unknown vartype detected");
      }
   }
   
   void SetAdj() {
      adj_.clear();
      adj_.resize(num_spins);
      for (int64_t i = 0; i < num_interactions_; ++i) {
         for (const auto &index: poly_key_list_[i]) {
            adj_[index].push_back(i);
         }
      }
   }
   
   void ResetZeroCount() {
      zero_count_.resize(num_interactions_);
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         int64_t zero_count = 0;
         for (const auto &index: poly_key_list_[i]) {
            if (spin[index] == 0) {
               zero_count++;
            }
         }
         zero_count_[i] = zero_count;
      }
   }
   
   
   
   
   
   
   
};

//! @brief Helper function for ClassicalIsingPolynomial constructor
//! @tparam GraphType
//! @param init_spin const graph::Spins&. The initial spin/binaries.
//! @param init_interaction GraphType&. The initial interactions.
template<typename GraphType>
auto make_k_local_polynomial(const graph::Spins &init_spin, const GraphType &init_interaction) {
   return KLocalPolynomial<GraphType>(init_spin, init_interaction);
}

//! @brief Helper function for ClassicalIsingPolynomial constructor by using nlohmann::json object
//! @tparam FloatType
//! @param init_spin const graph::Spins&. The initial spin/binaries.
//! @param init_obj nlohmann::json&
auto make_k_local_polynomial(const graph::Spins &init_spin, const nlohmann::json &init_obj) {
   return KLocalPolynomial<graph::Polynomial<double>>(init_spin, init_obj);
}


} //namespace system
} //namespace openjij

#endif /* k_local_polynomial_hpp */
