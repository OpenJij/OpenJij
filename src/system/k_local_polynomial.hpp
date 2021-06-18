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
   
   //! @brief system type
   using system_type = classical_system;
   
   //! @brief The number of spins/binaries
   const int64_t num_binaries;
      
   //! @brief Spin/binary configurations
   graph::Binaries binaries;
   
   KLocalPolynomial(const graph::Binaries &initial_binaries, const graph::Polynomial<FloatType> &poly_graph): num_binaries(initial_binaries.size()) {
      if (poly_graph.get_vartype() != cimod::Vartype::BINARY) {
         throw std::runtime_error("Only Binary variables are supported");
      }
      
      vartype_ = poly_graph.get_vartype();
      
      const auto &poly_key_list   = poly_graph.get_keys();
      const auto &poly_value_list = poly_graph.get_values();
      
      if (poly_key_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }

      poly_key_list_.clear();
      poly_value_list_.clear();

      for (std::size_t i = 0; i < poly_key_list.size(); ++i) {
         if (poly_value_list[i] != 0) {
            poly_key_list_.push_back(poly_key_list[i]);
            poly_value_list_.push_back(poly_value_list[i]);
         }
      }
      num_interactions_ = static_cast<int64_t>(poly_key_list_.size());
      
      SetAdj();
      reset_spins(initial_binaries);
   }
   
   KLocalPolynomial(const graph::Binaries &initial_binaries, const nlohmann::json &j) :num_binaries(initial_binaries.size()) {
      
      vartype_ = ConvertVartype(j.at("vartype"));
      
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
      
      SetAdj();
      reset_spins(initial_binaries);
   }
   
   void reset_spins(const graph::Binaries &init_binaries) {
      binaries    = init_binaries;
      binaries_v_ = init_binaries;
      ResetZeroCount();
      reset_dE();
   }
   
   void reset_dE() {
      dE_.clear();
      dE_v_.clear();
      dE_.resize(num_binaries);
      dE_v_.resize(num_binaries);
      
#pragma omp parallel for
      for (int64_t index_binary = 0; index_binary < num_binaries; ++index_binary) {
         FloatType val = 0.0;
         const graph::Binary binary = binaries[index_binary];
         for (const auto &index_key: adj_[index_binary]) {
            if (zero_count_[index_key] + binary == 1) {
               val += poly_value_list_[index_key];
            }
         }
         dE_[index_binary]   = (-2*binary + 1)*val;
         dE_v_[index_binary] = dE_[index_binary];
      }
   }
   
   inline FloatType dE_single(const graph::Index index_binary) const {
      return dE_[index_binary];
   }
   
   FloatType dE_k_local(const std::size_t index_key) {
      FloatType dE = 0.0;
      for (const auto &index_binary: poly_key_list_[index_key]) {
         if (binaries_v_[index_binary] == 0) {
            dE += dE_v_[index_binary];
            virtual_update_system_single(index_binary);
         }
      }
      return dE;
   }
   
   void update_system_k_local() {
      for (const auto &index_binary: update_index_binaries_v_) {
         binaries[index_binary] = binaries_v_[index_binary];
      }
      for (const auto &index_zero_count: update_index_zero_count_v_) {
         zero_count_[index_zero_count] = zero_count_v_[index_zero_count];
      }
      for (const auto &index_dE: update_index_dE_v_) {
         dE_[index_dE] = dE_v_[index_dE];
      }
      update_index_binaries_v_.clear();
      update_index_zero_count_v_.clear();
      update_index_dE_v_.clear();
   }
   
   void reset_virtual_system() {
      for (const auto &index_binary: update_index_binaries_v_) {
         binaries_v_[index_binary] = binaries[index_binary];
      }
      for (const auto &index_zero_count: update_index_zero_count_v_) {
         zero_count_v_[index_zero_count] = zero_count_[index_zero_count];
      }
      for (const auto &index_dE: update_index_dE_v_) {
         dE_v_[index_dE] = dE_[index_dE];
      }
      update_index_binaries_v_.clear();
      update_index_zero_count_v_.clear();
      update_index_dE_v_.clear();
   }
   
   void virtual_update_system_single(const graph::Index index_update_binary) {
      const graph::Binary update_binary = binaries_v_[index_update_binary];
      const int coeef = -2*update_binary + 1;
      const int count = +2*update_binary - 1;
      for (const auto &index_key: adj_[index_update_binary]) {
         FloatType val = poly_value_list_[index_key];
         for (const auto &index_binary: poly_key_list_[index_key]) {
            const graph::Binary binary = binaries_v_[index_binary];
            if (zero_count_v_[index_key] + update_binary + binary == 2 && index_binary != index_update_binary) {
               dE_v_[index_binary] += coeef*(-2*binary + 1)*val;
               update_index_dE_v_.emplace(index_binary);
            }
         }
         zero_count_v_[index_key] += count;
         update_index_zero_count_v_.emplace(index_key);
      }
      dE_v_[index_update_binary] *= -1;
      update_index_dE_v_.emplace(index_update_binary);
      binaries_v_[index_update_binary] = 1 - binaries_v_[index_update_binary];
      update_index_binaries_v_.push_back(index_update_binary);
   }
   
   void update_system_single(const graph::Index index_update_binary) {
      const graph::Binary update_binary = binaries[index_update_binary];
      const int coeef = -2*update_binary + 1;
      const int count = +2*update_binary - 1;
      for (const auto &index_key: adj_[index_update_binary]) {
         FloatType val = poly_value_list_[index_key];
         for (const auto &index_binary: poly_key_list_[index_key]) {
            const graph::Binary binary = binaries[index_binary];
            if (zero_count_[index_key] + update_binary + binary == 2 && index_binary != index_update_binary) {
               dE_[index_binary]   += coeef*(-2*binary + 1)*val;
               dE_v_[index_binary] += coeef*(-2*binary + 1)*val;
            }
         }
         zero_count_[index_key]   += count;
         zero_count_v_[index_key] += count;
      }
      dE_[index_update_binary]   *= -1;
      dE_v_[index_update_binary] *= -1;
      binaries[index_update_binary]    = 1 - binaries[index_update_binary];
      binaries_v_[index_update_binary] = 1 - binaries_v_[index_update_binary];
   }
   
   inline int64_t GetNumInteractions() const {
      return num_interactions_;
   }
   
   inline int64_t GetZeroCount(const std::size_t index_key) const {
      return zero_count_[index_key];
   }
   
   inline FloatType GetPolyValue(const std::size_t index_key) const {
      return poly_value_list_[index_key];
   }
   
   inline const std::vector<graph::Index> &GetAdj(const std::size_t index_binary) const {
      return adj_[index_binary];
   }
   
   cimod::Vartype get_vartype() const {
      return vartype_;
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
      for (std::size_t i = 0; i < num_binaries; ++i) {
         printf("adj[%ld]=", i);
         for (const auto &index_key: adj_[i]) {
            printf("%ld(%+lf), ", index_key, poly_value_list_[index_key]);
         }
         printf("\n");
      }
   }
   
private:
   cimod::Vartype vartype_;
   
   int64_t num_interactions_;
   
   std::vector<FloatType> dE_;
   
   std::vector<FloatType> dE_v_;
   
   graph::Binaries binaries_v_;
   
   std::unordered_set<std::size_t> update_index_dE_v_;
   
   std::unordered_set<std::size_t> update_index_zero_count_v_;
   
   std::vector<std::size_t> update_index_binaries_v_;
      
   std::vector<int64_t> zero_count_v_;
   
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
   
   void SetInteractions(const cimod::PolynomialKeyList<graph::Index> &poly_key_list,
                        const cimod::PolynomialValueList<FloatType>  &poly_value_list) {
      
      if (poly_key_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }

      poly_key_list_.clear();
      poly_value_list_.clear();

      for (std::size_t i = 0; i < poly_key_list.size(); ++i) {
         if (poly_value_list[i] != 0) {
            poly_key_list_.push_back(poly_key_list[i]);
            poly_value_list_.push_back(poly_value_list[i]);
         }
      }
      num_interactions_ = static_cast<int64_t>(poly_key_list_.size());
   }
   
   void SetAdj() {
      adj_.clear();
      adj_.resize(num_binaries);
      for (int64_t i = 0; i < num_interactions_; ++i) {
         for (const auto &index: poly_key_list_[i]) {
            adj_[index].push_back(i);
         }
      }
      
      //sort
      auto compare = [this](std::size_t i1, std::size_t i2) { return poly_value_list_[i1] < poly_value_list_[i2]; };
      
      int64_t adj_size = static_cast<int64_t>(adj_.size());
#pragma omp parallel for
      for (int64_t i = 0; i < adj_size; ++i) {
         std::sort(adj_[i].begin(), adj_[i].end(), compare);
      }
   }
   
   void ResetZeroCount() {
      zero_count_.resize(num_interactions_);
      zero_count_v_.resize(num_interactions_);
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         int64_t zero_count = 0;
         for (const auto &index: poly_key_list_[i]) {
            if (binaries[index] == 0) {
               zero_count++;
            }
         }
         zero_count_[i]   = zero_count;
         zero_count_v_[i] = zero_count;
      }
   }
   
   
   
   
   
   
   
};

//! @brief Helper function for ClassicalIsingPolynomial constructor
//! @tparam GraphType
//! @param init_binaries const graph::Binaries&. The initial binaries.
//! @param init_interaction GraphType&. The initial interactions.
template<typename GraphType>
auto make_k_local_polynomial(const graph::Binaries &init_binaries, const GraphType &init_interaction) {
   return KLocalPolynomial<GraphType>(init_binaries, init_interaction);
}

//! @brief Helper function for ClassicalIsingPolynomial constructor by using nlohmann::json object
//! @tparam FloatType
//! @param init_binaries const graph::Binaries&. The initial binaries.
//! @param init_obj nlohmann::json&
auto make_k_local_polynomial(const graph::Binaries &init_binaries, const nlohmann::json &init_obj) {
   return KLocalPolynomial<graph::Polynomial<double>>(init_binaries, init_obj);
}


} //namespace system
} //namespace openjij

#endif /* k_local_polynomial_hpp */
