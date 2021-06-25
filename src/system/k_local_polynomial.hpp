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
#include <sstream>

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
   
   //! @brief The number of binaries/binaries
   const int64_t num_binaries;
   
   //! @brief k-local  update is activated per rate_call_k_local times
   int rate_call_k_local = 10;
   
   //! @brief Counter of calling updater
   int64_t count_call_updater = 0;
      
   //! @brief Spin/binary configurations
   graph::Binaries binaries;
   
   const cimod::Vartype vartype = cimod::Vartype::BINARY;
   
   KLocalPolynomial(const graph::Binaries &init_binaries, const graph::Polynomial<FloatType> &poly_graph): num_binaries(init_binaries.size()), binaries(init_binaries), binaries_v_(init_binaries) {
            
      const auto &poly_key_list   = poly_graph.get_keys();
      const auto &poly_value_list = poly_graph.get_values();
      
      if (poly_key_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }

      std::unordered_set<graph::Index> active_binary_set;
      
      poly_key_list_.clear();
      poly_value_list_.clear();
      
      for (std::size_t i = 0; i < poly_key_list.size(); ++i) {
         if (poly_value_list[i] != 0) {
            poly_key_list_.push_back(poly_key_list[i]);
            poly_value_list_.push_back(poly_value_list[i]);
            for (const auto &it: poly_key_list[i]) {
               active_binary_set.emplace(it);
            }
         }
      }
      num_interactions_ = static_cast<int64_t>(poly_key_list_.size());
      active_binaries_ = std::vector<graph::Index>(active_binary_set.begin(), active_binary_set.end());
      std::sort(active_binaries_.begin(), active_binaries_.end());
      
      SetAdj();
      ResetZeroCount();
      reset_dE();
   }
   
   KLocalPolynomial(const graph::Binaries &init_binaries, const nlohmann::json &j) :num_binaries(init_binaries.size()), binaries(init_binaries), binaries_v_(init_binaries) {
      
      if (j.at("vartype") != "BINARY") {
         throw std::runtime_error("Only binary variables are supported");
      }
            
      const auto &v_k_v = graph::json_parse_polynomial<FloatType>(j);
      const auto &poly_key_list   = std::get<0>(v_k_v);
      const auto &poly_value_list = std::get<1>(v_k_v);
      
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
      
      active_binaries_.resize(num_binaries);
      std::iota(active_binaries_.begin(),active_binaries_.end(), 0);
      
      SetAdj();
      ResetZeroCount();
      reset_dE();
   }
   
   void reset_binaries(const graph::Binaries &init_binaries) {
      if (init_binaries.size() != binaries.size()) {
         throw std::runtime_error("The size of initial binaries does not equal to system size");
      }
      for (const auto &index_binary: active_binaries_) {
         if (binaries[index_binary] != init_binaries[index_binary]) {
            update_system_single(index_binary);
         }
         if (binaries[index_binary] != init_binaries[index_binary]) {
            std::stringstream ss;
            ss << "Unknown error detected in " << __func__;
            throw std::runtime_error(ss.str());
         }
      }
   }
   
   void reset_dE() {
      dE_.clear();
      dE_v_.clear();
      dE_.resize(num_binaries);
      dE_v_.resize(num_binaries);
      
      max_effective_dE = std::abs(poly_value_list_.front());
      min_effective_dE = std::abs(poly_value_list_.front());
      
      for (const auto &index_binary: active_binaries_) {
         FloatType val     = 0.0;
         FloatType abs_val = 0.0;
         bool flag = false;
         const graph::Binary binary = binaries[index_binary];
         for (const auto &index_key: adj_[index_binary]) {
            if (zero_count_[index_key] + binary == 1) {
               val     += poly_value_list_[index_key];
               abs_val += std::abs(poly_value_list_[index_key]);
               flag = true;
            }
         }
         dE_[index_binary]   = (-2*binary + 1)*val;
         dE_v_[index_binary] = dE_[index_binary];
         
         if (flag && max_effective_dE < abs_val) {
            max_effective_dE = abs_val;
         }
         if (flag && min_effective_dE > abs_val) {
            min_effective_dE = abs_val;
         }
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
         const FloatType val = poly_value_list_[index_key];
         for (const auto &index_binary: poly_key_list_[index_key]) {
            const graph::Binary binary = binaries_v_[index_binary];
            if (zero_count_v_[index_key] + update_binary + binary == 2 && index_binary != index_update_binary) {
               dE_v_[index_binary] += coeef*(-2*binary + 1)*val;
               update_index_dE_v_.emplace(index_binary);
            }
         }
         zero_count_v_[index_key] += count;
         update_index_zero_count_v_.push_back(index_key);
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
         const FloatType val = poly_value_list_[index_key];
         for (const auto &index_binary: poly_key_list_[index_key]) {
            const graph::Binary binary = binaries[index_binary];
            if (zero_count_[index_key] + update_binary + binary == 2 && index_binary != index_update_binary) {
               dE_[index_binary]   += coeef*(-2*binary + 1)*val;
               dE_v_[index_binary]  = dE_[index_binary];
            }
         }
         zero_count_[index_key]   += count;
         zero_count_v_[index_key]  = zero_count_[index_key];
      }
      dE_[index_update_binary]   *= -1;
      dE_v_[index_update_binary]  = dE_[index_update_binary];
      binaries[index_update_binary]    = 1 - binaries[index_update_binary];
      binaries_v_[index_update_binary] = binaries[index_update_binary];
   }
   
   void set_rate_call_k_local(int rate_k_local) {
      if (rate_k_local <= 0) {
         throw std::runtime_error("rate_k_local is larger than zero");
      }
      rate_call_k_local = rate_k_local;
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
   
   inline const std::vector<graph::Index> &get_adj(const std::size_t index_binary) const {
      return adj_[index_binary];
   }

   inline const std::vector<graph::Index> &get_active_binaries() const {
      return active_binaries_;
   }
   
   FloatType get_max_effective_dE() const {
      return max_effective_dE;
   }
   
   FloatType get_min_effective_dE() const {
      return min_effective_dE;
   }
   
   const cimod::PolynomialValueList<FloatType> &get_values() const {
      return poly_value_list_;
   }
   
   const cimod::PolynomialKeyList<graph::Index> &get_keys() const {
      return poly_key_list_;
   }
   
   const std::vector<std::vector<graph::Index>> &get_adj() const {
      return adj_;
   }
   
   std::string get_vartype_string() const {
      return "BINARY";
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
   
   int64_t num_interactions_;
   
   std::vector<FloatType> dE_;

   std::vector<int64_t> zero_count_;
   
   std::vector<std::vector<graph::Index>> adj_;
   
   cimod::PolynomialKeyList<graph::Index> poly_key_list_;
   
   cimod::PolynomialValueList<FloatType>  poly_value_list_;
   
   std::vector<graph::Index> active_binaries_;

   std::vector<FloatType> dE_v_;
   
   graph::Binaries binaries_v_;
   
   std::unordered_set<std::size_t> update_index_dE_v_;
   
   std::vector<std::size_t> update_index_zero_count_v_;
   
   std::vector<std::size_t> update_index_binaries_v_;
      
   std::vector<int64_t> zero_count_v_;
   
   FloatType max_effective_dE;
   
   FloatType min_effective_dE;
      
   void SetAdj() {
      adj_.clear();
      adj_.resize(num_binaries);
      for (int64_t i = 0; i < num_interactions_; ++i) {
         for (const auto &index: poly_key_list_[i]) {
            adj_[index].push_back(i);
         }
      }
      
      //sort
      auto compare = [this](const int64_t i1, const int64_t i2) { return poly_value_list_[i1] < poly_value_list_[i2]; };
      
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
