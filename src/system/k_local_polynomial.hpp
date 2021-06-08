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
   
   KLocalPolynomial(const graph::Spins &initial_spins, const graph::Polynomial<FloatType> &poly_graph):
   num_spins(poly_graph.size()), vartype(poly_graph.get_vartype()), spin(initial_spins) {
      
      if (vartype != cimod::Vartype::BINARY ) {
         throw std::runtime_error("Only binary variables are currently supported.");
      }
      
      const auto &poly_key_list   = poly_graph.get_keys();
      const auto &poly_value_list = poly_graph.get_values();
      
      //TO DO
      //Check Input Interactions
      
      if (poly_key_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      
      poly_key_list_.resize(poly_key_list.size());
      poly_value_list_.resize(poly_value_list.size());
      
      num_interactions_ = poly_key_list.size();
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         poly_key_list_[i]   = poly_key_list[i];
         poly_value_list_[i] = poly_value_list[i];
      }
      
      SetAdjacency();
      //reset_dE();
      reset_dE_interactions();
   }
   
   
   //! @brief Constructor of KLocalPolynomial
   //! @param init_spins graph::Spins&. The initial spin/binary configurations.
   //! @param j const nlohmann::json object
   KLocalPolynomial(const graph::Spins &initial_spins, const nlohmann::json &j):num_spins(initial_spins.size()), spin(initial_spins), vartype(j.at("vartype") == "SPIN" ? cimod::Vartype::SPIN : cimod::Vartype::BINARY) {
      
      if (vartype != cimod::Vartype::BINARY ) {
         throw std::runtime_error("Only binary variables are currently supported.");
      }

      const auto &v_k_v = graph::json_parse_polynomial<FloatType>(j);
      const auto &poly_key_list   = std::get<1>(v_k_v);
      const auto &poly_value_list = std::get<2>(v_k_v);
   
      if (poly_key_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      
      poly_key_list_.resize(poly_key_list.size());
      poly_value_list_.resize(poly_value_list.size());
      
      num_interactions_ = poly_key_list.size();
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         poly_key_list_[i]   = poly_key_list[i];
         poly_value_list_[i] = poly_value_list[i];
      }
      
      SetAdjacency();
      //reset_dE();
      reset_dE_interactions();
   }
   
   int64_t GetNumInteractions() const {
      return num_interactions_;
   }
   
   const std::vector<std::vector<int64_t>> &GetAdjacency() const {
      return adj_key_;
   }
   
   const std::vector<int64_t> &GetZeroCount() const {
      return binary_zero_count_poly_;
   }
   
   const std::vector<std::vector<std::vector<graph::Index>>> &GetUpdatedIndex() const {
      return adj_dE_;
   }
   
   const cimod::PolynomialKeyList<graph::Index> &GetPolyKeyList() const {
      return poly_key_list_;
   }
   
   const std::vector<graph::Index> &GetPolyKey(const std::size_t index) const {
      return poly_key_list_[index];
   }
   
   const cimod::PolynomialValueList<FloatType> &GetPolyValueList() const {
      return poly_value_list_;
   }
   
   inline const std::vector<std::size_t> &GetKey(const std::size_t index) const {
      return poly_key_list_[index];
   }
      
   inline bool IncludeBinaryWithZero(const std::size_t index_interaction) const {
      if (num_binary_with_one_[index_interaction] == poly_key_list_[index_interaction].size()) {
         return false;
      }
      else {
         return true;
      }
   }
   
   void reset_dE_interactions() {
      dE_interactions_.clear();
      adj_dE_.clear();
      adj_dE_.resize(num_interactions_);
      
      std::vector<std::unordered_set<graph::Index>> poly_key_set(num_interactions_);
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         poly_key_set[i] = std::unordered_set<graph::Index>(poly_key_list_[i].begin(), poly_key_list_[i].end());
      }
      
      for (const auto &key: poly_key_list_) {
                  
         for (std::size_t i = 0; i < key.size(); ++i) {
            std::vector<graph::Index> exclude_key;
            const auto index_dE = std::vector<graph::Index>(key.begin(), key.begin() + i + 1);
            if (dE_interactions_.count(index_dE) == 0) {
               FloatType temp_energy = 0.0;
               if (spin[index_dE.back()] == 0) {
                  for (const auto &index_key: adj_key_[index_dE.back()]) {
                     bool flag_exclude = true;
                     for (const auto &index_exclude_key: exclude_key) {
                        if (poly_key_set[index_key].count(index_exclude_key) != 0) {
                           flag_exclude = false;
                           break;
                        }
                     }
                     
                     printf("size - 1: %ld <-> %ld :one_count\n", poly_key_list_[index_key].size() - 1, num_binary_with_one_[index_key]);
                     
                     if (flag_exclude && (poly_key_list_[index_key].size() - 1 == num_binary_with_one_[index_key])) {
                        temp_energy += poly_value_list_[index_key];
                        printf("≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈\n");
                     }
                     if (flag_exclude) {
                        adj_dE_[index_key].push_back(index_dE);
                     }
                  }
               }
               
               dE_interactions_[index_dE] = temp_energy;
            }
            exclude_key.push_back(index_dE.back());
         }
      }
   }
   
   void reset_dE() {
      dE_binary_.clear();
      dE_interactions_.clear();
      adj_dE_.clear();
      dE_binary_.resize(num_spins);
      adj_dE_.resize(num_interactions_);
      
      std::vector<std::unordered_set<graph::Index>> poly_key_set(num_interactions_);
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         poly_key_set[i] = std::unordered_set<graph::Index>(poly_key_list_[i].begin(), poly_key_list_[i].end());
      }
      
      //Set Ordinary dE
      for (graph::Index i = 0; i < num_spins; ++i) {
         FloatType temp_energy = 0.0;
         const auto temp_spin = spin[i];
         const auto sign = Sign(temp_spin);
         for (const auto &adj_interaction_index: adj_key_[i]) {
            temp_energy += sign*poly_value_list_[adj_interaction_index]*ZeroOrOne(temp_spin, binary_zero_count_poly_[adj_interaction_index]);
         }
         dE_binary_[i] = temp_energy;
         if (temp_spin == 0) {
            dE_interactions_[std::vector<graph::Index>{i}] = temp_energy;
         }
         else {
            dE_interactions_[std::vector<graph::Index>{i}] = 0.0;
         }
      }
      
      //Set the other remaining dE
      for (int64_t i = 0; i < num_interactions_; ++i) {
         std::vector<graph::Index> exclude_key(0);
         for (int64_t j = 0; j < poly_key_list_[i].size(); ++j) {
            const auto include_index = poly_key_list_[i][j];
            const auto temp_spin = spin[include_index];
            const auto key = std::vector<graph::Index>(poly_key_list_[i].begin(), std::next(poly_key_list_[i].begin(), j + 1));
            if (dE_interactions_.count(key) == 0) {
               FloatType temp_energy = 0.0;
               for (const auto &adj_interaction_index: adj_key_[include_index]) {
                  bool flag_exclude = true;
                  for (const auto &exclude_index: exclude_key) {
                     if (poly_key_set[adj_interaction_index].count(exclude_index) != 0) {
                        flag_exclude = false;
                        break;
                     }
                  }
                  if (flag_exclude) {
                     temp_energy += poly_value_list_[adj_interaction_index]*Sign(temp_spin)*ZeroOrOne(temp_spin, binary_zero_count_poly_[adj_interaction_index]);
                     adj_dE_[adj_interaction_index].push_back(key);
                  }
               }
               
               if (temp_spin == 0) {
                  printf("%lf\n", temp_energy);
                  dE_interactions_[key] = temp_energy;
               }
               else {
                  dE_interactions_[key] = 0.0;
               }
            }
            exclude_key.push_back(include_index);
         }
      }
   }
   
   void SetAdjacency() {
      adj_key_.clear();
      binary_zero_count_poly_.clear();
      num_binary_with_one_.clear();
      adj_key_.resize(num_spins);
      binary_zero_count_poly_.resize(num_interactions_);
      num_binary_with_one_.resize(num_interactions_);
      for (int64_t i = 0; i < num_interactions_; ++i) {
         int64_t zero_count = 0;
         int64_t one_count  = 0;
         for (const auto &index: poly_key_list_[i]) {
            adj_key_[index].push_back(i);
            if (spin[index] == 0) {
               zero_count++;
            }
            else {
               one_count++;
            }
         }
         binary_zero_count_poly_[i] = zero_count;
         num_binary_with_one_[i] = one_count;
      }
   }
   
   FloatType dE_k_local(const std::size_t index_key) const {
      FloatType dE_out = 0.0;
      for (std::size_t i = 0; i < poly_key_list_[index_key].size(); ++i) {
         const auto key = std::vector<graph::Index>(poly_key_list_[index_key].begin(), std::next(poly_key_list_[index_key].begin(), i + 1));
         dE_out += dE_interactions_.at(key);
      }
      return dE_out;
   }
   
   inline FloatType dE_single(const std::size_t index_binary) const {
      return dE_binary_[index_binary];
   }
    
   void update_dE_interactions(const std::size_t index_updated_binary) {
      
      if (spin[index_updated_binary] == 1) {
         return;
      }
      
      dE_interactions_[std::vector<graph::Index>{index_updated_binary}] = 0.0;
      
      for (const auto &index_key: adj_key_[index_updated_binary]) {
         num_binary_with_one_[index_key]++;
         const std::size_t size_key    = poly_key_list_[index_key].size();
         const bool        flag_active = (size_key - 1 == num_binary_with_one_[index_key]);
         const FloatType   val         = poly_value_list_[index_key];
         if (size_key > 1 && flag_active) {
            for (const auto &index_binary: poly_key_list_[index_key]) {
               if (index_binary != index_updated_binary) {
                  dE_interactions_[std::vector<graph::Index>{index_binary}] += val*(1 - spin[index_binary]);
               }
            }
         }
         
         if (flag_active) {
            for (const auto &index_dE: adj_dE_[index_key]) {
               dE_interactions_[index_dE] += val*(1 - spin[index_dE.back()]);
            }
         }

      }
      
   }
   
   void update_system(const std::size_t index_binary) {
      graph::Binary x = spin[index_binary];
      for (const auto &index_interaction: adj_key_[index_binary]) {
         FloatType val = poly_value_list_[index_interaction];
         for (const auto &include_index_binary: poly_key_list_[index_interaction]) {
            graph::Binary y          = spin[include_index_binary];
            int64_t       zero_count = binary_zero_count_poly_[index_interaction];
            dE_binary_[include_index_binary] += Sign(x + y)*val*ZeroOrOne(x, y, zero_count);
            printf("ddE[%ld] for %lld id %d\n",include_index_binary, index_interaction, ZeroOrOne(x, y, zero_count));
            if (x == 0) {
               dE_interactions_[std::vector<graph::Index>{include_index_binary}] = 0.0;
            }
            else {
               dE_interactions_[std::vector<graph::Index>{include_index_binary}] += Sign(x + y)*val*ZeroOrOne(x, y, zero_count);
            }
         }
         
         //x will be updated to 0
         for (const auto &key: adj_dE_[index_interaction]) {
            graph::Binary y          = spin[key.back()];
            int64_t       zero_count = binary_zero_count_poly_[key.back()];
            if (y == 0) {
               dE_interactions_[key] += Sign(x + y)*val*ZeroOrOne(x, y, zero_count);
            }
            else {
               dE_interactions_[key]  = 0.0;
            }
         }

      }

      if (x == 0) {
         spin[index_binary] = 1;
         for (const auto &index_interaction: adj_key_[index_binary]) {
            binary_zero_count_poly_[index_interaction]--;
         }
      }
      else if (x == 1) {
         spin[index_binary] = 0;
         for (const auto &index_interaction: adj_key_[index_binary]) {
            binary_zero_count_poly_[index_interaction]++;
         }
      }
      else {
         throw std::runtime_error("Invalid binary variable (!= 0 and !=1) detected");
      }
   }
   
   //For debagging
   void PrintInfo() const {
      printf("NumInteractions: %lld\n", num_interactions_);
      
      for (std::size_t i = 0; i < num_interactions_; ++i) {
         printf("[%ld]: Key[", i);
         for (std::size_t j = 0; j < poly_key_list_[i].size(); ++j) {
            printf("%ld,", poly_key_list_[i][j]);
         }
         printf("]=%lf\n", poly_value_list_[i]);
      }
      
      for (std::size_t i = 0; i < spin.size(); ++i) {
         printf("Spin[%ld]=%d\n", i, spin[i]);
      }
      
      for (std::size_t i = 0; i < adj_key_.size(); ++i) {
         for (std::size_t j = 0; j < adj_key_[i].size(); ++j) {
            printf("Adj[%ld][%ld]=%lld\n", i, j, adj_key_[i][j]);
         }
      }
      
      for (std::size_t i = 0; i < binary_zero_count_poly_.size(); ++i) {
         printf("ZeroCount[%ld]=%lld\n", i, binary_zero_count_poly_[i]);
      }
      
      for (std::size_t i = 0; i < adj_dE_.size(); ++i) {
         printf("ToBeUpdatedIndex[%ld]=\n", i);
         for (std::size_t j = 0; j < adj_dE_[i].size(); ++j) {
            for (std::size_t k = 0; k < adj_dE_[i][j].size(); ++k) {
               printf("%ld,", adj_dE_[i][j][k]);
            }
            printf("\n");
         }
      }
      
      for (const auto &it: dE_interactions_) {
         printf("dE_inter[");
         for (const auto &vec: it.first) {
            printf("%ld,", vec);
         }
         printf("]=%lf\n", it.second);
      }
      
      for (std::size_t i = 0; i < dE_binary_.size(); ++i) {
         printf("dE_binary_[%ld]=%lf\n", i, dE_binary_[i]);
      }
      
   }
   
   //! @brief The number of spins/binaries
   const int64_t num_spins;
   
   //! @brief The model's type. cimod::vartype::SPIN or  cimod::vartype::BINARY
   const cimod::Vartype vartype;
   
   //! @brief Spin/binary configurations
   graph::Spins spin;

   
private:

   
   //! @brief Store the information about the energy difference when flipping a spin/binary
   std::unordered_map<std::vector<graph::Index>, FloatType, cimod::vector_hash> dE_interactions_;
   
   std::vector<FloatType> dE_binary_;
   
   //! @brief The number of the interactions
   int64_t num_interactions_;
   
   std::vector<std::vector<int64_t>> adj_key_;
      
   std::vector<int64_t> binary_zero_count_poly_;
   
   std::vector<int64_t> num_binary_with_one_;
   
   std::vector<std::vector<std::vector<graph::Index>>> adj_dE_;
      
   //! @brief The list of the indices of the polynomial interactions (namely, the list of keys of the polynomial interactions as std::unordered_map) as std::vector<std::vector>>.
   cimod::PolynomialKeyList<graph::Index> poly_key_list_;
   
   //! @brief The list of the values of the polynomial interactions (namely, the list of values of the polynomial interactions as std::unordered_map) as std::vector.
   cimod::PolynomialValueList<FloatType>  poly_value_list_;
   
   //! @brief Return -1 or +1 in accordance with the input binary
   //! @param binary graph::Binary
   //! @return -1 if binary is 1 number, otherwise +1
   inline int Sign(graph::Binary binary) {
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
