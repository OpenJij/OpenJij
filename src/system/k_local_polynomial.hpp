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
      
      SetAdjacencyList();
      reset_dE();
      
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
      
      SetAdjacencyList();
      reset_dE();
      
   }
   
   int64_t GetNumInteractions() const {
      return num_interactions_;
   }
   
   const std::vector<std::vector<int64_t>> &GetAdjacency() const {
      return adjacency_;
   }
   
   const std::vector<int64_t> &GetZeroCount() const {
      return binary_zero_count_poly_;
   }
   
   const std::vector<std::vector<std::vector<graph::Index>>> &GetUpdatedIndex() const {
      return to_be_updated_index;
   }
   
   const cimod::PolynomialKeyList<graph::Index> &GetPolyKeyList() const {
      return poly_key_list_;
   }
   
   const cimod::PolynomialValueList<FloatType> &GetPolyValueList() const {
      return poly_value_list_;
   }
   
   void reset_dE() {
      dE.clear();
      to_be_updated_index.resize(num_interactions_);
      
      std::vector<std::unordered_set<graph::Index>> poly_key_set(num_interactions_);
#pragma omp parallel for
      for (int64_t i = 0; i < num_interactions_; ++i) {
         poly_key_set[i] = std::unordered_set<graph::Index>(poly_key_list_[i].begin(), poly_key_list_[i].end());
      }
      
      //Set Ordinary dE
#pragma omp parallel for
      for (graph::Index i = 0; i < num_spins; ++i) {
         FloatType temp_energy = 0.0;
         auto temp_spin = spin[i];
         for (const auto &adj_interaction_index: adjacency_[i]) {
            temp_energy += poly_value_list_[adj_interaction_index]*Sign(temp_spin)*ZeroOrOne(temp_spin, binary_zero_count_poly_[adj_interaction_index]);
         }
         dE[std::vector<graph::Index>{i}] = temp_energy;
      }
      //Set the other remaining dE
      for (int64_t i = 0; i < num_interactions_; ++i) {
         std::vector<graph::Index> exclude_key(0);
         for (int64_t j = 0; j < poly_key_list_[i].size(); ++j) {
            const auto include_index = poly_key_list_[i][j];
            const auto temp_spin = spin[i];
            FloatType temp_energy = 0.0;
            const auto key = std::vector<graph::Index>(poly_key_list_[i].begin(), std::next(poly_key_list_[i].begin(), j + 1));
            if (dE.count(key) == 0) {
               for (const auto &adj_interaction_index: adjacency_[include_index]) {
                  bool flag_exclude = true;
                  for (const auto &exclude_index: exclude_key) {
                     if (poly_key_set[adj_interaction_index].count(exclude_index) != 0) {
                        flag_exclude = false;
                        break;
                     }
                  }
                  if (flag_exclude) {
                     temp_energy += poly_value_list_[adj_interaction_index]*Sign(temp_spin)*ZeroOrOne(temp_spin, binary_zero_count_poly_[adj_interaction_index]);
                     to_be_updated_index[adj_interaction_index].push_back(key);
                  }
               }
               dE[key] = temp_energy;
            }
            exclude_key.push_back(include_index);
         }
      }
   }
   
   

   
   
   //! @brief The number of spins/binaries
   const int64_t num_spins;
   
   //! @brief The model's type. cimod::vartype::SPIN or  cimod::vartype::BINARY
   const cimod::Vartype vartype;
   
   //! @brief Store the information about the energy difference when flipping a spin/binary
   std::unordered_map<std::vector<graph::Index>, FloatType, cimod::vector_hash> dE;
   
   //! @brief Spin/binary configurations
   graph::Spins spin;
   
private:
   //! @brief The number of the interactions
   int64_t num_interactions_;
   
   std::vector<std::vector<int64_t>> adjacency_;
   
   std::vector<int64_t> binary_zero_count_poly_;
   
   std::vector<std::vector<std::vector<graph::Index>>> to_be_updated_index;
      
   //! @brief The list of the indices of the polynomial interactions (namely, the list of keys of the polynomial interactions as std::unordered_map) as std::vector<std::vector>>.
   cimod::PolynomialKeyList<graph::Index> poly_key_list_;
   
   //! @brief The list of the values of the polynomial interactions (namely, the list of values of the polynomial interactions as std::unordered_map) as std::vector.
   cimod::PolynomialValueList<FloatType>  poly_value_list_;
   
   void SetAdjacencyList() {
      adjacency_.resize(num_spins);
      binary_zero_count_poly_.resize(num_interactions_);
      for (int64_t i = 0; i < num_interactions_; ++i) {
         int64_t zero_count = 0;
         for (const auto &index: poly_key_list_[i]) {
            adjacency_[index].push_back(i);
            if (spin[index] == 0) {
               zero_count++;
            }
         }
         binary_zero_count_poly_[i] = zero_count;
      }
   }
   
   //! @brief Return -1 or +1 in accordance with the input binary
   //! @param binary graph::Binary
   //! @return -1 if binary is 1 number, otherwise +1
   inline int Sign(graph::Binary binary) {
      return -2*binary + 1;
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
