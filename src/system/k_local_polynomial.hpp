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

//! @brief ClassicalIsingPolynomial class
template<typename FloatType>
class KLocalPolynomial<graph::Polynomial<FloatType>> {
   
   
public:
   
   //! @brief system type
   using system_type = classical_system;
   
   //! @brief Constructor of ClassicalIsingPolynomial
   //! @param init_spins graph::Spins&. The initial spin/binary configurations.
   //! @param j const nlohmann::json object
   ClassicalIsingPolynomial(const graph::Spins &initial_spins, const nlohmann::json &j):num_spins(initial_spins.size()), spin(initial_spins), vartype_(j.at("vartype") == "SPIN" ? cimod::Vartype::SPIN : cimod::Vartype::BINARY) {
      
      const auto &v_k_v = graph::json_parse_polynomial<FloatType>(j);
      const auto &poly_key_list   = std::get<1>(v_k_v);
      const auto &poly_value_list = std::get<2>(v_k_v);
      
      if (j.at("vartype") != "BINARY" ) {
         throw std::runtime_error("Only binary variables are currently supported.");
      }
      
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
      
   }
   
   
   //! @brief Set delta E (dE), which is used to determine whether to flip the spin/binary or not.
   void reset_dE() {
      dE.resize(num_interactions_);
      for (int64_t i = 0; i < num_interactions_; ++i) {
         FloatType temp_energy = 0.0;
         graph::Binary binary = spin[i];
         for (const auto &it: variable_to_key_list_index_[i]) {
            if (binary_zero_count_poly_[it] == static_cast<int64_t>(1 - binary)) {
               temp_energy += poly_value_list_[it]*Sign(binary);
            }
         }
         dE_single[i] = temp_energy;
      }
      
   }
   
   
   
   //! @brief The number of spins/binaries
   const std::size_t num_spins;
   
   //! @brief The model's type. cimod::vartype::SPIN or  cimod::vartype::BINARY
   const cimod::Vartype vartype;
   
   //! @brief Store the information about the energy difference when flipping a spin/binary
   std::vector<FloatType> dE_k_local;
   
   std::vector<FloatType> dE_single;
   
   //! @brief Spin/binary configurations
   graph::Spins spin;
   
private:
   //! @brief The number of the interactions
   int64_t num_interactions_;
   
   std::vector<int64_t> variable_to_key_list_index_;
   
   std::vector<int64_t> binary_zero_count_poly_;
   
   //! @brief The list of the indices of the polynomial interactions (namely, the list of keys of the polynomial interactions as std::unordered_map) as std::vector<std::vector>>.
   cimod::PolynomialKeyList<graph::Index> poly_key_list_;
   
   //! @brief The list of the values of the polynomial interactions (namely, the list of values of the polynomial interactions as std::unordered_map) as std::vector.
   cimod::PolynomialValueList<FloatType>  poly_value_list_;
   
   void SetAdjacencyList() {
      variable_to_key_list_index_.resize(num_spins);
      binary_zero_count_poly_.resize(num_interactions_);
      for (int64_t i = 0; i < num_interactions_; ++i) {
         int64_t zero_count = 0;
         for (const auto &index: poly_key_list_[i]) {
            variable_to_key_list_index_[index].push_back(i);
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
   
};



} //namespace system
} //namespace openjij

#endif /* k_local_polynomial_hpp */
