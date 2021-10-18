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

/**
 * @mainpage cimod
 *
 * @section s_overview Overview
 * cimod is a C++ library for a binary quadratic model.
 * This library provides a binary quadratic model class which contains an Ising model or a quadratic unconstrained binary optimization (QUBO) model.
 * It also provides utilities for constructing a model and transforming to some other interfaces.
 * This library is created based on dimod (https://github.com/dwavesystems/dimod).
 *
 * @section s_bqm Binary quadratic model
 * A binary quadratic model class can contain an Ising model or a QUBO model.
 *
 * @subsection ss_ising Ising model
 * An energy of an Ising model \f$E_{\mathrm{Ising}}\f$ is represented by
 * \f[
 * E_{\mathrm{Ising}} = \sum_{i} h_i s_i + \sum_{i \neq j} J_{ij} s_i s_j + \delta_{\mathrm{Ising}},
 * \f]
 * where \f$s_i \in \{+1, -1\}\f$ denotes a spin at the site \f$i\f$, \f$h_i\f$ denotes an external magnetic field parameter, \f$J_{ij}\f$ denotes an interaction parameter and \f$\delta_{\mathrm{Ising}}\f$ denotes an offset.
 * Note that this library assumes that the interaction is not symmetric, i.e., \f$J_{ij} \neq J_{ji}\f$.
 *
 * @subsection ss_qubo QUBO model
 * An evaluation function of a QUBO model \f$E_{\mathrm{QUBO}}\f$ is represented by
 * \f[
 * E_{\mathrm{QUBO}} = \sum_{i, j} Q_{ij} x_i x_j + \delta_{\mathrm{QUBO}},
 * \f]
 * where \f$x_i \in \{0, 1\}\f$ denotes a decision variable, \f$Q_{ij}\f$ denotes a quadratic bias and \f$\delta_{\mathrm{QUBO}}\f$ denotes an offset.
 * Note that this library assumes that the quadratic bias is not symmetric, i.e., \f$Q_{ij} \neq Q_{ji}\f$ if \f$i \neq j\f$.
 *
 * @section s_bpm Binary polynomial model
 * A binary polynomial model, which can be regarded as an extended model of the binary quadratic model, can handle Ising and HUBO models.
 * @subsection ss_bpm_Ising Ising model
 * An energy of an "extended" Ising model \f$E_{\mathrm{Ising}}\f$ is represented by
 * \f[
 * E_{\mathrm{Ising}} = \sum_{i} h_i s_i + \sum_{i \neq j} J_{ij} s_i s_j +  \sum_{i \neq j \neq k} J_{ijk} s_i s_j s_k + \ldots
 * \f]
 * Here \f$s_i \in \{+1, -1\}\f$ denotes the spin at the site \f$i\f$, \f$ h_i \f$ denotes the external magnetic field at the site \f$ i \f$, and \f$J_{ijk\ldots}\f$ represents the interaction between the sites.
 * Note that \f$ i \neq j \neq k \f$ means \f$ i \neq j \f$, \f$ j \neq k \f$, and \f$ i \neq k \f$.
 * This library assumes that the interaction is not symmetric. For example, \f$J_{ij} \neq J_{ji}\f$ for \f$  i\neq j\f$, \f$J_{ijk} \neq J_{jik}\f$ for \f$ i \neq j \neq k \f$, and so on.
 *
 * @subsection ss_bpm_hubo HUBO model
 * An energy of an "extended" QUBO model \f$ E_{\mathrm{HUBO}}\f$, here we call polynomial unconstrained binary optimization (HUBO), is represented by
 * \f[
 * E_{\mathrm{HUBO}} = \sum_{i \neq j} Q_{ij} x_i x_j +  \sum_{i \neq j \neq k} Q_{ijk} x_i x_j x_k + \ldots
 * \f]
 * Here \f$ x_i \in \{0, 1\} \f$ denotes the spin at the site \f$ i \f$ and \f$Q_{ijk\ldots}\f$ represents the interaction between the sites.
 * Note that \f$ i \neq j \neq k \f$ means \f$ i \neq j \f$, \f$ j \neq k \f$, and \f$ i \neq k \f$.
 * This library assumes that the interaction is not symmetric. For example, \f$Q_{ij} \neq Q_{ji}\f$ for \f$  i\neq j\f$, \f$Q_{ijk} \neq Q_{jik}\f$ for \f$ i \neq j \neq k \f$, and so on.
 *
 * @section s_example Example
 * @code
 * #include "../src/binary_quadratic_model.hpp"
 * #include "../src/binary_polynomial_model.hpp"
 *
 * using namespace cimod;
 *
 * int main() {
 *
 *   // Set linear biases and quadratic biases
 *   Linear<uint32_t, double> linear{ {1, 1.0}, {2, 2.0}, {3, 3.0}, {4, 4.0} };
 *   Quadratic<uint32_t, double> quadratic {
 *        {std::make_pair(1, 2), 12.0}, {std::make_pair(1, 3), 13.0}, {std::make_pair(1, 4), 14.0},
 *        {std::make_pair(2, 3), 23.0}, {std::make_pair(2, 4), 24.0},
 *        {std::make_pair(3, 4), 34.0}
 *    };
 *
 *   // Set variable type
 *   Vartype vartype = Vartype::BINARY;
 *
 *   // Set offset
 *   double offset = 0.0;
 *
 *   // Create a BinaryQuadraticModel instance
 *   BinaryQuadraticModel<uint32_t, double> bqm(linear, quadratic, offset, vartype);
 *
 *   // Print informations of bqm
 *   bqm.print();
 *
 *   //Set polynomial biases
 *   Polynomial<uint32_t, double> polynomial {
 *      //Linear biases
 *      {{1}, 1.0}, {{2}, 2.0}, {{3}, 3.0},
 *      //Quadratic biases
 *      {{1, 2}, 12.0}, {{1, 3}, 13.0}, {{2, 3}, 23.0},
 *      //Polynomial bias
 *      {{1, 2, 3}, 123.0}
 *   };
 *
 *   // Create a BinaryPolynominalModel instance
 *   BinaryPolynomialModel<uint32_t, double> bpm(polynomial, vartype);
 *
 *   // Print informations of bpm
 *   bpm.print();
 *
 *   return 0;
 * }
 *
 * @endcode
 */



#ifndef binary_polynomial_model_hpp
#define binary_polynomial_model_hpp

#include "vartypes.hpp"
#include "hash.hpp"
#include "utilities.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <typeinfo>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include <bitset>

namespace cimod {

//! @brief Type alias for the polynomial interactions as std::unordered_map.
//! @tparam IndexType
//! @tparam FloatType
template <typename IndexType, typename FloatType>
using Polynomial = std::unordered_map<std::vector<IndexType>, FloatType, vector_hash>;

//! @brief Type alias for the indices of the polynomial interactions (namely, the list of keys of the polynomial interactions as std::unordered_map) as std::vector<std::vector>>.
//! @tparam IndexType
template <typename IndexType>
using PolynomialKeyList = std::vector<std::vector<IndexType>>;

//! @brief Type alias for the values of the polynomial interactions (namely, the list of values of the polynomial interactions as std::unordered_map) as std::vector.
//! @tparam FloatType
template <typename FloatType>
using PolynomialValueList = std::vector<FloatType>;

//! @brief Type alias for sample, which represents the spin or binary configurations.
//! @tparam IndexType
template <typename IndexType>
using Sample = std::unordered_map<IndexType, int32_t>;

//! @brief Class for BinaryPolynomialModel.
//! @tparam IndexType
//! @tparam FloatType
template <typename IndexType, typename FloatType>
class BinaryPolynomialModel {
   
public:
   
   //! @brief BinaryPolynomialModel constructor.
   //! @param poly_map
   //! @param vartype
   BinaryPolynomialModel(const Polynomial<IndexType, FloatType> &poly_map, const Vartype vartype): vartype_(vartype) {
      if (vartype_ == Vartype::NONE) {
         throw std::runtime_error("Unknown vartype detected");
      }
      AddInteractionsFrom(poly_map);
      UpdateVariablesToIntegers();
   }
   
   //! @brief BinaryPolynomialModel constructor.
   //! @param key_list
   //! @param value_list
   //! @param vartype
   BinaryPolynomialModel(PolynomialKeyList<IndexType> &key_list, const PolynomialValueList<FloatType> &value_list, const Vartype vartype): vartype_(vartype) {
      if (vartype_ == Vartype::NONE) {
         throw std::runtime_error("Unknown vartype detected");
      }
      AddInteractionsFrom(key_list, value_list);
      UpdateVariablesToIntegers();
   }
   
   //! @brief BinaryPolynomialModel constructor.
   //! @param key_list
   //! @param value_list
   //! @param vartype
   BinaryPolynomialModel(const PolynomialKeyList<IndexType> &key_list, const PolynomialValueList<FloatType> &value_list, const Vartype vartype): vartype_(vartype) {
      if (vartype_ == Vartype::NONE) {
         throw std::runtime_error("Unknown vartype detected");
      }
      AddInteractionsFrom(key_list, value_list);
      UpdateVariablesToIntegers();
   }
   
   //! @brief BinaryPolynomialModel constructor.
   //! @details This constructor dose not check the input arguments are valid.
   //! @param variables
   //! @param poly_key_distance_list
   //! @param poly_value_list
   //! @param vartype
   BinaryPolynomialModel(const std::vector<IndexType>         &variables,
                         const PolynomialKeyList<std::size_t> &poly_key_distance_list,
                         const PolynomialValueList<FloatType> &poly_value_list,
                         const Vartype vartype
                         ): vartype_(vartype) {
      if (vartype_ == Vartype::NONE) {
         throw std::runtime_error("Unknown vartype detected");
      }
      
      if (poly_key_distance_list.size() != poly_value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }

      variables_ = std::unordered_set<IndexType>(variables.begin(), variables.end());
      
      if (variables_.size() != variables.size()) {
         throw std::runtime_error("Unknown error. It seems that the input variables contain the same variables");
      }
            
      std::size_t num_interactions = poly_key_distance_list.size();
      poly_key_list_.resize(num_interactions);
      poly_value_list_.resize(num_interactions);
      
#pragma omp parallel for
      for (int64_t i = 0; i < (int64_t)num_interactions; ++i) {
         std::vector<IndexType> temp;
         for (const auto &it: poly_key_distance_list[i]) {
            temp.push_back(variables[it]);
         }
         std::sort(temp.begin(), temp.end());
         poly_key_list_[i]   = temp;
         poly_value_list_[i] = poly_value_list[i];
      }
      
      for (std::size_t i = 0; i < num_interactions; ++i) {
         poly_key_inv_[poly_key_list_[i]] = i;
         for (const auto &it: poly_key_list_[i]) {
            each_variable_num_[it]++;
         }
      }
      
      UpdateVariablesToIntegers();
      
   }
   
   //! @brief Get the Polynomial object.
   //! @return Polynomial object as std::unordered_map.
   Polynomial<IndexType, FloatType> GetPolynomial() const {
      Polynomial<IndexType, FloatType> poly_map;
      for (std::size_t i = 0; i < poly_key_list_.size(); ++i) {
         poly_map[poly_key_list_[i]] = poly_value_list_[i];
      }
      return poly_map;
   }
   
   //! @brief Get the specific value of the interaction according to the key representing the indices of the polynomial interactions.
   //! @details If the interaction corresponding to the key dose not exist, return 0
   //! @param key
   //! @return Corresponding value of the interaction
   FloatType GetPolynomial(std::vector<IndexType> &key) const {
      FormatPolynomialKey(&key, vartype_);
      if (poly_key_inv_.count(key) != 0) {
         return poly_value_list_[poly_key_inv_.at(key)];
      }
      else {
         return 0;
      }
   }
   
   //! @brief Get the specific value of the interaction according to the key representing the indices of the polynomial interactions.
   //! @details If the interaction corresponding to the key dose not exist, return 0
   //! @param key
   //! @return Corresponding value of the interaction
   FloatType GetPolynomial(const std::vector<IndexType> &key) const {
      std::vector<IndexType> copied_key = key;
      return GetPolynomial(copied_key);
   }
   
   //! @brief Get variables_to_integers object
   //! @details This function may need O(N) calculation time (N is the number of the variables).
   //! @return variables_to_integers object, which represents the correspondence from variables to integer numbers
   const std::unordered_map<IndexType, int64_t> &GetVariablesToIntegers() {
      if (relabel_flag_for_variables_to_integers_) {
         UpdateVariablesToIntegers();
      }
      return variables_to_integers_;
   }
   
   //! @brief Get variables_to_integers object
   //! @details This function may need O(N) calculation time (N is the number of the variables).
   //! @return variables_to_integers, which represents the correspondence from variables to integer numbers
   std::unordered_map<IndexType, int64_t> GetVariablesToIntegers() const {
      if (relabel_flag_for_variables_to_integers_) {
         return GenerateVariablesToIntegers();
      }
      else {
         return variables_to_integers_;
      }
   }
   
   //! @brief Get the specific integer number corresponding to the input variable (index).
   //! @details This function may need O(N) calculation time (N is the number of the variables).
   //! @param index
   //! @return Non-negative integer number if the input variable is in the BinaryPolynomialModel, else -1
   int64_t GetVariablesToIntegers(const IndexType &index) {
      if (relabel_flag_for_variables_to_integers_) {
         UpdateVariablesToIntegers();
      }
      if (variables_to_integers_.count(index) == 0) {
         return -1;
      }
      else {
         return variables_to_integers_.at(index);
      }
   }
   
   //! @brief Get the specific integer number corresponding to the input variable (index).
   //! @details This function may need O(N) calculation time (N is the number of the variables).
   //! @param index
   //! @return Non-negative integer number if the input variable is in the BinaryPolynomialModel, else -1
   int64_t GetVariablesToIntegers(const IndexType &index) const {
      if (variables_.count(index) == 0) {
         return -1;
      }
      else {
         std::vector<IndexType> sorted_variables = GetSortedVariables();
         return std::distance(sorted_variables.begin(), std::lower_bound(sorted_variables.begin(), sorted_variables.end(), index));
      }
   }
   
   //! @brief Get the PolynomialKeyList object.
   //! @return PolynomialKeyList object as std::vector<std::vector>>.
   const PolynomialKeyList<IndexType> &GetKeyList() const {
      return poly_key_list_;
   }
   
   //! @brief Get the PolynomialValueList object.
   //! @return PolynomialValueList object as std::vector.
   const PolynomialValueList<FloatType> &GetValueList() const {
      return poly_value_list_;
   }

   //! @brief Get The inverse key list, which indicates the index of the poly_key_list_ and poly_value_list_.
   //! @return The inverse key list.
   const std::unordered_map<std::vector<IndexType>, std::size_t, vector_hash> &GetKeysInv() const {
      return poly_key_inv_;
   }
   
   //! @brief Return the variables as std::unordered_set.
   //! @return variables
   const std::unordered_set<IndexType> &GetVariables() const {
      return variables_;
   }
   
   //! @brief Return the sorted variables as std::vector.
   //! @details This function may need O(N) calculation time (N is the number of the variables).
   //! @return sorted variables as std::vector.
   const std::vector<IndexType> &GetSortedVariables() {
      if (relabel_flag_for_variables_to_integers_) {
         UpdateVariablesToIntegers();
      }
      return sorted_variables_;
   }
   
   //! @brief Return the sorted variables as std::vector.
   //! @details This function may need O(N) calculation time (N is the number of the variables).
   //! @return sorted variables as std::vector.
   std::vector<IndexType> GetSortedVariables() const {
      if (relabel_flag_for_variables_to_integers_) {
         return GenerateSortedVariables();
      }
      else {
         return sorted_variables_;
      }
   }
   
   //! @brief Return the maximum degree of interaction.
   //! @return degree
   std::size_t GetDegree() const {
      std::size_t degree = 0;
      for (const auto &it: poly_key_list_) {
         if (degree < it.size()) {
            degree = it.size();
         }
      }
      return degree;
   }
   
   //! @brief Return the offset.
   //! @return The offset
   FloatType GetOffset() const {
      return GetPolynomial(std::vector<IndexType>{});
   }
   
   //! @brief Return the vartype.
   //! @return The vartype
   Vartype GetVartype() const {
      return vartype_;
   }
   
   //! @brief Return the number of the interactions.
   //! @return The number of the interactions.
   std::size_t GetNumInteractions() const {
      return poly_key_list_.size();
   }
   
   //! @brief Return the number of variables.
   //! @return The number of the variables.
   std::size_t GetNumVariables() const {
      return variables_.size();
   }
   
   //! @brief Create an empty BinaryPolynomialModel.
   //! @param vartype
   //! @return The empty BinaryPolynomialModel.
   BinaryPolynomialModel Empty(const Vartype vartype) const {
      return BinaryPolynomialModel({}, vartype);
   }
   
   //! @brief Clear the BinaryPolynomialModel.
   void Clear() {
      each_variable_num_.clear();
      variables_to_integers_.clear();
      PolynomialKeyList<IndexType>().swap(poly_key_list_);
      PolynomialValueList<FloatType>().swap(poly_value_list_);
      std::unordered_set<IndexType>().swap(variables_);
      poly_key_inv_.clear();
      relabel_flag_for_variables_to_integers_ = true;
   }
   
   //! @brief Remove the specified interaction from the BinaryPolynomialModel.
   //! @param key
   void RemoveInteraction(std::vector<IndexType> &key) {
      FormatPolynomialKey(&key, vartype_);
      if (poly_key_inv_.count(key) == 0) {
         return;
      }
      
      for (const auto &index: key) {
         if (each_variable_num_[index] >= 2) {
            each_variable_num_[index]--;
         }
         else if (each_variable_num_[index] == 1) {
            each_variable_num_.erase(index);
            variables_.erase(index);
            relabel_flag_for_variables_to_integers_ = true;
         }
      }
      
      std::size_t inv = poly_key_inv_[key];
      
      std::swap(poly_key_inv_[key], poly_key_inv_[poly_key_list_.back()]);
      poly_key_inv_.erase(key);
      
      std::swap(poly_key_list_[inv], poly_key_list_.back());
      poly_key_list_.pop_back();
      
      std::swap(poly_value_list_[inv], poly_value_list_.back());
      poly_value_list_.pop_back();
      
   }
   
   //! @brief Remove the specified interaction from the BinaryPolynomialModel.
   //! @param key
   void RemoveInteraction(const std::vector<IndexType> &key) {
      std::vector<IndexType> copied_key = key;
      RemoveInteraction(copied_key);
   }
   
   //! @brief Remove the specified interactions from the BinaryPolynomialModel.
   //! @param key_list
   void RemoveInteractionsFrom(PolynomialKeyList<IndexType> &key_list) {
      for (auto &&key: key_list) {
         RemoveInteraction(key);
      }
   }
   
   //! @brief Remove the specified interactions from the BinaryPolynomialModel.
   //! @param key_list
   void RemoveInteractionsFrom(const PolynomialKeyList<IndexType> &key_list) {
      for (const auto &key: key_list) {
         RemoveInteraction(key);
      }
   }
   
   //! @brief Set the offset of the BinaryPolynomialModel to zero.
   void RemoveOffset() {
      RemoveInteraction(std::vector<IndexType>{});
   }
   
   //! @brief Remove a variable from the BinaryPolynomialModel.
   //! @param index
   void RemoveVariable(const IndexType &index) {
      for (auto &&key: poly_key_list_) {
         if (std::binary_search(key.begin(), key.end(), index)) {
            RemoveInteraction(key);
         }
      }
   }
   
   //! @brief Remove the specified variables from the BinaryPolynomialModel.
   //! @param key
   void RemoveVariablesFrom(const std::vector<IndexType> &key) {
      for (const auto &index: key) {
         RemoveVariable(index);
      }
   }
   
   //! @brief Add an interaction to the BinaryPolynomialModel.
   //! @param key
   //! @param value
   //! @param vartype
   void AddInteraction(std::vector<IndexType> &key, const FloatType &value, const Vartype vartype = Vartype::NONE) {
      if (std::abs(value) <= 0.0) {
         return;
      }
      
      if (vartype_ == vartype || vartype == Vartype::NONE) {
         FormatPolynomialKey(&key, vartype_);
         SetKeyAndValue(key, value);
      }
      else {
         const std::size_t original_key_size     = key.size();
         const std::size_t changed_key_list_size = IntegerPower(2, original_key_size);
         
         if (vartype_ == Vartype::SPIN && vartype == Vartype::BINARY) {
            FormatPolynomialKey(&key, vartype);
            for (std::size_t i = 0; i < changed_key_list_size; ++i) {
               const auto changed_key = GenerateChangedKey(key, i);
               int sign = ((original_key_size - changed_key.size())%2 == 0) ? 1.0 : -1.0;
               SetKeyAndValue(changed_key, value*IntegerPower(2, changed_key.size())*sign);
            }
         }
         else if (vartype_ == Vartype::BINARY && vartype == Vartype::SPIN) {
            FormatPolynomialKey(&key, vartype);
            FloatType changed_value = value*(1.0/changed_key_list_size);
            for (std::size_t i = 0; i < changed_key_list_size; ++i) {
               SetKeyAndValue(GenerateChangedKey(key, i), changed_value);
            }
         }
         else {
            throw std::runtime_error("Unknown vartype error");
         }
      }
   }
   
   //! @brief Add an interaction to the BinaryPolynomialModel.
   //! @param key
   //! @param value
   //! @param vartype
   void AddInteraction(const std::vector<IndexType> &key, const FloatType &value, const Vartype vartype = Vartype::NONE) {
      std::vector<IndexType> copied_key = key;
      AddInteraction(copied_key, value, vartype);
   }
   
   //! @brief Add interactions to the BinaryPolynomialModel.
   //! @param poly_map
   //! @param vartype
   void AddInteractionsFrom(const Polynomial<IndexType, FloatType> &poly_map, const Vartype vartype = Vartype::NONE) {
      for (const auto &it: poly_map) {
         AddInteraction(it.first, it.second, vartype);
      }
   }
   
   //! @brief Add interactions to the BinaryPolynomialModel.
   //! @param key_list
   //! @param value_list
   //! @param vartype
   void AddInteractionsFrom(PolynomialKeyList<IndexType> &key_list, const PolynomialValueList<FloatType> &value_list, const Vartype vartype = Vartype::NONE) {
      if (key_list.size() != value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      for (std::size_t i = 0; i < key_list.size(); ++i) {
         AddInteraction(key_list[i], value_list[i], vartype);
      }
   }
   
   //! @brief Add interactions to the BinaryPolynomialModel.
   //! @param key_list
   //! @param value_list
   //! @param vartype
   void AddInteractionsFrom(const PolynomialKeyList<IndexType> &key_list, const PolynomialValueList<FloatType> &value_list, const Vartype vartype = Vartype::NONE) {
      if (key_list.size() != value_list.size()) {
         throw std::runtime_error("The sizes of key_list and value_list must match each other");
      }
      for (std::size_t i = 0; i < key_list.size(); ++i) {
         std::vector<IndexType> copied_key = key_list[i];
         AddInteraction(copied_key, value_list[i], vartype);
      }
   }
   
   //! @brief Add specified value to the offset of the BinaryPolynomialModel.
   //! @param offset
   void AddOffset(FloatType offset) {
      AddInteraction(std::vector<IndexType>{}, offset);
   }
   
   //! @brief Determine the energy of the specified sample of the BinaryPolynomialModel.
   //! @details When omp_flag is true, the OpenMP is used to calculate the energy in parallel.
   //! @param sample
   //! @param omp_flag
   //! @return An energy with respect to the sample.
   FloatType Energy(const Sample<IndexType> &sample, bool omp_flag = true) const {
      if (sample.size() != GetNumVariables()) {
         throw std::runtime_error("The size of sample must be equal to num_variables");
      }
      
      if (GetNumInteractions() == 0) {
         return 0.0;
      }
      
      std::size_t num_interactions = GetNumInteractions();
      FloatType val = 0.0;
      
      if (omp_flag) {
#pragma omp parallel for reduction (+: val)
         for (int64_t i = 0; i < (int64_t)num_interactions; ++i) {
            int32_t spin_multiple = 1;
            for (const auto &index: poly_key_list_[i]) {
               spin_multiple *= sample.at(index);
               if (spin_multiple == 0.0) {
                  break;
               }
            }
            val += spin_multiple*poly_value_list_[i];
         }
      }
      else {
         for (std::size_t i = 0; i < num_interactions; ++i) {
            int32_t spin_multiple = 1;
            for (const auto &index: poly_key_list_[i]) {
               spin_multiple *= sample.at(index);
               if (spin_multiple == 0.0) {
                  break;
               }
            }
            val += spin_multiple*poly_value_list_[i];
         }
      }
      return val;
   }
   
   //! @brief Determine the energy of the specified sample_vec (as std::vector) of the BinaryPolynomialModel.
   //! @details When omp_flag is true, the OpenMP is used to calculate the energy in parallel.
   //! @param sample_vec
   //! @param omp_flag
   //! @return An energy with respect to the sample.
   FloatType Energy(const std::vector<int32_t> &sample_vec, bool omp_flag = true) {
      if (sample_vec.size() != GetNumVariables()) {
         throw std::runtime_error("The size of sample must be equal to num_variables");
      }
      
      if (GetNumInteractions() == 0) {
         return 0.0;
      }
      
      if (relabel_flag_for_variables_to_integers_) {
         UpdateVariablesToIntegers();
      }
      
      std::size_t num_interactions = GetNumInteractions();
      FloatType val = 0.0;
      
      if (omp_flag) {
#pragma omp parallel for reduction (+: val)
         for (int64_t i = 0; i < (int64_t)num_interactions; ++i) {
            int32_t spin_multiple = 1;
            for (const auto &index: poly_key_list_[i]) {
               spin_multiple *= sample_vec[variables_to_integers_.at(index)];
               if (spin_multiple == 0.0) {
                  break;
               }
            }
            val += spin_multiple*poly_value_list_[i];
         }
      }
      else {
         for (std::size_t i = 0; i < num_interactions; ++i) {
            int32_t spin_multiple = 1;
            for (const auto &index: poly_key_list_[i]) {
               spin_multiple *= sample_vec[variables_to_integers_.at(index)];
               if (spin_multiple == 0.0) {
                  break;
               }
            }
            val += spin_multiple*poly_value_list_[i];
         }
      }
      return val;
   }
   
   //! @brief Determine the energies of the given samples.
   //! @param samples
   //! @return Energies with respect to the samples as std::vector
   PolynomialValueList<FloatType> Energies(const std::vector<Sample<IndexType>> &samples) const {
      PolynomialValueList<FloatType> val_list(samples.size());
#pragma omp parallel for
      for (int64_t i = 0; i < (int64_t)samples.size(); ++i) {
         val_list[i] = Energy(samples[i], false);
      }
      return val_list;
   }
   
   //! @brief Determine the energies of the given samples_vec.
   //! @param samples_vec
   //! @return Energies with respect to the samples as std::vector
   PolynomialValueList<FloatType> Energies(const std::vector<std::vector<int32_t>> &samples_vec) {
      PolynomialValueList<FloatType> val_list(samples_vec.size());
#pragma omp parallel for
      for (int64_t i = 0; i < (int64_t)samples_vec.size(); ++i) {
         val_list[i] = Energy(samples_vec[i], false);
      }
      return val_list;
   }
   
   //! @brief Multiply by the specified scalar all the values of the interactions of the BinaryPolynomialModel.
   //! @param scalar
   //! @param ignored_interactions
   //! @param ignored_offset
   void Scale(const FloatType scalar,
              const PolynomialKeyList<IndexType> &ignored_interactions = {},
              const bool ignored_offset = false) {
      
      std::size_t num_interactions = GetNumInteractions();
      
      for (std::size_t i = 0; i < num_interactions; ++i) {
         poly_value_list_[i] *= scalar;
      }
      
      FloatType scalar_inv = 1.0/scalar;
      for (const auto &key: ignored_interactions) {
         if (poly_key_inv_.count(key) != 0) {
            poly_value_list_[poly_key_inv_[key]] *= scalar_inv;
         }
      }
      
      if (ignored_offset == true && poly_key_inv_.count(std::vector<IndexType>{}) != 0 && std::count(ignored_interactions.begin(), ignored_interactions.end(), std::vector<IndexType>{}) == 0) {
         poly_value_list_[poly_key_inv_[std::vector<IndexType>{}]] *= scalar_inv;
      }
   }
   
   //! @brief Normalizes the values of the interactions of the BinaryPolynomialModel such that they fall in the provided range(s).
   //! @param range
   //! @param ignored_interactions
   //! @param ignored_offset
   void normalize(const std::pair<FloatType, FloatType> &range = {1.0, 1.0},
                  const PolynomialKeyList<IndexType> &ignored_interactions = {},
                  const bool ignored_offset = false) {
      
      if (GetNumInteractions() == 0) {
         return;
      }
      
      FloatType max_poly_value = poly_value_list_[0];
      FloatType min_poly_value = poly_value_list_[0];
      
      for (const auto &poly_value: poly_value_list_) {
         if (max_poly_value < poly_value) {
            max_poly_value = poly_value;
         }
         if (min_poly_value > poly_value) {
            min_poly_value = poly_value;
         }
      }
      
      FloatType inv_scale = std::max(min_poly_value/range.first, max_poly_value/range.second);
      
      if (inv_scale != 0.0) {
         Scale(1.0/inv_scale, ignored_interactions, ignored_offset);
      }
   }
   
   //! @brief Create a BinaryPolynomialModel with the specified vartype.
   //! @param vartype
   //! @param inplace
   //! @return A new instance of the BinaryPolynomialModel.
   BinaryPolynomialModel ChangeVartype(const Vartype vartype, const bool inplace) {
      
      if (vartype == Vartype::SPIN) {
         if (inplace) {
            *this = ToSpin();
            return *this;
         }
         else {
            return ToSpin();
         }
      }
      else if (vartype == Vartype::BINARY) {
         if (inplace) {
            *this = ToBinary();
            return *this;
         }
         else {
            return ToBinary();
         }
      }
      else {
         throw std::runtime_error("Unknown vartype error");
      }
   }
   
   //! @brief Change the vartype of the BinaryPolynomialModel.
   void ChangeVartype(const Vartype vartype) {
      if (vartype == Vartype::SPIN) {
         *this = ToSpin();
      }
      else if (vartype == Vartype::BINARY) {
         *this = ToBinary();
      }
      else {
         throw std::runtime_error("Unknown vartype error");
      }
   }
   
   //! @brief Check if the specified index is in the BinaryPolynomialModel.
   //! @param index
   //! @return true or false
   bool HasVariable(const IndexType &index) {
      if (variables_.count(index) != 0) {
         return true;
      }
      else {
         return false;
      }
   }
   
   //! @brief Generate the polynomial interactions corresponding to the vartype being BINARY from the BinaryPolynomialModel.
   //! @return The polynomial interaction as std::unordered_map.
   Polynomial<IndexType, FloatType> ToHubo() const {
      if (vartype_ == Vartype::BINARY) {
         return GetPolynomial();
      }
      Polynomial<IndexType, FloatType> poly_map;
      std::size_t num_interactions = GetNumInteractions();
      for (std::size_t i = 0; i < num_interactions; ++i) {
         const std::vector<IndexType> &original_key = poly_key_list_[i];
         const FloatType   original_value           = poly_value_list_[i];
         const std::size_t original_key_size        = original_key.size();
         const std::size_t changed_key_list_size    = IntegerPower(2, original_key_size);
         
         for (std::size_t j = 0; j < changed_key_list_size; ++j) {
            const auto changed_key = GenerateChangedKey(original_key, j);
            int sign = ((original_key_size - changed_key.size())%2 == 0) ? 1.0 : -1.0;
            FloatType changed_value = original_value*IntegerPower(2, changed_key.size())*sign;
            poly_map[changed_key] += changed_value;
            if (poly_map[changed_key] == 0.0) {
               poly_map.erase(changed_key);
            }
         }
      }
      return poly_map;
   }
   
   //! @brief Generate the polynomial interactions corresponding to the vartype being SPIN from the BinaryPolynomialModel.
   //! @return The polynomial interaction as std::unordered_map.
   Polynomial<IndexType, FloatType> ToHising() const {
      if (vartype_ == Vartype::SPIN) {
         return GetPolynomial();
      }
      Polynomial<IndexType, FloatType> poly_map;
      const std::size_t num_interactions = GetNumInteractions();
      for (std::size_t i = 0; i < num_interactions; ++i) {
         const std::vector<IndexType> &original_key = poly_key_list_[i];
         const FloatType   original_value           = poly_value_list_[i];
         const std::size_t original_key_size        = original_key.size();
         const std::size_t changed_key_list_size    = IntegerPower(2, original_key_size);
         const FloatType   changed_value            = original_value*(1.0/changed_key_list_size);
         
         for (std::size_t j = 0; j < changed_key_list_size; ++j) {
            const auto changed_key = GenerateChangedKey(original_key, j);
            poly_map[changed_key] += changed_value;
            if (poly_map[changed_key] == 0.0) {
               poly_map.erase(changed_key);
            }
         }
      }
      return poly_map;
   }
   
   //! @brief Convert the BinaryPolynomialModel to a serializable object
   //! @return An object that can be serialized (nlohmann::json)
   nlohmann::json ToSerializable() const {
      nlohmann::json output;
      if (vartype_ == Vartype::BINARY) {
         output["vartype"] = "BINARY";
      }
      else if (vartype_ == Vartype::SPIN) {
         output["vartype"] = "SPIN";
      }
      else {
         throw std::runtime_error("Variable type must be SPIN or BINARY.");
      }
      
      std::size_t num_interactions = GetNumInteractions();
      PolynomialKeyList<std::size_t> poly_key_distance_list(num_interactions);
      std::vector<IndexType> sorted_variables = GetSortedVariables();
      
#pragma omp parallel for
      for (int64_t i = 0; i < (int64_t)num_interactions; ++i) {
         std::vector<std::size_t> temp;
         for (const auto &it: poly_key_list_[i]) {
            auto it_index = std::lower_bound(sorted_variables.begin(), sorted_variables.end(), it);
            std::size_t index_distance = std::distance(sorted_variables.begin(), it_index);
            temp.push_back(index_distance);
         }
         poly_key_distance_list[i] = temp;
      }
            
      output["variables"]              = sorted_variables;
      output["poly_key_distance_list"] = poly_key_distance_list;
      output["poly_value_list"]        = poly_value_list_;
      output["type"]                   = "BinaryPolynomialModel";
      
      return output;
   }
   
   //! @brief Create a BinaryPolynomialModel instance from a serializable object.
   //! @tparam IndexType_serial
   //! @tparam FloatType_serial
   //! @param input
   //! @return BinaryPolynomialModel instance
   template <typename IndexType_serial = IndexType, typename FloatType_serial = FloatType>
   static BinaryPolynomialModel<IndexType_serial, FloatType_serial> FromSerializable(const nlohmann::json &input) {
      if(input.at("type") != "BinaryPolynomialModel") {
         throw std::runtime_error("Type must be \"BinaryPolynomialModel\".\n");
      }
      Vartype vartype;
      if (input.at("vartype") == "SPIN") {
         vartype = Vartype::SPIN;
      }
      else if (input.at("vartype") == "BINARY") {
         vartype = Vartype::BINARY;
      }
      else {
         throw std::runtime_error("Variable type must be SPIN or BINARY.");
      }
      return BinaryPolynomialModel<IndexType_serial, FloatType_serial>(input["variables"], input["poly_key_distance_list"], input["poly_value_list"], vartype);
   }
   
   //! @brief Create a BinaryPolynomialModel from a Hubo model.
   //! @param poly_map
   //! @return BinaryPolynomialModel instance with the vartype being BINARY.
   static BinaryPolynomialModel FromHubo(const Polynomial<IndexType, FloatType> &poly_map) {
      return BinaryPolynomialModel<IndexType, FloatType>(poly_map, Vartype::BINARY);
   }
   
   //! @brief Create a BinaryPolynomialModel from a Hubo model.
   //! @param key_list
   //! @param value_list
   //! @return BinaryPolynomialModel instance with the vartype being BINARY.
   static BinaryPolynomialModel FromHubo(const PolynomialKeyList<IndexType> &key_list, const PolynomialValueList<FloatType> &value_list) {
      return BinaryPolynomialModel<IndexType, FloatType>(key_list, value_list, Vartype::BINARY);
   }
   
   //! @brief Create a BinaryPolynomialModel from a Hubo model.
   //! @param key_list
   //! @param value_list
   //! @return BinaryPolynomialModel instance with the vartype being BINARY.
   static BinaryPolynomialModel FromHubo(PolynomialKeyList<IndexType> &key_list, const PolynomialValueList<FloatType> &value_list) {
      return BinaryPolynomialModel<IndexType, FloatType>(key_list, value_list, Vartype::BINARY);
   }
   
   //! @brief Create a BinaryPolynomialModel from a higher ordere Ising model.
   //! @param poly_map
   //! @return BinaryPolynomialModel instance with the vartype being SPIN.
   static BinaryPolynomialModel FromHising(const Polynomial<IndexType, FloatType> &poly_map) {
      return BinaryPolynomialModel<IndexType, FloatType>(poly_map, Vartype::SPIN);
   }
   
   //! @brief Create a BinaryPolynomialModel from a higher ordere Ising model.
   //! @param key_list
   //! @param value_list
   //! @return BinaryPolynomialModel instance with the vartype being SPIN.
   static BinaryPolynomialModel FromHising(const PolynomialKeyList<IndexType> &key_list, const PolynomialValueList<FloatType> &value_list) {
      return BinaryPolynomialModel<IndexType, FloatType>(key_list, value_list, Vartype::SPIN);
   }
   
   //! @brief Create a BinaryPolynomialModel from a higher ordere Ising model.
   //! @param key_list
   //! @param value_list
   //! @return BinaryPolynomialModel instance with the vartype being SPIN.
   static BinaryPolynomialModel FromHising(PolynomialKeyList<IndexType> &key_list, const PolynomialValueList<FloatType> &value_list) {
      return BinaryPolynomialModel<IndexType, FloatType>(key_list, value_list, Vartype::SPIN);
   }
   
   
protected:
   
   //! @brief Variable list as std::unordered_set.
   std::unordered_set<IndexType> variables_;
   
   //! @brief The list of the number of the variables appeared in the polynomial interactions as std::unordered_map.
   std::unordered_map<IndexType, std::size_t> each_variable_num_;
   
   //! @brief The correspondence from variables to the integer numbers.
   std::unordered_map<IndexType, int64_t> variables_to_integers_;
   
   //! @brief Sorted variables is represents the correspondence from integer numbers.to the variables.
   std::vector<IndexType> sorted_variables_;
   
   //! @brief If true variable_to_index must be relabeled.
   bool relabel_flag_for_variables_to_integers_ = true;
   
   //! @brief The list of the indices of the polynomial interactions (namely, the list of keys of the polynomial interactions as std::unordered_map) as std::vector<std::vector>>.
   PolynomialKeyList<IndexType> poly_key_list_;
   
   //! @brief The list of the values of the polynomial interactions (namely, the list of values of the polynomial interactions as std::unordered_map) as std::vector.
   PolynomialValueList<FloatType> poly_value_list_;
   
   //! @brief The inverse key list, which indicates the index of the poly_key_list_ and poly_value_list_
   std::unordered_map<std::vector<IndexType>, std::size_t, vector_hash> poly_key_inv_;
   
   //! @brief The model's type. SPIN or BINARY
   Vartype vartype_ = Vartype::NONE;
   
   //! @brief Set key and value.
   //! @details Note that the key is assumed to be sorted.
   //! @param key
   //! @param value
   void SetKeyAndValue(const std::vector<IndexType> &key, const FloatType &value) {
      //key is assumed to be sorted
      if (poly_key_inv_.count(key) == 0) {
         poly_key_inv_[key] = poly_value_list_.size();
         poly_key_list_.push_back(key);
         poly_value_list_.push_back(value);
      }
      else {
         if (poly_value_list_[poly_key_inv_[key]] + value == 0.0) {
            RemoveInteraction(key);
            return;
         }
         poly_value_list_[poly_key_inv_[key]] += value;
      }
      for (const auto &index: key) {
         each_variable_num_[index]++;
         variables_.emplace(index);
         relabel_flag_for_variables_to_integers_ = true;
      }
   }
   
   //! @brief Caluculate the base to the power of exponent (std::pow(base, exponent) is too slow).
   //! @param base
   //! @param exponent
   //! @return The base to the power of exponent
   std::size_t IntegerPower(std::size_t base, std::size_t exponent) const {
      std::size_t val = 1;
      for (std::size_t i = 0; i < exponent; ++i) {
         val *= base;
      }
      return val;
   }
   
   //! @brief Generate the num_of_key-th the key when the vartype is changed.
   //! @param original_key
   //! @param num_of_key
   //! @return The changed key
   std::vector<IndexType> GenerateChangedKey(const std::vector<IndexType> &original_key, const std::size_t num_of_key) const {
      if (original_key.size() >= UINT16_MAX) {
         throw std::runtime_error("Too large degree of the interaction");
      }
      const std::size_t original_key_size = original_key.size();
      std::bitset<UINT16_MAX> bs(num_of_key);
      std::vector<IndexType> changed_key;
      for (std::size_t i = 0; i < original_key_size; ++i) {
         if (bs[i]) {
            changed_key.push_back(original_key[i]);
         }
      }
      return changed_key;
   }
   
   //! @brief Generate BinaryPolynomialModel with the vartype being SPIN.
   //! @return BinaryPolynomialModel instance with the vartype being SPIN.
   BinaryPolynomialModel ToSpin() const {
      if (vartype_ == Vartype::SPIN) {
         return *this;
      }
      return BinaryPolynomialModel(ToHising(), Vartype::SPIN);
   }
   
   //! @brief Generate BinaryPolynomialModel with the vartype being BINARY.
   //! @return BinaryPolynomialModel instance with the vartype being BINARY.
   BinaryPolynomialModel ToBinary() const {
      if (vartype_ == Vartype::BINARY) {
         return *this;
      }
      return BinaryPolynomialModel(ToHubo(), Vartype::BINARY);
   }

   //! @brief Update sorted_variables_ and variables_to_integers_
   void UpdateVariablesToIntegers() {
      sorted_variables_ = GenerateSortedVariables();
      variables_to_integers_.clear();
      for (std::size_t i = 0; i < sorted_variables_.size(); ++i) {
         variables_to_integers_[sorted_variables_[i]] = i;
      }
      relabel_flag_for_variables_to_integers_ = false;
   }
   
   //! @brief Generate variables_to_integers
   //! @return variables_to_integers
   std::unordered_map<IndexType, int64_t> GenerateVariablesToIntegers() const {
      std::vector<IndexType> sorted_variables = GenerateSortedVariables();
      std::unordered_map<IndexType, int64_t> variables_to_integers;
      for (std::size_t i = 0; i < sorted_variables.size(); ++i) {
         variables_to_integers[sorted_variables[i]] = i;
      }
      return variables_to_integers;
   }
   
   //! @brief Generate sorted variables
   //! @return sorted_variables
   std::vector<IndexType> GenerateSortedVariables() const {
      std::vector<IndexType> sorted_variables(variables_.begin(), variables_.end());
      std::sort(sorted_variables.begin(), sorted_variables.end());
      return sorted_variables;
   }
   
};

}


#endif /* binary_polynomial_model_hpp */
