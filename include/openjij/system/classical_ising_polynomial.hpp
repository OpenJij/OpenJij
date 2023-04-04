//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <nlohmann/json.hpp>

#include "openjij/graph/all.hpp"
#include "openjij/graph/json/parse.hpp"
#include "openjij/utility/thres_hold.hpp"

namespace openjij {
namespace system {

//! @brief ClassicalIsingPolynomial class, which is a system to solve higher
//! order unconstrained binary optimization (HUBO) problems with vartype being
//! "SPIN" or "BINARY"
//! @tparam GraphType type of graph
template <typename GraphType> class ClassicalIsingPolynomial;

//! @brief ClassicalIsingPolynomial class
template <typename FloatType>
class ClassicalIsingPolynomial<graph::Polynomial<FloatType>> {

public:
  //! @brief system type
  using system_type = classical_system;

  //! @brief The number of spins/binaries
  const int64_t num_variables;

  //! @brief Spin/binary configurations
  graph::Spins variables;

  //! @brief The model's type. SPIN or BINARY
  const cimod::Vartype vartype;

  //! @brief Constructor of ClassicalIsingPolynomial
  //! @param init_variables graph::Spins& or graph::Binaries& (both are equal).
  //! The initial spin/binary configurations.
  //! @param poly_graph graph::Polynomial<FloatType>& (Polynomial graph class).
  //! The initial interacrtions.
  //! @param init_vartype const cimod::Vartype. The model's variable type. SPIN
  //! or BINARY.
  ClassicalIsingPolynomial(const graph::Spins &init_variables,
                           const graph::Polynomial<FloatType> &poly_graph,
                           const cimod::Vartype init_vartype)
      : num_variables(poly_graph.size()), variables(init_variables),
        vartype(init_vartype) {
    cimod::CheckVariables(variables, vartype);
    SetInteractions(poly_graph);
    SetAdj();
    ResetZeroCount();
    ResetSignKey();
    reset_dE();
    const FloatType thres_hold =
        std::abs(FindMaxInteraction().second * utility::THRESHOLD<FloatType>);
    min_effective_dE_ = std::abs(FindMinInteraction(thres_hold).second);
  }

  //! @brief Constructor of ClassicalIsingPolynomial
  //! @param init_variables graph::Spins& or graph::Binaries& (both are equal).
  //! The initial spin/binary configurations.
  //! @param poly_graph graph::Polynomial<FloatType>& (Polynomial graph class).
  //! The initial interacrtions.
  //! @param init_vartype const std::string. The model's variable type. "SPIN"
  //! or "BINARY".
  ClassicalIsingPolynomial(const graph::Spins &init_variables,
                           const graph::Polynomial<FloatType> &poly_graph,
                           const std::string init_vartype)
      : num_variables(poly_graph.size()), variables(init_variables),
        vartype(ConvertVartype(init_vartype)) {
    cimod::CheckVariables(variables, vartype);
    SetInteractions(poly_graph);
    SetAdj();
    ResetZeroCount();
    ResetSignKey();
    reset_dE();
    const FloatType thres_hold =
        std::abs(FindMaxInteraction().second * utility::THRESHOLD<FloatType>);
    min_effective_dE_ = std::abs(FindMinInteraction(thres_hold).second);
  }

  //! @brief Constructor of ClassicalIsingPolynomial
  //! @param init_variables graph::Spins& or graph::Binaries& (both are equal).
  //! The initial spin/binary configurations.
  //! @param j const nlohmann::json object
  ClassicalIsingPolynomial(const graph::Spins &init_variables,
                           const nlohmann::json &j)
      : num_variables(init_variables.size()), variables(init_variables),
        vartype(j.at("vartype") == "SPIN" ? cimod::Vartype::SPIN
                                          : cimod::Vartype::BINARY) {
    cimod::CheckVariables(variables, vartype);
    const auto &v_k_v = graph::json_parse_polynomial<FloatType>(j);
    const auto &poly_key_list = std::get<0>(v_k_v);
    const auto &poly_value_list = std::get<1>(v_k_v);

    if (poly_key_list.size() != poly_value_list.size()) {
      throw std::runtime_error(
          "The sizes of key_list and value_list must match each other");
    }
    if (poly_key_list.size() == 0) {
      throw std::runtime_error("The interaction is empty.");
    }
    if (num_variables == 0) {
      throw std::runtime_error("The number of variables is zero.");
    }

    num_interactions_ = static_cast<int64_t>(poly_key_list.size());

    poly_key_list_.resize(num_interactions_);
    poly_value_list_.resize(num_interactions_);

#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions_; ++i) {
      poly_key_list_[i] = poly_key_list[i];
      poly_value_list_[i] = poly_value_list[i];
    }

    active_variables_.resize(num_variables);
    std::iota(active_variables_.begin(), active_variables_.end(), 0);

    SetAdj();
    ResetZeroCount();
    ResetSignKey();
    reset_dE();
    const FloatType thres_hold =
        std::abs(FindMaxInteraction().second * utility::THRESHOLD<FloatType>);
    min_effective_dE_ = std::abs(FindMinInteraction(thres_hold).second);
  }

  //! @brief Reset ClassicalIsingPolynomial system with new spin/binary
  //! configurations.
  //! @param init_variables const graph::Spins& or const graph::Binaries (both
  //! are equal).
  void reset_variables(const graph::Spins &init_variables) {
    if (init_variables.size() != variables.size()) {
      throw std::runtime_error(
          "The size of initial spins/binaries does not equal to system size");
    }
    cimod::CheckVariables(init_variables, vartype);
    if (vartype == cimod::Vartype::SPIN) {
      for (const auto &index_variable : active_variables_) {
        if (variables[index_variable] != init_variables[index_variable]) {
          update_spin_system(index_variable);
        }
        if (variables[index_variable] != init_variables[index_variable]) {
          std::stringstream ss;
          ss << "Unknown error detected in " << __func__;
          throw std::runtime_error(ss.str());
        }
      }
    } else if (vartype == cimod::Vartype::BINARY) {
      for (const auto &index_variable : active_variables_) {
        if (variables[index_variable] != init_variables[index_variable]) {
          update_binary_system(index_variable);
        }
        if (variables[index_variable] != init_variables[index_variable]) {
          std::stringstream ss;
          ss << "Unknown error detected in " << __func__;
          throw std::runtime_error(ss.str());
        }
      }
    } else {
      throw std::runtime_error("Unknown vartype detected");
    }
  }

  //! @brief Reset energy differences (dE), which is used to determine whether
  //! to flip the spin/binary or not.
  void reset_dE() {
    dE_.clear();
    dE_.resize(num_variables);

    if (vartype == cimod::Vartype::SPIN) {
      // Initialize
      max_effective_dE_ = 2.0 * std::abs(poly_value_list_.front());
      for (const auto &index_binary : active_variables_) {
        FloatType val = 0.0;
        FloatType abs_val = 0.0;
        bool flag = false;
        for (const auto &index_key : adj_[index_binary]) {
          val += poly_value_list_[index_key] * sign_key_[index_key];
          abs_val += std::abs(poly_value_list_[index_key]);
          flag = true;
        }
        dE_[index_binary] = -2 * val;
        if (flag && max_effective_dE_ < 2 * abs_val) {
          max_effective_dE_ = 2 * abs_val;
        }
      }
    } else if (vartype == cimod::Vartype::BINARY) {
      // Initialize
      max_effective_dE_ = std::abs(poly_value_list_.front());
      for (const auto &index_binary : active_variables_) {
        FloatType val = 0.0;
        FloatType abs_val = 0.0;
        bool flag = false;
        const graph::Binary binary = variables[index_binary];
        for (const auto &index_key : adj_[index_binary]) {
          if (zero_count_[index_key] + binary == 1) {
            val += poly_value_list_[index_key];
          }
          flag = true;
          abs_val += std::abs(poly_value_list_[index_key]);
        }
        dE_[index_binary] = (-2 * binary + 1) * val;

        if (flag && max_effective_dE_ < abs_val) {
          max_effective_dE_ = abs_val;
        }
      }
    } else {
      throw std::runtime_error("Unknown vartype detected");
    }
  }

  //! @brief Flip specified spin by single spin flip. Note that this function is
  //! used when the model's type is SPIN.
  //! @param index_update_spin const graph::Index.
  void update_spin_system(const graph::Index index_update_spin) {
    for (const auto &index_key : adj_[index_update_spin]) {
      const FloatType val = 4.0 * poly_value_list_[index_key];
      const int8_t sign = sign_key_[index_key];
      for (const auto &index_spin : poly_key_list_[index_key]) {
        if (index_spin != index_update_spin) {
          dE_[index_spin] += val * sign;
        }
      }
      sign_key_[index_key] *= -1;
    }
    dE_[index_update_spin] *= -1;
    variables[index_update_spin] *= -1;
  }

  //! @brief Flip specified binary by single spin flip. Note that this function
  //! is used when the model's type is BINARY.
  //! @param index_update_binary const graph::Index.
  void update_binary_system(const graph::Index index_update_binary) {
    const graph::Binary update_binary = variables[index_update_binary];
    const int coeef = -2 * update_binary + 1;
    const int count = +2 * update_binary - 1;
    for (const auto &index_key : adj_[index_update_binary]) {
      const FloatType val = poly_value_list_[index_key];
      for (const auto &index_binary : poly_key_list_[index_key]) {
        const graph::Binary binary = variables[index_binary];
        if (zero_count_[index_key] + update_binary + binary == 2 &&
            index_binary != index_update_binary) {
          dE_[index_binary] += coeef * (-2 * binary + 1) * val;
        }
      }
      zero_count_[index_key] += count;
    }
    dE_[index_update_binary] *= -1;
    variables[index_update_binary] = 1 - variables[index_update_binary];
  }

  //! @brief Return the energy difference of single spin flip update.
  //! @param index_variable const graph::Index.
  //! @return the energy difference corresponding to "index_variable".
  inline FloatType dE(const graph::Index index_variable) const {
    return dE_[index_variable];
  }

  //! @brief Get "active_binaries_", which is the list of the binaries connected
  //! by at least one interaction.
  //! @return active_binaries_
  inline const std::vector<graph::Index> &get_active_variables() const {
    return active_variables_;
  }

  //! @brief Get the PolynomialValueList object, which is the list of the values
  //! of the polynomial interactions as std::vector<FloatType>.
  //! @return "poly_value_list_"
  const cimod::PolynomialValueList<FloatType> &get_values() const {
    return poly_value_list_;
  }

  //! @brief Get the PolynomialKeyList object, which is the list of the indices
  //! of the polynomial interactions as std::vector<std::vector<graph::Index>>.
  //! @return "poly_key_list_"
  const cimod::PolynomialKeyList<graph::Index> &get_keys() const {
    return poly_key_list_;
  }

  //! @brief Get the adjacency list, which is the list of the indices of
  //! polynomial interactions including specific spin/binary.
  //! @return adjacency list
  const std::vector<std::vector<graph::Index>> &get_adj() const { return adj_; }

  //! @brief Get "max_effective_dE", which is a upper bound of energy gap.
  //! @return max_effective_dE
  FloatType get_max_effective_dE() const { return max_effective_dE_; }

  //! @brief Get "min_effective_dE", which is a rough lower bound of energy gap.
  //! @return min_effective_dE
  FloatType get_min_effective_dE() const { return min_effective_dE_; }

  //! @brief Get the vartype as std::string.
  //! @return The model's type as std::string
  std::string get_vartype_string() const {
    if (vartype == cimod::Vartype::SPIN) {
      return "SPIN";
    } else if (vartype == cimod::Vartype::BINARY) {
      return "BINARY";
    } else {
      throw std::runtime_error("Unknown vartype detected");
    }
  }

private:
  //! @brief The number of the interactions.
  int64_t num_interactions_;

  //! @brief The energy differences when flipping a spin/binary.
  std::vector<FloatType> dE_;

  //! @brief The number of variables taking the zero in each interaction. Note
  //! that this variable is used when the model's type is BINARY.
  std::vector<int64_t> zero_count_;

  //! @brief The sign of product of spin variables in each interaction. Note
  //! that this variable is used when the model's type is SPIN.
  std::vector<int8_t> sign_key_;

  //! @brief Adjacency list, which is the list of the indices of polynomial
  //! interactions including specific spin/binary.
  std::vector<std::vector<graph::Index>> adj_;

  //! @brief The list of the indices of the polynomial interactions as
  //! std::vector<std::vector<graph::Index>>.
  cimod::PolynomialKeyList<graph::Index> poly_key_list_;

  //! @brief The list of the values of the polynomial interactions as
  //! std::vector<FloatType>.
  cimod::PolynomialValueList<FloatType> poly_value_list_;

  //! @brief The list of the binaries connected by at least one interaction.
  std::vector<graph::Index> active_variables_;

  //! @brief Upper bound of energy gap.
  FloatType max_effective_dE_;

  //! @brief Rough lower bound of energy gap.
  FloatType min_effective_dE_;

  //! @brief Set adjacency list.
  void SetAdj() {
    adj_.clear();
    adj_.resize(num_variables);
    for (int64_t i = 0; i < num_interactions_; ++i) {
      for (const auto &index : poly_key_list_[i]) {
        adj_[index].push_back(i);
      }
    }
  }

  //! @brief Set interactions from Polynomial graph.
  //! @param poly_graph const graph::Polynomial<FloatType>&.
  void SetInteractions(const graph::Polynomial<FloatType> &poly_graph) {
    const auto &poly_key_list = poly_graph.get_keys();
    const auto &poly_value_list = poly_graph.get_values();

    if (poly_key_list.size() != poly_value_list.size()) {
      throw std::runtime_error(
          "The sizes of key_list and value_list must match each other");
    }

    if (poly_key_list.size() == 0) {
      throw std::runtime_error("The interaction is empty.");
    }

    std::unordered_set<graph::Index> active_variable_set;

    poly_key_list_.clear();
    poly_value_list_.clear();

    for (std::size_t i = 0; i < poly_key_list.size(); ++i) {
      if (poly_value_list[i] != 0.0) {
        poly_key_list_.push_back(poly_key_list[i]);
        poly_value_list_.push_back(poly_value_list[i]);
        for (const auto &it : poly_key_list[i]) {
          active_variable_set.emplace(it);
        }
      }
    }
    num_interactions_ = static_cast<int64_t>(poly_key_list_.size());
    active_variables_ = std::vector<graph::Index>(active_variable_set.begin(),
                                                  active_variable_set.end());
    std::sort(active_variables_.begin(), active_variables_.end());
  }

  //! @brief Set zero_count_.
  void ResetZeroCount() {
    if (vartype != cimod::Vartype::BINARY) {
      return;
    }
    zero_count_.resize(num_interactions_);
#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions_; ++i) {
      int64_t zero_count = 0;
      for (const auto &index : poly_key_list_[i]) {
        if (variables[index] == 0) {
          zero_count++;
        }
      }
      zero_count_[i] = zero_count;
    }
  }

  //! @brief Set sign_key_.
  void ResetSignKey() {
    if (vartype != cimod::Vartype::SPIN) {
      return;
    }
    sign_key_.resize(num_interactions_);
#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions_; ++i) {
      int8_t sign_key = 1;
      for (const auto &index : poly_key_list_[i]) {
        sign_key *= variables[index];
      }
      sign_key_[i] = sign_key;
    }
  }

  //! @brief Convert vartype from std::string to cimod::Vartype
  //! @param vartype const std::string
  cimod::Vartype ConvertVartype(const std::string vartype) const {
    if (vartype == "SPIN") {
      return cimod::Vartype::SPIN;
    } else if (vartype == "BINARY") {
      return cimod::Vartype::BINARY;
    } else {
      throw std::runtime_error("Unknown vartype detected");
    }
  }

  //! @brief Return the key and value of the absolute maximum interaction.
  //! @return The key and value of the absolute maximum interaction as
  //! std::pair.
  std::pair<std::vector<graph::Index>, FloatType> FindMaxInteraction() const {
    if (poly_key_list_.size() == 0) {
      throw std::runtime_error("Interactions are empty.");
    }
    FloatType max_val = 0.0;
    std::vector<graph::Index> max_key = {};
    for (std::size_t i = 0; i < poly_key_list_.size(); ++i) {
      if (std::abs(max_val) < std::abs(poly_value_list_[i])) {
        max_val = poly_value_list_[i];
        max_key = poly_key_list_[i];
      }
    }
    return std::pair<std::vector<graph::Index>, FloatType>(max_key, max_val);
  }

  //! @brief Return the key and value of the absolute minimum interaction.
  //! @return The key and value of the absolute minimum interaction as
  //! std::pair.
  std::pair<std::vector<graph::Index>, FloatType>
  FindMinInteraction(const FloatType threshold = 0.0) const {
    if (poly_key_list_.size() == 0) {
      throw std::runtime_error("Interactions are empty.");
    }
    FloatType min_val = 0.0;
    std::vector<graph::Index> min_key;

    // Set initial value larger than threshold.
    bool flag_success_initialize = false;
    for (std::size_t i = 0; i < poly_key_list_.size(); ++i) {
      if (std::abs(threshold) < std::abs(poly_value_list_[i])) {
        min_val = poly_value_list_[i];
        min_key = poly_key_list_[i];
        flag_success_initialize = true;
        break;
      }
    }
    if (!flag_success_initialize) {
      std::stringstream ss;
      ss << "No interactions larger than threshold=" << std::abs(threshold)
         << std::endl;
      throw std::runtime_error(ss.str());
    }

    for (std::size_t i = 0; i < poly_key_list_.size(); ++i) {
      if (std::abs(threshold) < std::abs(poly_value_list_[i]) &&
          std::abs(poly_value_list_[i]) < std::abs(min_val)) {
        min_val = poly_value_list_[i];
        min_key = poly_key_list_[i];
      }
    }

    if (std::abs(min_val) <= std::abs(threshold)) {
      std::stringstream ss;
      ss << "Unknown error in " << __func__ << std::endl;
      throw std::runtime_error(ss.str());
    }
    return std::pair<std::vector<graph::Index>, FloatType>(min_key, min_val);
  }
};

//! @brief Helper function for ClassicalIsingPolynomial constructor
//! @tparam GraphType
//! @param init_variables const graph::Spins& or const graph::Binaries& (both
//! are equal). The initial spin/binarie configulations.
//! @param poly_graph graph::Polynomial<FloatType>& (Polynomial graph class).
//! The initial interacrtions.
//! @param init_vartype const cimod::Vartype. The model's variable type. SPIN or
//! BINARY.
template <typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins &init_variables,
                                     const GraphType &poly_graph,
                                     const cimod::Vartype init_vartype) {
  return ClassicalIsingPolynomial<GraphType>(init_variables, poly_graph,
                                             init_vartype);
}

//! @brief Helper function for ClassicalIsingPolynomial constructor
//! @tparam GraphType
//! @param init_variables const graph::Spins& or const graph::Binaries& (both
//! are equal). The initial spin/binarie configulations.
//! @param poly_graph graph::Polynomial<FloatType>& (Polynomial graph class).
//! The initial interacrtions.
//! @param init_vartype std::string. The model's variable type. SPIN or BINARY.
template <typename GraphType>
auto make_classical_ising_polynomial(const graph::Spins &init_variables,
                                     const GraphType &poly_graph,
                                     const std::string init_vartype) {
  return ClassicalIsingPolynomial<graph::Polynomial<double>>(
      init_variables, poly_graph, init_vartype);
}

//! @brief Helper function for ClassicalIsingPolynomial constructor by using
//! nlohmann::json object
//! @param init_variables const graph::Spins& or const graph::Binaries& (both
//! are equal). The initial spin/binarie configulations.
//! @param init_obj nlohmann::json&
auto make_classical_ising_polynomial(const graph::Spins &init_variables,
                                     const nlohmann::json &init_obj) {
  return ClassicalIsingPolynomial<graph::Polynomial<double>>(init_variables,
                                                             init_obj);
}

} // namespace system
} // namespace openjij
