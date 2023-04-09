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

#include <sstream>

#include <nlohmann/json.hpp>

#include "openjij/graph/all.hpp"
#include "openjij/graph/json/parse.hpp"
#include "openjij/utility/thres_hold.hpp"

namespace openjij {
namespace system {

//! @brief KLocalPolynomial class, which is a system to solve higher order
//! unconstrained binary optimization (HUBO) problems with vartype being
//! "BINARY"
//! @tparam GraphType type of graph
template <class GraphType> class KLocalPolynomial;

//! @brief KLocalPolynomial class
template <typename FloatType>
class KLocalPolynomial<graph::Polynomial<FloatType>> {

public:
  //! @brief system type
  using system_type = classical_system;

  //! @brief The number of binaries
  const int64_t num_binaries;

  //! @brief k-local  update is activated per rate_call_k_local times
  int rate_call_k_local = 10;

  //! @brief Counter of calling updater
  int64_t count_call_updater = 0;

  //! @brief Binary configurations
  graph::Binaries binaries;

  //! @brief The model's type. SPIN or BINARY
  const cimod::Vartype vartype = cimod::Vartype::BINARY;

  //! @brief Constructor of KLocalPolynomial system class
  //! @param init_binaries const graph::Binaries&. The initial binary
  //! configurations.
  //! @param poly_graph const graph::Polynomial<FloatType>& (Polynomial graph
  //! class). The initial interacrtions.
  KLocalPolynomial(const graph::Binaries &init_binaries,
                   const graph::Polynomial<FloatType> &poly_graph)
      : num_binaries(init_binaries.size()), binaries(init_binaries),
        binaries_v_(init_binaries) {

    cimod::CheckVariables(binaries, vartype);

    const auto &poly_key_list = poly_graph.get_keys();
    const auto &poly_value_list = poly_graph.get_values();

    if (poly_key_list.size() != poly_value_list.size()) {
      throw std::runtime_error(
          "The sizes of key_list and value_list must match each other");
    }
    if (poly_key_list.size() == 0) {
      throw std::runtime_error("The interaction is empty.");
    }

    std::unordered_set<graph::Index> active_binary_set;

    poly_key_list_.clear();
    poly_value_list_.clear();

    for (std::size_t i = 0; i < poly_key_list.size(); ++i) {
      if (poly_value_list[i] != 0.0) {
        poly_key_list_.push_back(poly_key_list[i]);
        poly_value_list_.push_back(poly_value_list[i]);
        for (const auto &it : poly_key_list[i]) {
          active_binary_set.emplace(it);
        }
      }
    }
    num_interactions_ = static_cast<int64_t>(poly_key_list_.size());
    SortInteractions();
    active_binaries_ = std::vector<graph::Index>(active_binary_set.begin(),
                                                 active_binary_set.end());
    std::sort(active_binaries_.begin(), active_binaries_.end());
    SetAdj();
    ResetZeroCount();
    reset_dE();
    const FloatType thres_hold =
        std::abs(FindMaxInteraction().second * utility::THRESHOLD<FloatType>);
    min_effective_dE_ = std::abs(FindMinInteraction(thres_hold).second);
  }

  //! @brief Constructor of KLocalPolynomial system class.
  //! @param init_binaries const graph::Binaries&. The initial binary
  //! configurations.
  //! @param j const nlohmann::json&
  KLocalPolynomial(const graph::Binaries &init_binaries,
                   const nlohmann::json &j)
      : num_binaries(init_binaries.size()), binaries(init_binaries),
        binaries_v_(init_binaries) {

    cimod::CheckVariables(binaries, vartype);

    if (j.at("vartype") != "BINARY") {
      throw std::runtime_error("Only binary variables are supported");
    }

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

    num_interactions_ = static_cast<int64_t>(poly_key_list.size());

    poly_key_list_.resize(num_interactions_);
    poly_value_list_.resize(num_interactions_);

#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions_; ++i) {
      poly_key_list_[i] = poly_key_list[i];
      poly_value_list_[i] = poly_value_list[i];
    }
    SortInteractions();

    active_binaries_.resize(num_binaries);
    std::iota(active_binaries_.begin(), active_binaries_.end(), 0);

    SetAdj();
    ResetZeroCount();
    reset_dE();
    const FloatType thres_hold =
        std::abs(FindMaxInteraction().second * utility::THRESHOLD<FloatType>);
    min_effective_dE_ = std::abs(FindMinInteraction(thres_hold).second);
  }

  //! @brief Reset KLocalPolynomial system with new binary configurations.
  //! @param init_binaries const graph::Binaries&
  void reset_binaries(const graph::Binaries &init_binaries) {

    cimod::CheckVariables(init_binaries, vartype);

    if (init_binaries.size() != binaries.size()) {
      throw std::runtime_error(
          "The size of initial binaries does not equal to system size");
    }
    for (const auto &index_binary : active_binaries_) {
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

  //! @brief Reset energy differences (dE), which is used to determine whether
  //! to flip the binary or not.
  void reset_dE() {
    dE_.clear();
    dE_v_.clear();
    dE_.resize(num_binaries);
    dE_v_.resize(num_binaries);

    // Initialize
    max_effective_dE_ = std::abs(poly_value_list_.front());

    for (const auto &index_binary : active_binaries_) {
      FloatType val = 0.0;
      FloatType abs_val = 0.0;
      bool flag = false;
      const graph::Binary binary = binaries[index_binary];
      for (const auto &index_key : adj_[index_binary]) {
        if (zero_count_[index_key] + binary == 1) {
          val += poly_value_list_[index_key];
        }
        flag = true;
        abs_val += std::abs(poly_value_list_[index_key]);
      }
      dE_[index_binary] = (-2 * binary + 1) * val;
      dE_v_[index_binary] = dE_[index_binary];

      if (flag && max_effective_dE_ < abs_val) {
        max_effective_dE_ = abs_val;
      }
    }
  }

  //! @brief Return the energy difference of single spin flip update
  //! @param index_binary const graph::Index
  //! @return the energy difference corresponding to "index_binary"
  inline FloatType dE_single(const graph::Index index_binary) const {
    return dE_[index_binary];
  }

  //! @brief Return the energy difference of k-local update
  //! @details Note that this function changes the internal state of
  //! KLocalPolynomial system. This function virtually update the system by
  //! k-local update.
  //! @param index_key const graph::Index
  //! @return the energy difference corresponding to "index_key"
  FloatType dE_k_local(const std::size_t index_key) {
    FloatType dE = 0.0;
    for (const auto &index_binary : poly_key_list_[index_key]) {
      if (binaries_v_[index_binary] == 0) {
        dE += dE_v_[index_binary];
        virtual_update_system_single(index_binary);
      }
      if (dE < 0.0) {
        break;
      }
    }
    return dE;
  }

  //! @brief Update binary configurations by k-local update
  void update_system_k_local() {
    for (const auto &index_binary : update_index_binaries_v_) {
      binaries[index_binary] = binaries_v_[index_binary];
    }
    for (const auto &index_zero_count : update_index_zero_count_v_) {
      zero_count_[index_zero_count] = zero_count_v_[index_zero_count];
    }
    for (const auto &index_dE : update_index_dE_v_) {
      dE_[index_dE] = dE_v_[index_dE];
    }
    update_index_binaries_v_.clear();
    update_index_zero_count_v_.clear();
    update_index_dE_v_.clear();
  }

  //! @brief Flip specified binary by single spin flip.
  //! @param index_update_binary const graph::Index
  void update_system_single(const graph::Index index_update_binary) {
    const graph::Binary update_binary = binaries[index_update_binary];
    const int coeef = -2 * update_binary + 1;
    const int count = +2 * update_binary - 1;
    for (const auto &index_key : adj_[index_update_binary]) {
      const FloatType val = poly_value_list_[index_key];
      for (const auto &index_binary : poly_key_list_[index_key]) {
        const graph::Binary binary = binaries[index_binary];
        if (zero_count_[index_key] + update_binary + binary == 2 &&
            index_binary != index_update_binary) {
          dE_[index_binary] += coeef * (-2 * binary + 1) * val;
          dE_v_[index_binary] = dE_[index_binary];
        }
      }
      zero_count_[index_key] += count;
      zero_count_v_[index_key] = zero_count_[index_key];
    }
    dE_[index_update_binary] *= -1;
    dE_v_[index_update_binary] = dE_[index_update_binary];
    binaries[index_update_binary] = 1 - binaries[index_update_binary];
    binaries_v_[index_update_binary] = binaries[index_update_binary];
  }

  //! @brief Virtually flip specified binary by single spin flip.
  //! @param index_update_binary const graph::Index.
  void virtual_update_system_single(const graph::Index index_update_binary) {
    const graph::Binary update_binary = binaries_v_[index_update_binary];
    const int coeef = -2 * update_binary + 1;
    const int count = +2 * update_binary - 1;
    for (const auto &index_key : adj_[index_update_binary]) {
      const FloatType val = poly_value_list_[index_key];
      for (const auto &index_binary : poly_key_list_[index_key]) {
        const graph::Binary binary = binaries_v_[index_binary];
        if (zero_count_v_[index_key] + update_binary + binary == 2 &&
            index_binary != index_update_binary) {
          dE_v_[index_binary] += coeef * (-2 * binary + 1) * val;
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

  //! @brief Reset binary configurations virtually updated by k-local update.
  void reset_virtual_system() {
    for (const auto &index_binary : update_index_binaries_v_) {
      binaries_v_[index_binary] = binaries[index_binary];
    }
    for (const auto &index_zero_count : update_index_zero_count_v_) {
      zero_count_v_[index_zero_count] = zero_count_[index_zero_count];
    }
    for (const auto &index_dE : update_index_dE_v_) {
      dE_v_[index_dE] = dE_[index_dE];
    }
    update_index_binaries_v_.clear();
    update_index_zero_count_v_.clear();
    update_index_dE_v_.clear();
  }

  //! @brief Set "rate_call_k_local". k-local update is activated per
  //! rate_call_k_local times.
  //! @param rate_k_local int.
  void set_rate_call_k_local(int rate_k_local) {
    if (rate_k_local <= 0) {
      throw std::runtime_error("rate_k_local must be larger than zero");
    }
    rate_call_k_local = rate_k_local;
  }

  //! @brief Get the number of interactions.
  inline int64_t GetNumInteractions() const { return num_interactions_; }

  //! @brief Return the number of variables taking the zero in the interaction
  //! specified by "index_key".
  //! @param index_key const std::size_t
  //! @return Corresponding number of variables taking the zero.
  inline int64_t GetZeroCount(const std::size_t index_key) const {
    return zero_count_[index_key];
  }

  //! @brief Get the value of the interaction specified by "index_key".
  //! @param index_key const std::size_t
  //! @return Corresponding value.
  inline FloatType GetPolyValue(const std::size_t index_key) const {
    return poly_value_list_[index_key];
  }

  //! @brief Get the adjacency list (the index of interactions) of the binary
  //! specified by "index_binary".
  //! @param index_binary const std::size_t
  //! @return Corresponding adjacency list.
  inline const std::vector<graph::Index> &
  get_adj(const std::size_t index_binary) const {
    return adj_[index_binary];
  }

  //! @brief Get "active_binaries_", which is the list of the binaries connected
  //! by at least one interaction.
  //! @return active_binaries_
  inline const std::vector<graph::Index> &get_active_binaries() const {
    return active_binaries_;
  }

  //! @brief Get "max_effective_dE_", which is a upper bound of energy gap.
  //! @return max_effective_dE_
  FloatType get_max_effective_dE() const { return max_effective_dE_; }

  //! @brief Get "min_effective_dE_", which is a rough lower bound of energy
  //! gap.
  //! @return min_effective_dE_
  FloatType get_min_effective_dE() const { return min_effective_dE_; }

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
  //! polynomial interactions including specific binary.
  //! @return adjacency list
  const std::vector<std::vector<graph::Index>> &get_adj() const { return adj_; }

  //! @brief Get the vartype as std::string, which must be "BINARY".
  //! @return "BINARY" as std::string
  std::string get_vartype_string() const { return "BINARY"; }

  ///----------------------------------------------------------------------------------------
  ///----------------The following functions are for debugging. Disable when
  ///release.----------------
  ///----------------------------------------------------------------------------------------
  /*
  void print_dE() const {
     for (std::size_t i = 0; i < dE_.size(); ++i) {
        printf("dE[%2ld]=%+.15lf\n", i, dE_[i]);
     }
  }

  void print_zero_count() const {
     for (int64_t i = 0; i < num_interactions_; ++i) {
        printf("zero_count[");
        for (const auto &index_binary: poly_key_list_[i]) {
           printf("%ld, ", index_binary);
        }
        printf("]=%lld\n", zero_count_[i]);
     }
  }

  void print_adj() const {
     for (int64_t i = 0; i < num_binaries; ++i) {
        printf("adj[%lld]=", i);
        for (const auto &index_key: adj_[i]) {
           printf("%ld(%+lf), ", index_key, poly_value_list_[index_key]);
        }
        printf("\n");
     }
  }

  void print_active_binaries() const {
     for (std::size_t i = 0; i < active_binaries_.size(); ++i) {
        printf("%d, ", binaries[active_binaries_[i]]);
     }
     printf("\n");
  }

  void print_interactions() const {
     for (int64_t i = 0; i < num_interactions_; ++i) {
        printf("%lld: size:%ld val: %lf\n", i, poly_key_list_[i].size(),
  poly_value_list_[i]);
     }
  }
  */
  ///----------------------------------------------------------------------------------------

private:
  //! @brief The number of the interactions.
  int64_t num_interactions_;

  //! @brief The energy differences when flipping a binary.
  std::vector<FloatType> dE_;

  //! @brief The number of variables taking the zero in each interaction.
  std::vector<int64_t> zero_count_;

  //! @brief Adjacency list, which is the list of the indices of polynomial
  //! interactions including specific binary.
  std::vector<std::vector<graph::Index>> adj_;

  //! @brief The list of the indices of the polynomial interactions as
  //! std::vector<std::vector<graph::Index>>.
  cimod::PolynomialKeyList<graph::Index> poly_key_list_;

  //! @brief The list of the values of the polynomial interactions as
  //! std::vector<FloatType>.
  cimod::PolynomialValueList<FloatType> poly_value_list_;

  //! @brief The list of the binaries connected by at least one interaction.
  std::vector<graph::Index> active_binaries_;

  //! @brief Upper bound of energy gap.
  FloatType max_effective_dE_;

  //! @brief Rough lower bound of energy gap.
  FloatType min_effective_dE_;

  ///------------------------------------------------------------------------------------------------------------------------------------------------------
  ///----------------The following member variables are used to virtually update
  ///the system for k-local update----------------
  ///------------------------------------------------------------------------------------------------------------------------------------------------------
  //! @brief The energy differences when flipping a binary, which is used to
  //! implement k-local update.
  std::vector<FloatType> dE_v_;

  //! @brief Virtually updated binary configulations, which is used to implement
  //! k-local update.
  graph::Binaries binaries_v_;

  //! @brief The list of virtually updated delta E's, which is used to implement
  //! k-local update.
  std::unordered_set<std::size_t> update_index_dE_v_;

  //! @brief The list of virtually updated index of zero_count_v_, which is used
  //! to implement k-local update.
  std::vector<std::size_t> update_index_zero_count_v_;

  //! @brief The list of virtually updated index of binaries_v_, which is used
  //! to implement k-local update.
  std::vector<std::size_t> update_index_binaries_v_;

  //! @brief The list of virtually updated the number of variables taking the
  //! zero in each interaction, which is used to implement k-local update.
  std::vector<int64_t> zero_count_v_;
  ///------------------------------------------------------------------------------------------------------------------------------------------------------

  //! @brief Sort interactions in accordance with its value and the degree of
  //! interactions (ascending order).
  void SortInteractions() {

    std::vector<graph::Index> index(num_interactions_);
#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions_; ++i) {
      index[i] = i;
    }

    auto compare_value = [this](const std::size_t i1, const std::size_t i2) {
      return poly_value_list_[i1] < poly_value_list_[i2];
    };
    auto compare_size = [this](const std::size_t i1, const std::size_t i2) {
      return poly_key_list_[i1].size() < poly_key_list_[i2].size();
    };

    std::stable_sort(index.begin(), index.end(), compare_size);
    std::stable_sort(index.begin(), index.end(), compare_value);

    cimod::PolynomialValueList<FloatType> vv = poly_value_list_;

#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions_; ++i) {
      poly_value_list_[i] = vv[index[i]];
    }

    cimod::PolynomialValueList<FloatType>().swap(vv);

    cimod::PolynomialKeyList<graph::Index> ii = poly_key_list_;

#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions_; ++i) {
      poly_key_list_[i] = ii[index[i]];
    }
  }

  //! @brief Set adjacency list. Note that the list is sorted in accordance with
  //! the value of the interactions.
  void SetAdj() {
    adj_.clear();
    adj_.resize(num_binaries);
    for (int64_t i = 0; i < num_interactions_; ++i) {
      for (const auto &index : poly_key_list_[i]) {
        adj_[index].push_back(i);
      }
    }

    // sort by value and key size
    auto compare_size = [this](const int64_t i1, const int64_t i2) {
      return poly_key_list_[i1].size() < poly_key_list_[i2].size();
    };
    auto compare_value = [this](const int64_t i1, const int64_t i2) {
      return poly_value_list_[i1] < poly_value_list_[i2];
    };

    int64_t adj_size = static_cast<int64_t>(adj_.size());
#pragma omp parallel for
    for (int64_t i = 0; i < adj_size; ++i) {
      std::stable_sort(adj_[i].begin(), adj_[i].end(), compare_size);
      std::stable_sort(adj_[i].begin(), adj_[i].end(), compare_value);
    }
  }

  //! @brief Set zero_count_ and zero_count_v_
  void ResetZeroCount() {
    zero_count_.resize(num_interactions_);
    zero_count_v_.resize(num_interactions_);
#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions_; ++i) {
      int64_t zero_count = 0;
      for (const auto &index : poly_key_list_[i]) {
        if (binaries[index] == 0) {
          zero_count++;
        }
      }
      zero_count_[i] = zero_count;
      zero_count_v_[i] = zero_count;
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
    FloatType min_val = poly_value_list_[0];
    std::vector<graph::Index> min_key = poly_key_list_[0];
    for (std::size_t i = 0; i < poly_key_list_.size(); ++i) {
      if (poly_value_list_[i] != 0.0 &&
          std::abs(poly_value_list_[i]) < std::abs(min_val) &&
          threshold < std::abs(poly_value_list_[i])) {
        min_val = poly_value_list_[i];
        min_key = poly_key_list_[i];
      }
    }

    if (std::abs(min_val) <= threshold) {
      throw std::runtime_error("No minimum value in interactions");
    }
    return std::pair<std::vector<graph::Index>, FloatType>(min_key, min_val);
  }
};

//! @brief Helper function for ClassicalIsingPolynomial constructor
//! @tparam GraphType
//! @param init_binaries const graph::Binaries&. The initial binaries.
//! @param init_interaction GraphType&. The initial interactions.
template <typename GraphType>
auto make_k_local_polynomial(const graph::Binaries &init_binaries,
                             const GraphType &init_interaction) {
  return KLocalPolynomial<GraphType>(init_binaries, init_interaction);
}

//! @brief Helper function for ClassicalIsingPolynomial constructor by using
//! nlohmann::json object
//! @tparam FloatType
//! @param init_binaries const graph::Binaries&. The initial binaries.
//! @param init_obj nlohmann::json&
auto make_k_local_polynomial(const graph::Binaries &init_binaries,
                             const nlohmann::json &init_obj) {
  return KLocalPolynomial<graph::Polynomial<double>>(init_binaries, init_obj);
}

} // namespace system
} // namespace openjij
