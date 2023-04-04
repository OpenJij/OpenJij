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

//! @file polynomial.hpp
//! @brief Graph class to represent polynomial unconstrained binary model or
//! Ising model with polynomial interactions.
//! @date 2021-03-11
//! @copyright Copyright (c) Jij Inc. 2021

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <cimod/binary_polynomial_model.hpp>

#include "openjij/graph/graph.hpp"
#include "openjij/graph/json/parse.hpp"
#include "openjij/utility/index_type.hpp"
#include "openjij/utility/pairhash.hpp"

namespace openjij {
namespace graph {


template<typename FloatType>
class BinaryPolynomialModel {
   static_assert(std::is_floating_point<FloatType>::value,
                 "Template parameter FloatType must be floating point type");
   
   
public:
   //! @brief The value type.
   using ValueType = FloatType;
   
   //! @brief The index type.
   using IndexType = utility::IndexType;
   
   //! @brief The hash for IndexType.
   using IndexHash = utility::IndexHash;
   
   //! @brief The variable type, which here represents binary variables \f$ x_i\in \{0, 1\} \f$
   using VariableType = std::int8_t;
   
   BinaryPolynomialModel(const std::vector<std::vector<IndexType>> &key_list,
                         const std::vector<ValueType> &value_list) {
      
      if (key_list.size() != value_list.size()) {
         throw std::runtime_error("The size of keys and values does not match each other.");
      }
      
      ValueType abs_max_interaction = -1;
      
      // Generate index list and store max interactions
      std::unordered_set<IndexType, IndexHash> index_set;
      
      for (std::size_t i = 0; i < key_list.size(); ++i) {
         if (std::abs(value_list[i]) > std::numeric_limits<ValueType>::epsilon()) {
            index_set.insert(key_list[i].begin(), key_list[i].end());
            if (std::abs(value_list[i]) > abs_max_interaction) {
               abs_max_interaction = std::abs(value_list[i]);
            }
         }
      }
      
      index_list_ = std::vector<IndexType>(index_set.begin(), index_set.end());
      std::sort(index_list_.begin(), index_list_.end());
      
      system_size_ = static_cast<std::int32_t>(index_list_.size());
      
      // Generate index map (from index to integer)
      index_map_.reserve(system_size_);
      for (std::int32_t i = 0; i < system_size_; ++i) {
         index_map_[index_list_[i]] = i;
      }
      
      // Generate interactions with integer index
      std::unordered_map<std::vector<std::int32_t>, ValueType, utility::VectorHash> poly;
      poly.reserve(key_list.size());
      for (std::size_t i = 0; i < key_list.size(); ++i) {
         if (std::abs(value_list[i]) > std::numeric_limits<ValueType>::epsilon()) {
            std::vector<std::int32_t> int_key(key_list[i].size());
            for (std::size_t j = 0; j < key_list[i].size(); ++j) {
               int_key[j] = index_map_.at(key_list[i][j]);
            }
            std::sort(int_key.begin(), int_key.end());
            int_key.erase(std::unique(int_key.begin(), int_key.end()), int_key.end());
            poly[int_key] += value_list[i];
            if (degree_ < int_key.size()) {
               degree_ = static_cast<std::int32_t>(int_key.size());
            }
         }
      }
      
      key_value_list_.reserve(poly.size());
      for (const auto &it: poly) {
         key_value_list_.push_back({it.first, it.second});
      }
      
      poly.clear();
      
      //Sort by keys.
      std::sort(key_value_list_.begin(), key_value_list_.end(), [](const auto &a, const auto &b) {
         return a.first < b.first;
      });
      
      adjacency_list_.resize(system_size_);
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         for (const auto &index: key_value_list_[i].first) {
            adjacency_list_[index].push_back(i);
         }
      }
      
      for (std::int32_t i = 0; i < system_size_; ++i) {
         adjacency_list_[i].shrink_to_fit();
         std::sort(adjacency_list_[i].begin(), adjacency_list_[i].end());
      }
      
      // Set relevant absolute minimum interaction
      // Apply threshold to avoid extremely minimum interaction derived from numerical errors.
      ValueType relevant_abs_min_interaction = abs_max_interaction*min_max_energy_difference_ratio_;
      estimated_min_energy_difference_ = std::numeric_limits<ValueType>::max();
      estimated_max_energy_difference_ = -1;
      
      for (std::int32_t i = 0; i < system_size_; ++i) {
         ValueType abs_row_sum_interaction = 0;
         for (const auto &interaction_index: adjacency_list_[i]) {
            if (std::abs(key_value_list_[interaction_index].second) >= relevant_abs_min_interaction) {
               abs_row_sum_interaction += std::abs(key_value_list_[interaction_index].second);
               if (std::abs(key_value_list_[interaction_index].second) < estimated_min_energy_difference_) {
                  estimated_min_energy_difference_ = std::abs(key_value_list_[interaction_index].second);
               }
            }
         }
         if (abs_row_sum_interaction > estimated_max_energy_difference_) {
            estimated_max_energy_difference_ = abs_row_sum_interaction;
         }
      }
      
      if (degree_ == 0) {
         estimated_min_energy_difference_ = 0;
         estimated_max_energy_difference_ = 0;
      }
      
   }
   
   //! @brief Get the degree of the polynomial interactions.
   //! @return The degree.
   std::int32_t GetDegree() const {
      return degree_;
   }
   
   //! @brief Get the system size.
   //! @return The system size.
   std::int32_t GetSystemSize() const {
      return static_cast<std::int32_t>(index_list_.size());
   }
   
   //! @brief Get the index list of the polynomial interactions.
   //! @return The index list.
   const std::vector<IndexType> &GetIndexList() const {
      return index_list_;
   }
   
   //! @brief Get the mapping from the index to the integer.
   //! @return The index map.
   const std::unordered_map<IndexType, std::int32_t, IndexHash> &GetIndexMap() const {
      return index_map_;
   }
   
   //! @brief Get the integer key and value list as pair.
   //! @return The integer key and value list as pair.
   const std::vector<std::pair<std::vector<std::int32_t>, ValueType>> &GetKeyValueList() const {
      return key_value_list_;
   }
   
   //! @brief Get the adjacency list, which stored the integer index of
   //! the polynomial interaction specified by the site index.
   //! @return The adjacency list.
   const std::vector<std::vector<std::size_t>> &GetAdjacencyList() const {
      return adjacency_list_;
   }
   
   //! @brief Get estimated minimum energy difference.
   //! @return The estimated minimum energy difference.
   ValueType GetEstimatedMinEnergyDifference() const {
      return estimated_min_energy_difference_;
   }
   
   //! @brief Get estimated maximum energy difference.
   //! @return The estimated maximum energy difference.
   ValueType GetEstimatedMaxEnergyDifference() const {
      return estimated_max_energy_difference_;
   }
   
   //! @brief Calculate energy corresponding to the variable configuration.
   //! @param variables The variable configuration.
   //! @return The energy.
   ValueType CalculateEnergy(const std::vector<VariableType> &variables) const {
      if (variables.size() != system_size_) {
         throw std::runtime_error("The size of variables is not equal to the system size");
      }
      ValueType val = 0;
      for (std::size_t i = 0; i < key_value_list_.size(); ++i) {
         VariableType hot_count = 0;
         for (const auto &index: key_value_list_[i].first) {
            if (variables[index] == 0) {
               break;
            }
            hot_count += variables[index];
         }
         if (hot_count == key_value_list_[i].first.size()) {
            val += key_value_list_[i].second;
         }
      }
      return val;
   }
   
   
private:
   //! @brief The degree of the interactions.
   std::int32_t degree_ = 0;
   
   //! @brief The system size.
   std::int32_t system_size_ = 0;
   
   //! @brief The index list of the interactions.
   std::vector<IndexType> index_list_;
   
   //! @brief The mapping from the index to the integer.
   std::unordered_map<IndexType, std::int32_t, IndexHash> index_map_;
   
   //! @brief The integer key and value list as pair.
   std::vector<std::pair<std::vector<std::int32_t>, ValueType>> key_value_list_;
   
   //! @brief The adjacency list, which stored the integer index of
   //! the polynomial interaction specified by the site index.
   std::vector<std::vector<std::size_t>> adjacency_list_;
   
   //! @brief The estimated minimum energy difference.
   ValueType estimated_min_energy_difference_ = 0;
   
   //! @brief The estimated maximum energy difference.
   ValueType estimated_max_energy_difference_ = 0;
   
   //! @brief The ratio of minimum and maximum energy difference set by 1e-08.
   //! \f[ {\rm ratio} = \frac{\Delat E_{{\rm min}}}{\Delat E_{{\rm max}}}\f]
   const ValueType min_max_energy_difference_ratio_ = 1e-08;
   
};


//! @brief Polynomial graph class, which can treat many-body interactions.
//! The Hamiltonian is like
//! \f[
//! H=\sum_{i \neq j} Q_{ij} x_i x_j +  \sum_{i \neq j \neq k} Q_{ijk} x_i x_j
//! x_k + \ldots \f] Note here that \f$ x_i \in \{0, 1\} \f$ or \f$ x_i \in
//! \{-1, +1\} \f$.
//! @tparam FloatType floating-point type
template <typename FloatType> class Polynomial : public Graph {
  static_assert(std::is_floating_point<FloatType>::value,
                "FloatType must be floating-point type.");

public:
  //! @brief Floating-point type
  using value_type = FloatType;

  //! @brief Constructor of Polynomial class to initialize variables and
  //! vartype.
  //! @param num_variables std::size_t
  explicit Polynomial(const std::size_t num_variables) : Graph(num_variables) {}

  //! @brief Constructor of Polynomial class to initialize num_variables, and
  //! interactions from json.
  //! @param j JSON object
  explicit Polynomial(const nlohmann::json &j)
      : Graph(j.at("variables").size()) {
    const auto &v_k_v = json_parse_polynomial<FloatType>(j);
    const auto &poly_key_list = std::get<0>(v_k_v);
    const auto &poly_value_list = std::get<1>(v_k_v);

    if (poly_key_list.size() != poly_value_list.size()) {
      throw std::runtime_error(
          "The sizes of key_list and value_list must match each other");
    }

    int64_t num_interactions = static_cast<int64_t>(poly_key_list.size());

    poly_key_list_.resize(num_interactions);
    poly_value_list_.resize(num_interactions);

#pragma omp parallel for
    for (int64_t i = 0; i < num_interactions; ++i) {
      poly_key_list_[i] = poly_key_list[i];
      poly_value_list_[i] = poly_value_list[i];
    }

    for (std::size_t i = 0; i < poly_key_list.size(); ++i) {
      poly_key_inv_[poly_key_list_[i]] = i;
    }
  }

  //! @brief Access the interaction corresponding to the input argument to set
  //! an interaction.
  //! @details Note that the input argument "key" will be sorted. If the
  //! interaction specified by "key" is empty, the zero will be added to the
  //! polynomial graph.
  //! @param key std::vector<Index>&
  //! @return The interaction corresponding to "key".
  FloatType &J(std::vector<Index> &key) {
    std::sort(key.begin(), key.end());
    CheckKeyValid(key);
    if (poly_key_inv_.count(key) == 0) {
      poly_key_inv_[key] = poly_value_list_.size();
      poly_key_list_.push_back(key);
      poly_value_list_.push_back(0.0);
    }
    return poly_value_list_[poly_key_inv_.at(key)];
  }

  //! @brief Access the interaction corresponding to the input argument to set
  //! an interaction.
  //! @details If the interaction specified by "key" is empty, the zero will be
  //! added to the polynomial graph.
  //! @param key const std::vector<Index>&
  //! @return The interaction corresponding to "key".
  FloatType &J(const std::vector<Index> &key) {
    std::vector<Index> copied_key = key;
    return J(copied_key);
  }

  //! @brief Return the interaction corresponding to the input argument.
  //! @details Note that the input argument "key" will be sorted. If the
  //! interaction specified by "key" is empty, this function returns the zero.
  //! @param key std::vector<Index>&
  //! @return The interaction corresponding to "key".
  FloatType J(std::vector<Index> &key) const {
    std::sort(key.begin(), key.end());
    CheckKeyValid(key);
    if (poly_key_inv_.count(key) == 0) {
      return 0.0;
    } else {
      return poly_value_list_[poly_key_inv_.at(key)];
    }
  }

  //! @brief Return the interaction corresponding to the input argument.
  //! @details If the interaction specified by "key" is empty, this function
  //! returns the zero.
  //! @param key const std::vector<Index>&
  //! @return The interaction corresponding to "key".
  FloatType J(const std::vector<Index> &key) const {
    std::vector<Index> copied_key = key;
    return J(copied_key);
  }

  //! @brief Access the interaction corresponding to the input argument to set
  //! an interaction.
  //! @details If the interaction specified by "args" is empty, the zero will be
  //! added to the polynomial graph.
  //! @param args parameter pack
  //! @return The interaction corresponding to "args".
  template <typename... Args> FloatType &J(Args... args) {
    std::vector<Index> copied_key{(Index)args...};
    return J(copied_key);
  }

  //! @brief Access the interaction corresponding to the input argument to set
  //! an interaction.
  //! @details If the interaction specified by "args" is empty, this function
  //! returns the zero.
  //! @param args parameter pack
  //! @return The interaction corresponding to "args".
  template <typename... Args> FloatType J(Args... args) const {
    std::vector<Index> copied_key{(Index)args...};
    return J(copied_key);
  }

  //! @brief Generate and return all the polynomial interactions as
  //! std::unordered_map<std::vector<Index>, FloatType>.
  //! @return All the interactions
  cimod::Polynomial<Index, FloatType> get_polynomial() const {
    cimod::Polynomial<Index, FloatType> poly_map;
    for (std::size_t i = 0; i < poly_key_list_.size(); ++i) {
      poly_map[poly_key_list_[i]] = poly_value_list_[i];
    }
    return poly_map;
  }

  //! @brief Get the PolynomialKeyList object (all the keys of polynomial
  //! interactions).
  //! @return PolynomialKeyList object as std::vector<std::vector<Index>>.
  const cimod::PolynomialKeyList<Index> &get_keys() const {
    return poly_key_list_;
  }

  //! @brief Get the PolynomialValueList object (all the values of polynomial
  //! interactions).
  //! @return PolynomialValueList object as std::vector<FloatType>.
  const cimod::PolynomialValueList<FloatType> &get_values() const {
    return poly_value_list_;
  }

  //! @brief Return the number of all the interactions.
  //! @return The number of all the interactions.
  std::size_t get_num_interactions() const { return poly_key_list_.size(); }

  //! @brief Return the total energy corresponding to the input variables, Spins
  //! or Binaries.
  //! @param spins const Spins& or const Binaries& (both are the same type)
  //! @param omp_flag if true OpenMP is enabled.
  //! @return The total energy
  FloatType energy(const Spins &spins, const bool omp_flag = true) const {
    if (spins.size() != Graph::size()) {
      throw std::out_of_range("The size of spins/binaries does not equal to "
                              "the size of polynomial graph");
    }

    FloatType energy = 0.0;

    int64_t num_interactions = static_cast<int64_t>(poly_key_list_.size());

    if (omp_flag) {
#pragma omp parallel for reduction(+ : energy)
      for (int64_t i = 0; i < num_interactions; ++i) {
        Spin spin_multiple = 1;
        for (const auto &index : poly_key_list_[i]) {
          spin_multiple *= spins[index];
          if (spin_multiple == 0.0) {
            break;
          }
        }
        energy += spin_multiple * poly_value_list_[i];
      }
    } else {
      for (int64_t i = 0; i < num_interactions; ++i) {
        Spin spin_multiple = 1;
        for (const auto &index : poly_key_list_[i]) {
          spin_multiple *= spins[index];
          if (spin_multiple == 0.0) {
            break;
          }
        }
        energy += spin_multiple * poly_value_list_[i];
      }
    }
    return energy;
  }

  //! @deprecated
  //! @brief Return the total energy corresponding to the input variables, Spins
  //! or Binaries.
  //! @param spins const Spins& or const Binaries& (both are the same type)
  //! @param omp_flag if true OpenMP is enabled.
  //! @return The total energy
  FloatType calc_energy(const Spins &spins, const bool omp_flag = true) const {
    return energy(spins, omp_flag);
  }

private:
  //! @brief The list of the indices of the polynomial interactions (namely, the
  //! list of key of the polynomial interactions as std::unordered_map) as
  //! std::vector<std::vector<Index>>.
  cimod::PolynomialKeyList<Index> poly_key_list_;

  //! @brief The list of the values of the polynomial interactions (namely, the
  //! list of value of the polynomial interactions as std::unordered_map) as
  //! std::vecto<FloatType>.
  cimod::PolynomialValueList<FloatType> poly_value_list_;

  //! @brief The inverse key list, which indicates the index of the
  //! poly_key_list_ and poly_value_list_
  std::unordered_map<std::vector<Index>, std::size_t, cimod::vector_hash>
      poly_key_inv_;

  //! @brief Check if the input keys are valid
  void CheckKeyValid(const std::vector<Index> &key) const {
    if (key.size() > Graph::size()) {
      std::stringstream ss;
      ss << "Too small system size. ";
      ss << "The degree of the input polynomial interaction is " << key.size();
      ss << ". But the system size is " << Graph::size();
      throw std::runtime_error(ss.str());
    }
    if (0 < key.size()) {
      // key is assumed to be sorted
      for (std::size_t i = 0; i < key.size() - 1; ++i) {
        if (key[i] == key[i + 1]) {
          throw std::runtime_error("No self-loops allowed");
        }
        if (key[i] >= Graph::size()) {
          std::stringstream ss;
          ss << "Too small system size. ";
          ss << "The index of a interaction: " << key[i] << " is out of range";
          throw std::runtime_error(ss.str());
        }
      }
      if (key.back() >= Graph::size()) {
        std::stringstream ss;
        ss << "Too small system size. ";
        ss << "The index of a interaction: " << key.back()
           << " is out of range";
        throw std::runtime_error(ss.str());
      }
    }
  }
};

} // namespace graph
} // namespace openjij
