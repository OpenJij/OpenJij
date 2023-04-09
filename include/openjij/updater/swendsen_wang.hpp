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
#include <cmath>
#include <random>
#include <unordered_map>

#include "openjij/graph/graph.hpp"
#include "openjij/system/classical_ising.hpp"
#include "openjij/utility/schedule_list.hpp"
#include "openjij/utility/union_find.hpp"

namespace openjij {
namespace updater {

/**
 * @brief swendsen wang updater
 *
 * @tparam System
 */
template <typename System> struct SwendsenWang;

/**
 * @brief swendsen wang updater for classical ising model (on Sparse graph)
 *
 * @tparam FloatType
 */
template <typename FloatType>
struct SwendsenWang<system::ClassicalIsing<graph::Sparse<FloatType>>> {

  using ClIsing = system::ClassicalIsing<graph::Sparse<FloatType>>;

  template <typename RandomNumberEngine>
  inline static void
  update(ClIsing &system, RandomNumberEngine &random_number_engine,
         const utility::ClassicalUpdaterParameter &parameter) {
    auto urd = std::uniform_real_distribution<>(0, 1.0);

    // num_spin = system size + additional spin
    const size_t num_spin = system.spin.size();

    // 1. update bonds
    auto union_find_tree = utility::UnionFind(num_spin);
    for (std::size_t node = 0; node < num_spin; ++node) {
      for (typename ClIsing::SparseMatrixXx::InnerIterator it(
               system.interaction, node);
           it; ++it) {
        // fetch adjacent node
        std::size_t adj_node = it.index();
        // fetch system.interaction(node, adj_node)
        const FloatType &J = it.value();
        if (node >= adj_node)
          continue;
        // check if bond can be connected
        if (J * system.spin(node) * system.spin(adj_node) > 0)
          continue;
        const auto unite_rate =
            std::max(static_cast<FloatType>(0.0),
                     static_cast<FloatType>(
                         1.0 - std::exp(-2.0 * parameter.beta * std::abs(J))));
        if (urd(random_number_engine) < unite_rate)
          union_find_tree.unite_sets(node, adj_node);
      }
    }

    // 2. make clusters
    const auto cluster_map = [num_spin, &union_find_tree]() {
      auto cluster_map = std::unordered_multimap<utility::UnionFind::Node,
                                                 utility::UnionFind::Node>();
      for (std::size_t node = 0; node < num_spin; ++node) {
        cluster_map.insert({union_find_tree.find_set(node), node});
      }
      return cluster_map;
    }();

    // 3. update spin states in each cluster
    for (auto &&c : union_find_tree.get_roots()) {
      const auto range = cluster_map.equal_range(c);

      // 3.1. decide spin state (flip with the probability 1/2)
      const FloatType probability = 1.0 / 2.0;
      if (urd(random_number_engine) < probability) {
        // 3.2. update spin states
        for (auto itr = range.first, last = range.second; itr != last; ++itr) {
          const auto idx = itr->second;
          system.spin(idx) *= -1;
        }
      }
    }

    return;
  }
};

/**
 * @brief swendsen wang updater for classical ising model (on CSR Sparse graph)
 *
 * @tparam FloatType
 */
template <typename FloatType>
struct SwendsenWang<system::ClassicalIsing<graph::CSRSparse<FloatType>>> {
  //TODO: duplicate code of `SwendsenWang<system::ClassicalIsing<graph::Sparse<FloatType>>>`, need fix.

  using ClIsing = system::ClassicalIsing<graph::CSRSparse<FloatType>>;

  template <typename RandomNumberEngine>
  inline static void
  update(ClIsing &system, RandomNumberEngine &random_number_engine,
         const utility::ClassicalUpdaterParameter &parameter) {
    auto urd = std::uniform_real_distribution<>(0, 1.0);

    // num_spin = system size + additional spin
    const size_t num_spin = system.spin.size();

    // 1. update bonds
    auto union_find_tree = utility::UnionFind(num_spin);
    for (std::size_t node = 0; node < num_spin; ++node) {
      for (typename ClIsing::SparseMatrixXx::InnerIterator it(
               system.interaction, node);
           it; ++it) {
        // fetch adjacent node
        std::size_t adj_node = it.index();
        // fetch system.interaction(node, adj_node)
        const FloatType &J = it.value();
        if (node >= adj_node)
          continue;
        // check if bond can be connected
        if (J * system.spin(node) * system.spin(adj_node) > 0)
          continue;
        const auto unite_rate =
            std::max(static_cast<FloatType>(0.0),
                     static_cast<FloatType>(
                         1.0 - std::exp(-2.0 * parameter.beta * std::abs(J))));
        if (urd(random_number_engine) < unite_rate)
          union_find_tree.unite_sets(node, adj_node);
      }
    }

    // 2. make clusters
    const auto cluster_map = [num_spin, &union_find_tree]() {
      auto cluster_map = std::unordered_multimap<utility::UnionFind::Node,
                                                 utility::UnionFind::Node>();
      for (std::size_t node = 0; node < num_spin; ++node) {
        cluster_map.insert({union_find_tree.find_set(node), node});
      }
      return cluster_map;
    }();

    // 3. update spin states in each cluster
    for (auto &&c : union_find_tree.get_roots()) {
      const auto range = cluster_map.equal_range(c);

      // 3.1. decide spin state (flip with the probability 1/2)
      const FloatType probability = 1.0 / 2.0;
      if (urd(random_number_engine) < probability) {
        // 3.2. update spin states
        for (auto itr = range.first, last = range.second; itr != last; ++itr) {
          const auto idx = itr->second;
          system.spin(idx) *= -1;
        }
      }
    }

    return;
  }
};
} // namespace updater
} // namespace openjij
