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
#include <cassert>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

#include "openjij/graph/all.hpp"
#include "openjij/system/continuous_time_ising.hpp"
#include "openjij/utility/schedule_list.hpp"
#include "openjij/utility/union_find.hpp"

namespace openjij {
namespace updater {

/**
 * @brief Continuous Time Swendsen Wang updater
 *
 * @tparam System
 */
template <typename System> struct ContinuousTimeSwendsenWang;

/**
 * @brief Continuous Time Swendsen Wang updater for CTQIsystem
 *
 * @tparam FloatType
 */
template <typename FloatType>
struct ContinuousTimeSwendsenWang<
    system::ContinuousTimeIsing<graph::Sparse<FloatType>>> {
  using CTIsing = system::ContinuousTimeIsing<graph::Sparse<FloatType>>;
  using GraphType = typename graph::Sparse<FloatType>;
  using CutPoint = typename system::ContinuousTimeIsing<GraphType>::CutPoint;
  using TimeType = typename system::ContinuousTimeIsing<GraphType>::TimeType;

  /**
   * @brief continuous time Swendsen-Wang updater for transverse ising model
   *
   */
  template <typename RandomNumberEngine>
  static void
  update(system::ContinuousTimeIsing<GraphType> &system,
         RandomNumberEngine &random_number_engine,
         const utility::TransverseFieldUpdaterParameter &parameter) {

    const graph::Index num_spin = system.num_spins;
    std::vector<graph::Index> index_helper;
    index_helper.reserve(num_spin + 1);
    index_helper.push_back(0);
    /* index_helper[i]+k gives 1 dimensionalized index of kth time point at ith
     * site. this helps use of union-find tree only available for 1D structure.
     */

    /* 1. remove old cuts and place new cuts for every site */
    for (graph::Index i = 0; i < num_spin; i++) {
      auto &timeline = system.spin_config[i];
      auto cuts =
          generate_poisson_points(0.5 * system.gamma * (1.0 - parameter.s),
                                  parameter.beta, random_number_engine);
      // assuming transverse field gamma is positive

      timeline = create_timeline(timeline, cuts);
      assert(timeline.size() > 0);

      index_helper.push_back(index_helper.back() + timeline.size());
    }

    /* 2. place spacial bonds */
    utility::UnionFind union_find_tree(index_helper.back());
    for (graph::Index i = 0; i < num_spin; i++) {
      for (typename CTIsing::SparseMatrixXx::InnerIterator it(
               system.interaction, i);
           it; ++it) {
        std::size_t j = it.index();
        const FloatType &J = it.value();
        if (i < j) {
          continue; // ignore duplicated interaction
                    // if adj_nodes are sorted, this "continue" can be replaced
                    // by "break"
        }

        const auto bonds =
            generate_poisson_points(std::abs(0.5 * J * parameter.s),
                                    parameter.beta, random_number_engine);
        for (const auto bond : bonds) {
          /* get time point indices just before the bond */
          auto ki = system.get_temporal_spin_index(i, bond);
          auto kj = system.get_temporal_spin_index(j, bond);

          if (system.spin_config[i][ki].second *
                  system.spin_config[j][kj].second * J <
              0) {
            union_find_tree.unite_sets(index_helper[i] + ki,
                                       index_helper[j] + kj);
          }
        }
      }
    }

    /* 3. make clusters */
    std::unordered_map<utility::UnionFind::Node,
                       std::vector<utility::UnionFind::Node>>
        cluster_map;
    // root-index to {contained nodes} map

    for (graph::Index i = 0; i < num_spin; i++) {
      for (size_t k = 0; k < system.spin_config[i].size(); k++) {
        auto index = index_helper[i] + k;
        auto root_index = union_find_tree.find_set(index);
        auto position = cluster_map.find(root_index);
        if (position == cluster_map.end()) {
          cluster_map.emplace(root_index,
                              std::vector<utility::UnionFind::Node>{index});
        } else {
          position->second.push_back(index);
        }
      }
    }

    /* 4. flip clusters */
    auto urd = std::uniform_real_distribution<>(0, 1.0);
    for (const auto &cluster : cluster_map) {
      // 4.1. decide spin state (flip with the probability 1/2)
      const FloatType probability = 1.0 / 2.0;
      if (urd(random_number_engine) < probability) {
        // 4.2. update spin states
        for (auto node : cluster.second) {
          auto i = std::distance(index_helper.begin(),
                                 std::upper_bound(index_helper.begin(),
                                                  index_helper.end(), node)) -
                   1;
          auto k = node - index_helper[i];
          system.spin_config[i][k].second *= -1;
        }
      }
    }
  }

  /**
   * @brief create new timeline; place kinks by ignoring old cuts and place new
   * cuts
   *
   */
  static std::vector<CutPoint>
  create_timeline(const std::vector<CutPoint> &old_timeline,
                  const std::vector<TimeType> &cuts) {
    /* remove redundant cuts*/
    std::vector<CutPoint> concatenated_timeline;
    auto current_spin = old_timeline.back().second;
    for (auto cut_point : old_timeline) {
      if (cut_point.second != current_spin) {
        concatenated_timeline.push_back(cut_point);
      }
      current_spin = cut_point.second;
    }

    /* if entire timeline is occupied by single spin state */
    std::vector<CutPoint> new_timeline;
    if (concatenated_timeline.empty()) {
      if (cuts.empty()) {
        new_timeline.push_back(old_timeline[0]);
      } else {
        for (auto cut : cuts) {
          new_timeline.push_back(CutPoint(cut, current_spin));
        }
      }
      return new_timeline;
    }

    current_spin = concatenated_timeline.back().second;
    auto timeline_itr = concatenated_timeline.begin();
    auto cuts_itr = cuts.begin();
    while (true) {
      /* if all cuts have been placed, add remaining old kinks and break loop */
      if (cuts_itr == cuts.end()) {
        std::for_each(timeline_itr, concatenated_timeline.end(),
                      [&](CutPoint it) { new_timeline.push_back(it); });
        break;
      }

      /* if all spin kinks have been placed, add remaining cuts and break loop
       */
      if (timeline_itr == concatenated_timeline.end()) {
        std::for_each(cuts_itr, cuts.end(), [&](TimeType it) {
          new_timeline.push_back(CutPoint(it, current_spin));
        });
        break;
      }

      /* add earlier of kink or cut to new timeline */
      if (*cuts_itr < timeline_itr->first) {
        new_timeline.push_back(CutPoint(*cuts_itr, current_spin));
        cuts_itr++;
      } else {
        new_timeline.push_back(*timeline_itr);
        current_spin = timeline_itr->second;
        timeline_itr++;
      }
    }

    return new_timeline;
  }

  /**
   * @brief easy but inefficient version of create_timeline()
   *
   */
  static std::vector<CutPoint>
  create_timeline_easy(const std::vector<CutPoint> &old_timeline,
                       const std::vector<TimeType> &cuts) {
    /* remove redundant cuts*/
    std::vector<CutPoint> new_timeline;
    auto current_spin = old_timeline.back().second;
    for (auto cut_point : old_timeline) {
      if (cut_point.second != current_spin) {
        new_timeline.push_back(cut_point);
      }
      current_spin = cut_point.second;
    }

    /* if entire timeline is occupied by single spin state */
    if (new_timeline.empty()) {
      if (cuts.empty()) {
        new_timeline.push_back(old_timeline[0]);
      } else {
        for (auto cut : cuts) {
          new_timeline.push_back(CutPoint(cut, current_spin));
        }
      }
      return new_timeline;
    }

    static const auto first_lt = [](CutPoint x, CutPoint y) {
      return x.first < y.first;
    };
    for (auto cut : cuts) {
      const auto dummy_cut = CutPoint(cut, 0); // dummy cut for binary search
      auto itr = std::upper_bound(new_timeline.begin(), new_timeline.end(),
                                  dummy_cut, first_lt);
      auto prev_itr =
          (itr == new_timeline.begin()) ? new_timeline.end() - 1 : itr - 1;

      new_timeline.insert(itr, CutPoint(cut, prev_itr->second));
    }

    return new_timeline;
  }

  /**
   * @brief generates Poisson points with density lambda in the range of
   * [0:beta)
   * @note might be better to move this function to utility
   *
   */
  template <typename RandomNumberEngine>
  static std::vector<TimeType>
  generate_poisson_points(const TimeType lambda, const TimeType beta,
                          RandomNumberEngine &random_number_engine) {
    std::uniform_real_distribution<> rand(0.0, 1.0);
    std::uniform_real_distribution<> rand_beta(0.0, beta);

    const TimeType coef = beta * lambda;
    TimeType n = 0;
    TimeType d = std::exp(-coef);
    TimeType p = d;
    TimeType xi = rand(random_number_engine);

    while (p < xi) {
      n += 1;
      d *= coef / n;
      p += d;
    }

    std::vector<TimeType> poisson_points(n);
    for (int k = 0; k < n; k++) {
      poisson_points[k] = rand_beta(random_number_engine);
    }
    std::sort(poisson_points.begin(), poisson_points.end());

    return poisson_points;
  }
};

/**
 * @brief Continuous Time Swendsen Wang updater for CTQIsystem on CSR Sparse graph
 *
 * @tparam FloatType
 */
template <typename FloatType>
struct ContinuousTimeSwendsenWang<
    system::ContinuousTimeIsing<graph::CSRSparse<FloatType>>> {
  using CTIsing = system::ContinuousTimeIsing<graph::CSRSparse<FloatType>>;
  using GraphType = typename graph::CSRSparse<FloatType>;
  using CutPoint = typename system::ContinuousTimeIsing<GraphType>::CutPoint;
  using TimeType = typename system::ContinuousTimeIsing<GraphType>::TimeType;

  //TODO: duplicate code of `system::ContinuousTimeIsing<graph::Sparse<FloatType>>>`, need fix.

  /**
   * @brief continuous time Swendsen-Wang updater for transverse ising model
   *
   */
  template <typename RandomNumberEngine>
  static void
  update(system::ContinuousTimeIsing<GraphType> &system,
         RandomNumberEngine &random_number_engine,
         const utility::TransverseFieldUpdaterParameter &parameter) {

    const graph::Index num_spin = system.num_spins;
    std::vector<graph::Index> index_helper;
    index_helper.reserve(num_spin + 1);
    index_helper.push_back(0);
    /* index_helper[i]+k gives 1 dimensionalized index of kth time point at ith
     * site. this helps use of union-find tree only available for 1D structure.
     */

    /* 1. remove old cuts and place new cuts for every site */
    for (graph::Index i = 0; i < num_spin; i++) {
      auto &timeline = system.spin_config[i];
      auto cuts =
          generate_poisson_points(0.5 * system.gamma * (1.0 - parameter.s),
                                  parameter.beta, random_number_engine);
      // assuming transverse field gamma is positive

      timeline = create_timeline(timeline, cuts);
      assert(timeline.size() > 0);

      index_helper.push_back(index_helper.back() + timeline.size());
    }

    /* 2. place spacial bonds */
    utility::UnionFind union_find_tree(index_helper.back());
    for (graph::Index i = 0; i < num_spin; i++) {
      for (typename CTIsing::SparseMatrixXx::InnerIterator it(
               system.interaction, i);
           it; ++it) {
        std::size_t j = it.index();
        const FloatType &J = it.value();
        if (i < j) {
          continue; // ignore duplicated interaction
                    // if adj_nodes are sorted, this "continue" can be replaced
                    // by "break"
        }

        const auto bonds =
            generate_poisson_points(std::abs(0.5 * J * parameter.s),
                                    parameter.beta, random_number_engine);
        for (const auto bond : bonds) {
          /* get time point indices just before the bond */
          auto ki = system.get_temporal_spin_index(i, bond);
          auto kj = system.get_temporal_spin_index(j, bond);

          if (system.spin_config[i][ki].second *
                  system.spin_config[j][kj].second * J <
              0) {
            union_find_tree.unite_sets(index_helper[i] + ki,
                                       index_helper[j] + kj);
          }
        }
      }
    }

    /* 3. make clusters */
    std::unordered_map<utility::UnionFind::Node,
                       std::vector<utility::UnionFind::Node>>
        cluster_map;
    // root-index to {contained nodes} map

    for (graph::Index i = 0; i < num_spin; i++) {
      for (size_t k = 0; k < system.spin_config[i].size(); k++) {
        auto index = index_helper[i] + k;
        auto root_index = union_find_tree.find_set(index);
        auto position = cluster_map.find(root_index);
        if (position == cluster_map.end()) {
          cluster_map.emplace(root_index,
                              std::vector<utility::UnionFind::Node>{index});
        } else {
          position->second.push_back(index);
        }
      }
    }

    /* 4. flip clusters */
    auto urd = std::uniform_real_distribution<>(0, 1.0);
    for (const auto &cluster : cluster_map) {
      // 4.1. decide spin state (flip with the probability 1/2)
      const FloatType probability = 1.0 / 2.0;
      if (urd(random_number_engine) < probability) {
        // 4.2. update spin states
        for (auto node : cluster.second) {
          auto i = std::distance(index_helper.begin(),
                                 std::upper_bound(index_helper.begin(),
                                                  index_helper.end(), node)) -
                   1;
          auto k = node - index_helper[i];
          system.spin_config[i][k].second *= -1;
        }
      }
    }
  }

  /**
   * @brief create new timeline; place kinks by ignoring old cuts and place new
   * cuts
   *
   */
  static std::vector<CutPoint>
  create_timeline(const std::vector<CutPoint> &old_timeline,
                  const std::vector<TimeType> &cuts) {
    /* remove redundant cuts*/
    std::vector<CutPoint> concatenated_timeline;
    auto current_spin = old_timeline.back().second;
    for (auto cut_point : old_timeline) {
      if (cut_point.second != current_spin) {
        concatenated_timeline.push_back(cut_point);
      }
      current_spin = cut_point.second;
    }

    /* if entire timeline is occupied by single spin state */
    std::vector<CutPoint> new_timeline;
    if (concatenated_timeline.empty()) {
      if (cuts.empty()) {
        new_timeline.push_back(old_timeline[0]);
      } else {
        for (auto cut : cuts) {
          new_timeline.push_back(CutPoint(cut, current_spin));
        }
      }
      return new_timeline;
    }

    current_spin = concatenated_timeline.back().second;
    auto timeline_itr = concatenated_timeline.begin();
    auto cuts_itr = cuts.begin();
    while (true) {
      /* if all cuts have been placed, add remaining old kinks and break loop */
      if (cuts_itr == cuts.end()) {
        std::for_each(timeline_itr, concatenated_timeline.end(),
                      [&](CutPoint it) { new_timeline.push_back(it); });
        break;
      }

      /* if all spin kinks have been placed, add remaining cuts and break loop
       */
      if (timeline_itr == concatenated_timeline.end()) {
        std::for_each(cuts_itr, cuts.end(), [&](TimeType it) {
          new_timeline.push_back(CutPoint(it, current_spin));
        });
        break;
      }

      /* add earlier of kink or cut to new timeline */
      if (*cuts_itr < timeline_itr->first) {
        new_timeline.push_back(CutPoint(*cuts_itr, current_spin));
        cuts_itr++;
      } else {
        new_timeline.push_back(*timeline_itr);
        current_spin = timeline_itr->second;
        timeline_itr++;
      }
    }

    return new_timeline;
  }

  /**
   * @brief easy but inefficient version of create_timeline()
   *
   */
  static std::vector<CutPoint>
  create_timeline_easy(const std::vector<CutPoint> &old_timeline,
                       const std::vector<TimeType> &cuts) {
    /* remove redundant cuts*/
    std::vector<CutPoint> new_timeline;
    auto current_spin = old_timeline.back().second;
    for (auto cut_point : old_timeline) {
      if (cut_point.second != current_spin) {
        new_timeline.push_back(cut_point);
      }
      current_spin = cut_point.second;
    }

    /* if entire timeline is occupied by single spin state */
    if (new_timeline.empty()) {
      if (cuts.empty()) {
        new_timeline.push_back(old_timeline[0]);
      } else {
        for (auto cut : cuts) {
          new_timeline.push_back(CutPoint(cut, current_spin));
        }
      }
      return new_timeline;
    }

    static const auto first_lt = [](CutPoint x, CutPoint y) {
      return x.first < y.first;
    };
    for (auto cut : cuts) {
      const auto dummy_cut = CutPoint(cut, 0); // dummy cut for binary search
      auto itr = std::upper_bound(new_timeline.begin(), new_timeline.end(),
                                  dummy_cut, first_lt);
      auto prev_itr =
          (itr == new_timeline.begin()) ? new_timeline.end() - 1 : itr - 1;

      new_timeline.insert(itr, CutPoint(cut, prev_itr->second));
    }

    return new_timeline;
  }

  /**
   * @brief generates Poisson points with density lambda in the range of
   * [0:beta)
   * @note might be better to move this function to utility
   *
   */
  template <typename RandomNumberEngine>
  static std::vector<TimeType>
  generate_poisson_points(const TimeType lambda, const TimeType beta,
                          RandomNumberEngine &random_number_engine) {
    std::uniform_real_distribution<> rand(0.0, 1.0);
    std::uniform_real_distribution<> rand_beta(0.0, beta);

    const TimeType coef = beta * lambda;
    TimeType n = 0;
    TimeType d = std::exp(-coef);
    TimeType p = d;
    TimeType xi = rand(random_number_engine);

    while (p < xi) {
      n += 1;
      d *= coef / n;
      p += d;
    }

    std::vector<TimeType> poisson_points(n);
    for (int k = 0; k < n; k++) {
      poisson_points[k] = rand_beta(random_number_engine);
    }
    std::sort(poisson_points.begin(), poisson_points.end());

    return poisson_points;
  }
};
} // namespace updater
} // namespace openjij
