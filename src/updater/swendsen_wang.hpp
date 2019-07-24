//    Copyright 2019 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef OPENJIJ_UPDATER_SWENDSEN_WANG_HPP__
#define OPENJIJ_UPDATER_SWENDSEN_WANG_HPP__

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

#include <graph/graph.hpp>
#include <system/classical_ising.hpp>
#include <utility/schedule_list.hpp>
#include <utility/union_find.hpp>

namespace openjij {
    namespace updater {

        /**
         * @brief swendsen wang updater
         *
         * @tparam System
         */
        template<typename System>
        struct SwendsenWang;

        /**
         * @brief swendsen wang updater for classical ising model (no Eigen implementation)
         *
         * @tparam GraphType type of graph (assume Dense, Sparse or derived class of them)
         */
        template<typename GraphType>
        struct SwendsenWang<system::ClassicalIsing<GraphType, false>> {

            using ClIsing = system::ClassicalIsing<GraphType, false>;
            using FloatType = typename GraphType::value_type;

            template<typename RandomNumberEngine>
            inline static void update(ClIsing& system,
                                 RandomNumberEngine& random_number_engine,
                                 const utility::ClassicalUpdaterParameter& parameter) {
                auto urd = std::uniform_real_distribution<>(0, 1.0);
                const auto num_spin = system.spin.size();

                // 1. update bonds
                auto union_find_tree = utility::UnionFind(num_spin);
                for (std::size_t node = 0; node < num_spin; ++node) {
                    for (auto&& adj_node : system.interaction.adj_nodes(node)) {
                        if (node >= adj_node) continue;
                        //check if bond can be connected
                        if (system.interaction.J(node, adj_node) * system.spin[node] * system.spin[adj_node] > 0) continue;
                        const auto unite_rate = std::max(0.0, 1.0 - std::exp( - 2.0 * parameter.beta * std::abs(system.interaction.J(node, adj_node))));
                        if (urd(random_number_engine) < unite_rate)
                            union_find_tree.unite_sets(node, adj_node);
                    }
                }

                // 2. make clusters
                const auto cluster_map = [num_spin, &union_find_tree](){
                    auto cluster_map = std::unordered_multimap<utility::UnionFind::Node, utility::UnionFind::Node>();
                    for (std::size_t node = 0; node < num_spin; ++node) {
                        cluster_map.insert({union_find_tree.find_set(node), node});
                    }
                    return cluster_map;
                }();

                // 3. update spin states in each cluster
                for (auto&& c : union_find_tree.get_roots()) {
                    const auto range = cluster_map.equal_range(c);

                    // 3.1. calculate energy \sum_{i \in C} h_i
                    double energy_magnetic = 0.0;
                    for (auto itr = range.first, last = range.second; itr != last; ++itr) {
                        const auto idx = itr->second;
                        energy_magnetic += system.interaction.h(idx)*system.spin[idx];
                    }

                    // 3.2. decide spin state
                    const FloatType probability = 1.0 / ( std::exp(-2 * parameter.beta * energy_magnetic) + 1.0 );
                    if(urd(random_number_engine) < probability){
                        // 3.3. update spin states
                        for (auto itr = range.first, last = range.second; itr != last; ++itr) {
                            const auto idx = itr->second;
                            system.spin[idx] *= -1;
                        }
                    }
                }

                // TODO: implement calculation of total energy difference
                // DEPRECATED: Dense::calc_energy
                //return system.interaction.calc_energy(system.spin);
                return;
            }
        };
    } // namespace updater
} // namespace openjij

#endif
