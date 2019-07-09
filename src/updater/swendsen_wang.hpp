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

#include <cmath>
#include <random>

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
         * @brief swendsen wang updater for classical ising model
         *
         * @tparam GraphType type of graph (assume Dense, Sparse or derived class of them)
         */
        template<typename GraphType>
        struct SwendsenWang<system::ClassicalIsing<GraphType>> {

            using ClIsing = system::ClassicalIsing<GraphType>;

            template<typename RandomNumberEngine>
            static void update(ClIsing& system,
                                 RandomNumberEngine& random_numder_engine,
                                 const utility::ClassicalUpdaterParameter& parameter) {
                const auto num_spin = system.spin.size();

                auto candidate_spin = graph::Spins(num_spin);
                auto urd = std::uniform_real_distribution<>(0, 1.0);
                for (auto& s : candidate_spin) {
                    s = urd(random_numder_engine) < 0.5 ? -1 : 1;
                }

                // make clusters
                const auto unite_rate = 1.0 - std::exp(-2.0 * parameter.beta);
                auto clusters = utility::UnionFind(num_spin);
                for (std::size_t i = 0; i < num_spin; ++i) {
                    // find adjacent spins
                    for (auto&& adj_index : system.interaction.adj_nodes(i)) {
                        // are signs of both spins the same?
                        // or
                        // is probability equal to or larger than p if the sign of each spin is different?
                        if (system.spin[i] * system.spin[adj_index] > 0 || urd(random_numder_engine) >= unite_rate) continue;
                        // unite two clusters corresponding to each spin
                        clusters.unite_sets(i, adj_index);
                    }
                }

                // update states in each cluster
                for (std::size_t i = 0; i < num_spin; ++i) {
                    system.spin[i] = candidate_spin[clusters.find_set(i)];
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
