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

#ifndef OPENJIJ_UPDATER_CONTINUOUS_TIME_SWENDSEN_WANG_HPP__
#define OPENJIJ_UPDATER_CONTINUOUS_TIME_SWENDSEN_WANG_HPP__

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <graph/all.hpp>
#include <system/continuous_time_ising.hpp>
#include <utility/schedule_list.hpp>
#include <utility/union_find.hpp>

namespace openjij {
    namespace updater {
        template<typename GraphType>
        struct ContinuousTimeSwendsenWang {
            using CutPoint = system::CutPoint;
            using TimeType = system::TimeType;
            using FloatType = typename GraphType::value_type;
            
            /**
             * @brief continuous time Swendsen-Wang updater for transverse ising model (no Eigen implementation)
             *
             */
            template <typename RandomNumberEngine>
            static void update(system::ContinuousTimeIsing<GraphType>& system,
                               RandomNumberEngine& random_number_engine,
                               const utility::TransverseFieldUpdaterParameter& parameter) {                

                const graph::Index num_spin = system.num_classical_spins;
                std::vector<graph::Index> index_helper;
                index_helper.reserve(num_spin+1);
                index_helper.push_back(0);
                /* index_helper[i]+k gives 1 dimensionalized index of kth time point at ith site.
                 * this helps use of union-find tree only available for 1D structure.
                 */

                /* 1. remove old cuts and place new cuts for every site */
                for(graph::Index i = 0;i < num_spin; i++) {
                    auto& timeline = system.spin_config[i];
                    auto cuts = generate_poisson_points(0.5*parameter.s, parameter.beta, random_number_engine);
                    // assuming transverse field s is positive
            
                    timeline = create_timeline(timeline, cuts);
                    index_helper.push_back(index_helper.back()+timeline.size());
                }

                const auto first_lt = [](CutPoint x, CutPoint y) { return x.first < y.first; };
                // function to compare time point (lt; less than)
                
                /* 2. place spacial bonds */
                utility::UnionFind union_find_tree(index_helper.back());
                for(graph::Index i = 0;i < num_spin; i++) {
                    for(auto&& j : system.interaction.adj_nodes(i)) {
                        if (i < j) {
                            continue; // ignore duplicated interaction
                                      // if adj_nodes are sorted, this "continue" can be replaced by "break"
                        }
                        
                        const auto bonds = generate_poisson_points(std::abs(0.5*system.interaction.J(i, j)),
                                                                   parameter.beta, random_number_engine);
                        for(const auto bond : bonds) {
                            const auto dummy_bond = CutPoint(bond, 0); // dummy variable for binary search                            
                            
                            /* find time point at ith site just before the bond */
                            auto ki = std::distance(system.spin_config[i].begin(),
                                                    std::lower_bound(system.spin_config[i].begin(),
                                                                     system.spin_config[i].end(),
                                                                     dummy_bond,
                                                                     first_lt));
                            if(ki == 0) { // if the bond lies before any time points
                                ki = system.spin_config[i].size() - 1; // periodic boundary condition
                            } else if(ki == system.spin_config[i].size()) { // if the bond lies after any time points
                                ki--;
                            }

                            /* find time point at jth site just before the bond */
                            auto kj = std::distance(system.spin_config[j].begin(),
                                                    std::lower_bound(system.spin_config[j].begin(),
                                                                     system.spin_config[j].end(),
                                                                     dummy_bond,
                                                                     first_lt));
                            if(kj == 0) {
                                kj = system.spin_config[j].size() - 1;
                            } else if(kj == system.spin_config[j].size()) {
                                kj--;
                            }

                            if(system.spin_config[i][ki].second * system.spin_config[j][kj].second * system.interaction.J(i, j) < 0) {
                                union_find_tree.unite_sets(index_helper[i]+ki, index_helper[j]+kj);
                            }
                        }
                    }
                }

                /* 3. make clusters */
                std::unordered_map<utility::UnionFind::Node, std::vector<utility::UnionFind::Node>> cluster_map;
                // root-index to {contained nodes} map
                
                for(graph::Index i = 0;i < num_spin;i++) {
                    for(size_t k = 0;k < system.spin_config[i].size();k++) {
                        auto index = index_helper[i] + k;
                        auto root_index = union_find_tree.find_set(index);
                        auto position = cluster_map.find(root_index);
                        if(position == cluster_map.end()) {
                            cluster_map.emplace(root_index, std::vector<utility::UnionFind::Node>{ index });
                        } else {
                            position->second.push_back(index);
                        }
                    }
                }

                /* 4. flip clusters */
                auto urd = std::uniform_real_distribution<>(0, 1.0);
                for(const auto& cluster : cluster_map) {
                    // 4.1. decide spin state (flip with the probability 1/2)
                    const FloatType probability = 1.0 / 2.0;
                    if(urd(random_number_engine) < probability){
                        // 4.2. update spin states
                        for(auto node : cluster.second) {
                            auto i = std::distance(index_helper.begin(),
                                                   std::upper_bound(index_helper.begin(), index_helper.end(), node)) - 1;
                            // TODO: check carefully the case which node == index_helper element
                            auto k = node - index_helper[i];
                            system.spin_config[i][k].second *= -1;
                        }
                    }
                }
            }

        private:
            /**
             * @brief create new timeline; place kinks by ignoring old cuts and place new cuts
             * 
             */
            static std::vector<CutPoint> create_timeline(const std::vector<CutPoint>& old_timeline,
                                                         const std::vector<TimeType>& cuts) {
                std::vector<CutPoint> new_timeline;
                new_timeline.reserve(old_timeline.size()); // not actual size, but decrease reallocation frequency
                size_t old_time_index = 0;
                size_t cut_index = 0;
                    
                while(true) {
                    /* add earlier of kink or cut to new timeline */
                    if(cuts[cut_index] < old_timeline[old_time_index].first) {
                        new_timeline.push_back(CutPoint{cuts[cut_index], old_timeline[old_time_index].second});
                        cut_index++;
                    } else {
                        /* if spin direction is different from previous one, place cut (kink) */
                        const auto prev_time_index = (old_time_index + 1 + old_timeline.size()) % old_timeline.size();
                        if(old_timeline[old_time_index].second != old_timeline[prev_time_index].second) {
                            new_timeline.push_back(old_timeline[old_time_index]);
                        }
                        old_time_index++;
                    }

                    /* when all cuts have been placed, add remaining old kinks and break loop */
                    if(cut_index >= cuts.size()) {
                        for(;old_time_index < old_timeline.size();old_time_index++) {
                            /* if spin direction is different from previous one, place cut (kink) */
                            const auto prev_time_index = (old_time_index + 1 + old_timeline.size()) % old_timeline.size();
                            if(old_timeline[old_time_index].second != old_timeline[prev_time_index].second) {
                                new_timeline.push_back(old_timeline[old_time_index]);
                            }
                        }
                        break;
                    }

                    /* when all spin kinks have been placed, add remaining cuts and break loop */
                    if(old_time_index >= old_timeline.size()) {
                        const auto final_spin = old_timeline[old_timeline.size()-1].second;
                        for(;cut_index < cuts.size();cut_index++) {
                            new_timeline.push_back(CutPoint{cuts[cut_index], final_spin});
                        }
                        break;
                    }
                }

                return new_timeline;
            }

            /**
             * @brief generates Poisson points with density lambda in the range of [0:beta)
             * @note might be better to move this function to utility
             *
             */
            template<typename TimeType, typename RandomNumberEngine>
            static std::vector<TimeType> generate_poisson_points(const TimeType lambda, const TimeType beta,
                                                                 RandomNumberEngine& random_number_engine) {
                std::uniform_real_distribution<> rand(0.0, 1.0);
                std::uniform_real_distribution<> rand_beta(0.0, beta);

                const TimeType coef = beta*lambda;
                TimeType n = 0;
                TimeType d = std::exp(-coef);
                TimeType p = d;
                TimeType xi = rand(random_number_engine);
  
                while(p < xi) {
                    n += 1;
                    d *= coef/n;
                    p += d;
                }

                std::vector<TimeType> poisson_points(n);
                for(int k = 0;k < n;k++) {
                    poisson_points[k] = rand_beta(random_number_engine);
                }
                std::sort(poisson_points.begin(), poisson_points.end());

                return poisson_points;
            }
        };
    } // namespace updater
} // namespace openjij

#endif
