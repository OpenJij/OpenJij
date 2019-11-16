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

#include <graph/all.hpp>
#include <system/continuous_time_ising.hpp>
#include <utility/schedule_list.hpp>

namespace openjij {
    namespace updater {
        template<typename GraphType>
        struct ContinuousTimeSwendsenWang {
            using CutPoint = system::CutPoint;
            using TimeType = system::TimeType;
            
            /**
             * @brief continuous time Swendsen-Wang updater for transverse ising model (no Eigen implementation)
             *
             */
            template <typename RandomNumberEngine>
            static void update(system::ContinuousTimeIsing<GraphType>& system,
                               RandomNumberEngine& random_number_engine,
                               const utility::TransverseFieldUpdaterParameter& parameter) {                

                const graph::Index num_spin = system.num_classical_spins;

                /* 1. remove old cuts and place new cuts for every site */
                for(graph::Index i = 0;i < num_spin; i++) {
                    auto& timeline = system.spin_config[i];
                    auto cuts = generate_poisson_points(0.5*parameter.s, parameter.beta, random_number_engine); // assuming transverse field s is positive
                    
                    timeline = create_timeline(timeline, cuts);
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
