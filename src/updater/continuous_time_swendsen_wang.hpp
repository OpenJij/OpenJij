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

#include <system/continuous_time_ising.hpp>
#include <utility/schedule_list.hpp>

namespace openjij {
    namespace updater {
        template<typename GraphType>
        struct ContinuousTimeSwendsenWang {
            /**
             * @brief continuous time Swendsen-Wang updater for transverse ising model (no Eigen implementation)
             *
             */
            template <typename RandomNumberEngine>
            static void update(system::ContinuousTimeIsing<GraphType>& system,
                               RandomNumberEngine& random_number_engine,
                               const utility::TransverseFieldUpdaterParameter& parameter) {
                
            }

        private:
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
