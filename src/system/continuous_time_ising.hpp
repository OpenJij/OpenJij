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

#ifndef OPENJIJ_SYSTEM_CONTINUOUS_TIME_ISING_HPP__
#define OPENJIJ_SYSTEM_CONTINUOUS_TIME_ISING_HPP__

#include <system/system.hpp>
#include <graph/all.hpp>
#include <cassert>
#include <vector>
#include <utility>

namespace openjij {
    namespace system {
        using TimeType = double;
        using CutPoint = std::pair<TimeType, graph::Spin>;
        /**
         * @brief spin configuration in real and continuous time space
         * spin_config[i][j] -> at i th site, j th pair of imaginary time point and spin value after the point
         */
        using SpinConfiguration = std::vector<std::vector<CutPoint>>;
        
        
        template<typename GraphType>
        struct ContinuousTimeIsing {
            using system_type = transverse_field_system;
            using FloatType = typename GraphType::value_type;
            
            /**
             * @brief ContinuousTimeIsing constructor
             *
             * @param init_spin_config
             * @param init_interaction
             */
            ContinuousTimeIsing(const SpinConfiguration& init_spin_config, const GraphType& init_interaction, FloatType gamma)
                : spin_config(init_spin_config), interaction(init_interaction), num_classical_spins(init_spin_config.size()), gamma(gamma){
                assert(num_classical_spins == init_interaction.get_num_spins());
            }

            /**
             * @brief reset spins with given spin configuration
             *
             * @param init_spin_config
             */
            void reset_spins(const SpinConfiguration& init_spin_config) {
                this->spin_config = init_spin_config;
            }

            /**
             * @brief reset spins with given spin configuration
             *
             * @param classical_spins
             */
            void reset_spins(const graph::Spins& classical_spins) {
                for(int i = 0;i < num_classical_spins;i++) {
                    this->spin_config[i] = std::vector<CutPoint> {
                        CutPoint(TimeType(), classical_spins[i]) // TimeType() is zero value of the type
                    };
                }
            }

            
            /* Member variables */
            
            /**
             * @brief spin configuration
             */
            SpinConfiguration spin_config;
            
            /**
             * @brief interaction 
             */
            const GraphType interaction;

            /**
             * @brief number of real classical spins
             */
            const std::size_t num_classical_spins;
            
            /**
             * @brief coefficient of transverse field term
             */
            FloatType gamma;
        };
    } // namespace system
} // namespace openjij

#endif
