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
             * @param init_auxiliary_spin_timeline,
             * @param init_interaction
             * @param gamma
             */
            ContinuousTimeIsing(const SpinConfiguration& init_spin_config,
                                const std::vector<CutPoint>& init_auxiliary_spin_timeline,
                                const GraphType& init_interaction,
                                const FloatType gamma)
                : spin_config(init_spin_config), num_spins(init_spin_config.size()+1), interaction(init_interaction.get_num_spins()+1), gamma(gamma) {
                assert(init_spin_config.size() == init_interaction.get_num_spins());

                for(graph::Index i = 0;i < init_interaction.get_num_spins();i++) {
                    for(auto j : init_interaction.adj_nodes(i)) {
                        if(i < j) {
                            continue;
                        }

                        // add actual interactions
                        this->interaction.J(i, j) = init_interaction.J(i, j);
                    }

                    this->interaction.J(i, this->num_spins-1) = init_interaction.h(i);
                    // add longitudinal magnetic field as interaction between ith spin and auxiliary spin
                }

                spin_config.push_back(init_auxiliary_spin_timeline);
            }


            /**
             * @brief ContinuousTimeIsing constructor, initializing each site to have only one cut at time zero with given spin
             *
             * @param init_spin
             * @param init_auxiliary_spin
             * @param init_interaction
             * @param gamma
             */
            ContinuousTimeIsing(const graph::Spins& init_spin,
                                const graph::Spin init_auxiriary_spin,
                                const GraphType& init_interaction,
                                const FloatType gamma)
                : spin_config(), num_spins(init_spin.size()+1), interaction(init_interaction.get_num_spins()+1), gamma(gamma) {
                assert(init_spin.size() == init_interaction.get_num_spins());

                for(graph::Index i = 0;i < init_interaction.get_num_spins();i++) {
                    for(auto j : init_interaction.adj_nodes(i)) {
                        if(i < j) {
                            continue;
                        }

                        // add actual interactions
                        this->interaction.J(i, j) = init_interaction.J(i, j);
                    }

                    this->interaction.J(i, this->num_spins-1) = init_interaction.h(i);
                    // add longitudinal magnetic field as interaction between ith spin and auxiliary spin
                }

                /* set spin configuration for each site, which has only one cut at time zero */
                for(auto spin : init_spin) {
                    spin_config.push_back(std::vector<CutPoint>{ CutPoint(TimeType(), spin) }); // TimeType() is zero value of the type
                }
                spin_config.push_back(std::vector<CutPoint>{ CutPoint(TimeType(), init_auxiriary_spin) });
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
                for(int i = 0;i < num_spins;i++) {
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
             * @brief number of spins, including auxiliary spin for longitudinal magnetic field
             */
            const std::size_t num_spins;
            
            /**
             * @brief interaction
             */
            GraphType interaction;
            
            /**
             * @brief coefficient of transverse field term, actual field would be gamma * s, where s = [0:1]
             */
            const FloatType gamma;
        };
    } // namespace system
} // namespace openjij

#endif
