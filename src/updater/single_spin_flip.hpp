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

#ifndef OPENJIJ_UPDATER_SINGLE_SPIN_FLIP_HPP__
#define OPENJIJ_UPDATER_SINGLE_SPIN_FLIP_HPP__

#include <random>

#include <system/classical_ising.hpp>
#include <system/quantum_ising.hpp>
#include <utility/schedule_list.hpp>

namespace openjij {
    namespace updater {

        /**
         * @brief updater (naive single spin flip)
         *
         * @tparam System type of system
         */
        template<typename System>
        struct SingleSpinFlip;


        /**
         * @brief single spin flip for sparse classical ising model
         */
        template<>
        struct SingleSpinFlip<system::ClassicalIsing<graph::Sparse<double>>> {
            
            /**
             * @brief ClassicalIsing with sparse interactions
             */
            using ClIsing = system::ClassicalIsing<graph::Sparse<double>>;

            /**
             * @brief operate single spin flip in a classical ising system
             *
             * @param system object of a classical ising system
             * @param random_number_engine random number gengine
             * @param parameter parameter object including inverse temperature \f\beta:=(k_B T)^{-1}\f
             *
             * @return energy difference \f\Delta E\f
             */
          template<typename RandomNumberEngine>
            static double update(ClIsing& system,
                                 RandomNumberEngine& random_numder_engine,
                                 const utility::ClassicalUpdaterParameter& parameter) {
                // set probability distribution object
                // to select candidate for flip at random
                static auto uid = std::uniform_int_distribution<std::size_t>(0, system.spin.size()-1);
                // to do Metroopolis
                static auto urd = std::uniform_real_distribution<>(0, 1.0);

                // energy difference
                auto total_dE = 0;

                for (std::size_t time = 0, num_spin = system.spin.size(); time < num_spin; ++time) {
                    // index of spin selected at random
                    const auto index = uid(random_numder_engine);

                    // local energy difference
                    auto dE = 0;
                    for (auto&& adj_index : system.interaction.adj_nodes(index)) {
                        dE += -2.0 * system.spin[index] * (index != adj_index ? (system.interaction.J(index, adj_index) * system.spin[adj_index])
                                                                              :  system.interaction.h(index));
                    }

                    // Flip the spin?
                    if (std::exp( -parameter.beta * dE) > urd(random_numder_engine)) {
                        system.spin[index] *= -1;
                        total_dE += dE;
                    }
                }

                return total_dE;
            }
        };

        template<>
        struct SingleSpinFlip<system::QuantumIsing<graph::Sparse<double>>> {

            using QIsing = system::QuantumIsing<graph::Sparse<double>>;
            
            /**
             * @brief operate single spin flip in a quantum ising system
             *
             * @param system object of a quantum ising system
             * @param random_number_engine random number gengine
             * @param parameter parameter object including inverse temperature \f\beta:=(k_B T)^{-1}\f and transverse magnetic field \f\gamma\f
             *
             * @return energy difference \f\Delta E\f
             */
            template<typename RandomNumberEngine>
            static double update(QIsing& system,
                                 RandomNumberEngine& random_numder_engine,
                                 const utility::QuantumUpdaterParameter& parameter) {
                double totaldE = 0;
                //size_t num_classical_spins = spins[0].size();
                //size_t num_trotter_slices = spins.size();

                ////default updater (single_spin_flip)
                //for(size_t i=0; i<num_classical_spins*num_trotter_slices; i++){
                //    size_t index_trot = uid_trotter(mt);
                //    size_t index = uid(mt);
                //    //do metropolis
                //    double dE = 0;
                //    //adjacent nodes
                //    for(auto&& adj_index : interaction.adj_nodes(index)){
                //        dE += -2 * s * (beta/num_trotter_slices) * spins[index_trot][index] * (index != adj_index ? (interaction.J(index, adj_index) * spins[index_trot][adj_index]) : interaction.h(index));
                //    }

                //    //trotter direction
                //    dE += -2 * (1/2.) * log(tanh(beta* gamma * (1.0-s) /num_trotter_slices)) * spins[index_trot][index]*(spins[mod_t((int64_t)index_trot+1)][index] + spins[mod_t((int64_t)index_trot-1)][index]);

                //    //metropolis 
                //    if(exp(-dE) > urd(mt)){
                //        spins[index_trot][index] *= -1;
                //        totaldE += dE;
                //    }

                //}

                return totaldE;
            }
        };
    } // namespace updater
} // namespace openjij

#endif
