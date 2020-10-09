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
#include <system/transverse_ising.hpp>
#include <utility/schedule_list.hpp>
#include <type_traits>

namespace openjij {
    namespace updater {

        /**
         * @brief naive single spin flip updater
         *
         * @tparam System type of system
         */
        template<typename System>
        struct SingleSpinFlip;

        /**
         * @brief single spin flip for classical ising model (with Eigen implementation)
         *
         * @tparam GraphType graph type (assume Dense<FloatType> or Sparse<FloatType>)
         */
        template<typename GraphType>
        struct SingleSpinFlip<system::ClassicalIsing<GraphType>> {
            
            /**
             * @brief ClassicalIsing with dense interactions
             */
            using ClIsing = system::ClassicalIsing<GraphType>;

            /**
             * @brief float type
             */
            using FloatType = typename GraphType::value_type;

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
            inline static void update(ClIsing& system,
                                 RandomNumberEngine& random_number_engine,
                                 const utility::ClassicalUpdaterParameter& parameter) {
                // set probability distribution object
                // to do Metroopolis
                auto urd = std::uniform_real_distribution<>(0, 1.0);

                bool flip_spin;
                // do a iteraction except for the auxiliary spin
                for (std::size_t index = 0; index < system.num_spins; ++index) {
                    
                    flip_spin = false;

                    if (system.dE(index) <= 0 || std::exp( -parameter.beta * system.dE(index)) > urd(random_number_engine)) {
                        flip_spin = true;
                    }

                    if(flip_spin){

                        // update dE
                        system.dE += 4 * system.spin(index) * (system.interaction.row(index).transpose().cwiseProduct(system.spin));

                        system.dE(index) *= -1;
                        system.spin(index) *= -1;

                    }

                    //assure that the dummy spin is not changed.
                    system.spin(system.num_spins) = 1;
                }
            }

        };

        /**
         * @brief single spin flip for transverse field ising model (with Eigen implementation)
         *
         * @tparam GraphType graph type (assume Dense<FloatType> or Sparse<FloatType>)
         */
        template<typename GraphType>
        struct SingleSpinFlip<system::TransverseIsing<GraphType>> {
            
            /**
             * @brief transverse field ising system
             */
            using QIsing = system::TransverseIsing<GraphType>;

            /**
             * @brief float type
             */
            using FloatType = typename GraphType::value_type;

            /**
             * @brief operate single spin flip in a transverse ising system
             *
             * @param system object of a transverse ising system
             * @param random_number_engine random number engine
             * @param parameter parameter object including inverse temperature \f\beta:=(k_B T)^{-1}\f and transverse magnetic field \f\s\f
             *
             * @return energy difference \f\Delta E\f
             */
            template<typename RandomNumberEngine>
                inline static void update(QIsing& system,
                        RandomNumberEngine& random_number_engine,
                        const utility::TransverseFieldUpdaterParameter& parameter) {

                    //get number of classical spins
                    std::size_t num_classical_spins = system.num_classical_spins;
                    //get number of trotter slices
                    std::size_t num_trotter_slices = system.trotter_spins.cols();

                    //do metropolis
                    auto urd = std::uniform_real_distribution<>(0, 1.0);

                    //aliases
                    auto& spins = system.trotter_spins;
                    auto& gamma = system.gamma;
                    auto& beta = parameter.beta;
                    auto& s = parameter.s;

                    // transverse factor
                    FloatType B = (1/2.) * log(tanh(beta* gamma * (1.0-s) /num_trotter_slices));

                    //initialize dE and dEtrot
                    system.dE = -2 * s * (beta/num_trotter_slices) * spins.cwiseProduct(system.interaction * spins);
                    system.dEtrot = QIsing::TrotterMatrix::Zero(num_classical_spins+1, num_trotter_slices);
                    for(std::size_t t=0; t<num_trotter_slices; t++){
                        system.dEtrot.col(t) = -2 * B * spins.col(t).cwiseProduct(
                                spins.col(mod_t((int64_t)t+1, num_trotter_slices)) +
                                spins.col(mod_t((int64_t)t-1, num_trotter_slices))
                                );
                    }

                    
                    for(std::size_t t=0; t<num_trotter_slices; t++){
                        for(std::size_t i=0; i<num_classical_spins; i++){
                            //calculate matrix dot product
                            
                        }
                    }
                }

            private: 
            template<typename RandomNumberEngine>
                inline static void do_job(QIsing& system,
                        RandomNumberEngine& random_number_engine,
                        const utility::TransverseFieldUpdaterParameter& parameter, size_t i, size_t t) {

                            //aliases
                            auto& spins = system.trotter_spins;
                            auto& gamma = system.gamma;
                            auto& beta = parameter.beta;
                            auto& s = parameter.s;

                            FloatType dE = system.dE(i, t) + system.dEtrot(i, t);

                            // for debugging

                            //FloatType testdE = 0;

                            ////do metropolis
                            //testdE += -2 * s * (beta/num_trotter_slices) * spins(i, t)*(system.interaction.row(i).dot(spins.col(t)));

                            ////trotter direction
                            //testdE += -2 * (1/2.) * log(tanh(beta* gamma * (1.0-s) /num_trotter_slices)) * spins(i, t) *
                            //    (    spins(i, mod_t((int64_t)t+1, num_trotter_slices)) 
                            //       + spins(i, mod_t((int64_t)t-1, num_trotter_slices)));

                            //assert(dE == testdE);

                            //metropolis 
                            if(dE < 0 || exp(-dE) > urd(random_number_engine)){

                                //update dE and dEtrot
                                system.dE.col(t) += 4 * s * (beta/num_trotter_slices) * spins(i, t) * (system.interaction.row(i).transpose().cwiseProduct(spins.col(t)));
                                system.dEtrot(i, mod_t(t+1, num_trotter_slices)) += 4 * B * spins(i, t) * spins(i, mod_t(t+1, num_trotter_slices));
                                system.dEtrot(i, mod_t(t-1, num_trotter_slices)) += 4 * B * spins(i, t) * spins(i, mod_t(t-1, num_trotter_slices));

                                system.dE(i, t)     *= -1;
                                system.dEtrot(i, t) *= -1;

                                //update spins
                                spins(i, t) *= -1;
                            }
                }
            

            inline static std::size_t mod_t(std::int64_t a, std::size_t num_trotter_slices){
                //a -> [-1:num_trotter_slices]
                //return a%num_trotter_slices (a>0), num_trotter_slices-1 (a==-1)
                return (a+num_trotter_slices)%num_trotter_slices;
            }
        };

    } // namespace updater
} // namespace openjij

#endif
