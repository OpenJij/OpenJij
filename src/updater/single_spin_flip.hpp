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
                                 RandomNumberEngine& random_numder_engine,
                                 const utility::ClassicalUpdaterParameter& parameter) {
                // set probability distribution object
                // to do Metroopolis
                auto urd = std::uniform_real_distribution<>(0, 1.0);

                bool flip_spin;
                // do a iteraction except for the auxiliary spin
                for (std::size_t index = 0; index < system.num_spins; ++index) {

                    flip_spin = false;

                    if (system.dE(index) <= 0 || std::exp( -parameter.beta * system.dE(index)) > urd(random_numder_engine)) {
                        flip_spin = true;
                    }

                    if(flip_spin){

                        // update dE
                        update_dE<GraphType>(system, index);
                        //for (std::size_t k=0; k < system.num_spins; ++k){
                        //    system.dE(k) += 4 * system.spin(index) * system.interaction.coeff(index, k) * system.spin(k);
                        //}

                        system.dE(index) *= -1;
                        system.spin(index) *= -1;
                    }

                    //assure that the dummy spin is not changed.
                    system.spin(system.num_spins) = 1;
                }
            }

          /**
           * @brief update_dE (for Dense Graph)
           *
           * @param system
           * @param graph::Dense
           */

          template<typename T>
          inline static std::enable_if_t<std::is_same_v<T, graph::Dense<FloatType>>>
          update_dE(ClIsing& system, size_t index){
              //specialized for Dense Graph
              system.dE += 4 * system.spin(index) * (system.interaction.row(index).transpose().cwiseProduct(system.spin));
              //for (std::size_t k=0; k < system.num_spins; ++k){
              //    system.dE(k) += 4 * system.spin(index) * system.interaction.coeff(index, k) * system.spin(k);
              //}
          }

          /**
           * @brief update_dE (for Sparse Graph)
           *
           * @brief 
           * @param system
           * @param graph::Sparse
           */
          template<typename T>
          inline static std::enable_if_t<std::is_same_v<T, graph::Sparse<FloatType>>>
          update_dE(ClIsing& system, size_t index){
              //specialized for Sparse Graph
              //for(typename ClIsing::SparseMatrixXx::InnerIterator it(system.interaction, index); it; ++it){
              //    // note that the matrix is row-major.
              //    system.dE(it.col()) += 4 * system.spin(index) * it.value() * system.spin(it.col());
              //}
              system.dE += 4 * system.spin(index) * (system.interaction.row(index).transpose().cwiseProduct(system.spin));
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
                        RandomNumberEngine& random_numder_engine,
                        const utility::TransverseFieldUpdaterParameter& parameter) {

                    //get number of classical spins
                    std::size_t num_classical_spins = system.num_classical_spins;
                    //get number of trotter slices
                    std::size_t num_trotter_slices = system.trotter_spins.cols();

                    auto uid = std::uniform_int_distribution<std::size_t>{0, num_classical_spins-1};
                    auto uid_trotter = std::uniform_int_distribution<std::size_t>{0, num_trotter_slices-1};

                    //do metropolis
                    auto urd = std::uniform_real_distribution<>(0, 1.0);

                    //aliases
                    auto& spins = system.trotter_spins;
                    auto& gamma = system.gamma;
                    auto& beta = parameter.beta;
                    auto& s = parameter.s;

                    for(std::size_t i=0; i<num_classical_spins*num_trotter_slices; i++){
                        //select random trotter slice
                        std::size_t index_trot = uid_trotter(random_numder_engine);
                        //select random classical spin index
                        std::size_t index = uid(random_numder_engine);
                        //do metropolis
                        FloatType dE = 0;
                        assert(index < num_classical_spins);
                        assert(index_trot < num_trotter_slices);
                        //calculate matrix dot product
                        dE += -2 * s * (beta/num_trotter_slices) * spins(index, index_trot)*(system.interaction.row(index).dot(spins.col(index_trot)));

                        //trotter direction
                        dE += -2 * (1/2.) * log(tanh(beta* gamma * (1.0-s) /num_trotter_slices)) * spins(index, index_trot)*
                            (  spins(index, mod_t((int64_t)index_trot+1, num_trotter_slices)) 
                             + spins(index, mod_t((int64_t)index_trot-1, num_trotter_slices)));

                        //metropolis 
                        if(dE < 0 || exp(-dE) > urd(random_numder_engine)){
                            spins(index, index_trot) *= -1;
                        }

                    }
                }

            private: 
            inline static std::size_t mod_t(std::int64_t a, std::size_t num_trotter_slices){
                //a -> [-1:num_trotter_slices]
                //return a%num_trotter_slices (a>0), num_trotter_slices-1 (a==-1)
                return (a+num_trotter_slices)%num_trotter_slices;
            }
        };

    } // namespace updater
} // namespace openjij

#endif
