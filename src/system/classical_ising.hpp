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

#ifndef OPENJIJ_SYSTEM_CLASSICAL_ISING_HPP__
#define OPENJIJ_SYSTEM_CLASSICAL_ISING_HPP__

#include <cassert>
#include <system/system.hpp>
#include <graph/all.hpp>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Dense>


namespace openjij {
    namespace system {

        /**
         * @brief ClassicalIsing structure (system for classical Ising model)
         *
         * @tparam GraphType type of graph
         */
        template<typename GraphType>
            struct ClassicalIsing {
                using system_type = classical_system;
                
                /**
                 * @brief Constructor to initialize spin and interaction
                 *
                 * @param spin
                 * @param interaction
                 */
                ClassicalIsing(const graph::Spins& init_spin, const GraphType& init_interaction)
                    : spin{init_spin}, interaction{init_interaction} {
                        assert(init_spin.size() == init_interaction.get_num_spins());
                    }

                graph::Spins spin;
                const GraphType interaction;
            };


        /**
         * @brief ClassicalIsing structure for Dense graph
         *
         * @tparam FloatType type of floating-point
         */
        template<typename FloatType>
            struct ClassicalIsing<graph::Dense<FloatType>>{
                using system_type = classical_system;

                using MatrixX = Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic>;
                using VectorX = Eigen::Matrix<FloatType, Eigen::Dynamic, 1>;

                /**
                 * @brief Constructor to initialize spin and interaction
                 *
                 * @param spin
                 * @param interaction
                 */
                ClassicalIsing(const graph::Spins& init_spin, const graph::Dense<FloatType>& init_interaction)
                : spin(init_interaction.get_num_spins()+1), interaction(init_interaction.get_num_spins()+1, init_interaction.get_num_spins()+1){
                    assert(init_spin.size() == init_interaction.get_num_spins());

                    //initialize spin
                    for(size_t i=0; i<init_spin.size(); i++){
                        spin(i) = init_spin[i];
                    }

                    //for local field
                    spin[init_spin.size()] = 1;

                    //initialize interaction
                    interaction = Eigen::MatrixXd::Zero(init_interaction.get_num_spins()+1, init_interaction.get_num_spins()+1);
                    for(size_t i=0; i<init_interaction.get_num_spins(); i++){
                        for(size_t j=i+1; j<init_interaction.get_num_spins(); j++){
                            interaction(i,j) = init_interaction.J(i,j);
                            interaction(j,i) = init_interaction.J(i,j);
                        }
                    }

                    //for local field
                    for(size_t i=0; i<init_interaction.get_num_spins(); i++){
                            interaction(i,init_interaction.get_num_spins()) = init_interaction.h(i);
                            interaction(init_interaction.get_num_spins(),i) = init_interaction.h(i);
                    }

                    //for local field
                    interaction(init_interaction.get_num_spins(),init_interaction.get_num_spins()) = 1;
                }

                VectorX spin;
                MatrixX interaction;
            };
    } // namespace system
} // namespace openjij

#endif
