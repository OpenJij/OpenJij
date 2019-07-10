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

#ifndef OPENJIJ_SYSTEM_QUANTUM_ISING_HPP__
#define OPENJIJ_SYSTEM_QUANTUM_ISING_HPP__

#include <system/system.hpp>
#include <graph/all.hpp>
#include <vector>
#include <utility>

namespace openjij {
    namespace system {

        /**
         * @brief trotterized spin (std::vector<Spins>)
         */
        using TrotterSpins = std::vector<graph::Spins>;

        /**
         * @brief naive TransverseIsing structure with discrete-time trotter spins
         *
         * @tparam GraphType
         */
        template<typename GraphType>
            struct TransverseIsing {
                using system_type = transverse_field_system;
                using FloatType = typename GraphType::value_type;

                /**
                 * @brief TransverseIsing Constructor
                 *
                 * @param init_trotter_spins
                 * @param init_interaction
                 */
                TransverseIsing(const TrotterSpins& init_trotter_spins, const GraphType& init_interaction, FloatType gamma)
                : trotter_spins(init_trotter_spins), interaction(init_interaction), gamma(gamma){
                }

                /**
                 * @brief TransverseIsing Constuctor with initial classical spins
                 *
                 * @param classical_spins initial classical spins
                 * @param init_interaction
                 * @param num_trotter_slices
                 */
                TransverseIsing(const graph::Spins& classical_spins, const GraphType& init_interaction, FloatType gamma, size_t num_trotter_slices)
                : trotter_spins(num_trotter_slices), interaction(init_interaction), gamma(gamma){
                    //initialize trotter_spins with classical_spins
                    for(auto& spins : trotter_spins){
                        spins = classical_spins;
                    }
                }

                /**
                 * @brief trotterlized spins
                 */
                TrotterSpins trotter_spins;

                /**
                 * @brief interaction 
                 */
                const GraphType interaction;

                /**
                 * @brief coefficient of transverse field term
                 */
                FloatType gamma;
            };

        template<typename GraphType>
            TransverseIsing<GraphType> make_transverse_ising(const TrotterSpins& init_trotter_spins, const GraphType& init_interaction, double gamma){
                return TransverseIsing<GraphType>(init_trotter_spins, init_interaction, static_cast<typename GraphType::value_type>(gamma));
            }

        template<typename GraphType>
            TransverseIsing<GraphType> make_transverse_ising(const graph::Spins& classical_spins, const GraphType& init_interaction, double gamma, std::size_t num_trotter_slices){
                return TransverseIsing<GraphType>(classical_spins, init_interaction, static_cast<typename GraphType::value_type>(gamma), num_trotter_slices);
            }
    } // namespace system
} // namespace openjij

#endif
