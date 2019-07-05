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

namespace openjij {
    namespace system {

        /**
         * @brief QuantumIsing structure with discrete-time trotter spins
         *
         * @tparam GraphType
         */
        template<typename GraphType>
            struct QuantumIsing {
                using system_type = quantum_system;
                using TrotterSpins = std::vector<graph::Spins>;

                /**
                 * @brief QuantumIsing Constructor
                 *
                 * @param init_trotter_spins
                 * @param init_interaction
                 */
                QuantumIsing(const TrotterSpins& init_trotter_spins, const GraphType& init_interaction)
                : trotter_spins(init_trotter_spins), interaction(init_interaction){
                }

                /**
                 * @brief QuantumIsing Constuctor with initial classical spins
                 *
                 * @param classical_spins initial classical spins
                 * @param init_interaction
                 * @param num_trotter_slices
                 */
                QuantumIsing(const graph::Spins& classical_spins, const GraphType& init_interaction, size_t num_trotter_slices)
                : trotter_spins(num_trotter_slices), interaction(init_interaction){
                    //initialize trotter_spins with classical_spins
                    for(auto& spins : trotter_spins){
                        spins = classical_spins;
                    }
                }

                TrotterSpins trotter_spins;
                const GraphType interaction;
            };
    } // namespace system
} // namespace openjij

#endif
