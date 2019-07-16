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

#ifndef OPENJIJ_SYSTEM_CHIMERA_GPU_TRANSVERSE_HPP__
#define OPENJIJ_SYSTEM_CHIMERA_GPU_TRANSVERSE_HPP__

#ifdef USE_CUDA

#include <system/system.hpp>
#include <system/transverse_ising.hpp>
#include <system/gpu/chimera_interactions.hpp>
#include <graph/all.hpp>
#include <utility/gpu/memory.hpp>

namespace openjij {
    namespace system {

        /**
         * @brief Chimera Transverse Ising structure with cuda
         *
         * @tparam FloatType
         */
        template<typename FloatType>
            struct ChimeraTransverseGPU {
                using system_type = transverse_field_system;

                /**
                 * @brief TransverseIsing Constructor
                 *
                 * @param init_trotter_spins
                 * @param init_interaction
                 */
                ChimeraTransverseGPU(const TrotterSpins& init_trotter_spins, const graph::Chimera<FloatType>& init_interaction, FloatType gamma)
                    :gamma(gamma), interaction(init_interaction.get_num_row()*init_interaction.get_num_column()*chimera_unitsize){
                    }

                /**
                 * @brief TransverseIsing Constuctor with initial classical spins
                 *
                 * @param classical_spins initial classical spins
                 * @param init_interaction
                 * @param num_trotter_slices
                 */
                ChimeraTransverseGPU(const graph::Spins& classical_spins, const graph::Chimera<FloatType>& init_interaction, FloatType gamma, size_t num_trotter_slices)
                    :gamma(gamma), interaction(init_interaction.get_num_row()*init_interaction.get_num_column()*chimera_unitsize){
                        //initialize trotter_spins with classical_spins
                        TrotterSpins trotter_spins;
                        assert(trotter_spins.size() >= 2);
                        for(auto& spins : trotter_spins){
                            spins = classical_spins;
                        }
                    }

                /**
                 * @brief number of spins in single chimera unit
                 */
                constexpr static std::size_t chimera_unitsize = 8;

                /**
                 * @brief coefficient of transverse field term
                 */
                FloatType gamma;

                const ChimeraInteractions<FloatType> interaction;

            };

    } // namespace system
} // namespace openjij

#endif
#endif
