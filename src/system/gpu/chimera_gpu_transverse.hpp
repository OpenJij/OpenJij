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

#include <cstddef>
#include <system/system.hpp>
#include <system/transverse_ising.hpp>
#include <graph/all.hpp>
#include <utility/gpu/memory.hpp>
#include <vector>

namespace openjij {
    namespace system {

        /**
         * @brief chimera interactions structure
         *
         * @tparam FloatType
         */
        template<typename FloatType>
            struct ChimeraInteractions{
                using value_type = FloatType;
                utility::cuda::unique_dev_ptr<FloatType[]> J_out_p;
                utility::cuda::unique_dev_ptr<FloatType[]> J_out_n;
                utility::cuda::unique_dev_ptr<FloatType[]> J_in_04;
                utility::cuda::unique_dev_ptr<FloatType[]> J_in_15;
                utility::cuda::unique_dev_ptr<FloatType[]> J_in_26;
                utility::cuda::unique_dev_ptr<FloatType[]> J_in_37;
                utility::cuda::unique_dev_ptr<FloatType[]> h;

                ChimeraInteractions(std::size_t n)
                    : J_out_p(utility::cuda::make_dev_unique<FloatType[]>(n)),
                    J_out_n(utility::cuda::make_dev_unique<FloatType[]>(n)),
                    J_in_04(utility::cuda::make_dev_unique<FloatType[]>(n)),
                    J_in_15(utility::cuda::make_dev_unique<FloatType[]>(n)),
                    J_in_26(utility::cuda::make_dev_unique<FloatType[]>(n)),
                    J_in_37(utility::cuda::make_dev_unique<FloatType[]>(n)),
                    h(utility::cuda::make_dev_unique<FloatType[]>(n)){
                    }
            };

        /**
         * @brief chimera information struct (row, column, num_trotter)
         */
        struct ChimeraInfo{
            std::size_t rows;
            std::size_t cols;
            std::size_t trotters;
        };

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
                    :gamma(gamma), 
                    info({init_interaction.get_num_row(), init_interaction.get_num_column(), init_trotter_spins.size()}),
                    interaction(init_interaction.get_num_row()*init_interaction.get_num_column()*chimera_unitsize),
                    spin(utility::cuda::make_dev_unique<std::int32_t[]>(init_interaction.get_num_row()*init_interaction.get_num_column()*chimera_unitsize*init_trotter_spins.size())){
                    }

                /**
                 * @brief TransverseIsing Constuctor with initial classical spins
                 *
                 * @param classical_spins initial classical spins
                 * @param init_interaction
                 * @param num_trotter_slices
                 */
                ChimeraTransverseGPU(const graph::Spins& classical_spins, const graph::Chimera<FloatType>& init_interaction, FloatType gamma, size_t num_trotter_slices)
                    :gamma(gamma), 
                    info({init_interaction.get_num_row(), init_interaction.get_num_column(), num_trotter_slices}),
                    interaction(init_interaction.get_num_row()*init_interaction.get_num_column()*chimera_unitsize),
                    spin(utility::cuda::make_dev_unique<std::int32_t[]>(init_interaction.get_num_row()*init_interaction.get_num_column()*chimera_unitsize*init_trotter_spins.size())){
                        //initialize trotter_spins with classical_spins
                        TrotterSpins trotter_spins;
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

                /**
                 * @brief chimera graph information
                 */
                const ChimeraInfo info;

                /**
                 * @brief interaction pointer to gpu memory. DO NOT ACCESS FROM CPU.
                 */
                const ChimeraInteractions<FloatType> interaction;

                /**
                 * @brief spin pointer to gpu memory. DO NOT ACCESS FROM CPU.
                 */
                utility::cuda::unique_dev_ptr<std::int32_t[]> spin;

                private:

                    /**
                     * @brief send interaction information to GPU device
                     *
                     * @param init_interaction
                     */
                    inline void send_interaction_to_gpu(const graph::Chimera<FloatType>& init_interaction){
                        //generate temporary interaction
                    }

            };

    } // namespace system
} // namespace openjij

#endif
#endif
