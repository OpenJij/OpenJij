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
#include <system/gpu/chimera_cuda/index.hpp>
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
                utility::cuda::unique_dev_ptr<FloatType[]> J_out_p; //previous 
                utility::cuda::unique_dev_ptr<FloatType[]> J_out_n; //next
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
         * @brief Chimera Transverse Ising structure with cuda
         *
         * @tparam FloatType
         */
        template<typename FloatType>
            struct ChimeraTransverseGPU {
                using system_type = transverse_field_system;

                /**
                 * @brief Chimera transverse ising constructor
                 *
                 * @param init_trotter_spins
                 * @param init_interaction
                 * @param gamma
                 * @param device_num
                 */
                ChimeraTransverseGPU(const TrotterSpins& init_trotter_spins, const graph::Chimera<FloatType>& init_interaction, FloatType gamma, int device_num=0)
                    :gamma(gamma), 
                    info({init_interaction.get_num_row(), init_interaction.get_num_column(), init_trotter_spins.size()}),
                    interaction(init_interaction.get_num_row()*init_interaction.get_num_column()*info.chimera_unitsize),
                    spin(utility::cuda::make_dev_unique<std::int32_t[]>(init_interaction.get_num_row()*init_interaction.get_num_column()*info.chimera_unitsize*init_trotter_spins.size())){
                        //initialize
                        initialize_gpu(init_interaction, init_trotter_spins, device_num);
                    }

                /**
                 * @brief Chimera transverse ising constructor
                 *
                 * @param classical_spins
                 * @param init_interaction
                 * @param gamma
                 * @param num_trotter_slices
                 * @param device_num
                 */
                ChimeraTransverseGPU(const graph::Spins& classical_spins, const graph::Chimera<FloatType>& init_interaction, FloatType gamma, size_t num_trotter_slices, int device_num=0)
                    :gamma(gamma), 
                    info({init_interaction.get_num_row(), init_interaction.get_num_column(), num_trotter_slices}),
                    interaction(init_interaction.get_num_row()*init_interaction.get_num_column()*info.chimera_unitsize),
                    spin(utility::cuda::make_dev_unique<std::int32_t[]>(init_interaction.get_num_row()*init_interaction.get_num_column()*info.chimera_unitsize*num_trotter_slices)){
                        //initialize trotter_spins with classical_spins
                        TrotterSpins trotter_spins;
                        for(auto& spins : trotter_spins){
                            spins = classical_spins;
                        }

                        //initialize
                        initialize_gpu(init_interaction, trotter_spins, device_num);
                    }

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
                    inline void initialize_gpu(const graph::Chimera<FloatType>& init_interaction, const TrotterSpins& trotter_spins, int gpu_num){

                        //specify gpu number
                        HANDLE_ERROR_CUDA(cudaSetDevice(gpu_num));

                        //generate temporary interaction and spin
                        const std::size_t localsize = info.rows*info.cols*info.chimera_unitsize;

                        auto J_out_p = utility::cuda::make_host_unique<FloatType[]>(localsize);
                        auto J_out_n = utility::cuda::make_host_unique<FloatType[]>(localsize);
                        auto J_in_04 = utility::cuda::make_host_unique<FloatType[]>(localsize);
                        auto J_in_15 = utility::cuda::make_host_unique<FloatType[]>(localsize);
                        auto J_in_26 = utility::cuda::make_host_unique<FloatType[]>(localsize);
                        auto J_in_37 = utility::cuda::make_host_unique<FloatType[]>(localsize);
                        auto       h = utility::cuda::make_host_unique<FloatType[]>(localsize);
                        auto temp_spin = utility::cuda::make_host_unique<int32_t[]>(localsize*info.trotters);

                        using namespace chimera_gpu;

                        //copy interaction info to std::vector variables
                        for(size_t r=0; r<info.rows; r++){
                            for(size_t c=0; c<info.cols; c++){
                                for(size_t i=0; i<info.chimera_unitsize; i++){
                                    if(r > 0 && i < 4){
                                        //MINUS_R
                                        J_out_p[glIdx(info,r,c,i)] = init_interaction.J(r,c,i,graph::ChimeraDir::MINUS_R);
                                    }
                                    if(c > 0 && 4 <= i){
                                        //MINUS_C
                                        J_out_p[glIdx(info,r,c,i)] = init_interaction.J(r,c,i,graph::ChimeraDir::MINUS_C);
                                    }
                                    if(r < info.rows-1 && i < 4){
                                        //PLUS_R
                                        J_out_n[glIdx(info,r,c,i)] = init_interaction.J(r,c,i,graph::ChimeraDir::PLUS_R);
                                    }
                                    if(c < info.cols-1 && 4 <= i){
                                        //PLUS_C
                                        J_out_n[glIdx(info,r,c,i)] = init_interaction.J(r,c,i,graph::ChimeraDir::PLUS_C);
                                    }

                                    //inside chimera unit
                                    J_in_04[glIdx(info,r,c,i)] = init_interaction.J(r,c,i,graph::ChimeraDir::IN_0or4);
                                    J_in_15[glIdx(info,r,c,i)] = init_interaction.J(r,c,i,graph::ChimeraDir::IN_1or5);
                                    J_in_26[glIdx(info,r,c,i)] = init_interaction.J(r,c,i,graph::ChimeraDir::IN_2or6);
                                    J_in_37[glIdx(info,r,c,i)] = init_interaction.J(r,c,i,graph::ChimeraDir::IN_3or7);

                                    //local field
                                    h[glIdx(info,r,c,i)] = init_interaction.h(r,c,i);
                                }
                            }
                        }

                        //copy spin info to std::vector variables
                        for(size_t t=0; t<info.trotters; t++){
                            for(size_t r=0; r<info.rows; r++){
                                for(size_t c=0; c<info.cols; c++){
                                    for(size_t i=0; i<info.chimera_unitsize; i++){
                                        temp_spin[glIdx(info,r,c,i,t)] = trotter_spins[t][init_interaction.to_ind(r,c,i)];
                                    }
                                }
                            }
                        }

                        //cudaMemcpy
                        HANDLE_ERROR_CUDA(cudaMemcpy(interaction.J_out_p.get(), J_out_p.get(), localsize*sizeof(FloatType), cudaMemcpyHostToDevice));
                        HANDLE_ERROR_CUDA(cudaMemcpy(interaction.J_out_n.get(), J_out_n.get(), localsize*sizeof(FloatType), cudaMemcpyHostToDevice));
                        HANDLE_ERROR_CUDA(cudaMemcpy(interaction.J_in_04.get(), J_in_04.get(), localsize*sizeof(FloatType), cudaMemcpyHostToDevice));
                        HANDLE_ERROR_CUDA(cudaMemcpy(interaction.J_in_15.get(), J_in_15.get(), localsize*sizeof(FloatType), cudaMemcpyHostToDevice));
                        HANDLE_ERROR_CUDA(cudaMemcpy(interaction.J_in_26.get(), J_in_26.get(), localsize*sizeof(FloatType), cudaMemcpyHostToDevice));
                        HANDLE_ERROR_CUDA(cudaMemcpy(interaction.J_in_37.get(), J_in_37.get(), localsize*sizeof(FloatType), cudaMemcpyHostToDevice));
                        HANDLE_ERROR_CUDA(cudaMemcpy(interaction.h.get(),             h.get(), localsize*sizeof(FloatType), cudaMemcpyHostToDevice));

                        HANDLE_ERROR_CUDA(cudaMemcpy(spin.get(), temp_spin.get(), localsize*info.trotters*sizeof(int32_t), cudaMemcpyHostToDevice));
                    }

            };

        template struct ChimeraTransverseGPU<double>;

    } // namespace system
} // namespace openjij

#endif
#endif
