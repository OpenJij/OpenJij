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

#ifndef OPENJIJ_RESULT_GET_SOLUTION_HPP__
#define OPENJIJ_RESULT_GET_SOLUTION_HPP__

#include <graph/all.hpp>
#include <system/all.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef USE_CUDA
#include <utility/gpu/memory.hpp>
#endif

namespace openjij {
    namespace result {

        /**
         * @brief get solution of classical ising system (no Eigen implementation)
         *
         * @tparam GraphType graph type
         * @param system classical ising system without Eigen implementation
         *
         * @return solution
         */
        template<typename GraphType>
        const graph::Spins get_solution(const system::ClassicalIsing<GraphType, false>& system){
            return system.spin;
        }

        /**
         * @brief get solution of classical ising system (with Eigen implementation)
         *
         * @tparam GraphType graph type
         * @param system classical ising system with Eigen implementation
         *
         * @return solution
         */
        template<typename GraphType>
        const graph::Spins get_solution(const system::ClassicalIsing<GraphType, true>& system){
            //convert from Eigen::Vector to std::vector
            graph::Spins ret_spins(system.num_spins);
            for(std::size_t i=0; i<system.num_spins; i++){
                ret_spins[i] = system.spin(i);
            }
            return ret_spins;
        }

        /**
         * @brief get solution of transverse ising system (no Eigen implementation)
         *
         * @tparam GraphType
         * @param system
         *
         * @return solution
         */
        template<typename GraphType>
        const graph::Spins get_solution(const system::TransverseIsing<GraphType, false>& system){
            std::size_t mininum_trotter = 0;
            double energy = 0.0;
            double min_energy = std::numeric_limits<double>::max();
            for (std::size_t t=0; t<system.trotter_spins.size(); t++){
                energy = system.interaction.calc_energy(system.trotter_spins[t]);
                if(energy < min_energy){
                    mininum_trotter = t;
                    min_energy = energy;
                }
            }
           return system.trotter_spins[mininum_trotter];
        }

        /**
         * @brief get solution of transverse ising system (with Eigen implementation)
         *
         * @tparam GraphType
         * @param system
         *
         * @return solution
         */
        template<typename GraphType>
        const graph::Spins get_solution(const system::TransverseIsing<GraphType, true>& system){
            std::size_t minimum_trotter = 0;
            //aliases
            auto& spins = system.trotter_spins;
            double energy = 0.0;
            double min_energy = std::numeric_limits<double>::max();
            //get number of trotter slices
            std::size_t num_trotter_slices = system.trotter_spins.cols();
            for (std::size_t t=0; t<num_trotter_slices; t++){
                // calculate classical energy in each classical spin
                energy = spins.col(t).transpose() * system.interaction * spins.col(t);
                if(energy < min_energy){
                    minimum_trotter = t;
                    min_energy = energy;
                }
            }

            //convert from Eigen::Vector to std::vector
            graph::Spins ret_spins(system.num_classical_spins);
            for(std::size_t i=0; i<system.num_classical_spins; i++){
                ret_spins[i] = spins(i, minimum_trotter);
            }
            return ret_spins;
        }

#ifdef USE_CUDA
        template<typename FloatType,
            std::size_t rows_per_block,
            std::size_t cols_per_block,
            std::size_t trotters_per_block>
        const graph::Spins get_solution(const system::ChimeraTransverseGPU<FloatType, rows_per_block, cols_per_block, trotters_per_block>& system){

            std::size_t localsize = system.info.rows*system.info.cols*system.info.chimera_unitsize;
            std::size_t globalsize = localsize * system.info.trotters;

            graph::Spins ret_spins(localsize);

            //host pinned memory
            auto temp_spin = utility::cuda::make_host_unique<int32_t[]>(globalsize);
            HANDLE_ERROR_CUDA(cudaMemcpy(temp_spin.get(), system.spin.get(), globalsize*sizeof(int32_t), cudaMemcpyDeviceToHost));

            for(std::size_t r=0; r<system.info.rows; r++){
                for(std::size_t c=0; c<system.info.cols; c++){
                    for(std::size_t i=0; i<system.info.chimera_unitsize; i++){
                        double mean = 0;
                        for(std::size_t t=0; t<system.info.trotters; t++){
                            mean += temp_spin[system::chimera_cuda::glIdx(system.info, r,c,i,t)];
                        }
                        mean /= (double)system.info.trotters;
                        ret_spins[system::chimera_cuda::glIdx(system.info, r,c,i)] = mean>0 ? 1 : mean<0 ? -1 : 1;
                    }
                }
            }

            return ret_spins;
        }
#endif


    } // namespace result
} // namespace openjij

#endif
