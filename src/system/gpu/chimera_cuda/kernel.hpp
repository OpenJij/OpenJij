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

#ifndef OPENJIJ_SYSTEM_GPU_CHIMERA_CUDA_KERNEL_HPP__
#define OPENJIJ_SYSTEM_GPU_CHIMERA_CUDA_KERNEL_HPP__

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <system/gpu/chimera_gpu_transverse.hpp>
#include <system/gpu/chimera_cuda/index.hpp>
#include <utility/random.hpp>

namespace openjij {
    namespace system {
        namespace chimera_cuda {

            //updater interface

            template<
                typename FloatType,
                std::size_t block_row,
                std::size_t block_col,
                std::size_t block_trot,
                curandRngType_t rng_type
                > 
            FloatType update(
                    ChimeraTransverseGPU<FloatType, block_row, block_col, block_trot>& system,
                    utility::cuda::CurandWrapper<FloatType, rng_type>& random_engine,
                    double beta, FloatType gamma, double s);

            void test();
        } // namespace chimera_cuda
    } // namespace system
} // namespace openjij

#endif
#endif
