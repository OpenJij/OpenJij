//    Copyright 2023 Jij Inc.

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

#include "openjij/system/gpu/chimera_cuda/index.hpp"

namespace openjij {
namespace system {
namespace chimera_cuda {

// updater interface

template <typename FloatType, std::size_t block_row, std::size_t block_col,
          std::size_t block_trot>
void metropolis_interface(int32_t sw, int32_t *spin, const FloatType *rand,
                          const FloatType *J_out_p, const FloatType *J_out_n,
                          const FloatType *J_in_04, const FloatType *J_in_15,
                          const FloatType *J_in_26, const FloatType *J_in_37,
                          const FloatType *h, const ChimeraInfo &info,
                          const dim3 &grid, const dim3 &block, double beta,
                          FloatType gamma, double s);
} // namespace chimera_cuda
} // namespace system
} // namespace openjij

#endif
#endif
