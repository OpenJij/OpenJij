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

#ifndef OPENJIJ_SYSTEM_GPU_CHIMERA_CUDA_INDEX_HPP__
#define OPENJIJ_SYSTEM_GPU_CHIMERA_CUDA_INDEX_HPP__

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>

namespace openjij {
    namespace system {

        /**
         * @brief chimera information struct (row, column, number of trotter slices)
         */
        struct ChimeraInfo{
            std::size_t rows;
            std::size_t cols;
            std::size_t trotters;
            constexpr static std::size_t chimera_unitsize = 8;
        };

        //for both cuda host and device (kernel)
        namespace chimera_gpu {

            __host__ __device__ __forceinline__ std::uint64_t glIdx_x(ChimeraInfo info, std::uint64_t r, std::uint64_t c, std::uint64_t i, std::uint64_t t){
                assert(r < info.rows);
                assert(c < info.cols);
                assert(i < info.chimera_unitsize);
                assert(t < info.trotters);
                return info.chimera_unitsize*c + i;
            }
            __host__ __device__ __forceinline__ std::uint64_t glIdx_y(ChimeraInfo info, std::uint64_t r, std::uint64_t c, std::uint64_t i, std::uint64_t t){
                assert(r < info.rows);
                assert(c < info.cols);
                assert(i < info.chimera_unitsize);
                assert(t < info.trotters);
                return r;
            }
            __host__ __device__ __forceinline__ std::uint64_t glIdx_z(ChimeraInfo info, std::uint64_t r, std::uint64_t c, std::uint64_t i, std::uint64_t t){
                assert(r < info.rows);
                assert(c < info.cols);
                assert(i < info.chimera_unitsize);
                assert(t < info.trotters);
                return t;
            }

            __host__ __device__ __forceinline__ std::uint64_t glIdx(ChimeraInfo info, std::uint64_t r, std::uint64_t c, std::uint64_t i, std::uint64_t t){
                return (info.chimera_unitsize*info.cols*info.rows) * glIdx_z(info,r,c,i,t)
                    +(info.chimera_unitsize*info.cols) * glIdx_y(info,r,c,i,t)
                    +glIdx_x(info,r,c,i,t);
            }

            __host__ __device__ __forceinline__ std::uint64_t glIdx(ChimeraInfo info, std::uint64_t r, std::uint64_t c, std::uint64_t i){
                return glIdx(info, r, c, i, 0);
            }

        } // namespace chimera_gpu
    } // namespace system
} // namespace openjij

#endif
#endif
