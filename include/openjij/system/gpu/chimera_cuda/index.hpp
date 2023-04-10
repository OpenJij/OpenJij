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

#ifndef OPENJIJ_SYSTEM_GPU_CHIMERA_CUDA_INDEX_HPP__
#define OPENJIJ_SYSTEM_GPU_CHIMERA_CUDA_INDEX_HPP__

#ifdef USE_CUDA

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

namespace openjij {
namespace system {

/**
 * @brief chimera information struct (row, column, number of trotter slices)
 */
struct ChimeraInfo {
  std::size_t rows;
  std::size_t cols;
  std::size_t trotters;

  /**
   * @brief number of spins in each chimera unit
   */
  constexpr static std::size_t chimera_unitsize = 8;
};

// for both cuda host and device (kernel)
namespace chimera_cuda {

/**
 * @brief get global x index
 *
 * @param info
 * @param r
 * @param c
 * @param i
 * @param t
 *
 * @return
 */
__host__ __device__ __forceinline__ std::uint64_t
glIdx_x(ChimeraInfo info, std::uint64_t r, std::uint64_t c, std::uint64_t i,
        std::uint64_t t) {
  assert(r < info.rows);
  assert(c < info.cols);
  assert(i < info.chimera_unitsize);
  assert(t < info.trotters);
  return info.chimera_unitsize * c + i;
}

/**
 * @brief get global y index
 *
 * @param info
 * @param r
 * @param c
 * @param i
 * @param t
 *
 * @return
 */
__host__ __device__ __forceinline__ std::uint64_t
glIdx_y(ChimeraInfo info, std::uint64_t r, std::uint64_t c, std::uint64_t i,
        std::uint64_t t) {
  assert(r < info.rows);
  assert(c < info.cols);
  assert(i < info.chimera_unitsize);
  assert(t < info.trotters);
  return r;
}

/**
 * @brief get global z index
 *
 * @param info
 * @param r
 * @param c
 * @param i
 * @param t
 *
 * @return
 */
__host__ __device__ __forceinline__ std::uint64_t
glIdx_z(ChimeraInfo info, std::uint64_t r, std::uint64_t c, std::uint64_t i,
        std::uint64_t t) {
  assert(r < info.rows);
  assert(c < info.cols);
  assert(i < info.chimera_unitsize);
  assert(t < info.trotters);
  return t;
}

/**
 * @brief get global index
 *
 * @param info
 * @param r
 * @param c
 * @param i
 * @param t
 *
 * @return
 */
__host__ __device__ __forceinline__ std::uint64_t
glIdx(ChimeraInfo info, std::uint64_t r, std::uint64_t c, std::uint64_t i,
      std::uint64_t t) {
  return (info.chimera_unitsize * info.cols * info.rows) *
             glIdx_z(info, r, c, i, t) +
         (info.chimera_unitsize * info.cols) * glIdx_y(info, r, c, i, t) +
         glIdx_x(info, r, c, i, t);
}

/**
 * @brief get block index
 *
 * @param info
 * @param r
 * @param c
 * @param i
 * @param t
 *
 * @return
 */
template <std::size_t block_row, std::size_t block_col, std::size_t block_trot>
__host__ __device__ __forceinline__ std::uint64_t
bkIdx(ChimeraInfo info, std::uint64_t b_r, std::uint64_t b_c, std::uint64_t i,
      std::uint64_t b_t) {
  return (info.chimera_unitsize * block_col * block_row) * b_t +
         (info.chimera_unitsize * block_col) * b_r +
         (info.chimera_unitsize) * b_c + i;
}

/**
 * @brief get extended (padding) block index
 *
 * @param info
 * @param r
 * @param c
 * @param i
 * @param t
 *
 * @return
 */
template <std::size_t block_row, std::size_t block_col, std::size_t block_trot>
__host__ __device__ __forceinline__ std::uint64_t
bkIdx_ext(ChimeraInfo info, std::int64_t b_r, std::int64_t b_c, std::int64_t i,
          std::int64_t b_t) {
  return bkIdx<block_row + 2, block_col + 2, block_trot + 2>(
      info, b_r + 1, b_c + 1, i, b_t + 1);
}

/**
 * @brief get global index
 *
 * @param info
 * @param r
 * @param c
 * @param i
 *
 * @return
 */
__host__ __device__ __forceinline__ std::uint64_t
glIdx(ChimeraInfo info, std::uint64_t r, std::uint64_t c, std::uint64_t i) {
  return glIdx(info, r, c, i, 0);
}

/**
 * @brief get index i
 *
 * @param info
 * @param x
 * @param y
 * @param z
 *
 * @return
 */
__host__ __device__ __forceinline__ std::uint64_t
idx_i(ChimeraInfo info, std::uint64_t x, std::uint64_t y, std::uint64_t z) {
  return x % info.chimera_unitsize;
}

/**
 * @brief get index c
 *
 * @param info
 * @param x
 * @param y
 * @param z
 *
 * @return
 */
__host__ __device__ __forceinline__ std::uint64_t
idx_c(ChimeraInfo info, std::uint64_t x, std::uint64_t y, std::uint64_t z) {
  return x / info.chimera_unitsize;
}

/**
 * @brief get index r
 *
 * @param info
 * @param x
 * @param y
 * @param z
 *
 * @return
 */
__host__ __device__ __forceinline__ std::uint64_t
idx_r(ChimeraInfo info, std::uint64_t x, std::uint64_t y, std::uint64_t z) {
  return y;
}

/**
 * @brief get index t
 *
 * @param info
 * @param x
 * @param y
 * @param z
 *
 * @return
 */
__host__ __device__ __forceinline__ std::uint64_t
idx_t(ChimeraInfo info, std::uint64_t x, std::uint64_t y, std::uint64_t z) {
  return z;
}

} // namespace chimera_cuda
} // namespace system
} // namespace openjij

#endif
#endif
