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

#ifndef OPENJIJ_UTILITY_GPU_MEMORY_HPP__
#define OPENJIJ_UTILITY_GPU_MEMORY_HPP__

#ifdef USE_CUDA

#include <memory>
#include <type_traits>

#include <cuda_runtime.h>

#include "openjij/utility/gpu/handle_error.hpp"

namespace openjij {
namespace utility {
namespace cuda {

/**
 * @brief deleter for cuda device memory
 */
struct deleter_dev {
  void operator()(void *ptr) const { HANDLE_ERROR_CUDA(cudaFree(ptr)); }
};

/**
 * @brief deleter for cuda pinned host memory (cudaHostFree)
 */
struct deleter_host {
  void operator()(void *ptr) const { HANDLE_ERROR_CUDA(cudaFreeHost(ptr)); }
};

/**
 * @brief unique_ptr for cuda device memory
 *
 * @tparam T
 */
template <typename T> using unique_dev_ptr = std::unique_ptr<T, deleter_dev>;

/**
 * @brief unique_ptr for cuda host memory
 *
 * @tparam T
 */
template <typename T> using unique_host_ptr = std::unique_ptr<T, deleter_host>;

/**
 * @brief make_unique for cuda device memory
 *
 * @tparam T
 * @param n
 *
 * @return unique_dev_ptr object
 */
template <typename T> cuda::unique_dev_ptr<T> make_dev_unique(std::size_t n) {
  static_assert(std::is_array<T>::value, "T must be an array.");
  using U = typename std::remove_extent<T>::type;
  U *ptr;
  HANDLE_ERROR_CUDA(cudaMalloc(reinterpret_cast<void **>(&ptr), sizeof(U) * n));
  return cuda::unique_dev_ptr<T>{ptr};
}

/**
 * @brief make_unique for cuda pinned host memory (page-locked memory)
 *
 * @tparam T
 * @param n
 *
 * @return unique_host_ptr object
 */
template <typename T> cuda::unique_host_ptr<T> make_host_unique(std::size_t n) {
  static_assert(std::is_array<T>::value, "T must be an array.");
  using U = typename std::remove_extent<T>::type;
  U *ptr;
  HANDLE_ERROR_CUDA(
      cudaMallocHost(reinterpret_cast<void **>(&ptr), sizeof(U) * n));
  return cuda::unique_host_ptr<T>{ptr};
}

} // namespace cuda
} // namespace utility
} // namespace openjij

#endif
#endif
