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

#ifndef OPENJIJ_UTILITY_GPU_HANDLE_ERROR_HPP__
#define OPENJIJ_UTILITY_GPU_HANDLE_ERROR_HPP__

#ifdef USE_CUDA

#include <iostream>


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

namespace openjij {
namespace utility {
namespace cuda {

// macro for detecting errors

#ifndef NDEBUG

#ifndef HANDLE_ERROR_CUDA
#define HANDLE_ERROR_CUDA(expr)                                                \
  {                                                                            \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess)                                                    \
      std::cerr << "cuda error_code: " << err                                  \
                << " err_name: " << cudaGetErrorString(err) << " at "          \
                << __FILE__ << " line " << __LINE__ << std::endl;              \
  }
#endif

#ifndef HANDLE_ERROR_CURAND
#define HANDLE_ERROR_CURAND(expr)                                              \
  {                                                                            \
    curandStatus_t st = (expr);                                                \
    if (st != CURAND_STATUS_SUCCESS)                                           \
      std::cerr << "curand_error: " << st << " at " << __FILE__ << " line "    \
                << __LINE__ << std::endl;                                      \
  }
#endif

#ifndef HANDLE_ERROR_CUBLAS
#define HANDLE_ERROR_CUBLAS(expr)                                              \
  {                                                                            \
    cublasStatus_t st = (expr);                                                \
    if (st != CUBLAS_STATUS_SUCCESS)                                           \
      std::cerr << "cublas_error: " << st << " at " << __FILE__ << " line "    \
                << __LINE__ << std::endl;                                      \
  }
#endif

#else

#ifndef HANDLE_ERROR_CUDA
#define HANDLE_ERROR_CUDA(expr) expr
#endif

#ifndef HANDLE_ERROR_CURAND
#define HANDLE_ERROR_CURAND(expr) expr
#endif

#ifndef HANDLE_ERROR_CUBLAS
#define HANDLE_ERROR_CUBLAS(expr) expr
#endif

#endif
} // namespace cuda
} // namespace utility
} // namespace openjij

#endif
#endif
