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

#ifndef OPENJIJ_UTILITY_GPU_CUBLAS_HPP__
#define OPENJIJ_UTILITY_GPU_CUBLAS_HPP__

#ifdef USE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "openjij/utility/gpu/handle_error.hpp"
#include "openjij/utility/gpu/memory.hpp"

namespace openjij {
namespace utility {
namespace cuda {

// cuda datatype
template <typename FloatType> struct cudaDataType_impl;

template <> struct cudaDataType_impl<float> {
  constexpr static cudaDataType_t type = CUDA_R_32F;
};

template <> struct cudaDataType_impl<double> {
  constexpr static cudaDataType_t type = CUDA_R_64F;
};

// cublas get maximal value
template <typename FloatType>
inline cublasStatus_t cublas_Iamax_impl(cublasHandle_t handle, int n,
                                        const FloatType *x, int incx,
                                        int *result);

template <>
inline cublasStatus_t cublas_Iamax_impl(cublasHandle_t handle, int n,
                                        const float *x, int incx, int *result) {
  return cublasIsamax(handle, n, x, incx, result);
}

template <>
inline cublasStatus_t cublas_Iamax_impl(cublasHandle_t handle, int n,
                                        const double *x, int incx,
                                        int *result) {
  return cublasIdamax(handle, n, x, incx, result);
}

// cublas dot product
template <typename FloatType>
inline cublasStatus_t
cublas_dot_impl(cublasHandle_t handle, int n, const FloatType *x, int incx,
                const FloatType *y, int incy, FloatType *result);

template <>
inline cublasStatus_t cublas_dot_impl(cublasHandle_t handle, int n,
                                      const float *x, int incx, const float *y,
                                      int incy, float *result) {
  return cublasSdot(handle, n, x, incx, y, incy, result);
}

template <>
inline cublasStatus_t
cublas_dot_impl(cublasHandle_t handle, int n, const double *x, int incx,
                const double *y, int incy, double *result) {
  return cublasDdot(handle, n, x, incx, y, incy, result);
}

/**
 * @brief cuBLAS wrapper
 */
class CuBLASWrapper {
public:
  CuBLASWrapper() {
    // generate cuBLAS instance
    HANDLE_ERROR_CUBLAS(cublasCreate(&_handle));
    // use tensor core if possible
    HANDLE_ERROR_CUBLAS(cublasSetMathMode(_handle, CUBLAS_TENSOR_OP_MATH));
  }

  CuBLASWrapper(CuBLASWrapper &&obj) noexcept {
    // move cuBLAS handler
    this->_handle = obj._handle;
    obj._handle = NULL;
  }

  ~CuBLASWrapper() {
    // destroy generator
    if (_handle != NULL)
      HANDLE_ERROR_CUBLAS(cublasDestroy(_handle));
  }

  template <typename FloatType>
  inline void SgemmEx(cublasOperation_t transa, cublasOperation_t transb, int m,
                      int n, int k, const float *alpha,
                      const utility::cuda::unique_dev_ptr<FloatType> &A,
                      int lda,
                      const utility::cuda::unique_dev_ptr<FloatType> &B,
                      int ldb, const float *beta,
                      utility::cuda::unique_dev_ptr<FloatType> &C, int ldc) {

    cublasPointerMode_t mode;
    HANDLE_ERROR_CUBLAS(cublasGetPointerMode(_handle, &mode));
    HANDLE_ERROR_CUBLAS(
        cublasSetPointerMode(_handle, CUBLAS_POINTER_MODE_HOST));
    HANDLE_ERROR_CUBLAS(cublasSgemmEx(
        _handle, transa, transb, m, n, k, alpha, A.get(),
        cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
        lda, B.get(),
        cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
        ldb, beta, C.get(),
        cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
        ldc));
    HANDLE_ERROR_CUBLAS(cublasSetPointerMode(_handle, mode));
  }

  /**
   * @brief matrix multiplication
   * C_mn = A_mk B_kn
   *
   * @tparam FloatType
   * @param m
   * @param k
   * @param n
   * @param A
   * @param B
   * @param C
   */
  template <typename FloatType>
  inline void matmul(int m, int k, int n,
                     const utility::cuda::unique_dev_ptr<FloatType> &A,
                     const utility::cuda::unique_dev_ptr<FloatType> &B,
                     utility::cuda::unique_dev_ptr<FloatType> &C,
                     cublasOperation_t transa = CUBLAS_OP_N,
                     cublasOperation_t transb = CUBLAS_OP_N) {
    typename std::remove_extent<FloatType>::type alpha = 1.0;
    typename std::remove_extent<FloatType>::type beta = 0;

    cublasPointerMode_t mode;
    HANDLE_ERROR_CUBLAS(cublasGetPointerMode(_handle, &mode));
    HANDLE_ERROR_CUBLAS(
        cublasSetPointerMode(_handle, CUBLAS_POINTER_MODE_HOST));
    HANDLE_ERROR_CUBLAS(cublasSgemmEx(
        _handle, transa, transb, m, n, k, &alpha, A.get(),
        cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
        m, B.get(),
        cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
        k, &beta, C.get(),
        cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
        m));
    HANDLE_ERROR_CUBLAS(cublasSetPointerMode(_handle, mode));
  }

  /**
   * @brief wrap function of cublasIsamax
   * Note: returned value will be 1-indexed!
   *
   * @tparam FloatType
   * @param n
   * @param x
   * @param incx
   * @param result
   */
  template <typename FloatType>
  inline void Iamax(int n, const FloatType *x, int incx, int *result) {
    cublasPointerMode_t mode;
    HANDLE_ERROR_CUBLAS(cublasGetPointerMode(_handle, &mode));
    // set pointermode to device
    HANDLE_ERROR_CUBLAS(
        cublasSetPointerMode(_handle, CUBLAS_POINTER_MODE_DEVICE));
    HANDLE_ERROR_CUBLAS(cublas_Iamax_impl(_handle, n, x, incx, result));
    // reset pointermode
    HANDLE_ERROR_CUBLAS(cublasSetPointerMode(_handle, mode));
  }

  /**
   * @brief return the index of maximal element
   * Note: returned value will be 1-indexed!
   *
   * @tparam FloatType
   * @param n
   * @param x
   * @param result
   */
  template <typename FloatType>
  inline void
  absmax_val_index(int n, const utility::cuda::unique_dev_ptr<FloatType[]> &x,
                   utility::cuda::unique_dev_ptr<int[]> &result) {
    Iamax(n, x.get(), 1, result.get());
  }

  /**
   * @brief wrap function of cublasXdot
   *
   * @tparam FloatType
   * @param n
   * @param x
   * @param incx
   * @param y
   * @param incy
   * @param result
   */
  template <typename FloatType>
  inline void dot(int n, const FloatType *x, int incx, const FloatType *y,
                  int incy, FloatType *result) {
    cublasPointerMode_t mode;
    HANDLE_ERROR_CUBLAS(cublasGetPointerMode(_handle, &mode));
    HANDLE_ERROR_CUBLAS(
        cublasSetPointerMode(_handle, CUBLAS_POINTER_MODE_DEVICE));
    // set pointermode to device
    HANDLE_ERROR_CUBLAS(cublas_dot_impl(_handle, n, x, incx, y, incy, result));
    // reset pointermode
    HANDLE_ERROR_CUBLAS(cublasSetPointerMode(_handle, mode));
  }

  /**
   * @brief return the dot product of x and y
   *
   * @tparam FloatType
   * @param n
   * @param x
   * @param y
   * @param result
   */
  template <typename FloatType>
  inline void dot(int n, const utility::cuda::unique_dev_ptr<FloatType[]> &x,
                  const utility::cuda::unique_dev_ptr<FloatType[]> &y,
                  utility::cuda::unique_dev_ptr<FloatType[]> &result) {
    dot(n, x.get(), 1, y.get(), 1, result.get());
  }

private:
  cublasHandle_t _handle;
};

} // namespace cuda
} // namespace utility
} // namespace openjij

#endif
#endif
