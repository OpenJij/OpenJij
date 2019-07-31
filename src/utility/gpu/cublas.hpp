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

#ifndef OPENJIJ_UTILITY_GPU_CUBLAS_HPP__
#define OPENJIJ_UTILITY_GPU_CUBLAS_HPP__

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <utility/gpu/handle_error.hpp>
#include <utility/gpu/memory.hpp>
#include "cublas_v2.h"

namespace openjij {
    namespace utility {
        namespace cuda {

            template<typename FloatType>
                struct cudaDataType_impl;

            template<>
                struct cudaDataType_impl<float>{
                    constexpr static cudaDataType_t type = CUDA_R_32F;
                };

            template<>
                struct cudaDataType_impl<double>{
                    constexpr static cudaDataType_t type = CUDA_R_64F;
                };
                

            /**
             * @brief cuBLAS wrapper
             */
            class CuBLASWrapper{
                public:
                    CuBLASWrapper(){
                        //generate cuBLAS instance
                        HANDLE_ERROR_CUBLAS(cublasCreate(&_handle));
                        //use tensor core if possible
                        HANDLE_ERROR_CUBLAS(cublasSetMathMode(_handle, CUBLAS_TENSOR_OP_MATH));
                    }

                    CuBLASWrapper(CuBLASWrapper&& obj) noexcept
                    {
                        //move cuBLAS handler
                        this->_handle = obj._handle;
                        obj._handle = NULL;
                    }

                    ~CuBLASWrapper(){
                        //destroy generator
                        if(_handle != NULL)
                            HANDLE_ERROR_CUBLAS(cublasDestroy(_handle));
                    }

                    template<typename FloatType>
                    inline void SgemmEx(
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int m,
                            int n,
                            int k,
                            const float    *alpha,
                            const utility::cuda::unique_dev_ptr<FloatType>& A,
                            int lda,
                            const utility::cuda::unique_dev_ptr<FloatType>& B,
                            int ldb,
                            const float    *beta,
                            utility::cuda::unique_dev_ptr<FloatType>& C,
                            int ldc){
                        HANDLE_ERROR_CUBLAS(cublasSgemmEx(
                                    _handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    alpha,
                                    A.get(),
                                    cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
                                    lda,
                                    B.get(),
                                    cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
                                    ldb,
                                    beta,
                                    C.get(),
                                    cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
                                    ldc)
                                );
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
                    template<typename FloatType>
                    inline void matmul(
                            int m,
                            int k,
                            int n,
                            const utility::cuda::unique_dev_ptr<FloatType>& A,
                            const utility::cuda::unique_dev_ptr<FloatType>& B,
                            utility::cuda::unique_dev_ptr<FloatType>& C,
                            cublasOperation_t transa = CUBLAS_OP_N,
                            cublasOperation_t transb = CUBLAS_OP_N
                            ){
                        typename std::remove_extent<FloatType>::type alpha = 1.0;
                        typename std::remove_extent<FloatType>::type beta = 0;

                        HANDLE_ERROR_CUBLAS(cublasSgemmEx(
                                    _handle,
                                    transa,
                                    transb,
                                    m,
                                    n,
                                    k,
                                    &alpha,
                                    A.get(),
                                    cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
                                    m,
                                    B.get(),
                                    cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
                                    k,
                                    &beta,
                                    C.get(),
                                    cudaDataType_impl<typename std::remove_extent<FloatType>::type>::type,
                                    m)
                                );
                    }

                private:
                    cublasHandle_t _handle;
            };

        }// namespace cuda
    } // namespace utility
} // namespace openjij

#endif
#endif
