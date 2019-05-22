#pragma once

#include <cuda_runtime.h>
#include <curand.h> 
#include "cublas_v2.h"
#include <iostream>

namespace openjij{
	namespace system{

		/***************************
		  macro for detecting errors 
		 ***************************/
#ifndef HANDLE_ERROR
#define HANDLE_ERROR(expr) {cudaError_t err=(expr); if(err != cudaSuccess) std::cout << "error_code: " << err << " err_name: " << cudaGetErrorString(err) << " at " << __FILE__ << " line " << __LINE__ << std::endl;}
#endif

#ifndef HANDLE_ERROR_CURAND
#define HANDLE_ERROR_CURAND(expr) {curandStatus_t st=(expr); if(st != CURAND_STATUS_SUCCESS) std::cout << "curand_error: " << st << " at " << __FILE__ << " line " << __LINE__ << std::endl;}
#endif

#ifndef HANDLE_ERROR_CUBLAS
#define HANDLE_ERROR_CUBLAS(expr) {cublasStatus_t bst=(expr); if(bst != CUBLAS_STATUS_SUCCESS) std::cout << "cublas_error: " << st << " at " << __FILE__ << " line " << __LINE__ << std::endl;}
#endif
	}
}
