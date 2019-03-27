#include <cuda_runtime.h>
#include <curand.h> 
#include <curand_kernel.h>
#include "../index.h"

//for test
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cfloat>

namespace openjij{
	namespace method{
		namespace chimera_gpu{

			//HANDLE ERROR
			static cudaError_t err;
			static curandStatus_t st;

			/***************************
			  macro for detecting errors 
			 ***************************/

#define HANDLE_ERROR(expr) err=(expr); if(err != cudaSuccess) std::cout << "error_code: " << err << " err_name: " << cudaGetErrorString(err) << " at " << __FILE__ << " line " << __LINE__ << std::endl;
#define HANDLE_ERROR_CURAND(expr) st=(expr); if(st != CURAND_STATUS_SUCCESS) std::cout << "curand_error: " << st << " at " << __FILE__ << " line " << __LINE__ << std::endl;

			/*************************
			  list of device variables
			 *************************/

			// inteactions
			//(row*col*8) = localsize
			static float* dev_J_out_p;
			//(row*col*8)
			static float* dev_J_out_n;
			//(row*col*8)
			static float* dev_J_in_0;
			//(row*col*8)
			static float* dev_J_in_1;
			//(row*col*8)
			static float* dev_J_in_2;
			//(row*col*8)
			static float* dev_J_in_3;

			// local magnetization
			//(row*col*8)
			static float* dev_H;

			//TODO: delete
			//// beta and gamma (for parallel tempering)
			////(num_beta)
			//static float* dev_beta;
			////(num_gamma)
			//static float* dev_gamma;

			// spins and randoms
			//(gammasize*betasize*trot*row*col*8) =totalsize
			static int32_t* dev_spin;
			//(gammasize*betasize*trot*row*col*8) =totalsize
			static float* dev_random;

			//TODO: delete
			////energy for PT
			////(num_gamma*num_beta)
			//static float* dev_energy_chimera;
			////(num_gamma*num_beta)
			//static float* dev_energy_trot;

			//energy = (beta/M) * dev_energy_chimera + J_trot * dev_energy_trot

			//energy flip flag (when PT)
			//(num_gamma*num_beta)
			//static int32_t* dev_flip_flag;

			//magnetizations
			//(num_gamma*num_beta)
			//static float* dev_magnetizations;

			//mean and variance of magnetizations (abs,2,4)
			//(num_gamma*num_beta)
			//static float* dev_magnetizations_mean_A;
			//static float* dev_magnetizations_var_A;
			//static float* dev_magnetizations_mean_S;
			//static float* dev_magnetizations_var_S;
			//static float* dev_magnetizations_mean_Q;
			//static float* dev_magnetizations_var_Q;

			//magnetizations for each index
			//(num_gamma*num_beta*num_row*num_col*8)
			//static float* dev_magnetizations_site;
			//
			////mean and variance of magnetizations for each index
			////(num_gamma*num_beta*num_row*num_col*8)
			//static float* dev_magnetizations_site_mean_A;
			//static float* dev_magnetizations_site_var_A;
			//static float* dev_magnetizations_site_mean_S;
			//static float* dev_magnetizations_site_var_S;
			//static float* dev_magnetizations_site_mean_Q;
			//static float* dev_magnetizations_site_var_Q;

			//curand generator
			static curandGenerator_t rng;

			/**********************
			  host variables (static)
			 **********************/

			// the size of the arrays
			//static uint32_t num_beta;
			//static uint32_t num_gamma;

			//the number of trotter slices
			static uint32_t num_trot;

			//the number of rows and cols of each Chimera graph
			static uint32_t num_row;
			static uint32_t num_col;

			//grids and blocks
			static dim3 grid;
			static dim3 block;

			//spinarray (page-locked memory)
			static int32_t* spinarray;

			////magnetization array (page-locked memory)
			//static float* marray;
			//
			////mean and variance
			//static float* marray_mean_A;
			//static float* marray_var_A;
			//static float* marray_mean_S;
			//static float* marray_var_S;
			//static float* marray_mean_Q;
			//static float* marray_var_Q;
			//
			////magnetization for each index (page-locked memory)
			////static float* marray_site;
			//
			////mean and variance
			//static float* marray_site_mean_A;
			//static float* marray_site_var_A;
			//static float* marray_site_mean_S;
			//static float* marray_site_var_S;
			//static float* marray_site_mean_Q;
			//static float* marray_site_var_Q;

			//count variable
			//static uint64_t g_count;
			//PT switch
			//static int32_t g_sw;

			/**********************
			  const variables 
			 **********************/

			constexpr static uint32_t unitspins = 8;

			constexpr static uint32_t block_row = 2;
			constexpr static uint32_t block_col = 2;
			constexpr static uint32_t block_trot = 2;

			/**********************
			  cuda device functions
			 **********************/

			//TODO: memory alignment?

			__device__ __forceinline__ uint32_t get_index(uint32_t tidx){ //inside-index of chimera unit
				return tidx%unitspins;
			}

			__device__ __forceinline__ uint32_t get_i(uint32_t bidy, uint32_t bdmy, uint32_t tidy){
				return bidy*bdmy+tidy;
			}

			__device__ __forceinline__ uint32_t get_j(uint32_t bidx, uint32_t bdmx, uint32_t tidx, uint32_t col){
				return ((bidx*bdmx+tidx)/unitspins)%col;
			}

			__device__ __forceinline__ uint32_t get_t(uint32_t bidz, uint32_t bdmz, uint32_t tidz){
				return bidz*bdmz+tidz;
			}

			//__device__ __forceinline__ uint32_t get_b(uint32_t bidx, uint32_t bdmx, uint32_t tidx, uint32_t col, uint32_t beta){
			//	return ((bidx*bdmx+tidx)/(unitspins*col))%beta;
			//}
			//
			//__device__ __forceinline__ uint32_t get_g(uint32_t bidx, uint32_t bdmx, uint32_t tidx, uint32_t col, uint32_t beta){
			//	return ((bidx*bdmx+tidx)/(unitspins*col*beta));
			//}

			__device__ __forceinline__ uint32_t get_glIdx(
					uint32_t bidx, uint32_t bdmx, uint32_t tidx,
					uint32_t bidy, uint32_t bdmy, uint32_t tidy,
					uint32_t bidz, uint32_t bdmz, uint32_t tidz,
					uint32_t trot, uint32_t row, uint32_t col
					){
				return glIdx_TRCI(trot, row, col,
						get_t(bidz, bdmz, tidz),
						get_i(bidy, bdmy, tidy),
						get_j(bidx, bdmx, tidx, col),
						get_index(tidx)
						);
			}


			__global__ void cuda_init_spins(int32_t* spin, const float* rand, 
					uint32_t trot, uint32_t row, uint32_t col
					){
				uint32_t ind = get_glIdx(
						blockIdx.x, blockDim.x, threadIdx.x,
						blockIdx.y, blockDim.y, threadIdx.y,
						blockIdx.z, blockDim.z, threadIdx.z,
						trot, row, col);

				spin[ind] = (rand[ind] > 0.5)?1:-1;
			}

			//#ifdef DEBUG
			//__global__ void cuda_show_spins(const int32_t* spin,
			//		uint32_t gamma, uint32_t beta, uint32_t trot, uint32_t row, uint32_t col){
			//	uint32_t ind = get_glIdx(
			//			blockIdx.x, blockDim.x, threadIdx.x,
			//			blockIdx.y, blockDim.y, threadIdx.y,
			//			blockIdx.z, blockDim.z, threadIdx.z,
			//			gamma, beta, trot, row, col);
			//
			//	printf("i= %lu, j=%lu, t=%lu, b=%lu, g=%lu, ind=%lu, total=%lu, spin=%d\n", 
			//			(unsigned long)get_i(blockIdx.y, blockDim.y, threadIdx.y),
			//			(unsigned long)get_j(blockIdx.x, blockDim.x, threadIdx.x, col),
			//			(unsigned long)get_t(blockIdx.z, blockDim.z, threadIdx.z),
			//			(unsigned long)get_b(blockIdx.x, blockDim.x, threadIdx.x, col, beta),
			//			(unsigned long)get_g(blockIdx.x, blockDim.x, threadIdx.x, col, beta),
			//			(unsigned long)get_index(threadIdx.x),
			//			(unsigned long)ind,
			//			spin[ind]
			//			);
			//}
			//#endif

			//magnetization process
			//__global__ void cuda_init_magnetizations_to_zero(
			//		float* magnetizations,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//	if(t==0 && i==0 && j==0 && ind==0){
			//		magnetizations[glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//	}
			//}
			//
			////mean and variance
			//__global__ void cuda_init_magnetizations_mean_var_to_zero(
			//		float* magnetizations_mean_A,
			//		float* magnetizations_var_A,
			//		float* magnetizations_mean_S,
			//		float* magnetizations_var_S,
			//		float* magnetizations_mean_Q,
			//		float* magnetizations_var_Q,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//	if(t==0 && i==0 && j==0 && ind==0){
			//		magnetizations_mean_A[glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//		magnetizations_var_A[ glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//		magnetizations_mean_S[glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//		magnetizations_var_S[ glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//		magnetizations_mean_Q[glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//		magnetizations_var_Q[ glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//	}
			//}
			//
			//__global__ void cuda_calc_magnetizations(
			//		//make sure that magnetizations is initialized to zero.
			//		const int32_t* spin,
			//		float* magnetizations,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//
			//	uint32_t blockind = glIdx_TRCI(block_trot,block_row,block_col, t%(block_trot), i%(block_row), j%(block_col), ind);
			//
			//	//sharedmem
			//	__shared__ float mag_cache[block_row * block_col * block_trot * unitspins];
			//
			//	//insert spin variables in mag_cache
			//	mag_cache[blockind] = spin[glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,ind)];
			//
			//	__syncthreads();
			//
			//	//reduction
			//	uint32_t c = block_row * block_col * block_trot * unitspins; //64
			//	uint32_t ti = threadIdx.z*(blockDim.y*blockDim.x)+threadIdx.y*(blockDim.x)+threadIdx.x;
			//
			//	c = c/2; //32
			//	if(ti < c){
			//		mag_cache[ti] += mag_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//	c = c/2; //16
			//	if(ti < c){
			//		mag_cache[ti] += mag_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//	c = c/2; //8
			//	if(ti < c){
			//		mag_cache[ti] += mag_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//	c = c/2; //4
			//	if(ti < c){
			//		mag_cache[ti] += mag_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//	c = c/2; //2
			//	if(ti < c){
			//		mag_cache[ti] += mag_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//	c = c/2; //1
			//	if(ti < c){
			//		mag_cache[ti] += mag_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//
			//	if(ti < c){
			//		atomicAdd(&magnetizations[glIdx_GB(ngamma, nbeta, g,b)], mag_cache[ti]/(float)(ntrot*row*col*unitspins));
			//	}
			//}
			//
			////update mean and variance of magnetization
			//__global__ void cuda_calc_magnetizations_mean_var(
			//		const float* magnetizations,
			//		float* magnetizations_mean_A,
			//		float* magnetizations_var_A,
			//		float* magnetizations_mean_S,
			//		float* magnetizations_var_S,
			//		float* magnetizations_mean_Q,
			//		float* magnetizations_var_Q,
			//		uint32_t count,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//	if(t==0 && i==0 && j==0 && ind==0){
			//		float mean_A = magnetizations_mean_A[glIdx_GB(ngamma, nbeta, g,b)];
			//		float var_A  = magnetizations_var_A[ glIdx_GB(ngamma, nbeta, g,b)];
			//		float mean_S = magnetizations_mean_S[glIdx_GB(ngamma, nbeta, g,b)];
			//		float var_S  = magnetizations_var_S[ glIdx_GB(ngamma, nbeta, g,b)];
			//		float mean_Q = magnetizations_mean_Q[glIdx_GB(ngamma, nbeta, g,b)];
			//		float var_Q  = magnetizations_var_Q[ glIdx_GB(ngamma, nbeta, g,b)];
			//		float m      = magnetizations[       glIdx_GB(ngamma, nbeta, g,b)];
			//		//TODO: m2 m4?
			//
			//		float mean_A_next = (1.0/(count+1))*(count*mean_A+fabs(m));
			//		float mean_S_next = (1.0/(count+1))*(count*mean_S+m*m);
			//		float mean_Q_next = (1.0/(count+1))*(count*mean_Q+m*m*m*m);
			//
			//		magnetizations_mean_A[glIdx_GB(ngamma, nbeta, g,b)] = mean_A_next;
			//		magnetizations_mean_S[glIdx_GB(ngamma, nbeta, g,b)] = mean_S_next;
			//		magnetizations_mean_Q[glIdx_GB(ngamma, nbeta, g,b)] = mean_Q_next;
			//
			//		magnetizations_var_A[glIdx_GB(ngamma, nbeta, g,b)] = (1.0/(count+1))*(count*(var_A+mean_A*mean_A)+m*m)-mean_A_next*mean_A_next;
			//		magnetizations_var_S[glIdx_GB(ngamma, nbeta, g,b)] = (1.0/(count+1))*(count*(var_S+mean_S*mean_S)+m*m*m*m)-mean_S_next*mean_S_next;
			//		magnetizations_var_Q[glIdx_GB(ngamma, nbeta, g,b)] = (1.0/(count+1))*(count*(var_Q+mean_Q*mean_Q)+m*m*m*m*m*m*m*m)-mean_Q_next*mean_Q_next;
			//	}
			//}
			//
			////magnetization for each site
			//__global__ void cuda_init_magnetizations_site_to_zero(
			//		float* magnetizations_site,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//	if(t==0){
			//		magnetizations_site[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = 0;
			//	}
			//}
			//
			//__global__ void cuda_init_magnetizations_site_mean_var_to_zero(
			//		float* magnetizations_site_mean_A,
			//		float* magnetizations_site_var_A,
			//		float* magnetizations_site_mean_S,
			//		float* magnetizations_site_var_S,
			//		float* magnetizations_site_mean_Q,
			//		float* magnetizations_site_var_Q,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//	if(t==0){
			//		magnetizations_site_mean_A[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = 0;
			//		magnetizations_site_var_A[ glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = 0;
			//		magnetizations_site_mean_S[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = 0;
			//		magnetizations_site_var_S[ glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = 0;
			//		magnetizations_site_mean_Q[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = 0;
			//		magnetizations_site_var_Q[ glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = 0;
			//	}
			//}
			//
			//__global__ void cuda_calc_magnetizations_site(
			//		//make sure that magnetizations is initialized to zero.
			//		const int32_t* spin,
			//		float* magnetizations_site,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//
			//	//atomicAdd
			//	atomicAdd(
			//			&magnetizations_site[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)], 
			//			spin[glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,ind)]/(float)(ntrot));
			//}
			//
			////update mean and variance of magnetization for each site
			//__global__ void cuda_calc_magnetizations_site_mean_var(
			//		const float* magnetizations_site,
			//		float* magnetizations_site_mean_A,
			//		float* magnetizations_site_var_A,
			//		float* magnetizations_site_mean_S,
			//		float* magnetizations_site_var_S,
			//		float* magnetizations_site_mean_Q,
			//		float* magnetizations_site_var_Q,
			//		uint32_t count,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//	if(t==0){
			//		float mean_A = magnetizations_site_mean_A[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)];
			//		float var_A  = magnetizations_site_var_A[ glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)];
			//		float mean_S = magnetizations_site_mean_S[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)];
			//		float var_S  = magnetizations_site_var_S[ glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)];
			//		float mean_Q = magnetizations_site_mean_Q[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)];
			//		float var_Q  = magnetizations_site_var_Q[ glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)];
			//		float m      = magnetizations_site[       glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)];
			//
			//		float mean_A_next = (1.0/(count+1))*(count*mean_A+fabs(m));
			//		float mean_S_next = (1.0/(count+1))*(count*mean_S+m*m);
			//		float mean_Q_next = (1.0/(count+1))*(count*mean_Q+m*m*m*m);
			//
			//		magnetizations_site_mean_A[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = mean_A_next;
			//		magnetizations_site_mean_S[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = mean_S_next;
			//		magnetizations_site_mean_Q[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = mean_Q_next;
			//
			//		magnetizations_site_var_A[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = (1.0/(count+1))*(count*(var_A+mean_A*mean_A)+m*m)-mean_A_next*mean_A_next;
			//		magnetizations_site_var_S[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = (1.0/(count+1))*(count*(var_S+mean_S*mean_S)+m*m*m*m)-mean_S_next*mean_S_next;
			//		magnetizations_site_var_Q[glIdx_GBRCI(ngamma, nbeta, row, col, g,b,i,j,ind)] = (1.0/(count+1))*(count*(var_Q+mean_Q*mean_Q)+m*m*m*m*m*m*m*m)-mean_Q_next*mean_Q_next;
			//	}
			//}
			//
			//__global__ void cuda_calc_init_energy_to_zero(
			//		float* energy_chimera,
			//		float* energy_trot,
			//		int32_t* flip_flag,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//
			//	if(t==0 && i==0 && j==0 && ind==0){
			//		//zero initialization
			//		energy_chimera[glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//		energy_trot[   glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//
			//		flip_flag[     glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//	}
			//}
			//
			//
			//__global__ void cuda_calc_init_energy(
			//		float* energy_chimera,
			//		float* energy_trot,
			//		const int32_t* spin,
			//		const float* J_out_p,
			//		const float* J_out_n,
			//		const float* J_in_0,
			//		const float* J_in_1,
			//		const float* J_in_2,
			//		const float* J_in_3,
			//		const float* H,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	//know who I am
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//
			//	uint32_t blockind = glIdx_TRCI(block_trot,block_row,block_col, t%(block_trot), i%(block_row), j%(block_col), ind);
			//
			//	//sharedmem
			//	__shared__ float energy_chimera_cache[block_row * block_col * block_trot * unitspins];
			//	__shared__ float energy_trot_cache[   block_row * block_col * block_trot * unitspins];
			//
			//	float temp_c = 0;
			//	float temp_t = 0;
			//
			//
			//	//inside chimera
			//	if(ind < 4){
			//		temp_c += 
			//			J_in_0[glIdx_RCI(row, col, i,j,ind)]
			//			*spin[ glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,ind)]
			//			*spin[ glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,4)]+
			//			J_in_1[glIdx_RCI(row, col, i,j,ind)]
			//			*spin[ glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,ind)]
			//			*spin[ glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,5)]+
			//			J_in_2[glIdx_RCI(row, col, i,j,ind)]
			//			*spin[ glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,ind)]
			//			*spin[ glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,6)]+
			//			J_in_3[glIdx_RCI(row, col, i,j,ind)]
			//			*spin[ glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,ind)]
			//			*spin[ glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,7)];
			//	}
			//
			//	//column
			//	if(j != col-1 && 4 <= ind && ind < 8){
			//		temp_c +=
			//			J_out_n[glIdx_RCI(row, col, i,j,ind)]
			//			*spin[  glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,  ind)]
			//			*spin[  glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j+1,ind)];
			//	}
			//
			//	//row
			//	if(i != row-1 && ind < 4){
			//		temp_c +=
			//			J_out_n[glIdx_RCI(row, col, i  ,j,ind)]
			//			*spin[  glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i  ,j,ind)]
			//			*spin[  glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i+1,j,ind)];
			//	}
			//
			//	//magnetization
			//	temp_c +=
			//		H[    glIdx_RCI(row, col, i  ,j,ind)]
			//		*spin[glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i  ,j,ind)];
			//
			//	//trot
			//	temp_t +=
			//		spin[ glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,  i,j,ind)]
			//		*spin[glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,(t != ntrot-1)?t+1:0,i,j,ind)];
			//
			//	energy_chimera_cache[blockind] = temp_c;
			//	energy_trot_cache[   blockind] = temp_t;
			//
			//	__syncthreads();
			//
			//	uint32_t c = block_row * block_col * block_trot * unitspins; //64
			//	uint32_t ti = threadIdx.z*(blockDim.y*blockDim.x)+threadIdx.y*(blockDim.x)+threadIdx.x;
			//
			//	c = c/2; //32
			//	if(ti < c){
			//		energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
			//		energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//	c = c/2; //16
			//	if(ti < c){
			//		energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
			//		energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//	c = c/2; //8
			//	if(ti < c){
			//		energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
			//		energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//	c = c/2; //4
			//	if(ti < c){
			//		energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
			//		energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//	c = c/2; //2
			//	if(ti < c){
			//		energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
			//		energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//	c = c/2; //1
			//	if(ti < c){
			//		energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
			//		energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
			//	}
			//	__syncthreads();
			//
			//	if(ti < c){
			//		//chimera
			//		atomicAdd(&energy_chimera[glIdx_GB(ngamma, nbeta, g,b)], energy_chimera_cache[ti]);
			//		//trot
			//		atomicAdd(&energy_trot[   glIdx_GB(ngamma, nbeta, g,b)], energy_trot_cache[ti]);
			//	}
			//}
			//
			//#ifdef DEBUG
			//__global__ void cuda_show_energy(
			//		float* energy_chimera,
			//		float* energy_trot,
			//		const int32_t* spin,
			//		const float* J_out_p,
			//		const float* J_out_n,
			//		const float* J_in_0,
			//		const float* J_in_1,
			//		const float* J_in_2,
			//		const float* J_in_3,
			//		const float* H,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//
			//	if(t==0 && i==0 && j==0 && ind==0){
			//		printf("i= %lu, j=%lu, t=%lu, b=%lu, g=%lu, ind=%lu, chimera=%lf trot=%lf\n", 
			//				(unsigned long)get_i(blockIdx.y, blockDim.y, threadIdx.y),
			//				(unsigned long)get_j(blockIdx.x, blockDim.x, threadIdx.x, col),
			//				(unsigned long)get_t(blockIdx.z, blockDim.z, threadIdx.z),
			//				(unsigned long)get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta),
			//				(unsigned long)get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta),
			//				(unsigned long)get_index(threadIdx.x),
			//				energy_chimera[glIdx_GB(ngamma, nbeta, g,b)],
			//				energy_trot[   glIdx_GB(ngamma, nbeta, g,b)]
			//				);
			//	}
			//}
			//#endif


			__global__ void cuda_metropolis(
					int32_t sw,
					int32_t* spin, const float* rand,
					const float* J_out_p,
					const float* J_out_n,
					const float* J_in_0,
					const float* J_in_1,
					const float* J_in_2,
					const float* J_in_3,
					const float* H,
					uint32_t ntrot, uint32_t row, uint32_t col,
					float beta, float gamma 
					){

				//switch
				//(0 -> 1st chimera unit (t==0, i==0, j==0) -> (0...3))
				//(1 -> 1st chimera unit (t==0, i==0, j==0) -> (4...7))

				//shared memory
				//spin with boundaries
				__shared__ int32_t spincache[(block_row+2) * (block_col+2) * (block_trot+2) * unitspins];

				__shared__ float     randcache[block_row * block_col * block_trot * unitspins];

				__shared__ float J_out_p_cache[block_row * block_col * block_trot * unitspins];
				__shared__ float J_out_n_cache[block_row * block_col * block_trot * unitspins];
				__shared__ float  J_in_0_cache[block_row * block_col * block_trot * unitspins];
				__shared__ float  J_in_1_cache[block_row * block_col * block_trot * unitspins];
				__shared__ float  J_in_2_cache[block_row * block_col * block_trot * unitspins];
				__shared__ float  J_in_3_cache[block_row * block_col * block_trot * unitspins];
				__shared__ float       H_cache[block_row * block_col * block_trot * unitspins];

				float J_trot;

				//get index parameters, know who I am.
				uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
				uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
				uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
				uint32_t ind = get_index(threadIdx.x);
				//total index
				uint32_t totind = glIdx_TRCI(ntrot, row, col, t,i,j,ind);
				//local index
				uint32_t locind = glIdx_RCI(row, col,i,j,ind);
				//block index
				uint32_t blockind = glIdx_TRCI(block_trot,block_row,block_col, t%(block_trot), i%(block_row), j%(block_col), ind);
				//block index specialized for spincache
				uint32_t spincacheind = glIdx_TRCI_ext(block_trot,block_row,block_col, t%(block_trot), i%(block_row), j%(block_col), ind);

				//import interactions from global memory to shared memory (or registers)

				J_trot = +0.5*log(tanh(beta*gamma/(float)ntrot)); //-(1/2)log(coth(beta*gamma/M))

				J_out_p_cache[blockind] = J_out_p[locind];
				J_out_n_cache[blockind] = J_out_n[locind];
				J_in_0_cache[blockind]  = J_in_0[locind];
				J_in_1_cache[blockind]  = J_in_1[locind];
				J_in_2_cache[blockind]  = J_in_2[locind];
				J_in_3_cache[blockind]  = J_in_3[locind];
				H_cache[blockind] 		= H[locind];

				randcache[blockind] 		= rand[totind];
				spincache[spincacheind] = spin[totind];

				////be sure to initialize sharedmem to zero
				//energy_chimera_cache[blockind] = 0;
				//energy_trot_cache[   blockind] = 0;

				//boundary spin
				//row
				if(i%(block_row) == 0 && i != 0){
					spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), -1, (j%(block_col)), ind)]
						= spin[glIdx_TRCI(ntrot, row, col, t,i-1,j,ind)];
				}
				if(i%(block_row) == block_row-1 && i != row-1){
					spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), block_row, (j%(block_col)), ind)]
						= spin[glIdx_TRCI(ntrot, row, col, t,i+1,j,ind)];
				}

				//col
				if(j%(block_col) == 0 && j != 0){
					spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (i%(block_row)), -1, ind)]
						= spin[glIdx_TRCI(ntrot, row, col, t,i,j-1,ind)];
				}
				if(j%(block_col) == block_col-1 && j != col-1){
					spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (i%(block_row)), block_col, ind)]
						= spin[glIdx_TRCI(ntrot, row, col, t,i,j+1,ind)];
				}

				//trotter slices
				if(t%(block_trot) == 0){
					spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, -1, (i%(block_row)), (j%(block_col)), ind)]
						= spin[glIdx_TRCI(ntrot, row, col, (t!=0)?t-1:ntrot-1,i,j,ind)]; //periodic boundary condition
				}
				if(t%(block_trot) == block_trot-1){
					spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, block_trot, (i%(block_row)), (j%(block_col)), ind)]
						= spin[glIdx_TRCI(ntrot, row, col, (t!=ntrot-1)?t+1:0,i,j,ind)]; //periodic boundary condition
				}

				__syncthreads();

				//metropolis

				if(((i+j+t)%2 == sw && ind <= 3) || ((i+j+t)%2 != sw && 4 <= ind && ind <= 7)){
					//calculate energy difference
					//dE = (beta/M)*dE_c + Jtrot*dE_t;
					float dE_c =      
						-2*spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (i%(block_row)), (j%(block_col)), ind)]*(

								//outside chimera unit
								J_out_p_cache[blockind]
								//0 to 3 -> go up 4 to 7 -> go left
								*spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (ind <= 3)?(i%(block_row)-1):(i%(block_row)), (4 <= ind && ind <= 7)?(j%(block_col)-1):(j%(block_col)), ind)]+

								J_out_n_cache[blockind]
								//0 to 3 -> go down 4 to 7 -> go right
								*spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (ind <= 3)?(i%(block_row)+1):(i%(block_row)), (4 <= ind && ind <= 7)?(j%(block_col)+1):(j%(block_col)), ind)]+ 

								//inside chimera unit
								J_in_0_cache[blockind] 
								*spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (i%(block_row)), (j%(block_col)), (ind <= 3)?4:0)]+
								J_in_1_cache[blockind]
								*spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (i%(block_row)), (j%(block_col)), (ind <= 3)?5:1)]+
								J_in_2_cache[blockind]
								*spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (i%(block_row)), (j%(block_col)), (ind <= 3)?6:2)]+
								J_in_3_cache[blockind]
								*spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (i%(block_row)), (j%(block_col)), (ind <= 3)?7:3)]+


								//local magnetization
								H_cache[blockind]);

					float dE_t =
						-2*spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (i%(block_row)), (j%(block_col)), ind)]*(
								//trotter slices
								spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot))+1, (i%(block_row)), (j%(block_col)), ind)]+
								spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot))-1, (i%(block_row)), (j%(block_col)), ind)]
								);

					//update!
					//dE = (beta/M)*dE_c + Jtrot*dE_t;
					if(exp(-((beta/(float)ntrot)*dE_c + J_trot*dE_t)) > randcache[blockind]){
						spin[totind] = -spincache[spincacheind];
					}

				}

				__syncthreads();

				//uint32_t c = block_row * block_col * block_trot * unitspins; //64
				//uint32_t ti = threadIdx.z*(blockDim.y*blockDim.x)+threadIdx.y*(blockDim.x)+threadIdx.x;

				//c = c/2; //32
				//if(ti < c){
				//	energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
				//	energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
				//}
				//__syncthreads();
				//c = c/2; //16
				//if(ti < c){
				//	energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
				//	energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
				//}
				//__syncthreads();
				//c = c/2; //8
				//if(ti < c){
				//	energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
				//	energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
				//}
				//__syncthreads();
				//c = c/2; //4
				//if(ti < c){
				//	energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
				//	energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
				//}
				//__syncthreads();
				//c = c/2; //2
				//if(ti < c){
				//	energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
				//	energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
				//}
				//__syncthreads();
				//c = c/2; //1
				//if(ti < c){
				//	energy_chimera_cache[ti] += energy_chimera_cache[ti+c]; 
				//	energy_trot_cache[ti] += energy_trot_cache[ti+c]; 
				//}
				//__syncthreads();

				//if(ti < c){
				//	//chimera
				//	atomicAdd(&energy_chimera[glIdx_GB(ngamma, nbeta, g,b)], energy_chimera_cache[ti]);
				//	//trot
				//	atomicAdd(&energy_trot[   glIdx_GB(ngamma, nbeta, g,b)], energy_trot_cache[ti]);
				//}
			}

			//__global__ void cuda_pt_beta(
			//		int32_t sw,
			//		int32_t* spin, const float* rand, //random variable should be initialized as numgamma*numbeta
			//		const float* energy_chimera, const float* energy_trot,
			//		int32_t* flip_flag,
			//		const float* J_out_p,
			//		const float* J_out_n,
			//		const float* J_in_0,
			//		const float* J_in_1,
			//		const float* J_in_2,
			//		const float* J_in_3,
			//		const float* H,
			//		const float* betaarray,
			//		const float* gammaarray,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	//parallel tempering wrt beta
			//
			//	//know who I am
			//
			//	float beta, gamma, beta_next;
			//
			//	//get index parameters, know who I am.
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//
			//	beta = betaarray[b];
			//	gamma = gammaarray[g];
			//
			//
			//	if(b%2 == sw && b != nbeta-1){
			//		beta_next = betaarray[b+1];
			//
			//		float J_trot 		= +0.5*log(tanh(beta     *gamma/(float)ntrot)); //-(1/2)log(coth(beta     *gamma/M))
			//		float J_trot_next = +0.5*log(tanh(beta_next*gamma/(float)ntrot)); //-(1/2)log(coth(beta_next*gamma/M))
			//
			//#ifdef DEBUG
			//		printf("i= %lu, j=%lu, t=%lu, b=%lu, g=%lu, ind=%lu, beta=%lf, gamma=%lf, betanext=%lf, Jtrot=%lf, Jtrot_next=%lf, energy_chimera[b]=%lf, energy_trot[b]=%lf, energy_chimera[b+1]=%lf, energy_trot[b+1]=%lf, rand=%lf\n", 
			//				(unsigned long)get_i(blockIdx.y, blockDim.y, threadIdx.y),
			//				(unsigned long)get_j(blockIdx.x, blockDim.x, threadIdx.x, col),
			//				(unsigned long)get_t(blockIdx.z, blockDim.z, threadIdx.z),
			//				(unsigned long)get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta),
			//				(unsigned long)get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta),
			//				(unsigned long)get_index(threadIdx.x),
			//				beta,
			//				gamma,
			//				beta_next,
			//				J_trot,
			//				J_trot_next,
			//				energy_chimera[glIdx_GB(ngamma, nbeta, g,b  )],
			//				energy_trot[glIdx_GB(ngamma, nbeta, g,b  )],
			//				energy_chimera[glIdx_GB(ngamma, nbeta, g,b+1)],
			//				energy_trot[glIdx_GB(ngamma, nbeta, g,b+1)],
			//				rand[glIdx_GB(ngamma, nbeta, g,b )]
			//				);
			//#endif
			//
			//		float dE = 
			//			+((beta_next/(float)ntrot)*energy_chimera[glIdx_GB(ngamma, nbeta, g,b  )]+J_trot_next*energy_trot[glIdx_GB(ngamma, nbeta, g,b  )])
			//			+((beta     /(float)ntrot)*energy_chimera[glIdx_GB(ngamma, nbeta, g,b+1)]+J_trot     *energy_trot[glIdx_GB(ngamma, nbeta, g,b+1)])
			//			-((beta     /(float)ntrot)*energy_chimera[glIdx_GB(ngamma, nbeta, g,b  )]+J_trot     *energy_trot[glIdx_GB(ngamma, nbeta, g,b  )])
			//			-((beta_next/(float)ntrot)*energy_chimera[glIdx_GB(ngamma, nbeta, g,b+1)]+J_trot_next*energy_trot[glIdx_GB(ngamma, nbeta, g,b+1)]);
			//
			//		if(exp(-dE) > rand[glIdx_GB(ngamma, nbeta, g,b )]){
			//			//spin exchange
			//
			//			int32_t tempspin;
			//			tempspin = spin[glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,ind)];
			//			spin[     glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,  t,i,j,ind)]
			//				= spin[glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b+1,t,i,j,ind)];
			//			spin[     glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b+1,t,i,j,ind)] = tempspin;
			//
			//			if(t==0 && i==0 && j==0 && ind==0){
			//				//enable flip flag
			//				flip_flag[glIdx_GB(ngamma, nbeta, g,b)] = 1;
			//
			//#ifdef DEBUG
			//				printf("will be changed: i= %lu, j=%lu, t=%lu, b=%lu, g=%lu, ind=%lu, beta=%lf, gamma=%lf, betanext=%lf, Jtrot=%lf, Jtrot_next=%lf, energy_chimera[b]=%lf, energy_trot[b]=%lf, energy_chimera[b+1]=%lf, energy_trot[b+1]=%lf, rand=%lf\n", 
			//						(unsigned long)get_i(blockIdx.y, blockDim.y, threadIdx.y),
			//						(unsigned long)get_j(blockIdx.x, blockDim.x, threadIdx.x, col),
			//						(unsigned long)get_t(blockIdx.z, blockDim.z, threadIdx.z),
			//						(unsigned long)get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta),
			//						(unsigned long)get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta),
			//						(unsigned long)get_index(threadIdx.x),
			//						beta,
			//						gamma,
			//						beta_next,
			//						J_trot,
			//						J_trot_next,
			//						energy_chimera[glIdx_GB(ngamma, nbeta, g,b  )],
			//						energy_trot[glIdx_GB(ngamma, nbeta, g,b  )],
			//						energy_chimera[glIdx_GB(ngamma, nbeta, g,b+1)],
			//						energy_trot[glIdx_GB(ngamma, nbeta, g,b+1)],
			//						rand[glIdx_GB(ngamma, nbeta, g,b )]
			//						);
			//#endif
			//			}
			//		}
			//	}
			//}
			//
			//__global__ void cuda_pt_beta_energy_swap(
			//		float* energy_chimera, float* energy_trot,
			//		int32_t* flip_flag,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//
			//	if(flip_flag[glIdx_GB(ngamma, nbeta, g,b)] == 1 && t==0 && i==0 && j==0 && ind==0){
			//		//swap
			//		float tempenergy;
			//		tempenergy = energy_chimera[glIdx_GB(ngamma, nbeta, g,b)];
			//		energy_chimera[glIdx_GB(ngamma, nbeta, g,b)] = energy_chimera[glIdx_GB(ngamma, nbeta, g,b+1)];
			//		energy_chimera[glIdx_GB(ngamma, nbeta, g,b+1)] = tempenergy;
			//
			//		tempenergy = energy_trot[glIdx_GB(ngamma, nbeta, g,b)];
			//		energy_trot[glIdx_GB(ngamma, nbeta, g,b)] = energy_trot[glIdx_GB(ngamma, nbeta, g,b+1)];
			//		energy_trot[glIdx_GB(ngamma, nbeta, g,b+1)] = tempenergy;
			//
			//		flip_flag[glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//	}
			//}
			//
			//
			//__global__ void cuda_pt_gamma(
			//		int32_t sw,
			//		int32_t* spin, const float* rand, //random variable should be initialized as numgamma*numbeta
			//		const float* energy_chimera, const float* energy_trot,
			//		int32_t* flip_flag,
			//		const float* J_out_p,
			//		const float* J_out_n,
			//		const float* J_in_0,
			//		const float* J_in_1,
			//		const float* J_in_2,
			//		const float* J_in_3,
			//		const float* H,
			//		const float* betaarray,
			//		const float* gammaarray,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	//parallel tempering wrt gamma
			//
			//	//know who I am
			//
			//	float beta, gamma, gamma_next;
			//
			//	//get index parameters, know who I am.
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//
			//	beta = betaarray[b];
			//	gamma = gammaarray[g];
			//
			//
			//	if(g%2 == sw && g != ngamma-1){
			//		gamma_next = gammaarray[g+1];
			//
			//		float J_trot 		= +0.5*log(tanh(beta*gamma     /(float)ntrot)); //-(1/2)log(coth(beta*gamma/M))
			//		float J_trot_next = +0.5*log(tanh(beta*gamma_next/(float)ntrot)); //-(1/2)log(coth(beta*gamma_next/M))
			//
			//#ifdef DEBUG
			//		printf("i= %lu, j=%lu, t=%lu, b=%lu, g=%lu, ind=%lu, beta=%lf, gamma=%lf, gammanext=%lf, Jtrot=%lf, Jtrot_next=%lf, energy_chimera[g]=%lf, energy_trot[g]=%lf, energy_chimera[g+1]=%lf, energy_trot[g+1]=%lf, rand=%lf\n", 
			//				(unsigned long)get_i(blockIdx.y, blockDim.y, threadIdx.y),
			//				(unsigned long)get_j(blockIdx.x, blockDim.x, threadIdx.x, col),
			//				(unsigned long)get_t(blockIdx.z, blockDim.z, threadIdx.z),
			//				(unsigned long)get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta),
			//				(unsigned long)get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta),
			//				(unsigned long)get_index(threadIdx.x),
			//				beta,
			//				gamma,
			//				gamma_next,
			//				J_trot,
			//				J_trot_next,
			//				energy_chimera[glIdx_GB(ngamma, nbeta, g,b  )],
			//				energy_trot[glIdx_GB(ngamma, nbeta, g,b  )],
			//				energy_chimera[glIdx_GB(ngamma, nbeta, g+1,b)],
			//				energy_trot[glIdx_GB(ngamma, nbeta, g+1,b)],
			//				rand[glIdx_GB(ngamma, nbeta, g,b)]
			//				);
			//#endif
			//
			//		float dE = 
			//			+((beta/(float)ntrot)*energy_chimera[glIdx_GB(ngamma, nbeta, g  ,b)]+J_trot_next*energy_trot[glIdx_GB(ngamma, nbeta, g  ,b)])
			//			+((beta/(float)ntrot)*energy_chimera[glIdx_GB(ngamma, nbeta, g+1,b)]+J_trot     *energy_trot[glIdx_GB(ngamma, nbeta, g+1,b)])
			//			-((beta/(float)ntrot)*energy_chimera[glIdx_GB(ngamma, nbeta, g  ,b)]+J_trot     *energy_trot[glIdx_GB(ngamma, nbeta, g  ,b)])
			//			-((beta/(float)ntrot)*energy_chimera[glIdx_GB(ngamma, nbeta, g+1,b)]+J_trot_next*energy_trot[glIdx_GB(ngamma, nbeta, g+1,b)]);
			//
			//		if(exp(-dE) > rand[glIdx_GB(ngamma, nbeta, g,b)]){
			//			//spin exchange
			//
			//			int32_t tempspin;
			//			tempspin = spin[glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,b,t,i,j,ind)];
			//			spin[     glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g,  b,t,i,j,ind)]
			//				= spin[glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g+1,b,t,i,j,ind)];
			//			spin[     glIdx_GBTRCI(ngamma, nbeta, ntrot, row, col, g+1,b,t,i,j,ind)] = tempspin;
			//
			//			if(t==0 && i==0 && j==0 && ind==0){
			//				//enable flip flag
			//				flip_flag[glIdx_GB(ngamma, nbeta, g,b)] = 1;
			//
			//#ifdef DEBUG
			//				printf("will be changed: i= %lu, j=%lu, t=%lu, b=%lu, g=%lu, ind=%lu, beta=%lf, gamma=%lf, gammanext=%lf, Jtrot=%lf, Jtrot_next=%lf, energy_chimera[g]=%lf, energy_trot[g]=%lf, energy_chimera[g+1]=%lf, energy_trot[g+1]=%lf, rand=%lf\n", 
			//						(unsigned long)get_i(blockIdx.y, blockDim.y, threadIdx.y),
			//						(unsigned long)get_j(blockIdx.x, blockDim.x, threadIdx.x, col),
			//						(unsigned long)get_t(blockIdx.z, blockDim.z, threadIdx.z),
			//						(unsigned long)get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta),
			//						(unsigned long)get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta),
			//						(unsigned long)get_index(threadIdx.x),
			//						beta,
			//						gamma,
			//						gamma_next,
			//						J_trot,
			//						J_trot_next,
			//						energy_chimera[glIdx_GB(ngamma, nbeta, g,b  )],
			//						energy_trot[glIdx_GB(ngamma, nbeta, g,b  )],
			//						energy_chimera[glIdx_GB(ngamma, nbeta, g+1,b)],
			//						energy_trot[glIdx_GB(ngamma, nbeta, g+1,b)],
			//						rand[glIdx_GB(ngamma, nbeta, g,b)]
			//						);
			//#endif
			//			}
			//		}
			//	}
			//}
			//
			//__global__ void cuda_pt_gamma_energy_swap(
			//		float* energy_chimera, float* energy_trot,
			//		int32_t* flip_flag,
			//		uint32_t ngamma, uint32_t nbeta, uint32_t ntrot, uint32_t row, uint32_t col
			//		){
			//	uint32_t g = get_g(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t b = get_b(blockIdx.x, blockDim.x, threadIdx.x, col, nbeta);
			//	uint32_t t = get_t(blockIdx.z, blockDim.z, threadIdx.z);
			//	uint32_t i = get_i(blockIdx.y, blockDim.y, threadIdx.y);
			//	uint32_t j = get_j(blockIdx.x, blockDim.x, threadIdx.x, col);
			//	uint32_t ind = get_index(threadIdx.x);
			//
			//	if(flip_flag[glIdx_GB(ngamma, nbeta, g,b)] == 1 && t==0 && i==0 && j==0 && ind==0){
			//		//swap
			//		float tempenergy;
			//		tempenergy = energy_chimera[glIdx_GB(ngamma, nbeta, g,b)];
			//		energy_chimera[glIdx_GB(ngamma, nbeta, g,b)] = energy_chimera[glIdx_GB(ngamma, nbeta, g+1,b)];
			//		energy_chimera[glIdx_GB(ngamma, nbeta, g+1,b)] = tempenergy;
			//
			//		tempenergy = energy_trot[glIdx_GB(ngamma, nbeta, g,b)];
			//		energy_trot[glIdx_GB(ngamma, nbeta, g  ,b)] = energy_trot[glIdx_GB(ngamma, nbeta, g+1,b)];
			//		energy_trot[glIdx_GB(ngamma, nbeta, g+1,b)] = tempenergy;
			//
			//		flip_flag[glIdx_GB(ngamma, nbeta, g,b)] = 0;
			//	}
			//}

			/**********************
			  cuda host functions
			 **********************/

			//set device

			void cuda_set_device(int device){
				HANDLE_ERROR(cudaSetDevice(device));
			}

			//intialize GPU 
			void cuda_init(
					uint32_t arg_num_trot,
					uint32_t arg_num_row,
					uint32_t arg_num_col
					){

				//copy variables to host variables (static)
				num_trot = arg_num_trot;
				num_row = arg_num_row;
				num_col = arg_num_col;

				//localsize: the number of spins in each chimera graph
				uint32_t localsize = num_row*num_col*unitspins;
				//totalsize: the number of spins in all chimera graph (trotter slices included)
				uint32_t totalsize = num_trot*localsize;

				//create random generator
				HANDLE_ERROR(cudaMalloc((void**)&dev_random, totalsize*sizeof(float)));
				//mersenne twister
				HANDLE_ERROR_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MT19937));
				//HANDLE_ERROR_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW));
				//set seed
				HANDLE_ERROR_CURAND(curandSetPseudoRandomGeneratorSeed(rng, time(NULL)));

				//cudaMalloc
				HANDLE_ERROR(cudaMalloc((void**)&dev_J_out_p,	localsize*sizeof(float)));
				HANDLE_ERROR(cudaMalloc((void**)&dev_J_out_n,	localsize*sizeof(float)));
				HANDLE_ERROR(cudaMalloc((void**)&dev_J_in_0,		localsize*sizeof(float)));
				HANDLE_ERROR(cudaMalloc((void**)&dev_J_in_1,		localsize*sizeof(float)));
				HANDLE_ERROR(cudaMalloc((void**)&dev_J_in_2,		localsize*sizeof(float)));
				HANDLE_ERROR(cudaMalloc((void**)&dev_J_in_3,  	localsize*sizeof(float)));
				HANDLE_ERROR(cudaMalloc((void**)&dev_H,  			localsize*sizeof(float)));

				//spin
				HANDLE_ERROR(cudaMalloc((void**)&dev_spin,  		totalsize*sizeof(int32_t)));

				//set grids and blocks
				grid = dim3(num_col/block_col, num_row/block_row, num_trot/block_trot);
				block = dim3(unitspins*block_col, block_row, block_trot);

				//generate random_number
				HANDLE_ERROR_CURAND(curandGenerateUniform(rng, dev_random, totalsize));
				//init spins
				cuda_init_spins<<<grid, block>>>(dev_spin, dev_random, num_trot, num_row, num_col);

				//initialize spinarray
				HANDLE_ERROR(cudaMallocHost((void**)&spinarray, sizeof(int32_t)*totalsize));

				//get printf buffer
#ifdef DEBUG
				cudaDeviceSetLimit(cudaLimitPrintfFifoSize, (uint32_t)(1 << 30));
#endif
			}

			void cuda_init_interactions(
					const float* J_out_p,
					const float* J_out_n,
					const float* J_in_0,
					const float* J_in_1,
					const float* J_in_2,
					const float* J_in_3,
					const float* H
					){

				//localsize: the number of spins in each chimera graph
				uint32_t localsize = num_row*num_col*unitspins;
				//cudaMemcpy
				HANDLE_ERROR(cudaMemcpy(dev_J_out_p, J_out_p, 	localsize*sizeof(float), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(dev_J_out_n, J_out_n, 	localsize*sizeof(float), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(dev_J_in_0, J_in_0,		localsize*sizeof(float), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(dev_J_in_1, J_in_1,		localsize*sizeof(float), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(dev_J_in_2, J_in_2,		localsize*sizeof(float), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(dev_J_in_3, J_in_3,		localsize*sizeof(float), cudaMemcpyHostToDevice));
				HANDLE_ERROR(cudaMemcpy(dev_H, H,					localsize*sizeof(float), cudaMemcpyHostToDevice));

#ifdef DEBUG
				cudaDeviceSynchronize(); 	
				std::cout << "----------- init energies -----------" << std::endl;
#endif
			}

			void cuda_init_spin(){
				//localsize: the number of spins in each chimera graph
				uint32_t localsize = num_row*num_col*unitspins;
				//totalsize: the number of spins in all chimera graph (trotter slices and PT replicas included)
				uint32_t totalsize = num_trot*localsize;
				//generate random_number
				HANDLE_ERROR_CURAND(curandGenerateUniform(rng, dev_random, totalsize));
				//init spins
				cuda_init_spins<<<grid, block>>>(dev_spin, dev_random, num_trot, num_row, num_col);

				//init energies
				//cuda_calc_init_energy_to_zero<<<grid, block>>>(
				//		dev_energy_chimera,
				//		dev_energy_trot,
				//		dev_flip_flag,
				//		num_gamma, num_beta, num_trot, num_row, num_col);

				//cuda_calc_init_energy<<<grid, block>>>(
				//		dev_energy_chimera,
				//		dev_energy_trot,
				//		dev_spin,
				//		dev_J_out_p,
				//		dev_J_out_n,
				//		dev_J_in_0,
				//		dev_J_in_1,
				//		dev_J_in_2,
				//		dev_J_in_3,
				//		dev_H,
				//		num_gamma, num_beta, num_trot, num_row, num_col);
			}

			// single MCS
			void cuda_run(float beta, float gamma){

				const uint32_t localsize = num_row*num_col*unitspins;
				const uint32_t totalsize = num_trot*localsize;

				//generate random number
				HANDLE_ERROR_CURAND(curandGenerateUniform(rng, dev_random, totalsize));

				//metropolis update
				cuda_metropolis<<<grid, block>>>(
						0,
						dev_spin, dev_random,
						dev_J_out_p,
						dev_J_out_n,
						dev_J_in_0,
						dev_J_in_1,
						dev_J_in_2,
						dev_J_in_3,
						dev_H,
						num_trot, num_row, num_col,
						beta, gamma);

				//generate random number
				HANDLE_ERROR_CURAND(curandGenerateUniform(rng, dev_random, totalsize));

				cuda_metropolis<<<grid, block>>>(
						1,
						dev_spin, dev_random,
						dev_J_out_p,
						dev_J_out_n,
						dev_J_in_0,
						dev_J_in_1,
						dev_J_in_2,
						dev_J_in_3,
						dev_H,
						num_trot, num_row, num_col,
						beta, gamma);
				return;
			}

			void copy_spins(){
				HANDLE_ERROR(cudaMemcpy(spinarray, dev_spin, num_trot*num_row*num_col*unitspins*sizeof(int32_t), cudaMemcpyDeviceToHost));
			}

			int32_t get_spin(uint32_t t, uint32_t i, uint32_t j, uint32_t ind){
				return spinarray[glIdx_TRCI(num_trot, num_row, num_col, t,i,j,ind)];
			}

			void cuda_free(){
				HANDLE_ERROR(cudaFree(dev_random));
				//cudaMalloc
				HANDLE_ERROR(cudaFree(dev_J_out_p));
				HANDLE_ERROR(cudaFree(dev_J_out_n));
				HANDLE_ERROR(cudaFree(dev_J_in_0));
				HANDLE_ERROR(cudaFree(dev_J_in_1));
				HANDLE_ERROR(cudaFree(dev_J_in_2));
				HANDLE_ERROR(cudaFree(dev_J_in_3));
				HANDLE_ERROR(cudaFree(dev_H));

				HANDLE_ERROR(cudaFree(dev_spin));

				//curand
				HANDLE_ERROR_CURAND(curandDestroyGenerator(rng));
				//page-locked memory
				HANDLE_ERROR(cudaFreeHost(spinarray));
			}
		}
	}
}
