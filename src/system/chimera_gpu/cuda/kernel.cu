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
	namespace system{
		namespace chimera_gpu{

			//HANDLE ERROR
			cudaError_t err;
			curandStatus_t st;

			/***************************
			  macro for detecting errors 
			 ***************************/
#ifndef HANDLE_ERROR
#define HANDLE_ERROR(expr) err=(expr); if(err != cudaSuccess) std::cout << "error_code: " << err << " err_name: " << cudaGetErrorString(err) << " at " << __FILE__ << " line " << __LINE__ << std::endl;
#endif

#ifndef HANDLE_ERROR_CURAND
#define HANDLE_ERROR_CURAND(expr) st=(expr); if(st != CURAND_STATUS_SUCCESS) std::cout << "curand_error: " << st << " at " << __FILE__ << " line " << __LINE__ << std::endl;
#endif

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
					float beta, float gamma, float s
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

				J_trot = +0.5*log(tanh(beta*gamma*(1-s)/(float)ntrot)); //-(1/2)log(coth(beta*gamma/M))

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
						-2*s*spincache[glIdx_TRCI_ext(block_trot,block_row,block_col, (t%(block_trot)), (i%(block_row)), (j%(block_col)), ind)]*(

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


			/**********************
			  cuda host functions
			 **********************/



			void cuda_init_spin(int32_t*& dev_spin, float*& dev_random, uint32_t num_trot, uint32_t num_row, uint32_t num_col, curandGenerator_t& rng, dim3& grid, dim3& block){
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
			void cuda_run(float beta, float gamma, float s,
					int32_t*& dev_spin,
					float*& dev_random,
					float*& dev_J_out_p,
					float*& dev_J_out_n,
					float*& dev_J_in_0,
					float*& dev_J_in_1,
					float*& dev_J_in_2,
					float*& dev_J_in_3,
					float*& dev_H,
					uint32_t num_trot, uint32_t num_row, uint32_t num_col,
					curandGenerator_t& rng, dim3& grid, dim3& block
					){

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
						beta, gamma, s);

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
						beta, gamma, s);
				return;
			}
		}
	}
}
