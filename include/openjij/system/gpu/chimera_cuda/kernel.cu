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

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include "openjij/system/gpu/chimera_cuda/kernel.hpp"
#include <iostream>

namespace openjij {
    namespace system {

        //for cuda device (kernel)
        namespace chimera_cuda {

            template<
                typename FloatType,
                std::size_t block_row,
                std::size_t block_col,
                std::size_t block_trot,
                std::size_t unitsize
                > 
            __global__ void metropolis(
                    int32_t sw,
                    int32_t* spin, const FloatType* rand,
                    const FloatType* J_out_p,
                    const FloatType* J_out_n,
                    const FloatType* J_in_04,
                    const FloatType* J_in_15,
                    const FloatType* J_in_26,
                    const FloatType* J_in_37,
                    const FloatType* h,
                    ChimeraInfo info,
                    double beta, FloatType gamma, double s
                    ){

                static_assert(block_row*block_col*block_trot*unitsize <= 1024, "max limit of the number of thread for each block is 1024.");

                //switch
                //(0 -> 1st chimera unit (t==0, i==0, j==0) -> (0...3))
                //(1 -> 1st chimera unit (t==0, i==0, j==0) -> (4...7))

                //shared memory
				//spin with boundaries
				__shared__ int32_t spincache[(block_row+2) * (block_col+2) * (block_trot+2) * unitsize];

                __shared__ FloatType     randcache[block_row * block_col * block_trot * unitsize];

				__shared__ FloatType J_out_p_cache[block_row * block_col * block_trot * unitsize];
				__shared__ FloatType J_out_n_cache[block_row * block_col * block_trot * unitsize];
				__shared__ FloatType J_in_04_cache[block_row * block_col * block_trot * unitsize];
				__shared__ FloatType J_in_15_cache[block_row * block_col * block_trot * unitsize];
				__shared__ FloatType J_in_26_cache[block_row * block_col * block_trot * unitsize];
				__shared__ FloatType J_in_37_cache[block_row * block_col * block_trot * unitsize];
				__shared__ FloatType       h_cache[block_row * block_col * block_trot * unitsize];
				//__shared__ FloatType      dE_cache[block_row * block_col * block_trot * unitsize];

                FloatType J_trot = 0;

                //know who and where we are
                int32_t r = idx_r(info, blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
                int32_t c = idx_c(info, blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
                int32_t i = idx_i(info, blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
                int32_t t = idx_t(info, blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);

                int32_t b_r = r%block_row;
                int32_t b_c = c%block_col;
                int32_t b_t = t%block_trot;

                int32_t global_index = glIdx(info, r, c, i, t);
                int32_t local_index = glIdx(info, r, c, i);
                int32_t block_index = bkIdx<block_row,block_col,block_trot>(info,b_r,b_c,i,b_t);
                int32_t spin_index = bkIdx_ext<block_row,block_col,block_trot>(info,b_r,b_c,i,b_t);

                if(info.trotters > 1){
                    J_trot = 0.5*log(tanh(beta*gamma*(1-s)/(FloatType)info.trotters)); //-(1/2)log(coth(beta*gamma/M))
                }

                J_out_p_cache[block_index] = J_out_p[local_index];
                J_out_n_cache[block_index] = J_out_n[local_index];
                J_in_04_cache[block_index] = J_in_04[local_index];
                J_in_15_cache[block_index] = J_in_15[local_index];
                J_in_26_cache[block_index] = J_in_26[local_index];
                J_in_37_cache[block_index] = J_in_37[local_index];
                      h_cache[block_index] =       h[local_index];

                randcache[block_index] = rand[global_index];
                spincache[spin_index] =  spin[global_index];

                //be sure that dE_cache is initialized with zero
                //dE_cache[block_index] = 0;

                //copy boundary spins to shared memory
                //row
                if(r%block_row == 0 && r != 0){
                    spincache[bkIdx_ext<block_row,block_col,block_trot>(info,-1,b_c,i,b_t)]
                        = spin[glIdx(info,r-1,c,i,t)];
                }
                if(r%block_row == block_row-1 && r != info.rows-1){
                    spincache[bkIdx_ext<block_row,block_col,block_trot>(info,block_row,b_c,i,b_t)]
                        = spin[glIdx(info,r+1,c,i,t)];
                }
                //col
                if(c%block_col == 0 && c != 0){
                    spincache[bkIdx_ext<block_row,block_col,block_trot>(info,b_r,-1,i,b_t)]
                        = spin[glIdx(info,r,c-1,i,t)];
                }
                if(c%block_col == block_col-1 && c != info.cols-1){
                    spincache[bkIdx_ext<block_row,block_col,block_trot>(info,b_r,block_col,i,b_t)]
                        = spin[glIdx(info,r,c+1,i,t)];
                }
                //trotter slices
                if(info.trotters > 1){
                    if(t%block_trot == 0){
                        spincache[bkIdx_ext<block_row,block_col,block_trot>(info,b_r,b_c,i,-1)]
                            = spin[glIdx(info,r,c,i,(t!=0)?t-1:info.trotters-1)];
                    }
                    if(t%block_trot == block_trot-1){
                        spincache[bkIdx_ext<block_row,block_col,block_trot>(info,b_r,b_c,i,block_trot)]
                            = spin[glIdx(info,r,c,i,(t!=info.trotters-1)?t+1:0)];
                    }
                }

                __syncthreads();

                //do metropolis
                if(((r+c+t)%2 == sw && i <= 3) || ((r+c+t)%2 != sw && 4 <= i)){
                    FloatType temp_dE = 
                        -2*s*spincache[spin_index]*beta/(FloatType)(info.trotters)*(
                                //outside chimera unit
                                J_out_p_cache[block_index]
                                //0 to 3 -> go up 4 to 7 -> go left
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>(info,(i<=3)?b_r-1:b_r, (4<=i)?b_c-1:b_c,i,b_t)]+

                                J_out_n_cache[block_index]
                                //0 to 3 -> go down 4 to 7 -> go right
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>(info,(i<=3)?b_r+1:b_r, (4<=i)?b_c+1:b_c,i,b_t)]+
                                //inside chimera unit
                                J_in_04_cache[block_index] 
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>(info,b_r,b_c,(i<=3)?4:0,b_t)]+
                                J_in_15_cache[block_index]
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>(info,b_r,b_c,(i<=3)?5:1,b_t)]+
                                J_in_26_cache[block_index]
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>(info,b_r,b_c,(i<=3)?6:2,b_t)]+
                                J_in_37_cache[block_index]
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>(info,b_r,b_c,(i<=3)?7:3,b_t)]+

                                //local magnetization
                                h_cache[block_index]);

                    //trotter slices
                    if(info.trotters > 1){
                        temp_dE +=
                            -2*spincache[spin_index]*J_trot*(
                                    //trotter slices
                                    spincache[bkIdx_ext<block_row,block_col,block_trot>(info,b_r,b_c,i,b_t+1)]+
                                    spincache[bkIdx_ext<block_row,block_col,block_trot>(info,b_r,b_c,i,b_t-1)]
                                    );
                    }

                    //update
                    spin[global_index] = ((exp(-temp_dE) > randcache[block_index])?-1:1)*spincache[spin_index];
                }

                __syncthreads();

                // reduction for calculating dE
                //uint32_t count = block_row * block_col * block_trot * unitsize; // <= 1024
                //// thread index
                //uint32_t ti = threadIdx.z*(blockDim.y*blockDim.x)+threadIdx.y*(blockDim.x)+threadIdx.x;

                //count = count/2; //1024 -> 512
                //if(ti < count){
                //    dE_cache[ti] += dE_cache[ti+count]; 
                //}
                //__syncthreads();
                //count = count/2; //512 -> 256
                //if(ti < count){
                //    dE_cache[ti] += dE_cache[ti+count]; 
                //}
                //__syncthreads();
                //count = count/2; //256 -> 128
                //if(ti < count){
                //    dE_cache[ti] += dE_cache[ti+count]; 
                //}
                //__syncthreads();
                //count = count/2; //128 -> 64
                //if(ti < count){
                //    dE_cache[ti] += dE_cache[ti+count]; 
                //}
                //__syncthreads();
                //count = count/2; //64 -> 32
                //if(ti < count){
                //    dE_cache[ti] += dE_cache[ti+count]; 
                //}
                //__syncthreads();
                //count = count/2; //32 -> 16
                //if(ti < count){
                //    dE_cache[ti] += dE_cache[ti+count]; 
                //}
                //__syncthreads();
                //count = count/2; //16 -> 8
                //if(ti < count){
                //    dE_cache[ti] += dE_cache[ti+count]; 
                //}
                //__syncthreads();
                //count = count/2; //8 -> 4
                //if(ti < count){
                //    dE_cache[ti] += dE_cache[ti+count]; 
                //}
                //__syncthreads();
                //count = count/2; //4 -> 2
                //if(ti < count){
                //    dE_cache[ti] += dE_cache[ti+count]; 
                //}
                //__syncthreads();
                //count = count/2; //2 -> 1
                //if(ti < count){
                //    dE_cache[ti] += dE_cache[ti+count]; 
                //}
                //__syncthreads();

                //if(ti == 0){
                //    //add 'em
                //    atomicAdd(&dE[0], dE_cache[ti]);
                //}

            }

            template<
                typename FloatType,
                std::size_t block_row,
                std::size_t block_col,
                std::size_t block_trot>
            void metropolis_interface(
                    int32_t sw,
                    int32_t* spin, const FloatType* rand,
                    const FloatType* J_out_p,
                    const FloatType* J_out_n,
                    const FloatType* J_in_04,
                    const FloatType* J_in_15,
                    const FloatType* J_in_26,
                    const FloatType* J_in_37,
                    const FloatType* h,
                    const ChimeraInfo& info, const dim3& grid, const dim3& block,
                    double beta, FloatType gamma, double s){

                metropolis<FloatType, block_row, block_col, block_trot, info.chimera_unitsize><<<grid, block>>>(
                        sw,
                        spin, rand,
                        J_out_p,
                        J_out_n,
                        J_in_04,
                        J_in_15,
                        J_in_26,
                        J_in_37,
                        h,
                        info,
                        beta, gamma, s
                        );
            }

            //template instantiation

#define FLOAT_ARGTYPE int32_t,int32_t*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,const ChimeraInfo&,const dim3&,const dim3&,double,float,double
#define DOUBLE_ARGTYPE int32_t,int32_t*,const double*,const double*,const double*,const double*,const double*,const double*,const double*,const double*,const ChimeraInfo&,const dim3&,const dim3&,double,double,double

            template void metropolis_interface<float,1,1,1>(FLOAT_ARGTYPE);
            template void metropolis_interface<float,2,2,2>(FLOAT_ARGTYPE);
            template void metropolis_interface<float,3,3,3>(FLOAT_ARGTYPE);
            template void metropolis_interface<float,4,4,4>(FLOAT_ARGTYPE);
            template void metropolis_interface<float,2,2,1>(FLOAT_ARGTYPE);
            template void metropolis_interface<float,3,3,1>(FLOAT_ARGTYPE);
            template void metropolis_interface<float,4,4,1>(FLOAT_ARGTYPE);
            template void metropolis_interface<double,1,1,1>(DOUBLE_ARGTYPE);
            template void metropolis_interface<double,2,2,2>(DOUBLE_ARGTYPE);
            template void metropolis_interface<double,3,3,3>(DOUBLE_ARGTYPE);
            template void metropolis_interface<double,4,4,4>(DOUBLE_ARGTYPE);
            template void metropolis_interface<double,2,2,1>(DOUBLE_ARGTYPE);
            template void metropolis_interface<double,3,3,1>(DOUBLE_ARGTYPE);
            template void metropolis_interface<double,4,4,1>(DOUBLE_ARGTYPE);
        } // namespace chimera_cuda
    } // namespace system
} // namespace openjij

#endif
