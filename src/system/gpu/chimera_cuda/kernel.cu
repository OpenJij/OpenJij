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

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <system/gpu/chimera_cuda/kernel.hpp>

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
                    FloatType* dE,
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
				__shared__ FloatType      dE_cache[block_row * block_col * block_trot * unitsize];

                FloatType J_trot = 0;

                //know who and where we are
                uint32_t r = idx_r(info, blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
                uint32_t c = idx_c(info, blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
                uint32_t i = idx_i(info, blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
                uint32_t t = idx_t(info, blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
                uint32_t global_index = glIdx(info, r, c, i, t);
                uint32_t local_index = glIdx(info, r, c, i);
                uint32_t block_index = bkIdx<block_row,block_col,block_trot>(info,r,c,i,t);
                uint32_t spin_index = bkIdx_ext<block_row,block_col,block_trot>(info,r,c,i,t);

                if(info.trotters > 1){
                    J_trot = 0.5*log(tanh(beta*gamma*(1-s)/(FloatType)info.trotters)); //-(1/2)log(coth(beta*gamma/M))
                }

                J_out_p_cache[block_index] = J_out_p[local_index];
                J_out_n_cache[block_index] = J_out_n[local_index];
                J_in_04_cache[block_index] = J_in_04[local_index];
                J_in_15_cache[block_index] = J_in_15[local_index];
                J_in_26_cache[block_index] = J_in_26[local_index];
                J_in_37_cache[block_index] = J_in_37[local_index];

                randcache[block_index] = rand[global_index];
                spincache[spin_index] =  spin[global_index];

                //be sure that dE_cache is initialized with zero
                dE_cache[block_index] = 0;

                //copy boundary spins to shared memory
                //row
                if(r%block_row == 0 && r != 0){
                    spincache[bkIdx_ext<block_row,block_col,block_trot>(info,-1,c,i,t)]
                        = spin[glIdx(info,r-1,c,i,t)];
                }
                if(r%block_row == block_row-1 && r != info.rows-1){
                    spincache[bkIdx_ext<block_row,block_col,block_trot>(info,block_row,c,i,t)]
                        = spin[glIdx(info,r+1,c,i,t)];
                }
                //col
                if(c%block_col == 0 && c != 0){
                    spincache[bkIdx_ext<block_row,block_col,block_trot>(info,r,-1,i,t)]
                        = spin[glIdx(info,r,c-1,i,t)];
                }
                if(c%block_col == block_col-1 && c != info.cols-1){
                    spincache[bkIdx_ext<block_row,block_col,block_trot>(info,r,block_col,i,t)]
                        = spin[glIdx(info,r,c+1,i,t)];
                }
                //trotter slices
                if(info.trotters > 1){
                    if(t%block_trot == 0){
                        spincache[bkIdx_ext<block_row,block_col,block_trot>(info,r,c,i,-1)]
                            = spin[glIdx(info,r,c,i,(t!=0)?t-1:info.trotters-1)];
                    }
                    if(t%block_trot == block_trot-1){
                        spincache[bkIdx_ext<block_row,block_col,block_trot>(info,r,block_col,i,block_trot)]
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
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>((i<=3)?r-1:r, (4<=i)?c-1:c,i,t)]+

                                J_out_n_cache[block_index]
                                //0 to 3 -> go down 4 to 7 -> go right
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>((i<=3)?r+1:r, (4<=i)?c+1:c,i,t)]+
                                //inside chimera unit
                                J_in_04_cache[block_index] 
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>(r,c,(i<=3)?4:0,t)]+
                                J_in_15_cache[block_index]
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>(r,c,(i<=3)?5:1,t)]+
                                J_in_26_cache[block_index]
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>(r,c,(i<=3)?6:2,t)]+
                                J_in_37_cache[block_index]
                                *spincache[bkIdx_ext<block_row,block_col,block_trot>(r,c,(i<=3)?7:3,t)]+

                                //local magnetization
                                h_cache[block_index]);

                    //trotter slices
                    if(info.trotters > 1){
                        temp_dE +=
                            -2*spincache[spin_index]*J_trot*(
                                    //trotter slices
                                    spincache[bkIdx_ext<block_row,block_col,block_trot>(r,c,i,t+1)]+
                                    spincache[bkIdx_ext<block_row,block_col,block_trot>(r,c,i,t-1)]
                                    );
                    }

                    //update
                    spin[global_index] = ((exp(-temp_dE) > randcache[block_index])?-1:1)*spincache[spin_index];
                }

                __syncthreads();

                // reduction for calculating dE
                constexpr uint32_t count = block_row * block_col * block_trot * unitsize; // <= 1024
                // thread index
                uint32_t ti = threadIdx.z*(blockDim.y*blockDim.x)+threadIdx.y*(blockDim.x)+threadIdx.x;

                count = count/2; //1024 -> 512
                if(ti < count){
                    dE_cache[ti] += dE_cache[ti+count]; 
                }
                count = count/2; //512 -> 256
                if(ti < count){
                    dE_cache[ti] += dE_cache[ti+count]; 
                }
                count = count/2; //256 -> 128
                if(ti < count){
                    dE_cache[ti] += dE_cache[ti+count]; 
                }
                count = count/2; //128 -> 64
                if(ti < count){
                    dE_cache[ti] += dE_cache[ti+count]; 
                }
                count = count/2; //64 -> 32
                if(ti < count){
                    dE_cache[ti] += dE_cache[ti+count]; 
                }
                count = count/2; //32 -> 16
                if(ti < count){
                    dE_cache[ti] += dE_cache[ti+count]; 
                }
                count = count/2; //16 -> 8
                if(ti < count){
                    dE_cache[ti] += dE_cache[ti+count]; 
                }
                count = count/2; //8 -> 4
                if(ti < count){
                    dE_cache[ti] += dE_cache[ti+count]; 
                }
                count = count/2; //4 -> 2
                if(ti < count){
                    dE_cache[ti] += dE_cache[ti+count]; 
                }
                count = count/2; //2 -> 1
                if(ti < count){
                    dE_cache[ti] += dE_cache[ti+count]; 
                }

                if(ti == 0){
                    //add 'em
                    atomicAdd(&dE[0], dE_cache[ti]);
                }

            }


            template<
                typename FloatType,
                std::size_t block_row,
                std::size_t block_col,
                std::size_t block_trot,
                curandRngType_t rng_type
                > 
            FloatType update(
                    ChimeraTransverseGPU<FloatType, block_row, block_col, block_trot>& system,
                    utility::cuda::CurandWrapper<FloatType, rng_type>& random_engine,
                    double beta, FloatType gamma, double s){

                static auto dE = utility::cuda::make_dev_unique<FloatType[]>(1);

                system.spin.get();

                FloatType ret_dE = 0;
                //initialize dE
                HANDLE_ERROR_CUDA(cudaMemcpy(dE.get(), &ret_dE, 1*sizeof(FloatType), cudaMemcpyHostToDevice));
                //generate uniform random sequence
                random_engine.generate_uniform(system.info.rows*system.info.cols*system.info.trotters*system.info.chimera_unitsize);
                //do metropolis
                metropolis<FloatType, block_row, block_col, block_trot, system.info.chimera_unitsize><<<system.grid, system.block>>>(
                        0,
                        system.spin.get(), random_engine.get(),
                        dE.get(),
                        system.interaction.J_out_p.get(),
                        system.interaction.J_out_n.get(),
                        system.interaction.J_in_04.get(),
                        system.interaction.J_in_15.get(),
                        system.interaction.J_in_26.get(),
                        system.interaction.J_in_37.get(),
                        system.interaction.h.get(),
                        system.info,
                        beta, gamma, s
                        );
                //generate uniform random sequence
                random_engine.generate_uniform(system.info.rows*system.info.cols*system.info.trotters*system.info.chimera_unitsize);
                //do metropolis
                metropolis<FloatType, block_row, block_col, block_trot, system.info.chimera_unitsize><<<system.grid, system.block>>>(
                        1,
                        system.spin.get(), random_engine.get(),
                        dE.get(),
                        system.interaction.J_out_p.get(),
                        system.interaction.J_out_n.get(),
                        system.interaction.J_in_04.get(),
                        system.interaction.J_in_15.get(),
                        system.interaction.J_in_26.get(),
                        system.interaction.J_in_37.get(),
                        system.interaction.h.get(),
                        system.info,
                        beta, gamma, s
                        );

                //retrieve dE
                HANDLE_ERROR_CUDA(cudaMemcpy(&ret_dE, dE.get(), 1*sizeof(FloatType), cudaMemcpyDeviceToHost));

                return ret_dE;
            }
        } // namespace chimera_cuda
    } // namespace system
} // namespace openjij

#endif
