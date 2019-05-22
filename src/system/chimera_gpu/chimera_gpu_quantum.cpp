#include "chimera_gpu_quantum.h"
#include <random>
#include "../../algorithm/sqa.h"
#include <cuda_runtime.h>
#include <curand.h> 
#include "kernel_frontend.h"
#include "index.h"
#include "../cuda_error.h"
#include <cassert> 
#include <cmath>
#include <iostream>

namespace openjij {
	namespace system {


		/**********************
		  cuda host functions
		 **********************/

		inline void ChimeraGPUQuantum::cuda_set_device(int device){
			HANDLE_ERROR(cudaSetDevice(device));
		}

		//intialize GPU 
		void ChimeraGPUQuantum::cuda_init(
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
			//xorwow
			//HANDLE_ERROR_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MT19937));
			HANDLE_ERROR_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW));
			//set seed
			HANDLE_ERROR_CURAND(curandSetPseudoRandomGeneratorSeed(rng, std::random_device()()));

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
			chimera_gpu::cuda_init_spin(dev_spin, dev_random, num_trot, num_row, num_col, rng, grid, block);

			//initialize spinarray
			HANDLE_ERROR(cudaMallocHost((void**)&spinarray, sizeof(int32_t)*totalsize));

		}

		void ChimeraGPUQuantum::cuda_init_interactions(
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
		}

		void ChimeraGPUQuantum::copy_spins() const{
			HANDLE_ERROR(cudaMemcpy(spinarray, dev_spin, num_trot*num_row*num_col*unitspins*sizeof(int32_t), cudaMemcpyDeviceToHost));
		}

		void ChimeraGPUQuantum::cuda_free(){
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


		ChimeraGPUQuantum::ChimeraGPUQuantum(const graph::Chimera<double>& interaction, size_t num_trotter_slices, int gpudevice)
		: interaction(interaction), num_trotter_slices(num_trotter_slices), row(interaction.get_num_row()), col(interaction.get_num_column()){
			//num_trotter_slices must be even.

			size_t num_in_chimera = interaction.get_num_in_chimera();

			assert(num_trotter_slices%2 == 0);
			//row and col must be even.
			assert(row%2 == 0);
			assert(col%2 == 0);
			//overflow?
			assert((uint64_t)0xFFFFFFFF >= (uint64_t)num_trotter_slices*row*col*8);
			//set device
			cuda_set_device(gpudevice);
			//init GPU
			cuda_init(num_trotter_slices, row, col);

			//init parameters;
			uint32_t localsize = row*col*8;
			std::vector<float> J_out_p(localsize); //prev
			std::vector<float> J_out_n(localsize); //next
			std::vector<float> J_in_0(localsize);
			std::vector<float> J_in_1(localsize);
			std::vector<float> J_in_2(localsize);
			std::vector<float> J_in_3(localsize);
			std::vector<float> H(localsize);

			for(size_t r=0; r<row; r++){
				for(size_t c=0; c<col; c++){
					for(size_t i=0; i<num_in_chimera; i++){
						//open boundary
						if(r > 0 && i < 4){
							//MINUS_R (0<=i<4)
							J_out_p[glIdx_RCI(row, col, r, c, i)] = (float)interaction.J(r, c, i, graph::ChimeraDir::MINUS_R);
						}
						if(c > 0 && 4 <= i){
							//MINUS_C (4<=i<8)
							J_out_p[glIdx_RCI(row, col, r, c, i)] = (float)interaction.J(r, c, i, graph::ChimeraDir::MINUS_C);
						}
						if(r < row-1 && i < 4){
							//PLUS_R (0<=i<4)
							J_out_n[glIdx_RCI(row, col, r, c, i)] = (float)interaction.J(r, c, i, graph::ChimeraDir::PLUS_R);
						}
						if(c < col-1 && 4 <= i){
							//PLUS_C (4<=i<8)
							J_out_n[glIdx_RCI(row, col, r, c, i)] = (float)interaction.J(r, c, i, graph::ChimeraDir::PLUS_C);
						}

						//inside chimera unit
						J_in_0[glIdx_RCI(row, col, r, c, i)]  = (float)interaction.J(r, c, i, graph::ChimeraDir::IN_0or4);
						J_in_1[glIdx_RCI(row, col, r, c, i)]  = (float)interaction.J(r, c, i, graph::ChimeraDir::IN_1or5);
						J_in_2[glIdx_RCI(row, col, r, c, i)]  = (float)interaction.J(r, c, i, graph::ChimeraDir::IN_2or6);
						J_in_3[glIdx_RCI(row, col, r, c, i)]  = (float)interaction.J(r, c, i, graph::ChimeraDir::IN_3or7);

						//local field
						H[glIdx_RCI(row, col, r, c, i)] = (float)interaction.h(r, c, i);
					}
				}
			}


			//init interactions
			cuda_init_interactions(
					J_out_p.data(),
					J_out_n.data(),
					J_in_0.data(),
					J_in_1.data(),
					J_in_2.data(),
					J_in_3.data(),
					H.data()
					);

			//init_spins
			chimera_gpu::cuda_init_spin(dev_spin, dev_random, num_trot, num_row, num_col, rng, grid, block);

		}

		ChimeraGPUQuantum::~ChimeraGPUQuantum(){
			cuda_free();
		}

		double ChimeraGPUQuantum::update(const double beta, const double gamma, const double s, const std::string& algo){
			if(algo == "gpu_metropolis" or algo == ""){
				chimera_gpu::cuda_run(beta, gamma, s,
						dev_spin, dev_random,
						dev_J_out_p,
						dev_J_out_n,
						dev_J_in_0,
						dev_J_in_1,
						dev_J_in_2,
						dev_J_in_3,
						dev_H,
						num_trot, num_row, num_col,
						rng, grid, block
						);

			}
			return 0;
		}

		void ChimeraGPUQuantum::simulated_quantum_annealing(const double beta, const double gamma, const size_t step_length, const size_t step_num, const std::string& algo) {
			algorithm::SQA sqa(beta, gamma, step_length, step_num);
			//do simulated quantum annealing
			sqa.run(*this, algo);
		}

		void ChimeraGPUQuantum::simulated_quantum_annealing(const double beta, const double gamma, const Schedule& schedule, const std::string& algo) {
			algorithm::SQA sqa(beta, gamma, schedule);
			//do simulated quantum annealing
			sqa.run(*this, algo);
		}

		TrotterSpins ChimeraGPUQuantum::get_spins() const{
			//cudaMemcpy
			copy_spins();
			TrotterSpins ret_spins(num_trotter_slices);
			for(size_t t=0; t<num_trotter_slices; t++){
				ret_spins[t] = graph::Spins(interaction.get_num_spins());
				for(size_t r=0; r<row; r++){
					for(size_t c=0; c<col; c++){
						for(size_t i=0; i<8; i++){
							ret_spins[t][interaction.to_ind(r,c,i)] = spinarray[glIdx_TRCI(num_trot, num_row, num_col, t,r,c,i)];
						}
					}
				}
			}

			return ret_spins;
		}

	} // namespace system
} // namespace openjij
