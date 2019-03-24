#include "chimera_gpu_quantum.h"
#include "../../algorithm/sqa.h"
#include "kernel_frontend.h"
#include "index.h"
#include <cassert>
#include <cmath>
#include <iostream>

namespace openjij {
	namespace method {

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
			chimera_gpu::cuda_set_device(gpudevice);
			//init GPU
			chimera_gpu::cuda_init(num_trotter_slices, row, col);

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
			chimera_gpu::cuda_init_interactions(
					J_out_p.data(),
					J_out_n.data(),
					J_in_0.data(),
					J_in_1.data(),
					J_in_2.data(),
					J_in_3.data(),
					H.data()
					);

			//init_spins
			chimera_gpu::cuda_init_spin();

		}

		ChimeraGPUQuantum::~ChimeraGPUQuantum(){
			chimera_gpu::cuda_free();
		}

		double ChimeraGPUQuantum::update(double beta, double gamma, const std::string& algo){
			if(algo == "gpu_metropolis" or algo == ""){
				chimera_gpu::cuda_run(beta, gamma);
			}
			return 0;
		}

		void ChimeraGPUQuantum::simulated_quantum_annealing(double beta, double gamma_min, double gamma_max, double step_length, size_t step_num, const std::string& algo){
			algorithm::SQA sqa(beta, gamma_min, gamma_max, step_length, step_num);
			//do simulated quantum annealing
			sqa.exec(*this, algo);
		}

		TrotterSpins ChimeraGPUQuantum::get_spins() const{
			//cudaMemcpy
			chimera_gpu::copy_spins();
			TrotterSpins ret_spins(num_trotter_slices);
			for(size_t t=0; t<num_trotter_slices; t++){
				ret_spins[t] = graph::Spins(interaction.get_num_spins());
				for(size_t r=0; r<row; r++){
					for(size_t c=0; c<col; c++){
						for(size_t i=0; i<8; i++){
							ret_spins[t][interaction.to_ind(r,c,i)] = chimera_gpu::get_spin(t, r, c, i);
						}
					}
				}
			}

			return ret_spins;
		}

	} // namespace method
} // namespace openjij
