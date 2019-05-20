#pragma once
#include "../../graph/chimera.h"
#include "../system.h"
#include "../../updater/quantum_updater.h"
#include <memory>
#include <cuda_runtime.h>
#include <curand.h> 

namespace openjij {
	namespace system {

		class ChimeraGPUQuantum : public System, public updater::QuantumUpdater{
			//general transverse-field ising model with discrete trotter slices

			private:
				using Schedule = std::vector<std::pair<double, size_t>>;

				graph::Chimera<double> interaction;
				size_t num_trotter_slices;
				size_t row;
				size_t col;


				/*************************
				  list of device variables
				 *************************/

				// inteactions
				//(row*col*8) = localsize
				float* dev_J_out_p;
				//(row*col*8)
				float* dev_J_out_n;
				//(row*col*8)
				float* dev_J_in_0;
				//(row*col*8)
				float* dev_J_in_1;
				//(row*col*8)
				float* dev_J_in_2;
				//(row*col*8)
				float* dev_J_in_3;

				// local magnetization
				//(row*col*8)
				float* dev_H;

				// spins and randoms
				//(gammasize*betasize*trot*row*col*8) =totalsize
				int32_t* dev_spin;
				//(gammasize*betasize*trot*row*col*8) =totalsize
				float* dev_random;

				//curand generator
				curandGenerator_t rng;

				/**********************
				  host variables (static)
				 **********************/

				//the number of trotter slices
				uint32_t num_trot;

				//the number of rows and cols of each Chimera graph
				uint32_t num_row;
				uint32_t num_col;

				//grids and blocks
				dim3 grid;
				dim3 block;

				//spinarray (page-locked memory)
				int32_t* spinarray;

				/**********************
				  const variables 
				 **********************/

				constexpr static uint32_t unitspins = 8;

				constexpr static uint32_t block_row = 2;
				constexpr static uint32_t block_col = 2;
				constexpr static uint32_t block_trot = 2;

				/**********************
				  cuda host functions
				 **********************/

				void cuda_set_device(int device);

				void cuda_init(
						uint32_t arg_num_trot,
						uint32_t arg_num_row,
						uint32_t arg_num_col
						);

				void cuda_init_interactions(
						const float* J_out_p,
						const float* J_out_n,
						const float* J_in_0,
						const float* J_in_1,
						const float* J_in_2,
						const float* J_in_3,
						const float* H
						);

				void copy_spins() const;
				
				void cuda_free();

			public:
				ChimeraGPUQuantum(const graph::Chimera<double>& interaction, size_t num_trotter_slices, int gpudevice=0);
				~ChimeraGPUQuantum();

				//disable copy constructor
				ChimeraGPUQuantum(const ChimeraGPUQuantum&) = delete;



				virtual double update(const double beta, const double gamma, const double s, const std::string& algo = "") override;

				void simulated_quantum_annealing(const double beta, const double gamma, const size_t step_length, const size_t step_num, const std::string& algo = "");
				void simulated_quantum_annealing(const double beta, const double gamma, const Schedule& schedule, const std::string& algo = "");

				TrotterSpins get_spins() const;

		};
	} // namespace system
} // namespace openjij
