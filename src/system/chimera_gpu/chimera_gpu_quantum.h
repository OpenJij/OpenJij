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

				//HANDLE ERROR
				cudaError_t err;
				curandStatus_t st;

				/*************************
				  list of device variables
				 *************************/

				// inteactions
				//(row*col*8) = localsize
				std::unique_ptr<float> dev_J_out_p;
				//(row*col*8)
				std::unique_ptr<float> dev_J_out_n;
				//(row*col*8)
				std::unique_ptr<float> dev_J_in_0;
				//(row*col*8)
				std::unique_ptr<float> dev_J_in_1;
				//(row*col*8)
				std::unique_ptr<float> dev_J_in_2;
				//(row*col*8)
				std::unique_ptr<float> dev_J_in_3;

				// local magnetization
				//(row*col*8)
				std::unique_ptr<float> dev_H;

				// spins and randoms
				//(gammasize*betasize*trot*row*col*8) =totalsize
				std::unique_ptr<int32_t> dev_spin;
				//(gammasize*betasize*trot*row*col*8) =totalsize
				std::unique_ptr<float> dev_random;

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
				std::unique_ptr<int32_t> spinarray;


			public:
				ChimeraGPUQuantum(const graph::Chimera<double>& interaction, size_t num_trotter_slices, int gpudevice=0);
				~ChimeraGPUQuantum();

				virtual double update(const double beta, const double gamma, const double s, const std::string& algo = "") override;

				void simulated_quantum_annealing(const double beta, const double gamma, const size_t step_length, const size_t step_num, const std::string& algo = "");
				void simulated_quantum_annealing(const double beta, const double gamma, const Schedule& schedule, const std::string& algo = "");

				//graph::Spin get_spins(uint32_t t, uint32_t r, uint32_t c, uint32_t ind) const;

				TrotterSpins get_spins() const;

		};
	} // namespace system
} // namespace openjij
