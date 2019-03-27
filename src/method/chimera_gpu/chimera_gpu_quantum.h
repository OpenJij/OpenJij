#pragma once
#include "../../graph/chimera.h"
#include "../method.h"
#include "../../updater/quantum_updater.h"

namespace openjij {
	namespace method {

		class ChimeraGPUQuantum : public Method, public updater::QuantumUpdater{
			//general transverse-field ising model with discrete trotter slices
			
			private:
				graph::Chimera<double> interaction;
				size_t num_trotter_slices;
				size_t row;
				size_t col;

			public:
				ChimeraGPUQuantum(const graph::Chimera<double>& interaction, size_t num_trotter_slices, int gpudevice=0);
				~ChimeraGPUQuantum();

				virtual double update(double beta, double gamma, const std::string& algo = "") override;

				void simulated_quantum_annealing(double beta, double gamma_min, double gamma_max, double step_length, size_t step_num, const std::string& algo = "");

				//graph::Spin get_spins(uint32_t t, uint32_t r, uint32_t c, uint32_t ind) const;

				TrotterSpins get_spins() const;

		};
	} // namespace method
} // namespace openjij
