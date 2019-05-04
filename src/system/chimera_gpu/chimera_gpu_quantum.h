#pragma once
#include "../../graph/chimera.h"
#include "../system.h"
#include "../../updater/quantum_updater.h"

namespace openjij {
	namespace system {

		class ChimeraGPUQuantum : public System, public updater::QuantumUpdater{
			//general transverse-field ising model with discrete trotter slices
			
			private:
				graph::Chimera<double> interaction;
				size_t num_trotter_slices;
				size_t row;
				size_t col;

			public:
				ChimeraGPUQuantum(const graph::Chimera<double>& interaction, size_t num_trotter_slices, int gpudevice=0);
				~ChimeraGPUQuantum();

				virtual double update(const double beta, const double gamma, const double s, const std::string& algo = "") override;

				//graph::Spin get_spins(uint32_t t, uint32_t r, uint32_t c, uint32_t ind) const;

				TrotterSpins get_spins() const;

		};
	} // namespace system
} // namespace openjij
