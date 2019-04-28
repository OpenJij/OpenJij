#pragma once
#include "../graph/dense.h"
#include "system.h"
#include "../updater/quantum_updater.h"
#include <random>

namespace openjij {
	namespace system {

		//TODO: double -> FloatType (template)
		class QuantumIsing : public System, public updater::QuantumUpdater{
			//general transverse-field ising model with discrete trotter slices

			private:
				TrotterSpins spins; //spins with trotter slices
				graph::Dense<double> interaction; //original interaction

				//random number generators
				//TODO: use MT or xorshift?
				std::mt19937 mt;
				std::uniform_int_distribution<> uid;
				std::uniform_int_distribution<> uid_trotter;
				std::uniform_real_distribution<> urd;

				//mod function for dealing trotter indices
				inline size_t mod_t(int64_t a) const;

			public:
				QuantumIsing(const graph::Dense<double>& interaction, size_t num_trotter_slices);
				QuantumIsing(const graph::Dense<double>& interaction, size_t num_trotter_slices, graph::Spins& classical_spins);

				void initialize_spins();
				void set_spins(graph::Spins& initial_spin);

				virtual double update(double beta, double gamma, const std::string& algo = "") override;

				void simulated_quantum_annealing(double beta, double gamma_min, double gamma_max, double step_length, size_t step_num, const std::string& algo = "");

				TrotterSpins get_spins() const;

		};
	} // namespace system
} // namespace openjij

