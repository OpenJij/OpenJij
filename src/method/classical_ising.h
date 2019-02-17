#pragma once
#include "../graph/sparse.h"
#include "method.h"
#include "../updater/classical_updater.h"
#include <random>

namespace openjij {
	namespace method {
		using namespace openjij::graph;
		using namespace openjij::updater;

		//TODO: double -> FloatType (template)
		class ClassicalIsing : public Method, public ClassicalUpdater{
			//general classical ising model
			private:
				Spins spins;
				Sparse<double> interaction;
				//random number generator
				//TODO: use MT or xorshift
				std::mt19937 mt;
				std::uniform_int_distribution<> uid;
				std::uniform_real_distribution<> urd;

			public:
				ClassicalIsing(const Sparse<double>& interaction);

				virtual double update(double beta, const std::string& algo = "") override;

				//do simulated annelaing
				void simulated_annealing(double beta_min, double beta_max, double step_length, size_t step_num, const std::string& algo = "");

				const Spins get_spins() const;
		};
	} // namespace method
} // namespace openjij

