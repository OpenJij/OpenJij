#pragma once

#include <random>
#include <utility>
#include <vector>

#include "../graph/dense.h"
#include "system.h"
#include "../updater/classical_updater.h"

namespace openjij {
	namespace system {

		//TODO: double -> FloatType (template)
		class ClassicalIsing : public System, public updater::ClassicalUpdater{
			//general classical ising model
			public:
				using Schedule = std::vector<std::pair<double, size_t>>;

				ClassicalIsing(const graph::Dense<double>& interaction);
				ClassicalIsing(const graph::Dense<double>& interaction, graph::Spins& spins);

				void initialize_spins();
				void set_spins(graph::Spins& initial_spins);

				virtual double update(const double beta, const std::string& algo = "") override;

				const graph::Spins get_spins() const;

			private:
				graph::Spins spins;
				graph::Dense<double> interaction;
				//random number generator
				//TODO: use MT or xorshift
				std::mt19937 mt;
				std::uniform_int_distribution<> uid;
				std::uniform_real_distribution<> urd;

		};
	} // namespace system
} // namespace openjij

