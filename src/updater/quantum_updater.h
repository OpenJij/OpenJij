#pragma once
#include "updater.h"
#include <string>

namespace openjij {
	namespace updater {
		class QuantumUpdater : public Updater{
			public:
				//beta -> inverse temperature
				//gamma -> transverse field
				virtual double update(double beta, double gamma, const std::string& algo = "") = 0;
		};
	} // namespace updater
} // namespace openjij
