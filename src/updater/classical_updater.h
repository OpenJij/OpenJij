#pragma once
#include "updater.h"
#include <string>

namespace openjij {
	namespace updater {
		class ClassicalUpdater : public Updater{
			public:
				//beta -> inverse temperature
				virtual double update(double beta, const std::string& algo = "") = 0;
		};
	} // namespace updater
} // namespace openjij
