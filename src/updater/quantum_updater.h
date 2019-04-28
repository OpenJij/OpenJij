#pragma once
#include "updater.h"

#include <string>

namespace openjij {
    namespace updater {
        class QuantumUpdater : public Updater{
            public:
                // beta -> inverse temperature
                // gamma -> transverse field
                virtual double update(const double beta, const double gamma, const std::string& algo = "") = 0;
        };
    } // namespace updater
} // namespace openjij
