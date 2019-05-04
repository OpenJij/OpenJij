#pragma once
#include "updater.h"

#include <string>

namespace openjij {
    namespace updater {
        class QuantumUpdater : public Updater{
            public:
                // beta -> inverse temperature
                // gamma -> transverse field
                // s -> annealing parameter
                virtual double update(const double beta, const double gamma, const double s, const std::string& algo = "") = 0;
        };
    } // namespace updater
} // namespace openjij
