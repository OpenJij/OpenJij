#ifndef OPENJIJ_SYSTEM_INCLUDE_CLASSICAL_ISING_HPP__
#define OPENJIJ_SYSTEM_INCLUDE_CLASSICAL_ISING_HPP__

#include "../graph/dense.h"

namespace openjij {
    namespace system {
        struct ClassicalIsing {
            /**
             * @brief Constructor to initialize spin and interaction
             *
             * @param spin
             * @param interaction
             */
            ClassicalIsing(const graph::Spins& spin, const graph::Dense<double>& interaction)
                : spin{spin}, interaction{interaction} {
            }

            graph::Spins spin;
            graph::Dense<double> interaction;
        };
    } // namespace system
} // namespace openjij

#endif
