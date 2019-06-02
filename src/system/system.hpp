#ifndef OPENJIJ_SYSTEM_SYSTEM_HPP__
#define OPENJIJ_SYSTEM_SYSTEM_HPP__

#include "system_traits.hpp"
#include "../graph/all.h"

namespace openjij {
    namespace system {
        template<typename SystemType, typename SpinsType, typename GraphType>
        struct System {
            using system_type = SystemType;

            explicit System(const SpinsType& spins, const GraphType& interactions) noexcept
                : spins{spins}, interactions{interactions} {
            }

            SpinsType spins;
            GraphType interactions;
        };

        using ClassicalIsing = System<classical_system, graph::Spins, graph::Dense<double>>;
        using QuantumIsing = System<quantum_system, graph::TrotterSpins, graph::Dense<double>>;
    } // namespace system
} // namespace openjij

#endif
