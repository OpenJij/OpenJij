#ifndef OPENJIJ_UPDATER_INCLUDE_SINGLE_SPIN_FLIP_HPP__
#define OPENJIJ_UPDATER_INCLUDE_SINGLE_SPIN_FLIP_HPP__

#include <type_traits>
#include <utility>

#include "classical_ising.hpp"
#include "quantum_ising.hpp"

namespace openjij {
    namespace updater {
        template<typename System> struct SingleSpinFlip;
        
        template<>
        struct SingleSpinFlip<system::ClassicalIsing> {
            static double update(system::ClassicalIsing& system, double parameter) {
                std::cout << "execute ClassicalIsing SingleSpinFlip" << std::endl;
                std::cout << "parameter: " << parameter << std::endl;
        
                return 0;
            }
        };
        
        template<>
        struct SingleSpinFlip<system::QuantumIsing> {
            static double update(system::QuantumIsing& system, double parameter) {
                std::cout << "execute QuantumIsing SingleSpinFlip" << std::endl;
                std::cout << "parameter: " << parameter << std::endl;
        
                return 0;
            }
        };
    } // namespace updater
} // namespace openjij

#endif
