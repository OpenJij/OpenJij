#ifndef OPENJIJ_UPDATER_INCLUDE_SINGLE_SPIN_FLIP_HPP__
#define OPENJIJ_UPDATER_INCLUDE_SINGLE_SPIN_FLIP_HPP__

#include <random>

#include <system/classical_ising.hpp>
#include <system/quantum_ising.hpp>

namespace openjij {
    namespace updater {
        template<typename System> struct SingleSpinFlip;

        template<>
        struct SingleSpinFlip<system::ClassicalIsing> {
            /**
             * @brief operate single spin flip in a classical ising system
             *
             * @param system object of a classical ising system
             * @param beta inverse temperature \f\beta:=(k_B T)^{-1}\f
             *
             * @return energy difference \f\Delta E\f
             */
            static double update(system::ClassicalIsing& system, double beta) {
                // TODO: separate random generator discribed below from this function
                //
                // initialize random number generator
                // set a seed value
                // TODO: The seed MUST be given by an argument of this function
                std::random_device rd;
                // set random generator engine
                // TODO: This engine MUST be given by an argument of this function
                std::mt19937 mt(rd());
                // to select candidate for flip at random
                auto uid = std::uniform_int_distribution<std::size_t>(0, system.spin.size()-1);
                // to do Metroopolis
                auto urd = std::uniform_real_distribution<>(0, 1.0);
                // end initialize random number generator

                // energy difference
                double total_dE = 0;

                for (std::size_t time = 0, num_spin = system.spin.size(); time < num_spin; ++time) {
                    // index of spin selected at random
                    const auto index = uid(mt);

                    // local energy difference
                    auto dE = 0;
                    for (auto&& adj_index : system.interaction.adj_nodes(index)) {
                        dE += -2.0 * system.spin[index] * (index != adj_index ? (system.interaction.J(index, adj_index) * system.spin[adj_index])
                                                                              :  system.interaction.h(index));
                    }

                    // Flip the spin?
                    if (std::exp( -beta * dE) > urd(mt)) {
                        system.spin[index] *= -1;
                        total_dE += dE;
                    }
                }

                return total_dE;
            }
        };

        template<>
        struct SingleSpinFlip<system::QuantumIsing> {
            /**
             * @brief operate single spin flip in a quantum ising system
             *
             * @param system object of a quantum ising system
             * @param beta inverse temperature \f\beta:=(k_B T)^{-1}\f
             *
             * @return energy difference \f\Delta E\f
             */
            static double update(system::QuantumIsing& system, double parameter) {
                std::cout << "execute QuantumIsing SingleSpinFlip" << std::endl;
                std::cout << "parameter: " << parameter << std::endl;

                return 0;
            }
        };
    } // namespace updater
} // namespace openjij

#endif
