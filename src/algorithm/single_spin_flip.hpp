#ifndef OPENJIJ_ALGORITHM_SINGLE_SPIN_FLIP_HPP__
#define OPENJIJ_ALGORITHM_SINGLE_SPIN_FLIP_HPP__

#include <algorithm>

#include "algorithm.hpp"
#include "../system/schedule_list.hpp"
#include "../system/system_traits.hpp"

namespace openjij {
    namespace algorithm {
        namespace detail {
			// double totaldE = 0;
			// size_t num_spins = spins.size();
			// 	for(size_t i=0; i<num_spins; i++){
			// 		size_t index = uid(mt);
			// 		//do metropolis
			// 		double dE = 0;
			// 		for(auto&& adj_index : interaction.adj_nodes(index)){
			// 			dE += -2 * spins[index] * (index != adj_index ? (interaction.J(index, adj_index) * spins[adj_index]) : interaction.h(index));
			// 		}
   //
			// 		//metropolis
			// 		if(exp(-beta*dE) > urd(mt)){
			// 			spins[index] *= -1;
			// 			totaldE += dE;
			// 		}
			// 	}
            template<typename System>
            void single_spin_flip_impl(System& system, const system::ScheduleParameter<System>& parameter, system::classical_system) {
                std::cout << "classical_system" << std::endl;
                const auto beta = parameter;
                std::cout << "beta:  " << beta  << std::endl;

      //           for (auto step = 0, num_total_spin = system.spins.size(); step < num_total_spin; step++) {
      //               const auto idx_target = uid(mt);
      //
      //               // calculate local differential energy
      //               auto dE = 0;
      //               for (auto&& idx_adj : system.interactions.adj_nodes(idx_target)) {
						// dE += -2 * system.spins[idx_target] * (idx_target != idx_adj ? (system.interactions.J(idx_target, idx_adj) * system.spins[idx_adj]) : system.interactions.h(idx_target));
      //               }
      //
      //               // determine whether flip the target spin or not by following the Metropolis method
      //               if (std::exp(-beta * dE) > urd(mt)) {
      //                   system.spins[idx_target] *= -1;
      //               }
      //           }
            };

            template<typename System>
            void single_spin_flip_impl(System& system, const system::ScheduleParameter<System>& parameter, system::quantum_system) {
                std::cout << "quantum_system" << std::endl;
                const auto beta        = parameter.first;
                const auto gamma       = parameter.second;

                std::cout << "beta:  " << beta  << std::endl;
                std::cout << "gamma: " << gamma << std::endl;
            };
        } // namespace detail

        struct SingleSpinFlip : public Algorithm<SingleSpinFlip> {
            template<typename System>
            void update(System& system, const system::ScheduleParameter<System>& parameter) const {
                std::cout << "SingleSpinFlip::update()" << std::endl;
                detail::single_spin_flip_impl(system, parameter, typename system::get_system_type<System>::type());
            }

        private:
        };
    } // namespace algorithm
} // namespace openjij

#endif
