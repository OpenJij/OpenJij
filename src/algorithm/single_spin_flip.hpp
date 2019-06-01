#ifndef OPENJIJ_ALGORITHM_SINGLE_SPIN_FLIP_HPP__
#define OPENJIJ_ALGORITHM_SINGLE_SPIN_FLIP_HPP__

#include "algorithm.hpp"
#include "../system/schedule_list.hpp"
#include "../system/system_traits.hpp"

namespace openjij {
    namespace algorithm {
        namespace detail {
            template<typename System>
            void update_impl(System& system, const system::ScheduleParameter<System>& parameter, system::classical_system) {
                std::cout << "classical_system" << std::endl;
                const auto one_mc_step = parameter.first;
                const auto beta = parameter.second;

                std::cout << "one_mc_step: " << one_mc_step << std::endl;
                std::cout << "beta:        " << beta        << std::endl;
            };

            template<typename System>
            void update_impl(System& system, const system::ScheduleParameter<System>& parameter, system::quantum_system) {
                std::cout << "quantum_system" << std::endl;
                const auto one_mc_step = std::get<0>(parameter);
                const auto beta        = std::get<1>(parameter);
                const auto gamma       = std::get<2>(parameter);

                std::cout << "one_mc_step: " << one_mc_step << std::endl;
                std::cout << "beta:        " << beta        << std::endl;
                std::cout << "gamma:       " << gamma<< std::endl;
            };
        } // namespace detail

        struct SingleSpinFlip : public Algorithm<SingleSpinFlip> {
            template<typename System>
            void update(System& system, const system::ScheduleParameter<System>& parameter) const {
                std::cout << "SingleSpinFlip::update()" << std::endl;
                detail::update_impl(system, parameter, typename system::get_system_type<System>::type());
            }

        private:
        };
    } // namespace algorithm
} // namespace openjij

#endif
