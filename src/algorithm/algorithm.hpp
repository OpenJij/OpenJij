#ifndef SYSTEM_ALGORITHM_ALGORITHM_HPP__
#define SYSTEM_ALGORITHM_ALGORITHM_HPP__

#include <iostream>
#include <tuple>
#include "../system/schedule_list.hpp"

namespace openjij {
    namespace algorithm {
        template<typename Updater>
        struct Algorithm {
            template<typename System>
            void run(System& system, const system::ScheduleList<System>& schedule_list) const {
                for (auto&& schedule : schedule_list) {
                    const auto one_mc_step = schedule.first;
                    const auto parameter = schedule.second;

                    std::cout << "one_mc_step: " << one_mc_step << std::endl;
                    for (auto i = 0; i < one_mc_step; i++) {
                        static_cast<const Updater&>(*this).update(system, parameter);
                    }
                    std::cout << std::endl;
                }
            }
        };
    } // namespace algorithm
} // namespace openjij

#endif
