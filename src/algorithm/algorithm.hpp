//    Copyright 2019 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef SYSTEM_ALGORITHM_ALGORITHM_HPP__
#define SYSTEM_ALGORITHM_ALGORITHM_HPP__

#include <system/system.hpp>
#include <utility/schedule_list.hpp>

namespace openjij {
    namespace algorithm {
        template<template<typename> class Updater>
        struct Algorithm {
            template<typename System, typename RandomNumberEngine>
            static void run(System& system,
                            RandomNumberEngine& random_number_engine,
                            const utility::ScheduleList<typename system::get_system_type<System>::type>& schedule_list) {
                for (auto&& schedule : schedule_list) {
                    for (std::size_t i = 0; i < schedule.one_mc_step; ++i) {
                        Updater<System>::update(system, random_number_engine, schedule.updater_parameter);
                    }
                }
            }
        };

        //alias (Monte Carlo method)
        //TODO: Algorithm class will be deprecated someday.
        template<template<typename> class Updater>
        using MCMC = Algorithm<Updater>;

    } // namespace algorithm
} // namespace openjij

#endif
