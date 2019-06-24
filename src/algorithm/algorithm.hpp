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

#include <iostream>
#include <tuple>
#include <vector>

#include <utility/schedule_list.hpp>

namespace openjij {
    namespace algorithm {
        template<template<typename> class Updater>
        struct Algorithm {
            template<typename System>
            static void run(System& system, const utility::ScheduleList& schedule_list) {
                for (auto&& schedule : schedule_list) {
                    const auto one_mc_step = schedule.first;
                    const auto parameters = schedule.second;

                    for (auto i = 0; i < one_mc_step; ++i) {
                    // std::cout << "one_mc_step: " << one_mc_step << std::endl;
                        Updater<System>::update(system, parameters);
                    }
                    // std::cout << std::endl;
                }
            }
        };
    } // namespace algorithm
} // namespace openjij

#endif
