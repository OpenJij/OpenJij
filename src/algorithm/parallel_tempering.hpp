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

#ifndef SYSTEM_ALGORITHM_PARALLEL_TEMPERING_HPP__
#define SYSTEM_ALGORITHM_PARALLEL_TEMPERING_HPP__

#include <functional>
#include <system/composite/composite.hpp>
#include <system/system.hpp>
#include <utility/schedule_list.hpp>

namespace openjij {
    namespace algorithm {
        template<template<typename> class Updater>
        struct ParallelTempering {
            //TODO: add callback
            template<typename System, typename RandomNumberEngine>
            static void run(System& comp_system,
                            RandomNumberEngine& random_number_engine,
                            std::size_t num_pt_freq,
                            const std::function<void(const System&, const utility::UpdaterParameter<typename system::get_system_type<System>::type>&)>& callback = nullptr) {
                // update for each system
                for(auto& elem : comp_system.mcunits_list){
                    Updater<typename System::inside_system>::update(elem.second, random_number_engine);
                }
            }
        };

        //type alias (Monte Carlo method)
        //TODO: Algorithm class will be deprecated shortly.
        template<template<typename> class Updater>
        using MCMC = Algorithm<Updater>;

    } // namespace algorithm
} // namespace openjij

#endif
