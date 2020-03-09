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
#include <type_traits>
#include <algorithm>
#include <system/composite/composite.hpp>
#include <system/system.hpp>
#include <utility/schedule_list.hpp>

namespace openjij {
    namespace algorithm {
        template<template<typename> class Updater>
        struct ParallelTempering {
            //TODO: add callback
            template<typename CompSystem, typename RandomNumberEngine,
                std::enable_if_t<std::is_same<typename CompSystem::elem_system::system_type, typename system::classical_system>::value> = nullptr>
            static void run(CompSystem& comp_system,
                            RandomNumberEngine& random_number_engine,
                            std::size_t num_pt_freq,
                            std::size_t num_mc) {
                // update for each system
                for(std::size_t rep=0; rep < num_mc; rep++){
                    //TODO: OpenMP or MPI?
                    for(auto& elem : comp_system.mcunits_list){
                        Updater<typename CompSystem::elem_system>::update(elem.second, random_number_engine);
                    }
                    if(rep%num_pt_freq == 0){
                        //do parallel tempering
                        //the composite system is guaranteed to be a classical system
                        //sort by parameters
                        std::sort(comp_system.mcunits_list.begin(), comp_system.mcunits_list.end(), );
                    }
                }
            }
        };

    } // namespace algorithm
} // namespace openjij

#endif
