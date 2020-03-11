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

#ifndef OPENJIJ_UPDATER_PARALLEL_TEMPERING_HPP__
#define OPENJIJ_UPDATER_PARALLEL_TEMPERING_HPP__

#include <random>
#include <type_traits>
#include <algorithm>
#include <system/composite/composite.hpp>
#include <system/composite/calc_mc_energy.hpp>

namespace openjij {
    namespace updater {

        /**
         * @brief parallel tempering updater
         *
         * @tparam CompSystem type of composite system
         */
        template<typename CompSystem>
        struct ParallelTempering;

        /**
         * @brief parallel tempering updater for composite system
         *
         * @tparam System system in the composite system
         */
        template<typename System>
        struct ParallelTempering<system::Composite<System>> {

            using Comp = system::Composite<System>;
            
            /**
             * @brief operate parallel tempering (classical system)
             *
             * @param system object of a classical ising system
             * @param random_number_engine random number gengine
             *
             * @return energy difference \f\Delta E\f
             */
          template<typename RandomNumberEngine,
              std::enable_if_t<std::is_same<typename Comp::elem_system::system_type, typename system::classical_system>::value> = nullptr>
            inline static void update(Comp& system, RandomNumberEngine& random_number_engine) {

                auto urd = std::uniform_real_distribution<>(0, 1.0);

                // sort mcunits (key = param.beta, descending order)
                // TODO: add is_sorted option
                std::sort(system.mcunits.begin(), system.mcunits.end(), 
                        [](const typename Comp::MCUnit& lhs, const typename Comp::MCUnit& rhs){return rhs.first.beta < lhs.first.beta;});

                //replica exchange process
                //TODO: OpenMP?
                for(std::size_t i=0; i<system.mcunits.size()-1; i+=2){
                    double dE = system::calc_mc_energy(system.mcunits[i+1].second, system.mcunits[i].first) + system::calc_mc_energy(system.mcunits[i].second, system.mcunits[i+1].first)
                              -(system::calc_mc_energy(system.mcunits[i].second, system.mcunits[i].first) + system::calc_mc_energy(system.mcunits[i+1].second, system.mcunits[i+1].first));

                    if(dE < 0 || std::exp(-dE) > urd(random_number_engine)){
                        std::swap(system.mcunits[i].second, system.mcunit[i+1].second);
                    }
                }
                for(std::size_t i=1; i<system.mcunits.size()-1; i+=2){
                    double dE = system::calc_mc_energy(system.mcunits[i+1].second, system.mcunits[i].first) + system::calc_mc_energy(system.mcunits[i].second, system.mcunits[i+1].first)
                              -(system::calc_mc_energy(system.mcunits[i].second, system.mcunits[i].first) + system::calc_mc_energy(system.mcunits[i+1].second, system.mcunits[i+1].first));

                    if(dE < 0 || std::exp(-dE) > urd(random_number_engine)){
                        std::swap(system.mcunits[i].second, system.mcunit[i+1].second);
                    }
                }

            }
        };
        

    } // namespace updater
} // namespace openjij

#endif
