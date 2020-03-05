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

#ifndef OPENJIJ_SYSTEM_COMPOSITE_CALC_MC_ENERGY_HPP__
#define OPENJIJ_SYSTEM_COMPOSITE_CALC_MC_ENERGY_HPP__

#include <system/all.hpp>

namespace openjij {
    namespace system {

        /**
         * @brief calc mc energy of classical ising system (no Eigen implementation)
         *
         * @tparam GraphType graph type
         * @param system classical ising system without Eigen implementation
         *
         * @return value of energy
         */
        template<typename GraphType>
        double calc_mc_energy(const system::ClassicalIsing<GraphType, false>& system, const utility::UpdaterParameter<system::classical_system>& p){
            return p.beta*system.interaction.calc_energy(system.spin);
        }

        /**
         * @brief calc mc energy of classical ising system (with Eigen implementation)
         *
         * @tparam GraphType graph type
         * @param system classical ising system with Eigen implementation
         *
         * @return value of energy
         */
        template<typename GraphType>
        double calc_mc_energy(const system::ClassicalIsing<GraphType, true>& system, const utility::UpdaterParameter<system::classical_system>& p){
            //matrix calculation
            return p.beta*(1.0/2)*system.spin.dot(system.interaction*system.spin);
        }

    } // namespace system
} // namespace openjij

#endif
