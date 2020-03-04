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

#ifndef OPENJIJ_SYSTEM_COMPOSITE_HPP__
#define OPENJIJ_SYSTEM_COMPOSITE_HPP__

#include <system/all.hpp>
#include <utility/schedule_list.hpp>
#include <system/system.hpp>
#include <set>
#include <utility>

namespace openjij {
    namespace system {


        /**
         * @brief composite system
         *
         * @tparam System
         */
        template<typename System>
        struct Composite{
            using system_type = system::composite_mc_system;

            /**
             * @brief Monte Carlo Unit: system (as a reference) with sampling parameters
             */
            using MCUnit = std::pair<utility::UpdaterParameter<typename System::system_type>, std::reference_wrapper<System>>;

        };

    } // namespace system
} // namespace openjij

#endif
