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

#ifndef OPENJIJ_PRO_SYSTEM_COMPOSITE_MCUNIT_CONTAINER_HPP__
#define OPENJIJ_PRO_SYSTEM_COMPOSITE_MCUNIT_CONTAINER_HPP__

#include <set>
#include <utility>
#include <functional>
#include <utility/schedule_list.hpp>
#include <system/system.hpp>

namespace openjij {
    namespace system {

        /**
         * @brief Monte Carlo Unit: system (as a reference) with sampling parameters
         *
         * @tparam System
         */
        template<typename System>
            using MCUnit = std::pair<utility::UpdaterParameter<typename System::system_type>, std::reference_wrapper<System>>;


        /**
         * @brief Monte Carlo Unit container
         *
         * @tparam system_type
         */
        template<typename system_type>
            class MCUnitContainer{};

    } // namespace result
} // namespace openjij

#endif
