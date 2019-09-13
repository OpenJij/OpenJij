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

#include <system/system.hpp>
#include <system/all.hpp>
#include <unordered_map>

namespace openjij {
    namespace system {

        /**
         * @brief composite monte carlo system (used for parallel tempering, houdayer update...)
         *
         * @tparam System
         */
        template<typename System>
            using Composite = std::unordered_map


    } // namespace system
} // namespace openjij

#endif
