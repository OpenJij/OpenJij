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

#include <system/composite/composite.hpp>
#include <utility/schedule_list.hpp>

namespace openjij {
    namespace updater {

        /**
         * @brief parallel tempering updater
         *
         * @tparam System type of system
         */
        template<typename System>
        struct ParallelTempering;


    } // namespace updater
} // namespace openjij

#endif
