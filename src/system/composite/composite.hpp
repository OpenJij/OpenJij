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
#include <functional>

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
            using elem_system = System;

            /**
             * @brief Monte Carlo Unit: system (as a reference) with sampling parameters
             */
            using MCUnit = std::pair<utility::UpdaterParameter<typename System::system_type>, std::reference_wrapper<System>>;

            /**
             * @brief constructor for composite class
             *
             * @tparam Args arguments
             * @param schedule_list
             * @param args arguments for constructor of each system
             */
            template<typename... Args>
            Composite(const utility::ScheduleList<typename System::system_type>& schedule_list, Args&&... args){
                //emplace_back
                for(auto&& _ : schedule_list){
                    _system_list.emplace_back(std::forward<Args>(args)...);
                }
                //add mcunits and register references of systems
                for(std::size_t i=0; i<schedule_list.size(); i++){
                    mcunits.emplace_back(schedule_list[i], std::ref(_system_list[i]));
                }
            }

            /**
             * @brief list of Monte Carlo Units
             */
            std::vector<MCUnit> mcunits;

            private:

            /**
             * @brief list of systems
             */
            std::vector<System> _system_list;
        };


        /**
         * @brief helper function for Composite constructor
         *
         * @tparam ScheduleList
         * @tparam Func
         * @tparam Args
         * @param schedule_list
         * @param helper_func helper function (such as make_classical_ising)
         * @param args arguments of helper_func
         *
         * @return Composite object
         */
        template<typename ScheduleList, typename Func, typename... Args>
            auto make_composite(const ScheduleList& schedule_list, const Func& helper_func, Args&&... args){
                return Composite<decltype(helper_func(std::forward<Args>(args)...))>(schedule_list, std::forward<Args>(args)...);
            }

    } // namespace system
} // namespace openjij

#endif
