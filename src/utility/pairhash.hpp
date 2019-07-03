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

#ifndef OPENJIJ_UTILITY_PAIRHASH_HPP__
#define OPENJIJ_UTILITY_PAIRHASH_HPP__

#include <cstddef>
#include <utility>
#include <unordered_map>

#include <graph/graph.hpp>

namespace openjij {
    namespace utility {

        /**
         * @brief hash class for unordered_map with std::pair 
         */
        struct PairHash{

            /**
             * @brief generate hash for std::pair
             *
             * @param p std::pair variable
             *
             * @return generated hash
             */
            inline size_t operator()(const std::pair<graph::Index, graph::Index> & p) const{
                const auto h1 = std::hash<graph::Index>()(p.first);
                const auto h2 = std::hash<graph::Index>()(p.second);
                return h1 ^ (h2 << 1);
            }
        };
    } // namespace utility
} // namespace openjij

#endif
