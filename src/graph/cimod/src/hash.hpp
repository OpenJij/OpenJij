//    Copyright 2020 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

/**
 * @file hash.hpp
 * @author Fumiya Watanabe
 * @brief Hash function for std::pair
 * @version 1.0.0
 * @date 2020-03-13
 * 
 * @copyright Copyright (c) Jij Inc. 2020
 * 
 */

#ifndef HASH_HPP__
#define HASH_HPP__

#include <utility>
#include <cstdint>
#include <iostream>

namespace cimod
{
/**
 * @brief Hash function for std::unordered_map
 * 
 */
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        std::size_t lhs = std::hash<T1>()(p.first), rhs = std::hash<T2>()(p.second);
        return lhs^(rhs+0x9e3779b9+(lhs<<6)+(lhs>>2));
    }
};

struct vector_hash {
   template <class T>
   std::size_t operator() (const std::vector<T> &V) const {
      std::size_t hash = V.size();
      for (auto &i : V) {
         hash ^= std::hash<T>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
   }
};

}
#endif
