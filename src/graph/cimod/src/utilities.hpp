//    Copyright 2021 Jij Inc.

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
 * @file utilities.hpp
 * @author Fumiya Watanabe
 * @brief Utilities for cpp_dimod
 * @version 1.0.0
 * @date 2020-03-24
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef UTILITIES_HPP__
#define UTILITIES_HPP__

#include <unordered_map>

namespace cimod
{
/**
 * @brief Insert or assign a element of unordered_map (for C++14 or C++11)
 * 
 * @tparam C_key 
 * @tparam C_value 
 * @param um 
 * @param key 
 * @param val 
 */

template <class C_key, class C_value, class Hash>
void insert_or_assign
(
    std::unordered_map<C_key, C_value, Hash> &um,
    const C_key &key,
    const C_value &val
)
{
    // insert
    if(um.count(key)==0)
    {
        um.insert({{key, val}});
    }
    // assign
    else
    {
        um[key] = val;
    }
}
}

#endif