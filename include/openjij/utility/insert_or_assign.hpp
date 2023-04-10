//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include <unordered_map>

namespace openjij {
namespace utility {

/**
 * @brief Insert or assign a element of unordered_map (for C++14 or C++11)
 *
 * @tparam C_key
 * @tparam C_value
 * @tparam Hash
 * @param um
 * @param key
 * @param val
 *
 */
template <class C_key, class C_value, class Hash>
inline void insert_or_assign(std::unordered_map<C_key, C_value, Hash> &um,
                             const C_key &key, const C_value &val) {
  // insert
  if (um.count(key) == 0) {
    um.insert({{key, val}});
  }
  // assign
  else {
    um[key] = val;
  }
}
} // namespace utility
} // namespace openjij
