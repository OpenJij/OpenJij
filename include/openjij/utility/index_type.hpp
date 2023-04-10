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

#include <variant>
#include <string>
#include <vector>

namespace openjij {
namespace utility {

//! @brief Vector type of integer and string.
using AnyTupleType = std::vector<std::variant<std::int32_t, std::string>>;

//! @brief The index type of binary variables.
using IndexType = std::variant<std::int32_t, std::string, AnyTupleType>;


} // namespace utility
} // namespace openjij
