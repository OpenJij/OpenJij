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

#include <cstddef>
#include <type_traits>

namespace openjij {
namespace utility {

/**
 * @brief check if two template template parameters are the same.
 *
 */
template <template <typename...> class, template <typename...> class>
struct is_same_template : std::false_type {};

/**
 * @brief check if two template template parameters are the same.
 *
 */
template <template <typename...> class T>
struct is_same_template<T, T> : std::true_type {};

/**
 * @brief meta function for linear-searching base class of "derived_class" from
 * the variadic template list (head_base_class ... tail_base_class). if no base
 * class founds, returned type will be std::nullptr_t.
 *
 * @tparam derived_class
 * @tparam head_base_class
 * @tparam tail_base_classes
 */
template <typename derived_class, typename head_base_class,
          typename... tail_base_classes>
struct get_base_class {
  using type = typename std::conditional<
      std::is_base_of<head_base_class, derived_class>::value, head_base_class,
      typename get_base_class<derived_class, tail_base_classes...>::type>::type;
};

/**
 * @brief meta function for linear-searching base class of "derived_class" from
 * the variadic template list (head_base_class ... tail_base_class). if no base
 * class founds, returned type will be std::nullptr_t.
 *
 * @tparam derived_class
 * @tparam head_base_class
 * @tparam tail_base_classes
 */
template <typename derived_class, typename base_class>
struct get_base_class<derived_class, base_class> {
  using type = typename std::conditional<
      std::is_base_of<base_class, derived_class>::value, base_class,
      std::nullptr_t>::type;
};

} // namespace utility
} // namespace openjij
