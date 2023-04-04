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

#include "openjij/utility/type_traits.hpp"

namespace openjij {
namespace system {

/**
 * @brief monte carlo system tag
 */
struct mc_system {};

/**
 * @brief classical monte carlo system (using beta (inverse temperature) for
 * annealing parameter)
 */
struct classical_system : public mc_system {};

/**
 * @brief classical monte carlo system with a constraint (using beta (inverse
 * temperature) for annealing parameter)
 */
struct classical_constraint_system : public mc_system {};

/**
 * @brief quantum monte carlo system
 */
struct quantum_system : public mc_system {};

/**
 * @brief ordinary homogeneous transverse field system (using beta and annealing
 * schedule (s) for annealing parameter)
 */
struct transverse_field_system : public quantum_system {};

/**
 * @brief inhomogeneous transverse field system
 */
struct inhomogeneous_transverse_field_system : public quantum_system {};

/**
 * @brief nonstoquastic quantum system
 */
struct nonstoquastic_system : public quantum_system {};

/**
 * @brief system tag with single command execution (e.g. D-Wave, digital
 * annealer, ...)
 */
struct single_command_system {};

/**
 * @brief system tag for solving real time schrodinger dynamics
 */
struct realtime_dynamics_system {};

/**
 * @brief meta function for getting system type
 *
 * @tparam System system
 */
template <typename System> struct get_system_type {
  using type = typename System::system_type;
};

/**
 * @brief meta function to check whether system type is derived from
 * classical_system or quantum_system. If system type is derived from neither
 * classical_system nor quantum_system, returned type will be std::nullptr_t.
 *
 * @tparam System system
 */
template <typename System> struct get_classical_or_quantum_system_type {
  using type =
      typename utility::get_base_class<typename System::system_type,
                                       classical_system, quantum_system>::type;
};

/**
 * @brief meta function to check whether system type is derived from mc_system,
 * single_command_system, or real_time_dynamics_system. If system type is derive
 * from none of these systems, returned type will be std::nullptr_t.
 *
 * @tparam System system
 */
template <typename System> struct get_base_system_type {
  using type =
      typename utility::get_base_class<typename System::system_type, mc_system,
                                       single_command_system,
                                       realtime_dynamics_system>::type;
};

/**
 * @brief USE_EIGEN flag
 */
constexpr static bool USE_EIGEN = true;

} // namespace system
} // namespace openjij
