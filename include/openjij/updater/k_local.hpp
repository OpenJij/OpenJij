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

#include <random>

#include "openjij/system/k_local_polynomial.hpp"
#include "openjij/utility/schedule_list.hpp"

namespace openjij {
namespace updater {

template <typename System> struct KLocal;

//! @brief K-local update for higher order unconstrained binary optimization
//! (HUBO) problems with vartype being "BINARY".
//! @tparam GraphType graph type for Polynomial graph class
template <typename GraphType>
struct KLocal<system::KLocalPolynomial<GraphType>> {

  //! @brief floating point type
  using FloatType = typename GraphType::value_type;

  //! @brief Operate k-local update for HUBO.
  //! @param system ClPIsing&. Object of a KLocalPolynomial system.
  //! @param random_number_engine RandomNumberEngine&. Eandom number engine.
  //! @param parameter const utility::ClassicalUpdaterParameter&. Parameter
  //! object including inverse temperature \f\beta:=(k_B T)^{-1}\f.
  template <typename RandomNumberEngine>
  inline static void
  update(system::KLocalPolynomial<GraphType> &system,
         RandomNumberEngine &random_number_engine,
         const utility::ClassicalUpdaterParameter &parameter) {
    auto urd = std::uniform_real_distribution<>(0, 1.0);

    int64_t count = 0;

    for (const auto &index_binary : system.get_active_binaries()) {
      const FloatType dE_s = system.dE_single(index_binary);
      if (system.count_call_updater % system.rate_call_k_local == 0 &&
          dE_s == 0.0) {
        for (const auto &index_key : system.get_adj(index_binary)) {
          if (system.GetPolyValue(index_key) >= 0.0) {
            break;
          }
          const FloatType dE_i = system.dE_k_local(index_key);
          if (dE_i <= 0.0 ||
              std::exp(-parameter.beta * dE_i) > urd(random_number_engine)) {
            system.update_system_k_local();
          } else {
            system.reset_virtual_system();
          }
        }
      } else if (dE_s <= 0.0 ||
                 std::exp(-parameter.beta * dE_s) > urd(random_number_engine)) {
        system.update_system_single(index_binary);
      }

      if (dE_s >= 0.0) {
        count++;
      }
    }

    if (count == static_cast<int64_t>(system.get_active_binaries().size()) &&
        system.count_call_updater % system.rate_call_k_local != 0) {
      for (const auto &index_binary : system.get_active_binaries()) {
        const FloatType dE_s = system.dE_single(index_binary);
        if (dE_s == 0.0 && system.binaries[index_binary] == 1) {
          system.update_system_single(index_binary);
        }
      }
      for (int64_t index_key = 0; index_key < system.GetNumInteractions();
           index_key++) {
        if (system.GetPolyValue(index_key) >= 0.0) {
          break;
        }
        if (system.GetZeroCount(index_key) != 0) {
          const FloatType dE_i = system.dE_k_local(index_key);
          if (dE_i < 0.0) {
            system.update_system_k_local();
          } else {
            system.reset_virtual_system();
          }
        }
      }
    }

    system.count_call_updater++;
  }
};

} // namespace updater
} // namespace openjij
