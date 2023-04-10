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

#include <algorithm>
#include <cmath>
#include <limits>

#include "openjij/graph/all.hpp"
#include "openjij/system/all.hpp"

#ifdef USE_CUDA
#include "openjij/utility/gpu/memory.hpp"
#endif

namespace openjij {
namespace result {

/**
 * @brief get solution of classical ising system
 *
 * @tparam GraphType graph type
 * @param system classical ising system with Eigen implementation
 *
 * @return solution
 */
template <typename GraphType>
const graph::Spins
get_solution(const system::ClassicalIsing<GraphType> &system) {
  // convert from Eigen::Vector to std::vector
  graph::Spins ret_spins(system.num_spins);
  for (std::size_t i = 0; i < system.num_spins; i++) {
    ret_spins[i] = static_cast<graph::Spin>(system.spin(i) *
                                            system.spin(system.num_spins));
  }
  return ret_spins;
}

/**
 * @brief get solution of transverse ising system
 *
 * @tparam GraphType
 * @param system
 *
 * @return solution
 */
template <typename GraphType>
const graph::Spins
get_solution(const system::TransverseIsing<GraphType> &system) {
  std::size_t minimum_trotter = 0;
  // aliases
  auto &spins = system.trotter_spins;
  double energy = 0.0;
  double min_energy = std::numeric_limits<double>::max();
  // get number of trotter slices
  std::size_t num_trotter_slices = system.trotter_spins.cols();
  for (std::size_t t = 0; t < num_trotter_slices; t++) {
    // calculate classical energy in each classical spin
    energy = spins.col(t).transpose() * system.interaction * spins.col(t);
    if (energy < min_energy) {
      minimum_trotter = t;
      min_energy = energy;
    }
  }

  // convert from Eigen::Vector to std::vector
  graph::Spins ret_spins(system.num_classical_spins);
  for (std::size_t i = 0; i < system.num_classical_spins; i++) {
    ret_spins[i] = static_cast<graph::Spin>(spins(i, minimum_trotter));
  }
  return ret_spins;
}

template <typename GraphType>
const graph::Spins
get_solution(const system::ClassicalIsingPolynomial<GraphType> &system) {
  return system.variables;
}

template <typename GraphType>
const graph::Spins
get_solution(const system::KLocalPolynomial<GraphType> &system) {
  return system.binaries;
}

/**
 * @brief get solution of continuous time Ising system
 *
 * @tparam GraphType
 * @param system
 *
 * @return solution
 */
template <typename GraphType>
graph::Spins
get_solution(const system::ContinuousTimeIsing<GraphType> &system) {
  auto spins = system.get_slice_at(0.0);

  if (system.get_auxiliary_spin(0.0) < 0) {
    /*if auxiliary spin is negative, flip all spins*/
    for (auto &spin : spins) {
      spin *= -1;
    }
  }

  return spins;
}

#ifdef USE_CUDA

/**
 * @brief get solution of chimera transverse gpu system
 *
 * @tparam FloatType
 * @tparam rows_per_block
 * @tparam cols_per_block
 * @tparam trotters_per_block
 * @param system
 *
 * @return solution
 */
template <typename FloatType, std::size_t rows_per_block,
          std::size_t cols_per_block, std::size_t trotters_per_block>
const graph::Spins
get_solution(const system::ChimeraTransverseGPU<FloatType, rows_per_block,
                                                cols_per_block,
                                                trotters_per_block> &system) {

  std::size_t localsize =
      system.info.rows * system.info.cols * system.info.chimera_unitsize;

  graph::Spins ret_spins(localsize);

  size_t select_t = system.info.trotters / 2;
  HANDLE_ERROR_CUDA(
      cudaMemcpy(ret_spins.data(), system.spin.get() + (localsize * select_t),
                 localsize * sizeof(int), cudaMemcpyDeviceToHost));

  return ret_spins;
}

/**
 * @brief get solution of chimera classical gpu system
 *
 * @tparam FloatType
 * @tparam rows_per_block
 * @tparam cols_per_block
 * @param system
 *
 * @return solution
 */
template <typename FloatType, std::size_t rows_per_block,
          std::size_t cols_per_block>
const graph::Spins
get_solution(const system::ChimeraClassicalGPU<FloatType, rows_per_block,
                                               cols_per_block> &system) {
  using Base = typename system::ChimeraClassicalGPU<FloatType, rows_per_block,
                                                    cols_per_block>::Base;

  return get_solution(static_cast<const Base &>(system));
}
#endif

} // namespace result
} // namespace openjij
