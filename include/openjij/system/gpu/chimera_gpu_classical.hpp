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

#ifndef OPENJIJ_SYSTEM_CHIMERA_GPU_CLASSICAL_HPP__
#define OPENJIJ_SYSTEM_CHIMERA_GPU_CLASSICAL_HPP__

#ifdef USE_CUDA

#include <cstddef>

#include "openjij/system/gpu/chimera_gpu_transverse.hpp"
#include "openjij/system/system.hpp"

namespace openjij {
namespace system {

/**
 * @brief Chimera Classical Ising structure with cuda
 *
 * @tparam FloatType
 * @tparam rows_per_block
 * @tparam cols_per_block
 */
template <typename FloatType, std::size_t rows_per_block = 2,
          std::size_t cols_per_block = 2>
struct ChimeraClassicalGPU
    : public ChimeraTransverseGPU<FloatType, rows_per_block, cols_per_block,
                                  1> {
  using system_type = classical_system;
  using Base =
      ChimeraTransverseGPU<FloatType, rows_per_block, cols_per_block, 1>;

  /**
   * @brief Chimera classical ising constructor
   *
   * @param init_spin
   * @param init_interaction
   * @param device_num
   */
  ChimeraClassicalGPU(const graph::Spins &init_spin,
                      const graph::Chimera<FloatType> &init_interaction,
                      int device_num = 0)
      : Base(init_spin, init_interaction, 1.0, 1, device_num) {}

  /**
   * @brief reset spins
   *
   * @param init__spin
   */
  void reset_spins(const graph::Spins &init_spin) {
    Base::reset_spins(init_spin);
  }
};

/**
 * @brief helper function for Chimera ClassicalIsing constructor
 *
 * @tparam rows_per_block
 * @tparam cols_per_block
 * @tparam FloatType
 * @param init__spin
 * @param init_interaction
 * @param device_num
 *
 * @return
 */
template <std::size_t rows_per_block = 2, std::size_t cols_per_block = 2,
          typename FloatType>
ChimeraClassicalGPU<FloatType, rows_per_block, cols_per_block>
make_chimera_classical_gpu(const graph::Spins &init_spin,
                           const graph::Chimera<FloatType> &init_interaction,
                           int device_num = 0) {
  return ChimeraClassicalGPU<FloatType, rows_per_block, cols_per_block>(
      init_spin, init_interaction, device_num);
}

} // namespace system
} // namespace openjij

#endif
#endif
