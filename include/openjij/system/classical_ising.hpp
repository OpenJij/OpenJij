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

#include <cassert>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cimod/utilities.hpp>

#include "openjij/graph/all.hpp"
#include "openjij/system/system.hpp"
#include "openjij/utility/eigen.hpp"

namespace openjij {
namespace system {

/**
 * @brief ClassicalIsing structure (system for classical Ising model)
 *
 * @tparam GraphType type of graph
 */
template <typename GraphType> struct ClassicalIsing;

/**
 * @brief ClassicalIsing structure for Dense graph (Eigen-based)
 *
 * @tparam FloatType type of floating-point
 */
template <typename FloatType> struct ClassicalIsing<graph::Dense<FloatType>> {
  using system_type = classical_system;

  // matrix (row major)
  using MatrixXx =
      Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // vector (col major)
  using VectorXx = Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>;

  /**
   * @brief Constructor to initialize spin and interaction
   *
   * @param spin
   * @param interaction
   */
  ClassicalIsing(const graph::Spins &init_spin,
                 const graph::Dense<FloatType> &init_interaction)
      : spin(utility::gen_vector_from_std_vector<FloatType, Eigen::ColMajor>(
            init_spin)),
        interaction(init_interaction.get_interactions()),
        num_spins(init_interaction.get_num_spins()) {
    cimod::CheckVariables(init_spin, cimod::Vartype::SPIN);
    assert(init_spin.size() == init_interaction.get_num_spins());
    reset_dE();
  }

  /**
   * @brief reset spins
   *
   * @param init_spin
   */
  void reset_spins(const graph::Spins &init_spin) {
    this->spin =
        utility::gen_vector_from_std_vector<FloatType, Eigen::ColMajor>(
            init_spin);
    reset_dE();
  }

  /**
   * @brief reset dE
   *
   */
  inline void reset_dE() {
    this->dE =
        -2.0 * this->spin.array() * (this->interaction * this->spin).array();
  }

  /**
   * @brief spins (Eigen Vector)
   */
  VectorXx spin;

  /**
   * @brief interactions (Eigen Matrix)
   */
  const MatrixXx interaction;

  /**
   * @brief number of real spins (dummy spin excluded)
   */
  const std::size_t num_spins; // spin.size()-1

  /**
   * @brief delta E for updater
   */
  VectorXx dE;
};

/**
 * @brief ClassicalIsing structure for Sparse graph (Eigen-based)
 *
 * @tparam FloatType type of floating-point
 */
template <typename FloatType> struct ClassicalIsing<graph::Sparse<FloatType>> {
  using system_type = classical_system;

  // matrix (row major)
  using SparseMatrixXx = Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;
  // vector (col major)
  using VectorXx = Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>;

  /**
   * @brief Constructor to initialize spin and interaction
   *
   * @param spin
   * @param interaction
   */
  ClassicalIsing(const graph::Spins &init_spin,
                 const graph::Sparse<FloatType> &init_interaction)
      : spin(utility::gen_vector_from_std_vector<FloatType, Eigen::ColMajor>(
            init_spin)),
        interaction(
            utility::gen_matrix_from_graph<Eigen::RowMajor>(init_interaction)),
        num_spins(init_interaction.get_num_spins()) {
    assert(init_spin.size() == init_interaction.get_num_spins());
    reset_dE();
  }

  /**
   * @brief reset spins
   *
   * @param init_spin
   */
  void reset_spins(const graph::Spins &init_spin) {
    this->spin =
        utility::gen_vector_from_std_vector<FloatType, Eigen::ColMajor>(
            init_spin);
    reset_dE();
  }

  /**
   * @brief reset dE
   *
   */
  void reset_dE() {
    this->dE =
        -2.0 * this->spin.array() * (this->interaction * this->spin).array();
  }

  /**
   * @brief spins (Eigen Vector)
   */
  VectorXx spin;

  /**
   * @brief interaction (Eigen SparseMatrix)
   */
  const SparseMatrixXx interaction;

  /**
   * @brief number of real spins (dummy spin excluded)
   */
  const std::size_t num_spins; // spin.size()-1

  /**
   * @brief delta E for updater
   */
  VectorXx dE;
};

/**
 * @brief ClassicalIsing structure for CSR Sparse graph (Eigen-based)
 *
 * @tparam FloatType type of floating-point
 */
template <typename FloatType> struct ClassicalIsing<graph::CSRSparse<FloatType>> {
  using system_type = classical_system;

  // matrix (row major)
  using SparseMatrixXx = Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;
  // vector (col major)
  using VectorXx = Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>;

  /**
   * @brief Constructor to initialize spin and interaction
   *
   * @param spin
   * @param interaction
   */
  ClassicalIsing(const graph::Spins &init_spin,
                 const graph::CSRSparse<FloatType> &init_interaction)
      : spin(utility::gen_vector_from_std_vector<FloatType, Eigen::ColMajor>(
            init_spin)),
        interaction(init_interaction.get_interactions()),
        num_spins(init_interaction.get_num_spins()) {
    assert(init_spin.size() == init_interaction.get_num_spins());
    reset_dE();
  }

  /**
   * @brief reset spins
   *
   * @param init_spin
   */
  void reset_spins(const graph::Spins &init_spin) {
    this->spin =
        utility::gen_vector_from_std_vector<FloatType, Eigen::ColMajor>(
            init_spin);
    reset_dE();
  }

  /**
   * @brief reset dE
   *
   */
  void reset_dE() {
    this->dE =
        -2.0 * this->spin.array() * (this->interaction * this->spin).array();
  }

  /**
   * @brief spins (Eigen Vector)
   */
  VectorXx spin;

  /**
   * @brief interaction (Eigen SparseMatrix)
   */
  const SparseMatrixXx interaction;

  /**
   * @brief number of real spins (dummy spin excluded)
   */
  const std::size_t num_spins; // spin.size()-1

  /**
   * @brief delta E for updater
   */
  VectorXx dE;
};

/**
 * @brief helper function for ClassicalIsing constructor
 *
 * @tparam GraphType
 * @param init_spin initial spin
 * @param init_interaction initial interaction
 *
 * @return generated object
 */
template <typename GraphType>
auto make_classical_ising(const graph::Spins &init_spin,
                          const GraphType &init_interaction) {
  return ClassicalIsing<GraphType>(init_spin, init_interaction);
}

} // namespace system
} // namespace openjij
