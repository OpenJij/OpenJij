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
#include <exception>
#include <vector>

#include "openjij/graph/all.hpp"
#include "openjij/system/system.hpp"
#include "openjij/utility/eigen.hpp"

namespace openjij {
namespace system {

/**
 * @brief trotterized spin (std::vector<Spins>)
 * trotter_spins[i][j] -> jth spin in ith trotter slice.
 */
using TrotterSpins = std::vector<graph::Spins>;

/**
 * @brief TransverseIsing structure with discrete-time trotter spins
 *
 * @tparam GraphType
 */
template <typename GraphType> struct TransverseIsing;

/**
 * @brief naive Dense TransverseIsing structure with discrete-time trotter spins
 * (with Eigen implementation) The original Hamiltonian is like \f[ H(s) =
 * s\left(\sum_{i<j}J_{ij} \sigma_i^z \sigma_j^z + \sum_{i}h_{i} \sigma_i^z
 * \right) + \Gamma (1-s)\sum_i \sigma_i^x, \f] where \f$ \Gamma \f$ shows the
 * strength of transverse field (referred to as `gamma`) and \f$ s \f$ shows the
 * annealing schedule (referred to as `s`).
 *
 * Quantum Monte Carlo simulation is performed under the finite (inverse)
 * temperature (\f$ \beta \f$ or `beta`).
 *
 * @tparam FloatType
 */
template <typename FloatType> struct TransverseIsing<graph::Dense<FloatType>> {
  using system_type = transverse_field_system;

  // matrix (row major)
  using MatrixXx =
      Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // trotter matrix (col major)
  using TrotterMatrix =
      Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  /**
   * @brief TransverseIsing Constructor
   *
   * @param init_trotter_spins
   * @param init_interaction
   */
  TransverseIsing(const TrotterSpins &init_trotter_spins,
                  const graph::Dense<FloatType> &init_interaction,
                  const FloatType gamma)
      : trotter_spins(
            utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
                init_trotter_spins)),
        interaction(init_interaction.get_interactions()),
        num_classical_spins(init_trotter_spins[0].size()), gamma(gamma) {
    if (!(init_trotter_spins.size() >= 2)) {
      throw std::invalid_argument(
          "trotter slices must be equal or larger than 2.");
    }

    // initialize rand_pool
    rand_pool = TrotterMatrix(num_classical_spins + 1, trotter_spins.cols());

    // reset dE
    reset_dE();
  }

  /**
   * @brief TransverseIsing Constuctor with initial classical spins
   *
   * @param classical_spins initial classical spins
   * @param init_interaction
   * @param num_trotter_slices
   */
  TransverseIsing(const graph::Spins &init_classical_spins,
                  const graph::Dense<FloatType> &init_interaction,
                  const FloatType gamma, const size_t num_trotter_slices)
      : interaction(init_interaction.get_interactions()),
        num_classical_spins(init_classical_spins.size()), gamma(gamma) {
    // initialize trotter_spins with classical_spins

    if (!(num_trotter_slices >= 2)) {
      throw std::invalid_argument(
          "trotter slices must be equal or larger than 2.");
    }

    TrotterSpins init_trotter_spins(num_trotter_slices);

    for (auto &spins : init_trotter_spins) {
      spins = init_classical_spins;
    }

    // init trotter_spins
    trotter_spins =
        utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
            init_trotter_spins);

    // initialize rand_pool
    rand_pool = TrotterMatrix(num_classical_spins + 1, trotter_spins.cols());

    // reset dE
    reset_dE();
  }

  /**
   * @brief reset spins with trotter spins
   *
   * @param init_trotter_spins
   */
  void reset_spins(const TrotterSpins &init_trotter_spins) {
    this->trotter_spins =
        utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
            init_trotter_spins);

    // reset dE
    reset_dE();
  }

  /**
   * @brief reset spins with trotter spins
   *
   * @param classical_spins
   */
  void reset_spins(const graph::Spins &classical_spins) {
    TrotterSpins init_trotter_spins(
        trotter_spins.cols()); // cols -> num_trotter_slices

    for (auto &spins : init_trotter_spins) {
      spins = classical_spins;
    }
    // init trotter_spins
    this->trotter_spins =
        utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
            init_trotter_spins);

    // reset dE
    reset_dE();
  }

  /**
   * @brief reset dE
   */
  inline void reset_dE() {
    // reset dE and dEtrot

    // initialize dE (spatial direction)
    // we store only dE for spatial direction
    const auto &spins = this->trotter_spins;
    // std::size_t num_trotter_slices = this->trotter_spins.cols();

    this->dE = -2 * spins.cwiseProduct(this->interaction * spins);

    // for trotter direction
    // this->dEtrot = TrotterMatrix::Zero(num_classical_spins+1,
    // num_trotter_slices); for(std::size_t t=0; t<num_trotter_slices; t++){
    //    this->dEtrot.col(t) = -2 * spins.col(t).cwiseProduct(
    //            spins.col(mod_t((int64_t)t+1, num_trotter_slices)) +
    //            spins.col(mod_t((int64_t)t-1, num_trotter_slices))
    //            );
    //}

    // diffE = s * (beta/num_trotter_slices) * this->dE + (1/2.) *
    // log(tanh(beta* gamma * (1.0-s) /num_trotter_slices)) * this->dEtrot
  }

  inline static std::size_t mod_t(std::int64_t a,
                                  std::size_t num_trotter_slices) {
    // a -> [-1:num_trotter_slices]
    // return a%num_trotter_slices (a>0), num_trotter_slices-1 (a==-1)
    return (a + num_trotter_slices) % num_trotter_slices;
  }

  /**
   * @brief trotterlized spins
   */
  TrotterMatrix trotter_spins;

  /**
   * @brief interaction
   */
  const MatrixXx interaction;

  /**
   * @brief number of real classical spins (dummy spin excluded)
   */
  std::size_t num_classical_spins; // trotter_spins[0].size()-1

  /**
   * @brief coefficient of transverse field term
   */
  FloatType gamma;

  /**
   * @brief rand_pool
   */
  TrotterMatrix rand_pool;

  /**
   * @brief trotterlized dE (spatial direction)
   */
  TrotterMatrix dE;
};

/**
 * @brief naive Sparse TransverseIsing structure with discrete-time trotter
 * spins (with Eigen implementation) The original Hamiltonian is like \f[ H(s) =
 * s\left(\sum_{i<j}J_{ij} \sigma_i^z \sigma_j^z + \sum_{i}h_{i} \sigma_i^z
 * \right) + \Gamma (1-s)\sum_i \sigma_i^x, \f] where \f$ \Gamma \f$ shows the
 * strength of transverse field (referred to as `gamma`) and \f$ s \f$ shows the
 * annealing schedule (referred to as `s`).
 *
 * Quantum Monte Carlo simulation is performed under the finite (inverse)
 * temperature (\f$ \beta \f$ or `beta`).
 *
 * @tparam FloatType
 */
template <typename FloatType> struct TransverseIsing<graph::Sparse<FloatType>> {
  using system_type = transverse_field_system;

  // matrix (row major)
  using SparseMatrixXx = Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;
  // trotter matrix (col major)
  using TrotterMatrix =
      Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  /**
   * @brief TransverseIsing Constructor
   *
   * @param init_trotter_spins
   * @param init_interaction
   */
  TransverseIsing(const TrotterSpins &init_trotter_spins,
                  const graph::Sparse<FloatType> &init_interaction,
                  FloatType gamma)
      : trotter_spins(
            utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
                init_trotter_spins)),
        interaction(
            utility::gen_matrix_from_graph<Eigen::RowMajor>(init_interaction)),
        num_classical_spins(init_trotter_spins[0].size()), gamma(gamma) {
    if (!(init_trotter_spins.size() >= 2)) {
      throw std::invalid_argument(
          "trotter slices must be equal or larger than 2.");
    }

    // initialize rand_pool
    rand_pool = TrotterMatrix(num_classical_spins + 1, trotter_spins.cols());

    // reset dE
    reset_dE();
  }

  /**
   * @brief TransverseIsing Constuctor with initial classical spins
   *
   * @param classical_spins initial classical spins
   * @param init_interaction
   * @param num_trotter_slices
   */
  TransverseIsing(const graph::Spins &init_classical_spins,
                  const graph::Sparse<FloatType> &init_interaction,
                  const FloatType gamma, const size_t num_trotter_slices)
      : interaction(
            utility::gen_matrix_from_graph<Eigen::RowMajor>(init_interaction)),
        num_classical_spins(init_classical_spins.size()), gamma(gamma) {
    // initialize trotter_spins with classical_spins

    // assert(init_classical_spins.size() >= 2);
    if (!(num_trotter_slices >= 2)) {
      throw std::invalid_argument(
          "trotter slices must be equal or larger than 2.");
    }

    TrotterSpins init_trotter_spins(num_trotter_slices);

    for (auto &spins : init_trotter_spins) {
      spins = init_classical_spins;
    }

    // init trotter_spins
    trotter_spins =
        utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
            init_trotter_spins);

    // initialize rand_pool
    rand_pool = TrotterMatrix(num_classical_spins + 1, trotter_spins.cols());

    // reset dE
    reset_dE();
  }

  /**
   * @brief reset spins with trotter spins
   *
   * @param init_trotter_spins
   */
  void reset_spins(const TrotterSpins &init_trotter_spins) {
    this->trotter_spins =
        utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
            init_trotter_spins);

    // reset dE
    reset_dE();
  }

  /**
   * @brief reset spins with trotter spins
   *
   * @param classical_spins
   */
  void reset_spins(const graph::Spins &classical_spins) {
    TrotterSpins init_trotter_spins(
        trotter_spins.cols()); // cols -> num_trotter_slices

    for (auto &spins : init_trotter_spins) {
      spins = classical_spins;
    }
    // init trotter_spins
    this->trotter_spins =
        utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
            init_trotter_spins);

    // reset dE
    reset_dE();
  }

  /**
   * @brief reset dE
   */
  inline void reset_dE() {
    // reset dE and dEtrot

    // initialize dE (spatial direction)
    // we store only dE for spatial direction
    const auto &spins = this->trotter_spins;
    // std::size_t num_trotter_slices = this->trotter_spins.cols();

    this->dE = -2 * spins.cwiseProduct(this->interaction * spins);

    // for trotter direction
    // this->dEtrot = TrotterMatrix::Zero(num_classical_spins+1,
    // num_trotter_slices); for(std::size_t t=0; t<num_trotter_slices; t++){
    //    this->dEtrot.col(t) = -2 * spins.col(t).cwiseProduct(
    //            spins.col(mod_t((int64_t)t+1, num_trotter_slices)) +
    //            spins.col(mod_t((int64_t)t-1, num_trotter_slices))
    //            );
    //}

    // diffE = s * (beta/num_trotter_slices) * this->dE + (1/2.) *
    // log(tanh(beta* gamma * (1.0-s) /num_trotter_slices)) * this->dEtrot
  }

  inline static std::size_t mod_t(const std::int64_t a,
                                  const std::size_t num_trotter_slices) {
    // a -> [-1:num_trotter_slices]
    // return a%num_trotter_slices (a>0), num_trotter_slices-1 (a==-1)
    return (a + num_trotter_slices) % num_trotter_slices;
  }

  /**
   * @brief trotterlized spins
   */
  TrotterMatrix trotter_spins;

  /**
   * @brief interaction
   */
  const SparseMatrixXx interaction;

  /**
   * @brief number of real classical spins (dummy spin excluded)
   */
  std::size_t num_classical_spins; // trotter_spins[0].size()-1

  /**
   * @brief coefficient of transverse field term
   */
  FloatType gamma;

  /**
   * @brief rand_pool
   */
  TrotterMatrix rand_pool;

  /**
   * @brief trotterlized dE (spatial direction)
   */
  TrotterMatrix dE;
};

/**
 * @brief CSR Sparse TransverseIsing structure with discrete-time trotter
 * spins (with Eigen implementation) The original Hamiltonian is like \f[ H(s) =
 * s\left(\sum_{i<j}J_{ij} \sigma_i^z \sigma_j^z + \sum_{i}h_{i} \sigma_i^z
 * \right) + \Gamma (1-s)\sum_i \sigma_i^x, \f] where \f$ \Gamma \f$ shows the
 * strength of transverse field (referred to as `gamma`) and \f$ s \f$ shows the
 * annealing schedule (referred to as `s`).
 *
 * Quantum Monte Carlo simulation is performed under the finite (inverse)
 * temperature (\f$ \beta \f$ or `beta`).
 *
 * @tparam FloatType
 */
template <typename FloatType> struct TransverseIsing<graph::CSRSparse<FloatType>> {
  using system_type = transverse_field_system;

  // matrix (row major)
  using SparseMatrixXx = Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;
  // trotter matrix (col major)
  using TrotterMatrix =
      Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  /**
   * @brief TransverseIsing Constructor
   *
   * @param init_trotter_spins
   * @param init_interaction
   */
  TransverseIsing(const TrotterSpins &init_trotter_spins,
                  const graph::CSRSparse<FloatType> &init_interaction,
                  FloatType gamma)
      : trotter_spins(
            utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
                init_trotter_spins)),
        interaction(init_interaction.get_interactions()),
        num_classical_spins(init_trotter_spins[0].size()), gamma(gamma) {
    if (!(init_trotter_spins.size() >= 2)) {
      throw std::invalid_argument(
          "trotter slices must be equal or larger than 2.");
    }

    // initialize rand_pool
    rand_pool = TrotterMatrix(num_classical_spins + 1, trotter_spins.cols());

    // reset dE
    reset_dE();
  }

  /**
   * @brief TransverseIsing Constuctor with initial classical spins
   *
   * @param classical_spins initial classical spins
   * @param init_interaction
   * @param num_trotter_slices
   */
  TransverseIsing(const graph::Spins &init_classical_spins,
                  const graph::CSRSparse<FloatType> &init_interaction,
                  const FloatType gamma, const size_t num_trotter_slices)
      : interaction(init_interaction.get_interactions()),
        num_classical_spins(init_classical_spins.size()), gamma(gamma) {
    // initialize trotter_spins with classical_spins

    // assert(init_classical_spins.size() >= 2);
    if (!(num_trotter_slices >= 2)) {
      throw std::invalid_argument(
          "trotter slices must be equal or larger than 2.");
    }

    TrotterSpins init_trotter_spins(num_trotter_slices);

    for (auto &spins : init_trotter_spins) {
      spins = init_classical_spins;
    }

    // init trotter_spins
    trotter_spins =
        utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
            init_trotter_spins);

    // initialize rand_pool
    rand_pool = TrotterMatrix(num_classical_spins + 1, trotter_spins.cols());

    // reset dE
    reset_dE();
  }

  /**
   * @brief reset spins with trotter spins
   *
   * @param init_trotter_spins
   */
  void reset_spins(const TrotterSpins &init_trotter_spins) {
    this->trotter_spins =
        utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
            init_trotter_spins);

    // reset dE
    reset_dE();
  }

  /**
   * @brief reset spins with trotter spins
   *
   * @param classical_spins
   */
  void reset_spins(const graph::Spins &classical_spins) {
    TrotterSpins init_trotter_spins(
        trotter_spins.cols()); // cols -> num_trotter_slices

    for (auto &spins : init_trotter_spins) {
      spins = classical_spins;
    }
    // init trotter_spins
    this->trotter_spins =
        utility::gen_matrix_from_trotter_spins<FloatType, Eigen::ColMajor>(
            init_trotter_spins);

    // reset dE
    reset_dE();
  }

  /**
   * @brief reset dE
   */
  inline void reset_dE() {
    // reset dE and dEtrot

    // initialize dE (spatial direction)
    // we store only dE for spatial direction
    const auto &spins = this->trotter_spins;
    // std::size_t num_trotter_slices = this->trotter_spins.cols();

    this->dE = -2 * spins.cwiseProduct(this->interaction * spins);

    // for trotter direction
    // this->dEtrot = TrotterMatrix::Zero(num_classical_spins+1,
    // num_trotter_slices); for(std::size_t t=0; t<num_trotter_slices; t++){
    //    this->dEtrot.col(t) = -2 * spins.col(t).cwiseProduct(
    //            spins.col(mod_t((int64_t)t+1, num_trotter_slices)) +
    //            spins.col(mod_t((int64_t)t-1, num_trotter_slices))
    //            );
    //}

    // diffE = s * (beta/num_trotter_slices) * this->dE + (1/2.) *
    // log(tanh(beta* gamma * (1.0-s) /num_trotter_slices)) * this->dEtrot
  }

  inline static std::size_t mod_t(const std::int64_t a,
                                  const std::size_t num_trotter_slices) {
    // a -> [-1:num_trotter_slices]
    // return a%num_trotter_slices (a>0), num_trotter_slices-1 (a==-1)
    return (a + num_trotter_slices) % num_trotter_slices;
  }

  /**
   * @brief trotterlized spins
   */
  TrotterMatrix trotter_spins;

  /**
   * @brief interaction
   */
  const SparseMatrixXx interaction;

  /**
   * @brief number of real classical spins (dummy spin excluded)
   */
  std::size_t num_classical_spins; // trotter_spins[0].size()-1

  /**
   * @brief coefficient of transverse field term
   */
  FloatType gamma;

  /**
   * @brief rand_pool
   */
  TrotterMatrix rand_pool;

  /**
   * @brief trotterlized dE (spatial direction)
   */
  TrotterMatrix dE;
};

/**
 * @brief helper function for TransverseIsing constructor
 *
 * @tparam GraphType
 * @param init_trotter_spins
 * @param init_interaction
 * @param gamma
 *
 * @return generated object
 */
template <typename GraphType>
auto make_transverse_ising(const TrotterSpins &init_trotter_spins,
                           const GraphType &init_interaction,
                           const typename GraphType::value_type gamma) {
  return TransverseIsing<GraphType>(init_trotter_spins, init_interaction,
                                    gamma);
}

/**
 * @brief helper function for TransverseIsing constructor
 *
 * @tparam GraphType
 * @param classical_spins
 * @param init_interaction
 * @param gamma
 * @param num_trotter_slices
 *
 * @return generated object
 */
template <typename GraphType>
auto make_transverse_ising(const graph::Spins &classical_spins,
                           const GraphType &init_interaction,
                           const typename GraphType::value_type gamma,
                           const std::size_t num_trotter_slices) {
  return TransverseIsing<GraphType>(classical_spins, init_interaction, gamma,
                                    num_trotter_slices);
}
} // namespace system
} // namespace openjij
