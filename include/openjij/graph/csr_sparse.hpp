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
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "openjij/graph/graph.hpp"

#include <Eigen/Sparse>

#include "openjij/utility/disable_eigen_warning.hpp"

namespace openjij {
namespace graph {

/**
 * @brief CSRSparse graph: just store CSR Sparse Matrix (Eigen::Sparse)
 * The Hamiltonian is like
 * \f[
 * H = \sum_{i<j}J_{ij} \sigma_i \sigma_j + \sum_{i}h_{i} \sigma_i
 * \f]
 *
 * @tparam FloatType floating-point type
 */
template <typename FloatType> class CSRSparse : public Graph {
  static_assert(std::is_floating_point<FloatType>::value,
                "FloatType must be floating-point type.");

public:
  /**
   * @brief interaction type
   */
  using Interactions =
      Eigen::SparseMatrix<FloatType, Eigen::RowMajor>;

  /**
   * @brief float type
   */
  using value_type = FloatType;

private:
  /**
   * @brief interactions (the number of intereactions is
   * num_spins*(num_spins+1)/2).
   * 
   * The stored matrix has the following triangular form:
   *
   * \f[
   * \begin{pmatrix}
   * J_{0,0} & J_{0,1} & \cdots & J_{0,N-1} & h_{0}\\
   * 0 & J_{1,1} & \cdots & J_{1,N-1} & h_{1}\\
   * \vdots & \vdots & \vdots & \vdots & \vdots \\
   * 0 & 0 & \cdots & J_{N-1,N-1} & h_{N-1}\\
   * 0 & 0 & \cdots & 0 & 1 \\
   * \end{pmatrix}
   * \f]
   */
  Interactions _J;

public:

  /**
   * @brief CSRSparse constructor
   *
   * The input matrix must have the form:
   *
   * \f[
   * \begin{pmatrix}
   * J_{0,0} & J_{0,1} & \cdots & J_{0,N-1} & h_{0}\\
   * 0 & J_{1,1} & \cdots & J_{1,N-1} & h_{1}\\
   * \vdots & \vdots & \vdots & \vdots & \vdots \\
   * 0 & 0 & \cdots & J_{N-1,N-1} & h_{N-1}\\
   * 0 & 0 & \cdots & 0 & 1 \\
   * \end{pmatrix}
   * \f]
   *
   * @param interaction input matrix
   */
  explicit CSRSparse(const Interactions &interaction): Graph(interaction.rows()-1) {
    if (interaction.rows() != interaction.cols()) {
      std::runtime_error("interaction.rows() != interaction.cols()");
    }
    _J = interaction.template selfadjointView<Eigen::Upper>();
  } 

  /**
   * @brief CSRSparse copy constructor
   *
   */
  CSRSparse(const CSRSparse<FloatType> &) = default;

  /**
   * @brief CSRSparse move constructor
   *
   */
  CSRSparse(CSRSparse<FloatType> &&) = default;

  /**
   * @brief calculate total energy
   *
   * @param spins
   * @deprecated use energy(spins)
   *
   * @return corresponding energy
   */
  FloatType calc_energy(const Spins &spins) const {
    return this->energy(spins);
  }

  FloatType calc_energy(const Eigen::Matrix<FloatType, Eigen::Dynamic, 1,
                                            Eigen::ColMajor> &spins) const {
    return this->energy(spins);
  }

  /**
   * @brief calculate total energy
   *
   * @param spins
   *
   * @return corresponding energy
   */
  FloatType energy(const Spins &spins) const {
    if (spins.size() != this->get_num_spins()) {
      throw std::out_of_range("Out of range in energy in CSRSparse graph.");
    }

    using Vec = Eigen::Matrix<FloatType, Eigen::Dynamic, 1, Eigen::ColMajor>;
    Vec s(get_num_spins() + 1);
    for (size_t i = 0; i < spins.size(); i++) {
      s(i) = spins[i];
    }
    s(get_num_spins()) = 1;

    // the energy must be consistent with BinaryQuadraticModel.
    return (s.transpose() *
            (_J.template triangularView<Eigen::Upper>() * s))(0, 0) -
           1;
  }

  FloatType energy(const Eigen::Matrix<FloatType, Eigen::Dynamic, 1,
                                       Eigen::ColMajor> &spins) const {
    graph::Spins temp_spins(get_num_spins());
    for (size_t i = 0; i < temp_spins.size(); i++) {
      temp_spins[i] = spins(i);
    }
    return energy(temp_spins);
  }

  /**
   * @brief get interactions (Eigen Matrix)
   *
   * The returned matrix has the following symmetric form:
   *
   * \f[
   * \begin{pmatrix}
   * J_{0,0} & J_{0,1} & \cdots & J_{0,N-1} & h_{0}\\
   * J_{0,1} & J_{1,1} & \cdots & J_{1,N-1} & h_{1}\\
   * \vdots & \vdots & \vdots & \vdots & \vdots \\
   * J_{0,N-1} & J_{N-1,1} & \cdots & J_{N-1,N-1} & h_{N-1}\\
   * h_{0} & h_{1} & \cdots & h_{N-1} & 1 \\
   * \end{pmatrix}
   * \f]
   *
   * @return Eigen Matrix
   */
  const Interactions get_interactions() const {
    return this->_J.template selfadjointView<Eigen::Upper>();
  }
};
} // namespace graph
} // namespace openjij
